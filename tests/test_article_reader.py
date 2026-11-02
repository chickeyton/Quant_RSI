"""Offline public-page extraction and transport boundary tests; no network access."""

import gzip
import hashlib
import socket
from datetime import datetime
from email.message import Message

import pytest

from quant_rsi.research import article_reader as reader


class Response:
    def __init__(self, body=b"A readable public article.", status=200, **headers):
        self.status = status
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = headers.pop("Content-Type", "text/html; charset=utf-8")
        for key, value in headers.items():
            self.headers[key] = value

    def read(self, limit):
        return self.body[:limit]


@pytest.fixture
def transport(monkeypatch):
    calls = {"dns": [], "connect": [], "tls": [], "request": [], "responses": []}

    def resolve(host, port, *, type):
        calls["dns"].append((host, port))
        ip = "127.0.0.1" if host == "private.test" else "93.184.216.34"
        return [(socket.AF_INET, type, socket.IPPROTO_TCP, "", (ip, port))]

    class Socket:
        def __init__(self, *args):
            pass

        def settimeout(self, timeout):
            assert timeout == reader.TIMEOUT

        def connect(self, address):
            calls["connect"].append(address)

        def close(self):
            pass

    class TLS:
        def wrap_socket(self, sock, *, server_hostname):
            calls["tls"].append(server_hostname)
            return sock

    class Connection:
        def __init__(self, host, port, *, timeout):
            self.sock = None

        def request(self, method, path, headers):
            assert self.sock is not None  # A prevalidated, connected socket is supplied.
            calls["request"].append((method, path, headers))

        def getresponse(self):
            return calls["responses"].pop(0)

        def close(self):
            pass

    monkeypatch.setattr(reader.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(reader.socket, "socket", Socket)
    monkeypatch.setattr(reader.ssl, "create_default_context", TLS)
    monkeypatch.setattr(reader.http.client, "HTTPConnection", Connection)
    return calls


def test_html_prefers_article_and_pins_public_address_with_tls_hostname(transport):
    body = b"""<head><title>Page chrome</title></head><nav>Menu</nav>
    <main>Related stories<article><header>Share</header><h1>Earnings &amp; outlook</h1>
    <p>Demand grew.<script>steal secrets</script><style>css</style></p>
    <footer>Subscribe</footer></article>Other links</main>"""
    transport["responses"].append(Response(body))
    result = reader.read_article("https://public.test/news?id=1#section")
    assert result["status"] == "available"
    assert result["text"] == "Earnings & outlook Demand grew."
    assert not result["text_truncated"]
    assert result["content_sha256"] == hashlib.sha256(body).hexdigest()
    assert datetime.fromisoformat(result["received_at"]).tzinfo is not None
    assert transport["dns"] == [("public.test", 443)]
    assert transport["connect"] == [("93.184.216.34", 443)]
    assert transport["tls"] == ["public.test"]
    assert transport["request"][0][1] == "/news?id=1"
    assert transport["request"][0][2]["Accept-Encoding"] == "gzip, identity"
    assert "not_verified_full_article" in result["content_extent"]


@pytest.mark.parametrize(
    "body,kind,expected",
    [
        (b"<main>Useful <b>report</b></main> Footer", "text/html", "Useful report"),
        (b"<p>Body only</p><nav>Links</nav>", "text/html", "Body only"),
        (b"Public\ntext\tarticle", "text/plain", "Public text article"),
        (b"<article>Actual text</article>", "application/xhtml+xml", "Actual text"),
    ],
)
def test_supported_content_and_text_fallback(transport, body, kind, expected):
    transport["responses"].append(Response(body, **{"Content-Type": kind}))
    assert reader.read_article("http://public.test/report")["text"] == expected
    assert transport["tls"] == []


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://public.test/article",
        "https://user:secret@public.test/news",
        "https://public.test:8443/news",
        "http://public.test:443/news",
        "https://[fe80::1%25eth0]/news",
        "https://public.test/\nInjected",
        "https://public.test/" + "a" * 2048,
    ],
)
def test_unsafe_urls_do_not_resolve_or_leak_credentials(transport, url):
    result = reader.read_article(url)
    assert result["status"] == "unavailable" and result["reason"] == "unsafe_url"
    assert result["url"] is None and result["final_url"] is None
    assert "secret" not in str(result)
    assert transport["dns"] == [] and transport["connect"] == []


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "224.0.0.1"])
def test_non_public_dns_answers_are_blocked_before_connect(monkeypatch, transport, ip):
    monkeypatch.setattr(
        reader.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))],
    )
    result = reader.read_article("https://public.test/report")
    assert result["reason"] == "non_public_address"
    assert transport["connect"] == []


def test_redirect_is_read_and_every_hop_resolved_before_connect(transport):
    transport["responses"].extend(
        [Response(status=302, Location="/new"), Response(b"<main>Updated article</main>")]
    )
    result = reader.read_article("https://public.test/old")
    assert result["url"] == "https://public.test/old"
    assert result["final_url"] == "https://public.test/new"
    assert result["text"] == "Updated article"
    assert transport["dns"] == [("public.test", 443), ("public.test", 443)]


def test_redirect_cannot_connect_to_private_network(transport):
    transport["responses"].append(Response(status=302, Location="http://private.test/admin"))
    result = reader.read_article("https://public.test/redirect")
    assert result["reason"] == "non_public_address"
    assert transport["connect"] == [("93.184.216.34", 443)]
    assert result["text"] == "" and result["content_sha256"] is None


def test_redirect_budget_is_bounded(transport):
    transport["responses"] = [Response(status=302, Location="/again") for _ in range(4)]
    result = reader.read_article("https://public.test/redirect")
    assert result["reason"] == "redirect_limit"
    assert len(transport["request"]) == 4


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(status=403), "http_403"),
        (Response(status=302), "redirect_without_location"),
        (Response(**{"Content-Type": "application/pdf"}), "unsupported_content_type"),
        (Response(**{"Content-Encoding": "br"}), "unsupported_content_encoding"),
        (Response(**{"Content-Encoding": "gzip"}), "invalid_compressed_body"),
        (Response(**{"Content-Type": "text/plain; charset=unknown-codec"}), "unsupported_charset"),
        (Response(b"x" * (reader.MAX_BYTES + 1)), "response_too_large"),
        (Response(b"<script>Only javascript</script>"), "no_readable_text"),
    ],
)
def test_unavailable_pages_retain_reason_without_invented_text(transport, response, reason):
    transport["responses"].append(response)
    result = reader.read_article("https://public.test/report")
    assert result["status"] == "unavailable"
    assert result["reason"] == reason
    assert result["text"] == "" and result["content_sha256"] is None
    assert result["received_at"].endswith("Z")


def test_text_cap_is_explicit_and_hash_covers_received_body(transport):
    body = b"<main>" + b"x" * (reader.MAX_TEXT + 50) + b"</main>"
    transport["responses"].append(Response(body))
    result = reader.read_article("https://public.test/report")
    assert result["status"] == "available" and result["text_truncated"]
    assert len(result["text"]) == reader.MAX_TEXT
    assert result["content_sha256"] == hashlib.sha256(body).hexdigest()


def test_gzip_body_is_bounded_and_hash_identifies_received_compressed_bytes(transport):
    body = gzip.compress(b"<article>Issuer outlook raised.</article>")
    transport["responses"].append(Response(body, **{"Content-Encoding": "gzip"}))
    result = reader.read_article("https://public.test/news")
    assert result["text"] == "Issuer outlook raised."
    assert result["content_sha256"] == hashlib.sha256(body).hexdigest()
    bomb = gzip.compress(b"a" * (reader.MAX_BYTES + 1))
    transport["responses"].append(Response(bomb, **{"Content-Encoding": "gzip"}))
    result = reader.read_article("https://public.test/oversized")
    assert result["reason"] == "decompressed_response_too_large"
    assert result["text"] == "" and result["content_sha256"] is None


def test_failed_redirect_retains_actual_host_for_source_recovery(transport):
    transport["responses"].extend(
        [Response(status=302, Location="https://publisher.test/story"), Response(status=403)]
    )
    result = reader.read_article("https://gateway.test/story")
    assert result["reason"] == "http_403"
    assert result["final_url"] == "https://publisher.test/story"


def test_network_failure_is_reported_without_exception_details(monkeypatch, transport):
    def fail(*args, **kwargs):
        raise OSError("a credential or internal endpoint must not appear in the report")

    monkeypatch.setattr(reader.socket, "getaddrinfo", fail)
    result = reader.read_article("https://public.test/report")
    assert result["reason"] == "request_failed"
    assert "credential" not in str(result)
