"""Bounded public-page reads for current research context, never historical training data."""

import gzip
import hashlib
import http.client
import io
import ipaddress
import socket
import ssl
import zlib
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

MAX_BYTES = 1_048_576
MAX_TEXT = 16_000
TIMEOUT = 15
MAX_REDIRECTS = 3


class _Unavailable(ValueError):
    pass


def _url(value):
    if not isinstance(value, str) or len(value) > 2048 or any(ord(c) < 33 for c in value):
        raise _Unavailable("unsafe_url")
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or "%" in parsed.hostname
            or parsed.port not in {None, 443 if parsed.scheme == "https" else 80}
        ):
            raise _Unavailable("unsafe_url")
        parsed.hostname.encode("idna")
    except (ValueError, UnicodeError):
        raise _Unavailable("unsafe_url") from None
    return parsed


def _public_target(parsed):
    port = 443 if parsed.scheme == "https" else 80
    addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise _Unavailable("request_failed")
    for _, _, _, _, address in addresses:
        ip = ipaddress.ip_address(address[0])
        if not ip.is_global or ip.is_multicast:
            raise _Unavailable("non_public_address")
    # Prefer IPv4 when both are available; connect to the validated address, not the hostname.
    return min(addresses, key=lambda item: item[0] != socket.AF_INET)


def _fetch(url):
    parsed = _url(url)
    family, kind, protocol, _, address = _public_target(parsed)
    connection = http.client.HTTPConnection(parsed.hostname, address[1], timeout=TIMEOUT)
    sock = socket.socket(family, kind, protocol)
    try:
        sock.settimeout(TIMEOUT)
        sock.connect(address)
        if parsed.scheme == "https":
            sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parsed.hostname)
        connection.sock = sock
        path = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request(
            "GET",
            path,
            headers={
                "User-Agent": "QuantRSI/0.1 (public research article reader)",
                "Accept": "text/html,text/plain,application/xhtml+xml",
                "Accept-Encoding": "gzip, identity",
            },
        )
        response = connection.getresponse()
        if response.status in {301, 302, 303, 307, 308}:
            return response.status, response.headers, b""
        if response.status != 200:
            raise _Unavailable(f"http_{response.status}")
        if response.headers.get_content_type() not in {
            "text/html",
            "text/plain",
            "application/xhtml+xml",
        }:
            raise _Unavailable("unsupported_content_type")
        if response.headers.get("Content-Encoding", "identity").lower() not in {"identity", "gzip"}:
            raise _Unavailable("unsupported_content_encoding")
        body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise _Unavailable("response_too_large")
        return response.status, response.headers, body
    finally:
        connection.close()
        sock.close()


class _ArticleText(HTMLParser):
    _ignored = {
        "head",
        "script",
        "style",
        "nav",
        "footer",
        "header",
        "noscript",
        "svg",
        "form",
        "aside",
        "template",
    }
    _void = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = {"article": [], "main": [], "body": []}

    def handle_starttag(self, tag, attrs):
        if tag not in self._void:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.stack:
            index = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[index:]

    def handle_data(self, data):
        if self._ignored.intersection(self.stack):
            return
        self.parts["body"].append(data)
        for tag in ("article", "main"):
            if tag in self.stack:
                self.parts[tag].append(data)

    def text(self):
        for tag in ("article", "main", "body"):
            value = " ".join(" ".join(self.parts[tag]).split())
            if value:
                return value
        return ""


def read_article(url):
    """Read at most one bounded public page, following at most three checked redirects.

    An available page is extracted text, not proof of completeness, publication
    time, source accuracy, or a verified relationship. No browser/JS or paywall bypass.
    """
    result = {
        "url": None,
        "final_url": None,
        "status": "unavailable",
        "text": "",
        "text_truncated": False,
        "content_sha256": None,
        "content_extent": "bounded_public_page_text_not_verified_full_article",
    }
    try:
        _url(url)
        result["url"] = url
        current = url
        for redirects in range(MAX_REDIRECTS + 1):
            _url(current)
            result["final_url"] = current
            status, headers, body = _fetch(current)
            if status in {301, 302, 303, 307, 308}:
                if redirects == MAX_REDIRECTS:
                    raise _Unavailable("redirect_limit")
                location = headers.get("Location")
                if not location:
                    raise _Unavailable("redirect_without_location")
                current = urljoin(current, location)
                continue
            content_hash = hashlib.sha256(body).hexdigest()
            if headers.get("Content-Encoding", "identity").lower() == "gzip":
                try:
                    with gzip.GzipFile(fileobj=io.BytesIO(body)) as compressed:
                        body = compressed.read(MAX_BYTES + 1)
                except (OSError, EOFError, zlib.error):
                    raise _Unavailable("invalid_compressed_body") from None
                if len(body) > MAX_BYTES:
                    raise _Unavailable("decompressed_response_too_large")
            charset = headers.get_content_charset() or "utf-8"
            try:
                decoded = body.decode(charset, errors="replace")
            except LookupError:
                raise _Unavailable("unsupported_charset") from None
            if headers.get_content_type() == "text/plain":
                text = " ".join(decoded.split())
            else:
                parser = _ArticleText()
                parser.feed(decoded)
                parser.close()
                text = parser.text()
            if not text:
                raise _Unavailable("no_readable_text")
            result.update(
                final_url=current,
                status="available",
                text=text[:MAX_TEXT],
                text_truncated=len(text) > MAX_TEXT,
                content_sha256=content_hash,
            )
            break
    except _Unavailable as exc:
        result["reason"] = str(exc)
    except (OSError, http.client.HTTPException, ValueError):
        result["reason"] = "request_failed"
    result["received_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return result
