import io
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import pytest

from quant_rsi import forecast_context_collection as collection
from quant_rsi import forecast_context_data as adapter
from quant_rsi.artifacts import canonical, decode, digest

NOW = datetime(2025, 1, 3, 22, tzinfo=timezone.utc)
FIRST = {"target": "SOXL", "industry": "XSD", "market": "SPY"}
SECOND = {"target": "XLE", "industry": "VDE", "market": "SPY"}


def chart(ticker):
    return canonical(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "symbol": ticker,
                            "currency": "USD",
                            "instrumentType": "ETF",
                            "exchangeName": "PCX",
                            "exchangeTimezoneName": "America/New_York",
                            "dataGranularity": "1d",
                        },
                        "timestamp": [1735828200, 1735914600],
                        "indicators": {
                            "quote": {
                                # Replaced below to keep this fixture explicit.
                            }
                        },
                        "events": {
                            "dividends": {"old": {"amount": 0.093, "date": 1704205800}},
                            "capitalGains": {"unknown": {"amount": 1, "date": 1704205800}},
                        },
                    }
                ],
            }
        }
    )


def response_bodies(roles):
    result = {name: b"<html>Official source fixture</html>" for name in collection.CALENDAR_SOURCES}
    for ticker in roles.values():
        value = decode(chart(ticker))
        value["chart"]["result"][0]["indicators"]["quote"] = [
            {"open": [30.1, None], "close": [31.2, None]}
        ]
        result[f"{ticker}_chart_5y.json"] = canonical(value)
    return result


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, url):
        super().__init__(body)
        self.url = url
        self.headers = {"Content-Type": "application/json", "ETag": "fixture"}

    def geturl(self):
        return self.url


@pytest.fixture
def http(monkeypatch):
    bodies = {}
    calls = []
    failures = {}
    monkeypatch.setattr(collection, "_now", lambda: NOW)

    def get(request, timeout):
        assert timeout == 15
        url = request.full_url
        calls.append(url)
        item = failures.get(url)
        if item is not None:
            raise item
        return Response(bodies[url], url)

    monkeypatch.setattr(collection, "urlopen", get)

    def configure(roles):
        source_bodies = response_bodies(roles)
        for source in collection._sources(roles):
            bodies[source["url"]] = source_bodies[source["name"]]
        return source_bodies

    return configure, calls, failures, bodies


def filemap(root):
    return {
        p.relative_to(root).as_posix(): digest(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("roles", [FIRST, SECOND])
def test_profiles_produce_adapter_manifest_and_reuse_without_http(tmp_path, http, roles):
    configure, calls, _, _ = http
    bodies = configure(roles)
    root = collection.collect_context(tmp_path / "audit", roles=roles)
    assert isinstance(root, Path)
    config = decode((root / "collection.json").read_bytes())
    assert config["roles"] == roles
    assert config["roles_ref"] == digest(canonical(roles))
    manifest = decode((root / "manifest.json").read_bytes())
    members = {row["path"]: row for row in manifest["files"]}
    evidence = tmp_path / "adapter-evidence"
    (evidence / "evidence").mkdir(parents=True)
    for source in collection._sources(roles):
        name = source["name"]
        assert (root / "raw" / name).read_bytes() == bodies[name]
        body, receipt, _, _ = adapter._response(root, members, name, source["url"], evidence, {})
        assert body == bodies[name]
        assert receipt["requested_at"] == receipt["received_at"] == NOW.isoformat()
        assert receipt["headers"]["ETag"] == "fixture"
        assert receipt["roles_ref"] == config["roles_ref"]
    assert len(calls) == 7
    before = filemap(root)
    assert collection.collect_context(root, roles=roles) == root
    assert len(calls) == 7 and filemap(root) == before


def test_failure_and_interruption_resume_reuse_prior_success(tmp_path, http):
    configure, calls, failures, _ = http
    configure(FIRST)
    sources = collection._sources(FIRST)
    failures[sources[1]["url"]] = KeyboardInterrupt("interrupted")
    root = tmp_path / "audit"
    with pytest.raises(KeyboardInterrupt):
        collection.collect_context(root, roles=FIRST)
    first_raw = (root / "raw" / sources[0]["name"]).read_bytes()
    failed = root / "attempts" / sources[1]["name"] / "0001" / "receipt.json"
    failed_body = failed.read_bytes()
    assert decode(failed_body)["error"]["type"] == "KeyboardInterrupt"
    assert not (root / "manifest.json").exists()
    del failures[sources[1]["url"]]
    collection.collect_context(root, roles=FIRST)
    assert calls.count(sources[0]["url"]) == 1
    assert calls.count(sources[1]["url"]) == 2
    assert (root / "raw" / sources[0]["name"]).read_bytes() == first_raw
    assert failed.read_bytes() == failed_body
    assert any(
        row["path"] == str(failed.relative_to(root))
        for row in decode((root / "manifest.json").read_bytes())["files"]
    )


def test_no_receipt_is_fabricated_for_orphan_response(tmp_path, monkeypatch, http):
    configure, calls, _, _ = http
    configure(FIRST)
    root = tmp_path / "audit"
    original = collection._write_new
    failed_once = False

    def interrupt(path, body):
        nonlocal failed_once
        if path.name == "receipt.json" and not failed_once:
            failed_once = True
            raise KeyboardInterrupt("crash after body, before receipt")
        original(path, body)

    monkeypatch.setattr(collection, "_write_new", interrupt)
    with pytest.raises(KeyboardInterrupt):
        collection.collect_context(root, roles=FIRST)
    first = root / "attempts" / "SOXL_chart_5y.json" / "0001"
    raw = (first / "body").read_bytes()
    assert not (first / "receipt.json").exists()
    monkeypatch.setattr(collection, "_write_new", original)
    collection.collect_context(root, roles=FIRST)
    assert (first / "body").read_bytes() == raw
    assert not (first / "receipt.json").exists()
    source = collection._sources(FIRST)[0]
    assert calls.count(source["url"]) == 2
    published = (root / "SOXL_chart_5y.json.receipt.json").read_bytes()
    assert published == (first.parent / "0002" / "receipt.json").read_bytes()


def test_committed_response_publication_recovers_without_recollection(tmp_path, monkeypatch, http):
    configure, calls, _, _ = http
    configure(FIRST)
    root = tmp_path / "audit"
    original = collection._write_new

    def interrupt(path, body):
        if path == root / "SOXL_chart_5y.json.receipt.json":
            raise KeyboardInterrupt("crash between audit copies")
        original(path, body)

    monkeypatch.setattr(collection, "_write_new", interrupt)
    with pytest.raises(KeyboardInterrupt):
        collection.collect_context(root, roles=FIRST)
    receipt = (root / "attempts/SOXL_chart_5y.json/0001/receipt.json").read_bytes()
    assert (root / "raw/SOXL_chart_5y.json").exists()
    monkeypatch.setattr(collection, "_write_new", original)
    collection.collect_context(root, roles=FIRST)
    assert calls.count(collection._sources(FIRST)[0]["url"]) == 1
    assert (root / "SOXL_chart_5y.json.receipt.json").read_bytes() == receipt


def test_failure_attempt_budget_is_persistent(tmp_path, http):
    configure, calls, failures, _ = http
    configure(FIRST)
    source = collection._sources(FIRST)[0]
    failures[source["url"]] = TimeoutError("offline timeout")
    root = tmp_path / "audit"
    for _ in range(3):
        with pytest.raises(ValueError, match="Context source failed"):
            collection.collect_context(root, roles=FIRST)
    before = filemap(root)
    with pytest.raises(ValueError, match="budget exhausted"):
        collection.collect_context(root, roles=FIRST)
    assert len(calls) == 3 and filemap(root) == before


@pytest.mark.parametrize("member", ["raw/SOXL_chart_5y.json", "manifest.json", "collection.json"])
def test_changed_or_corrupt_initialized_evidence_is_rejected_without_http(tmp_path, http, member):
    configure, calls, _, _ = http
    configure(FIRST)
    root = collection.collect_context(tmp_path / "audit", roles=FIRST)
    path = root / member
    path.write_bytes(b"{}")
    before = filemap(root)
    with pytest.raises((ValueError, KeyError)):
        collection.collect_context(root, roles=FIRST)
    assert len(calls) == 7 and filemap(root) == before


def test_roles_drift_is_rejected_before_http_or_writes(tmp_path, http):
    configure, calls, _, _ = http
    configure(FIRST)
    root = collection.collect_context(tmp_path / "audit", roles=FIRST)
    before = filemap(root)
    with pytest.raises(ValueError, match="roles or source policy changed"):
        collection.collect_context(root, roles=SECOND)
    assert len(calls) == 7 and filemap(root) == before


@pytest.mark.parametrize("roles", [{}, {**FIRST, "extra": "QQQ"}, {**FIRST, "target": "../SPY"}])
def test_invalid_roles_do_not_initialize_output(tmp_path, http, roles):
    with pytest.raises(ValueError):
        collection.collect_context(tmp_path / "audit", roles=roles)
    assert not (tmp_path / "audit").exists()
    assert http[1] == []


@pytest.mark.parametrize("mode", ["wrong_symbol", "unsupported_exchange", "oversize", "http"])
def test_bad_responses_are_preserved_and_not_successful(tmp_path, http, mode):
    configure, _, failures, bodies = http
    configure(FIRST)
    source = collection._sources(FIRST)[0]
    if mode in {"wrong_symbol", "unsupported_exchange"}:
        data = decode(bodies[source["url"]])
        meta = data["chart"]["result"][0]["meta"]
        meta["symbol" if mode == "wrong_symbol" else "exchangeName"] = "WRONG"
        bodies[source["url"]] = canonical(data)
    elif mode == "oversize":
        bodies[source["url"]] = b"x" * (collection.MAX_RESPONSE_BYTES + 10)
    else:
        failures[source["url"]] = HTTPError(
            source["url"], 503, "offline unavailable", {}, io.BytesIO(b"unavailable")
        )
    root = tmp_path / "audit"
    with pytest.raises(ValueError, match="Context source failed"):
        collection.collect_context(root, roles=FIRST)
    attempt = root / "attempts" / source["name"] / "0001"
    receipt = decode((attempt / "receipt.json").read_bytes())
    body = (attempt / "body").read_bytes()
    assert receipt["error"] and receipt["sha256"] == digest(body)
    assert receipt["bytes"] == len(body) <= collection.MAX_RESPONSE_BYTES + 1
    assert not (root / "raw" / source["name"]).exists()
    assert not (root / "manifest.json").exists()


def test_shared_explicit_ticker_is_fetched_once(tmp_path, http):
    roles = {"target": "SPY", "industry": "SPY", "market": "SPY"}
    configure, calls, _, _ = http
    configure(roles)
    root = collection.collect_context(tmp_path / "audit", roles=roles)
    assert len(calls) == 5
    assert decode((root / "collection.json").read_bytes())["roles"] == roles


def test_receipt_without_preceding_published_body_is_corruption(tmp_path, monkeypatch, http):
    configure, calls, _, _ = http
    configure(FIRST)
    root = tmp_path / "audit"
    original = collection._write_new

    def interrupt(path, body):
        if path.name == "manifest.json":
            raise KeyboardInterrupt("before final manifest")
        original(path, body)

    monkeypatch.setattr(collection, "_write_new", interrupt)
    with pytest.raises(KeyboardInterrupt):
        collection.collect_context(root, roles=FIRST)
    (root / "raw/SOXL_chart_5y.json").unlink()
    before = filemap(root)
    monkeypatch.setattr(collection, "_write_new", original)
    with pytest.raises(ValueError, match="published receipt has no preceding body"):
        collection.collect_context(root, roles=FIRST)
    assert len(calls) == 7 and filemap(root) == before


@pytest.mark.parametrize("phase", ["collection", "intent", "body", "raw", "receipt", "manifest"])
def test_atomic_write_remnants_survive_recovery_without_fabricated_receipts(
    tmp_path, monkeypatch, http, phase
):
    configure, calls, _, _ = http
    configure(FIRST)
    root = tmp_path / "collection"
    original = collection.atomic_write
    remnants = []

    def interrupted(path, body):
        chosen = {
            "collection": path.name == "collection.json",
            "intent": path.name == "intent.json",
            "body": path.name == "body",
            "raw": path.parent.name == "raw",
            "receipt": path.name.endswith(".receipt.json"),
            "manifest": path.name == "manifest.json",
        }[phase]
        if chosen and not remnants:
            path.parent.mkdir(parents=True, exist_ok=True)
            remnant = path.parent / ".pending-interrupted"
            remnant.write_bytes(body[: max(1, len(body) // 2)])
            remnants.append(remnant)
            raise SystemExit("simulated process death during atomic write")
        original(path, body)

    monkeypatch.setattr(collection, "atomic_write", interrupted)
    with pytest.raises(SystemExit, match="process death"):
        collection.collect_context(root, roles=FIRST)
    saved = remnants[0].read_bytes()
    result = collection.collect_context(root, roles=FIRST)
    assert remnants[0].read_bytes() == saved
    count = len(collection._sources(FIRST))
    assert len(calls) == count + (1 if phase == "body" else 0)
    manifest = decode((result / "manifest.json").read_bytes())
    assert any(
        row["path"] == remnants[0].relative_to(root).as_posix() and row["sha256"] == digest(saved)
        for row in manifest["files"]
    )
    before = filemap(root)
    collection.collect_context(root, roles=FIRST)
    assert filemap(root) == before


def test_uncommitted_symlink_is_not_accepted_as_crash_evidence(tmp_path, http):
    root = tmp_path / "collection"
    root.mkdir()
    (root / ".pending-invalid").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="uncommitted"):
        collection.collect_context(root, roles=FIRST)
    assert not http[1]
