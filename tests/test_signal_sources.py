"""Small offline provider-shaped fixtures; no network, fits, or market registry."""

import io
import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from xml.sax.saxutils import escape

import pytest

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.forecast.forecast_inputs import SignalSnapshot
from quant_rsi.market import signal_sources as sources

NOW = "2026-10-27T02:00:00+00:00"
OLD = "0001045810-25-000001"
NEW = "0001045810-26-000075"


def spec(*ids):
    items = {
        "rss": {"source_id": "nvidia_newsroom_rss", "entity": "NVDA"},
        "sec": {
            "source_id": "sec_companyfacts",
            "entity": "NVDA",
            "cik": "1045810",
            "concepts": [{"taxonomy": "us-gaap", "concept": "Revenues", "unit": "USD"}],
        },
    }
    return {
        "schema_version": 1,
        "kind": "forecast_signal_collection",
        "start_at": "2023-10-20T00:00:00Z",
        "end_at": "2026-10-27T00:00:00Z",
        "sources": [items[i] for i in ids],
    }


def rss(items):
    return (
        "<rss><channel>"
        + "".join(
            "<item>" + "".join(f"<{k}>{escape(v)}</{k}>" for k, v in item.items()) + "</item>"
            for item in items
        )
        + "</channel></rss>"
    ).encode()


def item(guid="one", published="Wed, 21 Oct 2026 10:11:00 GMT"):
    return {
        "guid": guid,
        "title": "Official issuer news",
        "description": "Actual RSS summary",
        "link": "https://nvidianews.nvidia.com/news/sample",
        "pubDate": published,
    }


def sec():
    submissions = {
        "cik": "1045810",
        "tickers": ["NVDA"],
        "filings": {
            "recent": {
                "accessionNumber": [OLD, NEW],
                "acceptanceDateTime": ["2025-08-26T20:00:00Z", "2026-08-26T20:36:00Z"],
                "filingDate": ["2025-08-26", "2026-08-26"],
                "form": ["10-Q", "10-Q"],
                "primaryDocument": ["old.htm", "new.htm"],
            },
            "files": [{"name": "older-submissions.json"}],
        },
    }
    common = {"start": "2025-04-28", "end": "2025-07-27", "form": "10-Q", "fp": "Q2"}
    # A later filing can restate the same report period; preserve both documents.
    facts = {
        "cik": 1045810,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {**common, "accn": OLD, "filed": "2025-08-26", "val": 100, "fy": 2026},
                            {**common, "accn": NEW, "filed": "2026-08-26", "val": 101, "fy": 2027},
                        ]
                    }
                }
            }
        },
    }
    return submissions, facts


class Response(io.BytesIO):
    def __init__(self, body, url, status=200):
        super().__init__(body)
        self.status, self.url = status, url
        self.headers = {"Content-Type": "application/json"}

    def geturl(self):
        return self.url


def network(monkeypatch, mapping):
    calls = []

    def fetch(request, timeout):
        assert timeout == 15
        url = request.full_url
        calls.append(url)
        result = mapping[url]
        if isinstance(result, Exception):
            raise result
        return Response(result, url)

    monkeypatch.setattr(sources, "urlopen", fetch)
    monkeypatch.setattr(sources, "_now", lambda: NOW)
    return calls


def sec_mapping(submissions, facts):
    return {
        sources.SEC_BASE + "/submissions/CIK0001045810.json": canonical(submissions),
        sources.SEC_BASE + "/api/xbrl/companyfacts/CIK0001045810.json": canonical(facts),
    }


def test_collect_contract_evidence_and_real_receipt_clock(tmp_path, monkeypatch):
    sub, facts = sec()
    calls = network(monkeypatch, {sources.RSS_URL: rss([item()]), **sec_mapping(sub, facts)})
    result = sources.collect_signals(spec("rss", "sec"), tmp_path / "batch")
    snapshot = SignalSnapshot.model_validate_json(canonical(result["snapshot"]))
    assert len(calls) == 3 and len(snapshot.events) == 3
    assert all(e.first_seen_at == datetime.fromisoformat(NOW) for e in snapshot.events)
    assert all(e.available_at == datetime.fromisoformat(NOW) for e in snapshot.events)
    assert all(c.status == "partial" for c in snapshot.coverage)
    assert result["qualifications"]
    for ref, path in result["evidence_paths"].items():
        assert digest(Path(path).read_bytes()) == ref
    for event in snapshot.events:
        receipt = json.loads(Path(result["evidence_paths"][event.receipt_ref]).read_text())
        assert "events" not in receipt and "receipt_ref" not in receipt
        assert event.payload_ref in result["evidence_paths"]
        for record in receipt["responses"]:
            assert record["raw_ref"] in result["evidence_paths"]
    final = json.loads(Path(result["receipt_path"]).read_text())
    assert final["snapshot_ref"] in result["evidence_paths"]
    assert final["query_ref"] in result["evidence_paths"]


def test_sec_preserves_versions_and_does_not_invent_publication(tmp_path, monkeypatch):
    sub, facts = sec()
    network(monkeypatch, sec_mapping(sub, facts))
    result = sources.collect_signals(spec("sec"), tmp_path)
    events = result["snapshot"]["events"]
    assert {e["fields"]["value"] for e in events} == {"100", "101"}
    assert len({e["event_id"] for e in events}) == 2
    assert {e["fields"]["accession"] for e in events} == {OLD, NEW}
    for e in events:
        assert e["revision"] == 0
        assert e["published_at"] is e["published_date"] is e["source_available_at"] is None
        assert e["publication_precision"] == "unknown"
        assert e["report_period_end"] == "2025-07-27"
        assert e["units"]["value"] == "USD"
        assert e["fields"]["acceptance_at"]
        assert e["fields"]["filed_date"]
    # Selecting this later query cannot turn the old report period into old availability.
    q = spec("sec")
    q["start_at"] = "2026-01-01T00:00:00Z"
    second = sources.collect_signals(q, tmp_path / "later")
    assert [e["fields"]["accession"] for e in second["snapshot"]["events"]] == [NEW]


def test_rss_half_open_dates_unknown_and_dedup(tmp_path, monkeypatch):
    feed = rss(
        [
            item("start", "Fri, 20 Oct 2023 00:00:00 GMT"),
            item("end", "Tue, 27 Oct 2026 00:00:00 GMT"),
            item("date", "2026-10-21"),
            item("unknown", ""),
            item("date", "2026-10-21"),
        ]
    )
    network(monkeypatch, {sources.RSS_URL: feed})
    result = sources.collect_signals(spec("rss"), tmp_path)
    events = {e["fields"]["guid"]: e for e in result["snapshot"]["events"]}
    assert set(events) == {"start", "date", "unknown"}
    assert events["date"]["published_at"] is None
    assert events["date"]["published_date"] == "2026-10-21"
    assert events["date"]["publication_precision"] == "day"
    assert events["unknown"]["publication_precision"] == "unknown"


@pytest.mark.parametrize("change", ["cik", "ticker", "columns", "fact_metadata"])
def test_sec_identity_and_join_fail_closed(tmp_path, monkeypatch, change):
    sub, facts = sec()
    if change == "cik":
        facts["cik"] = 1234
    elif change == "ticker":
        sub["tickers"] = ["OTHER"]
    elif change == "columns":
        sub["filings"]["recent"]["acceptanceDateTime"].pop()
    else:
        facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["form"] = "10-K"
    network(monkeypatch, sec_mapping(sub, facts))
    result = sources.collect_signals(spec("sec"), tmp_path)
    assert result["snapshot"]["events"] == []
    assert result["snapshot"]["coverage"][0]["status"] == "unknown"
    assert "parse rejected" in result["snapshot"]["coverage"][0]["reason"]
    assert len(list(tmp_path.glob("*.raw"))) == 2


def test_missing_concept_and_missing_acceptance_are_not_complete(tmp_path, monkeypatch):
    sub, facts = sec()
    sub["filings"]["recent"]["acceptanceDateTime"][0] = ""
    network(monkeypatch, sec_mapping(sub, facts))
    q = spec("sec")
    q["sources"][0]["concepts"].append({"taxonomy": "us-gaap", "concept": "Absent", "unit": "USD"})
    result = sources.collect_signals(q, tmp_path)
    assert len(result["snapshot"]["events"]) == 2
    coverage = result["snapshot"]["coverage"][0]
    assert coverage["status"] == "partial"
    assert "missing concept/units" in coverage["reason"]
    assert "1 accessions" in coverage["reason"]


@pytest.mark.parametrize("failure", ["http", "oversize", "timeout", "xml", "conflicting_guid"])
def test_failed_source_preserves_reason_and_never_claims_empty_complete(
    tmp_path, monkeypatch, failure
):
    if failure == "http":
        value = HTTPError(sources.RSS_URL, 403, "Forbidden", {}, io.BytesIO(b"denied"))
    elif failure == "oversize":
        monkeypatch.setattr(sources, "MAX_RESPONSE_BYTES", 32)
        value = b"x" * 100
    elif failure == "timeout":
        value = TimeoutError("offline timeout")
    elif failure == "xml":
        value = b"<!DOCTYPE rss><rss><channel/></rss>"
    else:
        changed = item()
        changed["title"] = "conflicting"
        value = rss([item(), changed])
    network(monkeypatch, {sources.RSS_URL: value})
    result = sources.collect_signals(spec("rss"), tmp_path)
    assert result["snapshot"]["events"] == []
    c = result["snapshot"]["coverage"][0]
    assert c["status"] == "unknown" and c["reason"]
    receipt = json.loads(Path(result["evidence_paths"][c["receipt_ref"]]).read_text())
    assert receipt["responses"][0]["received_at"] == NOW
    if failure == "http":
        assert (tmp_path / "nvidia_newsroom_rss-0.raw").read_bytes() == b"denied"
    if failure == "oversize":
        assert (tmp_path / "nvidia_newsroom_rss-0.raw").stat().st_size == 33
        assert receipt["responses"][0]["truncated"]


def test_decimal_fact_extraction_retains_literal_precision(tmp_path, monkeypatch):
    sub, facts = sec()
    mapping = sec_mapping(sub, facts)
    url = sources.SEC_BASE + "/api/xbrl/companyfacts/CIK0001045810.json"
    mapping[url] = mapping[url].replace(b'"val":100', b'"val":1.23456789123456789')
    network(monkeypatch, mapping)
    result = sources.collect_signals(spec("sec"), tmp_path)
    assert result["snapshot"]["events"][0]["fields"]["value"] == "1.23456789123456789"


@pytest.mark.parametrize("change", ["naive", "reversed", "unknown", "entity", "cik", "duplicate"])
def test_invalid_spec_before_network_or_output(tmp_path, monkeypatch, change):
    q = spec("sec")
    if change == "naive":
        q["start_at"] = "2023-10-20"
    elif change == "reversed":
        q["end_at"] = q["start_at"]
    elif change == "unknown":
        q["extra"] = True
    elif change == "entity":
        q = spec("rss")
        q["sources"][0]["entity"] = "OTHER"
    elif change == "cik":
        q["sources"][0]["cik"] = "../bad"
    else:
        q["sources"][0]["concepts"] *= 2
    calls = network(monkeypatch, {})
    with pytest.raises(ValueError):
        sources.collect_signals(q, tmp_path / "new")
    assert calls == [] and not (tmp_path / "new").exists()


def test_nonempty_output_cannot_overwrite(tmp_path, monkeypatch):
    (tmp_path / "existing").write_text("preserve")
    calls = network(monkeypatch, {})
    with pytest.raises(FileExistsError):
        sources.collect_signals(spec("rss"), tmp_path)
    assert calls == [] and (tmp_path / "existing").read_text() == "preserve"


def test_empty_valid_feed_is_partial_not_complete(tmp_path, monkeypatch):
    network(monkeypatch, {sources.RSS_URL: rss([])})
    result = sources.collect_signals(spec("rss"), tmp_path)
    assert result["snapshot"]["events"] == []
    assert result["snapshot"]["coverage"][0]["status"] == "partial"


def test_nullable_sec_fiscal_metadata_preserved_without_guessing(tmp_path, monkeypatch):
    sub, facts = sec()
    facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0].update(fp=None, fy=None, frame=None)
    network(monkeypatch, sec_mapping(sub, facts))
    result = sources.collect_signals(spec("sec"), tmp_path)
    event = result["snapshot"]["events"][0]
    for key in ["fiscal_year", "fiscal_period", "frame"]:
        assert key not in event["fields"] and key not in event["units"]
    payload = json.loads(Path(result["evidence_paths"][event["payload_ref"]]).read_text())
    assert payload["fact"]["fp"] is payload["fact"]["fy"] is payload["fact"]["frame"] is None


def test_invalid_event_field_is_source_failure_with_receipt(tmp_path, monkeypatch):
    sub, facts = sec()
    facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"][0]["fp"] = ["Q2"]
    network(monkeypatch, sec_mapping(sub, facts))
    result = sources.collect_signals(spec("sec"), tmp_path)
    assert result["snapshot"]["events"] == []
    assert result["snapshot"]["coverage"][0]["status"] == "unknown"
    assert "parse rejected" in result["snapshot"]["coverage"][0]["reason"]
    assert Path(result["receipt_path"]).is_file()


def test_receipt_clock_waits_for_both_sec_responses(tmp_path, monkeypatch):
    sub, facts = sec()
    network(monkeypatch, sec_mapping(sub, facts))
    clocks = iter(f"2026-10-27T02:00:0{i}+00:00" for i in range(5))
    monkeypatch.setattr(sources, "_now", lambda: next(clocks))
    result = sources.collect_signals(spec("sec"), tmp_path)
    for event in result["snapshot"]["events"]:
        assert datetime.fromisoformat(event["first_seen_at"]).second == 3


def test_date_only_boundary_is_conservative_not_utc_midnight(tmp_path, monkeypatch):
    network(monkeypatch, {sources.RSS_URL: rss([item("unknown-zone", "2026-10-27")])})
    result = sources.collect_signals(spec("rss"), tmp_path)
    event = result["snapshot"]["events"][0]
    assert event["published_at"] is None and event["published_date"] == "2026-10-27"
    assert event["publication_precision"] == "day"
