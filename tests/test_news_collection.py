"""Prospective collection mechanics with invented RSS and a mocked network transport."""

import json
from html import escape
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from quant_rsi.core.artifacts import ArtifactStore, canonical
from quant_rsi.factors import lifecycle
from quant_rsi.market import market_contracts, news_collection, news_feed

TITLE = "The Coca-Cola Company Reports Second Quarter 2025 Results"
START = "2025-07-01T00:00:00Z"
SEEN = "2025-07-08T00:00:00Z"


def rss(title=TITLE, *, guid="release-1", pub_date="Thu, 03 Jul 2025 12:00:00 GMT"):
    item = (
        f"<item><guid>{escape(guid)}</guid><title>{escape(title)}</title>"
        "<link>https://investors.coca-colacompany.com/release/1</link>"
        f"<pubDate>{escape(pub_date)}</pubDate></item>"
    )
    return f"<rss><channel>{item}</channel></rss>".encode()


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    clock, responses, calls = [SEEN], [rss()], []
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: clock[0])

    def fetch(source, limit):
        calls.append((source.url, limit))
        value = responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value, dict(received_at=clock[0], http_status=200, response_url=source.url)

    monkeypatch.setattr(news_feed, "fetch", fetch)
    plan = news_collection.CollectionPlan.model_validate_json(
        canonical(dict(campaign_id="issuer-feed", start_at=START, max_requests=4))
    )
    news_collection.initialize(plan)
    store = ArtifactStore(news_collection.collection_root(plan.campaign_id) / "artifacts")
    return SimpleNamespace(
        plan=plan, clock=clock, responses=responses, calls=calls, store=store, tmp_path=tmp_path
    )


def poll():
    return news_collection.poll("issuer-feed", "first")


def test_poll_preserves_raw_receipt_partial_coverage_and_terminal_replay(collection):
    result = poll()
    receipt = collection.store.json(result["receipt_ref"])
    snapshot = collection.store.json(result["snapshot_ref"])
    assert collection.store.get(receipt["raw_ref"]) == rss()
    assert snapshot["events"][0]["first_seen_at"] == SEEN
    assert snapshot["events"][0]["published_at"].startswith("2025-07-03")
    assert snapshot["coverage"][0]["status"] == "partial"
    assert result["retrieval_mode"] == "rss_polling"
    collection.clock[0] = "2025-07-09T00:00:00Z"
    assert poll() == result
    assert len(collection.calls) == 1
    report = news_collection.inspect("issuer-feed")
    assert report["reserved_bytes"] == collection.plan.response_bytes
    assert report["requests_reserved"] == 1
    assert not report["provider_search"]


def test_unchanged_item_keeps_first_observation_and_feed_omission_is_not_retraction(collection):
    first = poll()
    collection.clock[0] = "2025-07-09T00:00:00Z"
    collection.responses.append(rss())
    second = news_collection.poll("issuer-feed", "second")
    first_events = collection.store.json(first["snapshot_ref"])["events"]
    assert collection.store.json(second["snapshot_ref"])["events"] == first_events
    collection.responses.append(b"<rss><channel/></rss>")
    third = news_collection.poll("issuer-feed", "third")
    assert collection.store.json(third["snapshot_ref"])["events"] == first_events
    assert third["coverage_status"] == "partial"


def test_revision_processing_retires_instance_and_preserves_old_snapshot(collection):
    from test_news_factor import scalar

    first = poll()
    admitted = news_collection.process("issuer-feed", "first", "issuer-research")
    instance_id = admitted["outcomes"][0]["instance_id"]
    raw_ref = collection.store.json(first["receipt_ref"])["raw_ref"]
    assert raw_ref in collection.store.json(admitted["news_result_ref"])["evidence_hex"]
    assert raw_ref in lifecycle.snapshot(instance_id)["transitions"][0]["evidence_refs"]
    assert admitted["outcomes"][0]["status"] == "active_for_registered_research"
    assert scalar(collection.store.json(admitted["news_result_ref"]))["value"] is None
    old_snapshot = collection.store.json(first["snapshot_ref"])
    later = "2025-07-09T00:00:00Z"
    collection.clock[0] = later
    collection.responses.append(rss("Corrected: results will be released later"))
    revised = news_collection.poll("issuer-feed", "second")
    events = collection.store.json(revised["snapshot_ref"])["events"]
    assert [event["revision"] for event in events] == [0, 1]
    retired = news_collection.process("issuer-feed", "second", "issuer-research")
    assert retired["outcomes"][-1]["status"] == "inactive"
    assert lifecycle.value_at(lifecycle.snapshot(instance_id), SEEN)["value"] == 1
    assert lifecycle.value_at(lifecycle.snapshot(instance_id), later)["value"] == 0
    assert collection.store.json(first["snapshot_ref"]) == old_snapshot
    assert poll() == first
    assert len(collection.calls) == 2


def test_unresolved_observed_revision_suspends_instance(collection):
    poll()
    admitted = news_collection.process("issuer-feed", "first", "issuer-research")
    instance_id = admitted["outcomes"][0]["instance_id"]
    collection.clock[0] = "2025-07-09T00:00:00Z"
    collection.responses.append(rss("x" * 5000))
    news_collection.poll("issuer-feed", "second")
    result = news_collection.process("issuer-feed", "second", "issuer-research")
    assert result["outcomes"][-1]["status"] == "suspended"
    assert lifecycle.value_at(lifecycle.snapshot(instance_id), collection.clock[0])["value"] is None


def test_processing_does_not_renew_downstream_owner_and_is_network_free(collection):
    poll()
    result = news_collection.process("issuer-feed", "first", "issuer-research")
    assert news_collection.process("issuer-feed", "first", "issuer-research") == result
    collection.responses.append(rss())
    news_collection.poll("issuer-feed", "second")
    with pytest.raises(ValueError):
        news_collection.process("issuer-feed", "second", "renewed-research")
    assert len(collection.calls) == 2


def test_interrupted_network_attempt_is_charged_and_never_implicitly_retried(collection):
    collection.responses[0] = SystemExit("response may have been received")
    with pytest.raises(SystemExit):
        poll()
    assert news_collection.inspect("issuer-feed")["requests_reserved"] == 1
    with pytest.raises(ValueError, match="pending request"):
        news_collection.poll("issuer-feed", "second")
    resolved = poll()
    assert resolved["coverage_status"] == "unknown"
    assert resolved["reason"] == "interrupted_request_outcome_unknown"
    assert len(collection.calls) == 1
    assert poll() == resolved


def test_response_checkpoint_recovery_uses_original_receipt_clock(collection, monkeypatch):
    def interrupt(*args):
        raise SystemExit("response persisted; parse not committed")

    with monkeypatch.context() as patch:
        patch.setattr(news_collection, "_seal", interrupt)
        with pytest.raises(SystemExit):
            poll()
    collection.clock[0] = "2025-07-10T00:00:00Z"
    recovered = poll()
    assert recovered["received_at"] == SEEN
    assert recovered["coverage_status"] == "partial"
    assert len(collection.calls) == 1


def test_failed_and_empty_responses_cannot_renew_collection_allowances(collection):
    for i in range(4):
        if i:
            collection.responses.append(b"not XML")
        news_collection.poll("issuer-feed", f"request-{i}")
    with pytest.raises(ValueError, match="allowance exhausted"):
        news_collection.poll("issuer-feed", "extra")
    assert len(collection.calls) == 4
    changed = collection.plan.model_copy(update={"max_requests": 5})
    with pytest.raises(ValueError, match="different plan"):
        news_collection.initialize(changed)


@pytest.mark.parametrize("filename", ["manifest.json", "journal.sqlite"])
def test_missing_authority_cannot_be_reinitialized(collection, filename):
    poll()
    (news_collection.collection_root("issuer-feed") / filename).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        news_collection.initialize(collection.plan)
    assert len(collection.calls) == 1


def test_tampered_raw_response_is_detected_on_readonly_replay(collection):
    result = poll()
    ref = collection.store.json(result["receipt_ref"])["raw_ref"]
    (collection.store.root / ref).write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        news_collection.inspect("issuer-feed")


def test_clock_rollback_rejects_new_request_before_network(collection):
    poll()
    collection.clock[0] = "2025-07-07T23:59:59Z"
    with pytest.raises(ValueError, match="clock moved backwards"):
        news_collection.poll("issuer-feed", "second")
    assert len(collection.calls) == 1
    assert news_collection.inspect("issuer-feed")["requests_reserved"] == 1


def test_readonly_replay_survives_runtime_change_but_new_dispatch_does_not(collection, monkeypatch):
    from quant_rsi.core import controller

    poll()
    report = news_collection.inspect("issuer-feed")
    monkeypatch.setattr(controller, "runtime_identity", lambda: {"changed": True})
    assert news_collection.inspect("issuer-feed") == report
    with pytest.raises(ValueError, match="registered source/runtime"):
        news_collection.poll("issuer-feed", "second")
    assert len(collection.calls) == 1


def test_incomplete_response_receipt_preserves_failure_and_budget(collection, monkeypatch):
    monkeypatch.setattr(
        news_feed,
        "fetch",
        lambda source, limit: (b"error response", dict(received_at=SEEN, error="http_error")),
    )
    result = poll()
    assert result["coverage_status"] == "unknown"
    assert result["reason"] == "http_error"
    assert result["response_bytes"] == len(b"error response")
    assert news_collection.inspect("issuer-feed")["requests_reserved"] == 1


def test_registered_feed_evidence_cannot_drop_raw_response(collection):
    from quant_rsi.factors.news import retained_evidence_refs

    result = poll()
    evidence = {ref: collection.store.get(ref) for ref in result["evidence_refs"]}
    raw_ref = collection.store.json(result["receipt_ref"])["raw_ref"]
    del evidence[raw_ref]
    with pytest.raises(ValueError, match="missing transitive"):
        retained_evidence_refs([result["receipt_ref"]], evidence)


def test_item_bound_preserves_unknown_coverage_instead_of_silent_truncation():
    source = news_feed.source_for("KO")
    body = rss().replace(
        b"</channel>",
        rss(guid="release-2").split(b"<channel>")[1].split(b"</channel>")[0] + b"</channel>",
    )
    with pytest.raises(ValueError, match="item allowance"):
        news_feed.parse(body, source, 1)


@pytest.mark.parametrize("pub_date,precision", [("", "unknown"), ("2025-07-03", "day")])
def test_unknown_and_date_only_publications_do_not_invent_timestamps(
    collection, pub_date, precision
):
    collection.responses[0] = rss(pub_date=pub_date)
    result = poll()
    event = collection.store.json(result["snapshot_ref"])["events"][0]
    assert event["publication_precision"] == precision
    assert event["published_at"] is None
    assert event["first_seen_at"] == SEEN


@pytest.mark.parametrize(
    "body",
    [
        b"<html/>",
        b"<!DOCTYPE rss [<!ENTITY x 'title'>]><rss><channel/></rss>",
        "<rss><channel/></rss>".encode("utf-16"),
        rss().replace(b"investors.coca-colacompany.com", b"unregistered.example"),
    ],
)
def test_unusable_responses_preserve_unknown_coverage(collection, body):
    collection.responses[0] = body
    result = poll()
    assert result["coverage_status"] == "unknown"
    assert collection.store.json(result["snapshot_ref"])["events"] == []
    assert collection.store.get(collection.store.json(result["receipt_ref"])["raw_ref"]) == body


def test_conflicting_duplicate_items_reject_entire_response(collection):
    body = rss().replace(
        b"</channel>",
        rss("different").split(b"<channel>")[1].split(b"</channel>")[0] + b"</channel>",
    )
    collection.responses[0] = body
    result = poll()
    assert result["coverage_status"] == "unknown"
    assert "conflicting duplicate" in result["reason"]


def test_transport_reads_one_bounded_body_and_rejects_redirect(monkeypatch):
    reads, requests = [], []
    source = news_feed.source_for("KO")

    class Response:
        status = 200
        headers = {"Content-Type": "text/xml"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def geturl(self):
            return source.url

        def read(self, limit):
            reads.append(limit)
            return b"x" * limit

    def open_request(request, timeout):
        requests.append(request.full_url)
        return Response()

    monkeypatch.setattr(news_feed, "build_opener", lambda *args: SimpleNamespace(open=open_request))
    body, metadata = news_feed.fetch(source, 1024)
    assert len(body) == 1024 and reads == [1024] and requests == [source.url]
    assert metadata["error"] == "response_byte_limit_reached"
    with pytest.raises(HTTPError):
        news_feed.NoRedirect().redirect_request(
            SimpleNamespace(full_url=source.url), None, 302, "redirect", {}, "https://example.org"
        )


def test_cli_poll_process_and_readonly_replay(collection, capsys):
    from quant_rsi.cli import main

    plan_path = collection.tmp_path / "plan.json"
    plan_path.write_bytes(canonical(collection.plan.model_dump(mode="json")))
    assert main(["news-collect", "init", str(plan_path)]) == 0
    capsys.readouterr()
    assert main(["news-collect", "poll", "issuer-feed", "--request-id", "first"]) == 0
    assert json.loads(capsys.readouterr().out)["coverage_status"] == "partial"
    assert (
        main(
            [
                "news-collect",
                "process",
                "issuer-feed",
                "--request-id",
                "first",
                "--news-campaign-id",
                "issuer-research",
            ]
        )
        == 0
    )
    assert (
        json.loads(capsys.readouterr().out)["outcomes"][0]["status"]
        == "active_for_registered_research"
    )
    for command in ("status", "replay"):
        assert main(["news-collect", command, "issuer-feed"]) == 0
        assert json.loads(capsys.readouterr().out)["requests_reserved"] == 1
    assert len(collection.calls) == 1
