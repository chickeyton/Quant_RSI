from copy import deepcopy

import pytest

from quant_rsi.artifacts import canonical, digest
from quant_rsi.forecast_inputs import SignalSnapshot, signal_view


def snapshot_value():
    ref = digest(b"engineering source")
    return dict(
        schema_version=1,
        kind="forecast_signal_snapshot",
        events=[
            dict(
                schema_version=1,
                kind="news",
                source_id="wire",
                entity="issuer",
                event_id="story",
                revision=0,
                query_ref=ref,
                published_at="2026-10-20T09:00:00Z",
                published_date=None,
                publication_precision="second",
                source_available_at=None,
                first_seen_at="2026-10-21T10:00:00Z",
                report_period_end=None,
                payload_ref=ref,
                receipt_ref=ref,
                extractor_version="raw_v1",
                units={},
                fields={},
            )
        ],
        coverage=[
            dict(
                schema_version=1,
                source_id="wire",
                entity="issuer",
                query_ref=ref,
                start_at="2026-10-20T00:00:00Z",
                end_at="2026-10-21T00:00:00Z",
                first_seen_at="2026-10-21T10:00:00Z",
                status="complete",
                reason=None,
                receipt_ref=ref,
            )
        ],
    )


def view(value, as_of="2026-10-21T11:00:00Z", **overrides):
    snapshot = SignalSnapshot.model_validate_json(canonical(value))
    args = dict(
        source_id="wire",
        entity="issuer",
        query_ref=digest(b"engineering source"),
        start_at="2026-10-20T00:00:00Z",
        end_at="2026-10-21T00:00:00Z",
        as_of=as_of,
    )
    args.update(overrides)
    return signal_view(snapshot, **args)


def test_later_retrieved_news_never_enters_earlier_knowledge():
    value = snapshot_value()
    before = view(value, as_of="2026-10-21T09:00:00Z")
    assert before["coverage"] == "unknown"
    assert before["events"] == [] and before["news_count"] is None
    after = view(value)
    assert after["coverage"] == "complete" and after["news_count"] == 1


def test_unknown_coverage_is_not_zero_and_is_query_entity_interval_specific():
    value = snapshot_value()
    value["events"] = []
    assert view(value)["news_count"] == 0
    for change in (
        dict(entity="another_issuer"),
        dict(query_ref=digest(b"another_query")),
        dict(start_at="2026-10-19T00:00:00Z"),
    ):
        result = view(value, **change)
        assert result["coverage"] == "unknown" and result["news_count"] is None
    value["coverage"][0].update(status="partial", reason="pagination incomplete")
    assert view(value)["news_count"] is None


def test_date_only_or_unknown_publication_never_gets_an_invented_timestamp():
    value = snapshot_value()
    value["events"][0].update(
        published_at=None,
        published_date="2026-10-20",
        publication_precision="day",
    )
    result = view(value)
    assert result["events"] == [] and result["news_count"] is None
    assert result["publication_time_unresolved"][0]["published_at"] is None
    value["events"][0]["published_at"] = "2026-10-20T00:00:00Z"
    with pytest.raises(ValueError, match="invent"):
        view(value)
    value["events"][0].update(
        published_at=None, published_date=None, publication_precision="unknown"
    )
    assert view(value)["news_count"] is None


def test_signal_revisions_respect_their_receipts_and_availability():
    value = snapshot_value()
    later = deepcopy(value["events"][0])
    later.update(revision=1, first_seen_at="2026-10-22T10:00:00Z")
    value["events"].append(later)
    assert view(value)["events"][0]["revision"] == 0
    assert view(value, as_of="2026-10-22T11:00:00Z")["events"][0]["revision"] == 1
    value["events"][0]["source_available_at"] = "2026-10-23T00:00:00Z"
    value["events"].pop()
    assert view(value)["events"] == []


def test_coverage_and_fundamental_contracts_reject_false_completeness():
    value = snapshot_value()
    value["coverage"][0]["first_seen_at"] = "2026-10-20T23:00:00Z"
    with pytest.raises(ValueError, match="before its end"):
        view(value)
    value = snapshot_value()
    value["events"][0]["kind"] = "fundamental"
    with pytest.raises(ValueError, match="report period"):
        view(value)
    value["events"][0].update(
        report_period_end="2026-09-30",
        fields={"revenue": "100"},
        units={},
    )
    with pytest.raises(ValueError, match="unit"):
        view(value)


def test_new_coverage_receipt_does_not_rewrite_earlier_knowledge():
    value = snapshot_value()
    correction = deepcopy(value["coverage"][0])
    correction.update(
        first_seen_at="2026-10-22T10:00:00Z",
        status="partial",
        reason="source later reports incomplete pagination",
    )
    value["coverage"].append(correction)
    assert view(value)["news_count"] == 1
    later = view(value, as_of="2026-10-22T11:00:00Z")
    assert later["coverage"] == "partial" and later["news_count"] is None


def test_news_interval_is_closed_at_start_and_open_at_end():
    value = snapshot_value()
    value["events"][0]["published_at"] = "2026-10-20T00:00:00Z"
    assert view(value)["news_count"] == 1
    value["events"][0]["published_at"] = "2026-10-21T00:00:00Z"
    assert view(value)["news_count"] == 0
