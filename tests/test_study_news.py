"""Receipt-time and bounded-archive oracles, with no network or predictive claims."""

from copy import deepcopy
from datetime import timedelta

import pytest

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.data import synthetic_data
from quant_rsi.core.schemas import DataSpec
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.research import study_news
from quant_rsi.research.study_news import (
    NewsHistory,
    load_archive,
    retain_snapshot,
    simulated_history,
)

RECIPE = dict(op="headline_keyword_count", keywords=["earnings"], window_days=7)
CUTOFF = "2026-10-29T16:00:00+00:00"


def snapshot(received="2026-10-29T15:00:00+00:00", title="EARNINGS growth", symbol="KO"):
    return dict(
        provider="issuer_rss",
        scope="target_issuer_feed_only",
        source_url=f"https://example.com/{symbol}/rss",
        received_at=received,
        items=[
            dict(
                id=f"{symbol}.1",
                symbol=symbol,
                title=title,
                summary="earnings in summary only",
                url=f"https://example.com/{symbol}/1",
                received_at=received,
                published_at=utc_text(timestamp(received) - timedelta(hours=1)),
                published_date=None,
                publication_precision="second",
            )
        ],
    )


def multi(target, peer, *, unavailable=False):
    sources = []
    for value in (target, peer):
        sources.append(
            dict(
                symbol=value["items"][0]["symbol"],
                status="available",
                source_url=value["source_url"],
                received_at=value["received_at"],
            )
        )
    items = target["items"] + peer["items"]
    if unavailable:
        sources[0] = dict(symbol="KO", status="unavailable", reason="transport failed")
        items = peer["items"]
    return dict(
        provider="issuer_rss",
        scope="target_and_configured_related_issuers",
        coverage=sources,
        items=items,
    )


def test_distinct_case_insensitive_literal_headlines_and_real_zero():
    data = snapshot(title="Earnings [beat]")
    data["items"].append(deepcopy(data["items"][0]))
    history = NewsHistory([data])
    assert history.evaluate(RECIPE, "KO", CUTOFF)["value"] == 1
    assert history.evaluate(RECIPE | {"keywords": ["[beat]"]}, "KO", CUTOFF)["value"] == 1
    zero = history.evaluate(RECIPE | {"keywords": ["summary"]}, "KO", CUTOFF)
    assert zero["value"] == 0 and zero["reasons"] == [] and zero["evidence_news_ids"] == []
    assert history.evaluate(RECIPE, "OTHER", CUTOFF)["value"] is None


def test_future_receipts_never_backfill_and_newest_snapshot_replaces_corpus():
    old = snapshot()
    future = snapshot("2026-10-30T15:00:00+00:00", title="Demand weakness")
    assert NewsHistory([old, future]).evaluate(RECIPE, "KO", CUTOFF)["value"] == 1
    assert NewsHistory([future]).evaluate(RECIPE, "KO", CUTOFF)["value"] is None
    latest = snapshot("2026-10-29T15:30:00+00:00", title="Demand weakness")
    assert NewsHistory([old, latest]).evaluate(RECIPE, "KO", CUTOFF)["value"] == 0


def test_exact_publication_window_and_exclusive_cutoff():
    data = snapshot()
    data["items"][0]["published_at"] = "2026-10-22T16:00:00+00:00"
    assert NewsHistory([data]).evaluate(RECIPE, "KO", CUTOFF)["value"] == 1
    data["items"][0]["published_at"] = CUTOFF
    assert NewsHistory([data]).evaluate(RECIPE, "KO", CUTOFF)["value"] == 0


@pytest.mark.parametrize("precision", ["day", "unknown"])
def test_imprecise_publication_is_missing_independently_of_recipe(precision):
    data = snapshot()
    data["items"][0].update(
        published_at=None,
        publication_precision=precision,
        published_date="2026-10-01" if precision == "day" else None,
    )
    for recipe in (RECIPE, RECIPE | {"window_days": 1, "keywords": ["absent"]}):
        result = NewsHistory([data]).evaluate(recipe, "KO", CUTOFF)
        assert result["value"] is None and result["reasons"] == ["publication_time_not_precise"]


def test_24_hour_freshness_includes_boundary_only():
    data = snapshot("2026-10-28T16:00:00+00:00")
    assert NewsHistory([data]).evaluate(RECIPE, "KO", CUTOFF)["value"] == 1
    result = NewsHistory([data]).evaluate(RECIPE, "KO", "2026-10-29T16:00:01+00:00")
    assert result["value"] is None and result["reasons"] == ["snapshot_older_than_24_hours"]


def test_source_receipts_and_latest_unavailable_target_do_not_fall_back():
    old = snapshot("2026-10-29T14:00:00+00:00")
    peer = snapshot("2026-10-29T15:00:00+00:00", symbol="MNST")
    result = NewsHistory([old, multi(old, peer, unavailable=True)]).evaluate(RECIPE, "KO", CUTOFF)
    assert result["value"] is None and result["reasons"] == ["target_coverage_unavailable"]
    future_peer = snapshot("2026-10-30T15:00:00+00:00", symbol="MNST")
    assert NewsHistory([multi(old, future_peer)]).evaluate(RECIPE, "KO", CUTOFF)["value"] == 1


def test_conflicting_same_snapshot_ids_rejected_but_later_corrections_preserved():
    old = snapshot()
    conflicting = deepcopy(old)
    conflicting["items"].append(old["items"][0] | {"title": "Corrected demand"})
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        NewsHistory([conflicting])
    revised = snapshot("2026-10-29T15:30:00+00:00", title="Corrected demand")
    history = NewsHistory([old, revised])
    assert history.evaluate(RECIPE, "KO", "2026-10-29T15:15:00+00:00")["value"] == 1
    assert history.evaluate(RECIPE, "KO", CUTOFF)["value"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("received_at", "2026-10-29T15:00:00"),
        ("published_at", "2026-13-01T12:00:00Z"),
        ("received_at", "2026-10-29T15:01:00Z"),
        ("url", "file:///secret"),
    ],
)
def test_malformed_item_rejected_before_archive_write(tmp_path, field, value):
    data = snapshot()
    data["items"][0][field] = value
    with pytest.raises(ValueError):
        retain_snapshot(tmp_path / "archive", data)
    assert not (tmp_path / "archive").exists()


def test_archive_preserves_exact_receipts_deduplicates_and_detects_corruption(tmp_path):
    data = snapshot()
    name = retain_snapshot(tmp_path, data)
    assert retain_snapshot(tmp_path, deepcopy(data)) == name
    assert load_archive(tmp_path) == [data]
    assert len(list(tmp_path.iterdir())) == 1
    next(tmp_path.iterdir()).write_bytes(canonical(snapshot(title="changed")))
    with pytest.raises(ValueError, match="does not match"):
        load_archive(tmp_path)


def test_archive_file_total_and_count_bounds(tmp_path, monkeypatch):
    data = snapshot()
    retain_snapshot(tmp_path, data)
    monkeypatch.setattr(study_news, "MAX_FILES", 1)
    with pytest.raises(ValueError, match="capacity"):
        retain_snapshot(tmp_path, snapshot("2026-10-29T15:30:00Z"))
    monkeypatch.setattr(study_news, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="bound"):
        load_archive(tmp_path)
    monkeypatch.setattr(study_news, "MAX_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(study_news, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="bound"):
        load_archive(tmp_path)


def test_history_is_frozen_and_ambiguous_same_receipt_is_missing():
    data = snapshot()
    history = NewsHistory([data])
    data["items"][0]["title"] = "Changed outside history"
    assert history.evaluate(RECIPE, "KO", CUTOFF)["value"] == 1
    ambiguous = NewsHistory([snapshot(), data]).evaluate(RECIPE, "KO", CUTOFF)
    assert ambiguous["value"] is None
    assert ambiguous["reasons"] == ["ambiguous_snapshots_at_same_receipt"]


def test_simulated_history_is_labeled_and_available_before_every_synthetic_close():
    data = synthetic_data(
        DataSpec(source="synthetic", sessions=24, instruments=1, train_cutoff=12, seed=7)
    )
    snapshots = simulated_history(data, "KO")
    history = NewsHistory(snapshots)
    assert history.providers == ("simulated_news_archive",)
    assert history.snapshot_count == 24
    values = [
        history.evaluate(RECIPE, "KO", data.timestamp(i, "close"))["value"] for i in range(24)
    ]
    assert values == [float(i % 2) for i in range(24)]
    bad = deepcopy(snapshots[0])
    bad["provider"] = "issuer_rss"
    with pytest.raises(ValueError, match="explicitly labeled"):
        NewsHistory([bad])
    assert digest(canonical(simulated_history(data, "KO"))) == digest(canonical(snapshots))
