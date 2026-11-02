"""Intraday session summaries as daily study factors: values, alignment and model use."""

import json
import math
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from intraday_helpers import EASTERN, build_snapshot, sessions

from quant_rsi.market.market_contracts import MarketSession
from quant_rsi.research.intraday_drivers import (
    METRICS,
    combine,
    prepare_intraday,
    summarise_session,
)


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    gap = sessions(6)[3].session_id
    store, ref = build_snapshot(
        tmp_path_factory.mktemp("drivers"), 6, missing={("KO", gap): range(25, 30)}
    )
    return store.load(ref)


def test_session_summary_matches_a_direct_computation(snapshot):
    session = snapshot.contract.calendar.sessions[0]
    columns = snapshot.block("KO", "decision", session.session_id).columns
    o, h, lo, c, v = (columns[k] for k in ("open", "high", "low", "close", "volume"))
    returns = np.diff(np.log(np.concatenate(([o[0]], c))))
    vwap = ((h + lo + c) / 3 * v).sum() / v.sum()
    expected = {
        "intraday_realized_vol": math.sqrt((returns**2).sum()),
        "intraday_first_hour_return": math.log(c[11] / o[0]),
        "intraday_last_hour_return": math.log(c[-1] / c[-13]),
        "intraday_close_location": (c[-1] - lo.min()) / (h.max() - lo.min()),
        "intraday_close_vwap_gap": c[-1] / vwap - 1,
        "intraday_last_hour_volume_share": v[-12:].sum() / v.sum(),
    }
    got = summarise_session(snapshot.block("KO", "decision", session.session_id), 300)
    assert got == pytest.approx(expected, rel=1e-12)
    assert set(got) == set(METRICS)


def test_rows_use_only_their_own_complete_closed_session(snapshot):
    calendar = snapshot.contract.calendar.sessions
    ids = [s.session_id for s in calendar] + ["2026-02-02"]
    cutoffs = [s.close_at.isoformat() for s in calendar] + ["2026-02-02T21:00:00+00:00"]
    cutoffs[1] = (calendar[1].close_at - timedelta(minutes=5)).isoformat()  # before its close
    bundle = prepare_intraday(snapshot, "KO", ids, cutoffs, required=[0, 2, 4])
    column = bundle["columns"]["external.intraday_last_hour_return"]
    first = summarise_session(snapshot.block("KO", "decision", ids[0]), 300)
    assert column[0] == first["intraday_last_hour_return"]
    assert column[1] is None  # the session had not closed at the cutoff
    assert column[3] is None  # one whole decision bar is missing
    assert column[6] is None  # no session in the snapshot
    assert all(isinstance(column[i], float) for i in (0, 2, 4, 5))
    assert all(d["runnable"] and d["missing_required_rows"] == 0 for d in bundle["definitions"])
    shifted = prepare_intraday(snapshot, "KO", ids, cutoffs, required=[0, 1, 3])
    assert not any(d["runnable"] for d in shifted["definitions"])
    assert {d["source"]["snapshot_sha256"] for d in bundle["definitions"]} == {snapshot.ref}
    assert {d["recipe"]["op"] for d in bundle["definitions"]} == {"external"}
    spy = prepare_intraday(snapshot, "SPY", ids, cutoffs, required=[0])
    assert spy["columns"] != bundle["columns"]
    with pytest.raises(ValueError, match="does not contain"):
        prepare_intraday(snapshot, "XOM", ids, cutoffs, required=[0])
    with pytest.raises(ValueError, match="timezones"):
        prepare_intraday(snapshot, "KO", ids[:1], ["2026-01-05T21:00:00"], required=[0])


def test_combined_bundles_keep_each_provider_and_every_column(snapshot):
    calendar = snapshot.contract.calendar.sessions
    ids = [s.session_id for s in calendar]
    intraday = prepare_intraday(
        snapshot, "KO", ids, [s.close_at.isoformat() for s in calendar], [0]
    )
    other = {
        "provider": "fixture_drivers",
        "definitions": [{"factor_id": "external.fixture"}],
        "columns": {"external.fixture": [1.0] * len(ids)},
        "evidence": [],
        "snapshot_sha256": "fixture",
        "failures": [{"reason": "fixture"}],
        "notice": "Fixture.",
    }
    merged = combine([other, intraday])
    assert merged["provider"] == "fixture_drivers+intraday_snapshot_summary"
    assert set(merged["columns"]) == {"external.fixture", *intraday["columns"]}
    assert merged["snapshot_sha256"]["intraday_snapshot_summary"] == snapshot.ref
    assert merged["failures"] == [{"reason": "fixture"}]
    assert combine([intraday]) is intraday


def early_sessions(dates):
    """Fixture sessions that close before the synthetic study's 16:00 UTC cutoffs."""
    result = []
    for day in dates:
        day = date.fromisoformat(day)
        opened = datetime(day.year, day.month, day.day, 4, 30, tzinfo=EASTERN)
        result.append(
            MarketSession(
                session_id=day.isoformat(),
                open_at=opened,
                close_at=opened + timedelta(hours=6),
            )
        )
    return tuple(result)


def test_intraday_columns_enter_real_study_fits(tmp_path):
    from pathlib import Path

    from quant_rsi.research.demo import load_catalog
    from quant_rsi.research.demo_model import LightGBMDemo

    library = load_catalog(Path(__file__).resolve().parents[1] / "instruments/shared/equities.json")
    model = LightGBMDemo(library, evaluation_sessions=20)
    model.configure_peers({"definitions": [], "columns": {}, "support": []})
    dates = list(model.data.dates)
    store, ref = build_snapshot(tmp_path, session_list=early_sessions(dates))
    cutoffs = [model.data.timestamp(i, "close") for i in range(len(dates))]
    bundle = prepare_intraday(store.load(ref), "KO", dates, cutoffs, [*model.train, *model.origins])
    assert all(d["runnable"] for d in bundle["definitions"])
    model.configure_external(bundle)
    model.price_info = {"latest_target_session": "2030-01-01"}
    price = model.library["definitions"][0]
    first, second = bundle["definitions"][:2]
    one = model.predict({"symbol": "KO"}, [price, first], "Intraday volatility")
    two = model.predict({"symbol": "KO"}, [price, second], "Opening hour")
    assert one["training_feature_columns"] == two["training_feature_columns"] == 2
    for fit, factor in zip(model._fits, [first, second], strict=True):
        np.testing.assert_allclose(
            fit[1][:, 1], np.asarray(bundle["columns"][factor["factor_id"]], dtype=float)
        )
    source = "publication_dated_external_drivers"
    usage = next(item for item in one["input_usage"] if item["source"] == source)
    assert usage["status"] == "used" and usage["factor_ids"] == [first["factor_id"]]


def test_cli_study_uses_and_ablates_intraday_summaries(tmp_path, capsys):
    from test_research_study import CATALOG, command

    from quant_rsi.cli import main
    from quant_rsi.research.demo import load_catalog
    from quant_rsi.research.demo_model import LightGBMDemo
    from quant_rsi.research.study_resume import resume_arguments

    dates = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20).data.dates
    store, ref = build_snapshot(tmp_path / "intraday", session_list=early_sessions(dates))
    root = tmp_path / "study"
    arguments = ("--intraday-store", str(store.root), "--intraday-snapshot", ref)
    assert main(command(root, "--llm", "simulated", "--no-ui", *arguments)) == 0
    result = json.loads((root / "result.json").read_bytes())
    selected = result["iterations"]["attempts"][result["iterations"]["selected_round"] - 1]
    used = [x for x in selected["prediction"]["factor_ids"] if x.startswith("external.intraday")]
    assert used
    comparison = result["study"]["external_comparison"]
    assert comparison["status"] == "completed" and comparison["removed_factor_ids"] == used
    assert result["configuration"]["comparison_fit_budget"] == 1
    assert result["study"]["external_data"]["provider"] == "intraday_snapshot_summary"
    assert resume_arguments(root).intraday_store == store.root.resolve()
    bad = command(tmp_path / "bad", "--llm", "simulated", "--no-ui", "--intraday-snapshot", ref)
    assert main(bad) == 1
    assert "must be given together" in capsys.readouterr().err
