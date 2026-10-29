"""Standalone intraday CLI commands and the real-time loop, against fixtures and fakes."""

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from ibkr_fakes import FakeBroker, config
from intraday_helpers import build_snapshot, plan, sessions

from quant_rsi.brokers.ibkr.orders import IbkrPaperBackend
from quant_rsi.brokers.ibkr.transport import Message
from quant_rsi.intraday.__main__ import main
from quant_rsi.intraday.contracts import IntradayDevelopmentPlan
from quant_rsi.intraday.live import run_session
from quant_rsi.intraday.paper import EpisodeJournal, LiveFeed, PaperPlan, PaperRunner
from quant_rsi.intraday.policy import signal_digest
from quant_rsi.intraday.profile import IntradayWorkflowConfig
from quant_rsi.intraday.schedule import anchor_for

ROOT = Path(__file__).parents[1]


def test_checked_in_examples_validate():
    IntradayWorkflowConfig.model_validate_json(
        (ROOT / "instruments/US/KO/intraday.json").read_bytes()
    )
    IntradayDevelopmentPlan.model_validate_json(
        (ROOT / "examples/intraday_development.json").read_bytes()
    )


def test_profile_requires_latency_to_cover_buffer():
    raw = json.loads((ROOT / "instruments/US/KO/intraday.json").read_text())
    raw["decision_latency_seconds"] = 2
    with pytest.raises(ValueError):
        IntradayWorkflowConfig.model_validate(raw)


def test_evaluate_and_fit_commands(tmp_path, capsys):
    store, ref = build_snapshot(tmp_path, count=32)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan().model_dump_json())
    assert (
        main(
            [
                "evaluate",
                "--store",
                str(tmp_path / "store"),
                "--snapshot",
                ref,
                "--plan",
                str(plan_path),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["kind"] == "IntradayDevelopmentEvaluation" and len(report["folds"]) == 2
    model_path = tmp_path / "model.txt"
    assert (
        main(
            [
                "fit",
                "--store",
                str(tmp_path / "store"),
                "--snapshot",
                ref,
                "--plan",
                str(plan_path),
                "--output",
                str(model_path),
            ]
        )
        == 0
    )
    fitted = json.loads(capsys.readouterr().out)
    assert len(fitted["model_sha256"]) == 64 and model_path.read_text().startswith("tree")


def test_live_feed_routes_updates_by_request(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    snapshot = store.load(ref)
    broker = FakeBroker()
    session = broker.session(client_id=11)
    today = sessions(4)[-1]
    feed = LiveFeed(session, snapshot.contract, today)
    req_id, route = next(iter(feed.routes.items()))
    sent = [r for r in broker.transport.sent if r.kind == "historical_data"]
    assert all(r.params["keep_up_to_date"] for r in sent) and len(sent) == 4
    broker.transport.push(
        Message("historical_update", req_id, {"date": "1"}),
        Message("historical_update", 999999, {"date": "2"}),
    )
    polled = feed.poll()
    assert [u for u in polled if u[2] == "update"] == [(*route, "update", {"date": "1"})]
    # The fake answers each subscription with an end marker: backfill completion is routed.
    assert len([u for u in polled if u[2] == "backfill_end"]) == 4


def test_run_session_with_silent_feed_closes_session_and_reconciles(tmp_path):
    from quant_rsi.intraday.features import build_features
    from quant_rsi.intraday.model import fit, label_vector

    store, ref = build_snapshot(tmp_path, count=25)
    snapshot = store.load(ref)
    development = plan()
    table = build_features(snapshot, development.leaves)
    model = fit(
        table.values, label_vector(snapshot, table, development), table.names, seed=1, min_rows=50
    )
    today = sessions(26)[-1]
    paper = PaperPlan.model_validate(
        {
            "schema_version": 1,
            "episode_id": "loop",
            "purpose": "observational_baseline_only",
            "snapshot_ref": ref,
            "model_sha256": model.model_digest,
            "signal_sha256": signal_digest(development.signal),
            "leaves": development.leaves,
            "context_missing_tolerance": 0,
            "timing": development.timing,
            "signal": development.signal,
            "costs": development.costs,
            "capital": "100000",
            "limit_offset_bps": "10",
            "risk": {
                "max_gross_notional": "1",
                "max_position_notional": "1",
                "max_orders_per_minute": 1,
                "max_daily_loss_fraction": "0.1",
                "max_consecutive_rejects": 1,
                "max_clock_skew_seconds": 2,
            },
            "schedule": {
                "schema_version": 1,
                "rule": "block_4w_dev_4w_reserved_v1",
                "anchor_monday": anchor_for(date.fromisoformat(today.session_id)),
                "development_weeks": 4,
                "reserved_weeks": 4,
                "embargo_sessions": 1,
            },
            "expires_at": datetime(2027, 1, 1, tzinfo=timezone.utc),
            "max_lag_seconds": 330,
        },
        strict=False,
    )
    orders, data = FakeBroker(), FakeBroker()
    backend = IbkrPaperBackend(
        orders.session(), config(), {b.instrument: b for b in snapshot.contract.instruments}
    )
    runner = PaperRunner(
        EpisodeJournal(tmp_path / "j.sqlite"), paper, model.model_text, snapshot, backend
    )
    start = today.open_at.astimezone(timezone.utc) - timedelta(minutes=1)
    runner.register(start)
    runner.activate(start)
    ticks = iter(start + timedelta(minutes=m) for m in range(0, 500))
    result = run_session(
        runner, data.session(client_id=11), today, clock=lambda: next(ticks), sleep=lambda s: None
    )
    journal = runner.journal
    assert journal.events("loop", "feed_stale_feed")
    assert journal.events("loop", "session_closed")
    assert result["reconciliation"] == {"findings": []}
    assert result["availability"]["observations"] == 0
