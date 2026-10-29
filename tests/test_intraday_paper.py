"""Paper episode runner against a scripted fake paper broker; synthetic fixtures only."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest
from ibkr_fakes import FakeBroker, config
from intraday_helpers import blocks_for, build_snapshot, plan, sessions

from quant_rsi.brokers.ibkr.guard import LiveAccountRefused
from quant_rsi.brokers.ibkr.orders import IbkrPaperBackend
from quant_rsi.intraday.features import build_features
from quant_rsi.intraday.model import fit, label_vector
from quant_rsi.intraday.paper import EpisodeJournal, PaperPlan, PaperRunner
from quant_rsi.intraday.schedule import anchor_for
from quant_rsi.intraday.storage import VALID

HISTORY = 25


def _today():
    return sessions(HISTORY + 1)[-1]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("paper")
    store, ref = build_snapshot(root, count=HISTORY)
    snapshot = store.load(ref)
    development = plan()
    table = build_features(snapshot, development.leaves)
    model = fit(
        table.values, label_vector(snapshot, table, development), table.names, seed=1, min_rows=50
    )
    rng = np.random.default_rng(99)
    today = _today()
    live = {}
    for name, price in (("KO", 60.0), ("SPY", 500.0)):
        blocks, _ = blocks_for(name, today, rng, price, 300)
        live[name] = {b.grid: b for b in blocks}
    return snapshot, model, live


def _plan(snapshot, model, **overrides):
    base = plan()
    value = {
        "schema_version": 1,
        "episode_id": "ko-p1",
        "purpose": "observational_baseline_only",
        "snapshot_ref": snapshot.ref,
        "model_sha256": model.model_digest,
        "leaves": base.leaves,
        "context_missing_tolerance": 0,
        "timing": base.timing,
        "signal": base.signal,
        "costs": base.costs,
        "capital": "100000",
        "limit_offset_bps": "10",
        "risk": {
            "max_gross_notional": "60000",
            "max_position_notional": "60000",
            "max_orders_per_minute": 10,
            "max_daily_loss_fraction": "0.5",
            "max_consecutive_rejects": 3,
            "max_clock_skew_seconds": 2.0,
        },
        "schedule": {
            "schema_version": 1,
            "rule": "block_4w_dev_4w_reserved_v1",
            "anchor_monday": anchor_for(date.fromisoformat(_today().session_id)),
            "development_weeks": 4,
            "reserved_weeks": 4,
            "embargo_sessions": 1,
        },
        "expires_at": datetime(2027, 1, 1, tzinfo=timezone.utc),
        "max_lag_seconds": 400,
    }
    value.update(overrides)
    return PaperPlan.model_validate(value, strict=False)


def _runner(tmp_path, world, broker=None, **plan_overrides):
    snapshot, model, _ = world
    broker = broker or FakeBroker()
    bindings = {b.instrument: b for b in snapshot.contract.instruments}
    backend = IbkrPaperBackend(broker.session(), config(), bindings)
    journal = EpisodeJournal(tmp_path / "paper.sqlite")
    runner = PaperRunner(
        journal, _plan(snapshot, model, **plan_overrides), model.model_text, snapshot, backend
    )
    return runner, broker, journal


def _payload(block, i):
    cols = block.columns
    return {
        "date": str(cols["start"][i]),
        "open": float(cols["open"][i]),
        "high": float(cols["high"][i]),
        "low": float(cols["low"][i]),
        "close": float(cols["close"][i]),
        "volume": str(cols["volume"][i]),
    }


def _run_day(
    runner,
    live,
    *,
    until=None,
    step_seconds=30,
    skip=None,
    session=None,
    stop_when=None,
    start_session=True,
    start=None,
):
    today = session or _today()
    if start_session:
        runner.start_session(today)
    sent = set()
    start = start or today.open_at.astimezone(timezone.utc)
    end = until or today.close_at.astimezone(timezone.utc) + timedelta(seconds=60)
    now, outputs = start, []
    while now <= end:
        if stop_when is not None and stop_when(now):
            return now
        for name, grids in live.items():
            for grid in ("decision", "execution"):
                block = grids[grid]
                seconds = 300 if grid == "decision" else 60
                for i in np.flatnonzero(block.columns["status"] == VALID):
                    bar_end = int(block.columns["start"][i]) + seconds
                    received = datetime.fromtimestamp(bar_end - 1, timezone.utc)
                    if received <= now and (name, grid, i) not in sent:
                        sent.add((name, grid, i))
                        runner.on_update(name, grid, _payload(block, i), received)
        if skip is None or not skip(now):
            outputs.append(runner.step(now))
        now += timedelta(seconds=step_seconds)
    return outputs


def _register(runner, now=None):
    now = now or _today().open_at.astimezone(timezone.utc) - timedelta(minutes=5)
    runner.register(now)
    runner.activate(now)


def test_full_day_trades_inside_windows_and_ends_flat(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    _run_day(runner, world[2])
    intents = journal.intents("ko-p1")
    assert intents, "the always-enter fixture policy should trade"
    submitted = {e["payload"]["intent_id"]: e["at"] for e in journal.events("ko-p1", "submitted")}
    for record in intents:
        intent = record["intent"]
        if intent.intent_id in submitted:
            at = datetime.fromisoformat(submitted[intent.intent_id])
            assert intent.submit_at <= at < intent.submit_deadline
        assert intent.submit_at.timestamp() % 60 == 0  # fill-bar starts are minute boundaries
    assert runner.positions() == {}
    closed = journal.events("ko-p1", "session_closed")[-1]["payload"]
    assert closed["open_positions"] == {} and closed["purpose"] == "development_observation"
    assert journal.status("ko-p1") == "active"
    fees = sum(Decimal(f["fee"]) for f in journal.fills("ko-p1"))
    assert fees == len(journal.fills("ko-p1"))  # fake broker charges 1.0 per execution
    assert runner.cash() == Decimal(closed["equity"])
    assert all(r["shortfall_bps"] is not None for r in runner.shadow_report())


def test_entries_are_never_submitted_after_their_deadline(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    today = _today()
    gap_start = today.open_at.astimezone(timezone.utc) + timedelta(minutes=60)
    gap_end = gap_start + timedelta(minutes=40)
    _run_day(runner, world[2], skip=lambda now: gap_start <= now < gap_end)
    abandoned = {
        e["payload"]["intent_id"] for e in journal.events("ko-p1", "abandoned_unsubmitted")
    }
    placed = {
        r.params["order"]["order_ref"] for r in broker.transport.sent if r.kind == "place_order"
    }
    skipped = journal.events("ko-p1", "decision_window_passed")
    assert skipped, "decisions whose window passed during the gap are skipped, not traded"
    assert not abandoned & placed
    gap_bars = {e["payload"]["bar"] for e in skipped}
    traded_bars = {r["intent"].decision_index for r in journal.intents("ko-p1")}
    assert not gap_bars & traded_bars


def test_unfilled_exits_suspend_with_overnight_breach(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world, broker=FakeBroker(fill_sides=("BUY",)))
    _register(runner)
    _run_day(runner, world[2])
    assert runner.positions()
    assert journal.status("ko-p1") == "suspended"
    assert journal.events("ko-p1", "status_suspended")[-1]["payload"]["reason"] == (
        "overnight_breach"
    )
    # Unfilled exits are cancelled at their deadline (cancelling -> cancelled), then retried.
    assert journal.events("ko-p1", "cancelled")


def test_one_active_episode_per_paper_account(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    snapshot, model, _ = world
    other = PaperRunner(
        journal,
        _plan(snapshot, model, episode_id="ko-p2"),
        model.model_text,
        snapshot,
        runner.backend,
    )
    with pytest.raises(ValueError, match="already has active episode"):
        other.register(_today().open_at)


def test_stop_finishes_flat_then_terminates_and_frees_the_account(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    halfway = _today().open_at.astimezone(timezone.utc) + timedelta(hours=3)
    _run_day(runner, world[2], until=halfway)
    runner.stop("stop-001", halfway)
    runner.stop("stop-001", halfway)  # idempotent
    assert journal.status("ko-p1") == "stopping"
    entries_before = len([r for r in journal.intents("ko-p1") if r["intent"].purpose == "entry"])
    today = _today()
    end = today.close_at.astimezone(timezone.utc) + timedelta(seconds=60)
    now = halfway
    while now <= end:
        runner.step(now)
        now += timedelta(seconds=30)
    entries_after = len([r for r in journal.intents("ko-p1") if r["intent"].purpose == "entry"])
    assert entries_after == entries_before
    assert journal.status("ko-p1") == "terminated"
    snapshot, model, _ = world
    successor = PaperRunner(
        journal,
        _plan(snapshot, model, episode_id="ko-p2"),
        model.model_text,
        snapshot,
        runner.backend,
    )
    successor.register(end)
    assert journal.status("ko-p2") == "registered"


def test_registration_refuses_preexisting_position_in_episode_instrument(tmp_path, world):
    broker = FakeBroker()
    broker.extra_positions[100] = Decimal(5)  # KO con_id in the fixture contract
    runner, _, _ = _runner(tmp_path, world, broker=broker)
    with pytest.raises(ValueError, match="cannot be attributed"):
        runner.register(_today().open_at)


def test_duplicate_executions_after_reconnect_are_ignored(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    halfway = _today().open_at.astimezone(timezone.utc) + timedelta(hours=3)
    _run_day(runner, world[2], until=halfway)
    fills_before = journal.fills("ko-p1")
    order_id = max(broker.orders)
    broker.transport.push(*broker.fill(order_id, duplicate=True))
    runner.step(halfway + timedelta(seconds=1))
    assert len(journal.fills("ko-p1")) == len(fills_before)


def test_reconcile_suspends_on_position_mismatch(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    halfway = _today().open_at.astimezone(timezone.utc) + timedelta(hours=3)
    _run_day(runner, world[2], until=halfway)
    broker.extra_positions[100] = Decimal(7)
    result = runner.reconcile(halfway)
    assert "position_mismatch:KO" in result["findings"]
    assert journal.status("ko-p1") == "suspended"


def test_live_account_is_refused_before_any_episode_state(tmp_path, world):
    with pytest.raises(LiveAccountRefused):
        _runner(tmp_path, world, broker=FakeBroker(accounts=("U7654321",)))
    assert not (tmp_path / "paper.sqlite").exists()


def test_model_digest_and_snapshot_are_bound(tmp_path, world):
    snapshot, model, _ = world
    broker = FakeBroker()
    backend = IbkrPaperBackend(
        broker.session(), config(), {b.instrument: b for b in snapshot.contract.instruments}
    )
    journal = EpisodeJournal(tmp_path / "paper.sqlite")
    with pytest.raises(ValueError, match="sealed model digest"):
        PaperRunner(journal, _plan(snapshot, model), model.model_text + " ", snapshot, backend)


def test_risk_block_stops_entries_but_records_reason(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    runner.clock_skew = lambda: 5.0
    outputs = _run_day(runner, world[2])
    assert any("clock_skew" in o.get("risk", []) for o in outputs)
    assert not [r for r in journal.intents("ko-p1") if r["intent"].purpose == "entry"]
