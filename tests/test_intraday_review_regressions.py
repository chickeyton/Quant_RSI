"""Regressions for the first implementation review (order safety, recovery, features, data)."""

# `world` is a module-scoped pytest fixture imported from test_intraday_paper.
# ruff: noqa: F811

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest
from ibkr_fakes import FakeBroker, config
from intraday_helpers import binding, blocks_for, build_snapshot, plan, sessions
from test_intraday_paper import (  # noqa: F401  (world is a pytest fixture)
    HISTORY,
    _payload,
    _plan,
    _register,
    _run_day,
    _runner,
    _today,
    world,
)

from quant_rsi.brokers.ibkr.config import in_restart_window
from quant_rsi.brokers.ibkr.connection import ClockSkewMonitor
from quant_rsi.brokers.ibkr.orders import IbkrPaperBackend
from quant_rsi.brokers.ibkr.pacing import PacingLimiter
from quant_rsi.brokers.ibkr.transport import Message
from quant_rsi.intraday.collect import collect_intraday
from quant_rsi.intraday.features import session_features
from quant_rsi.intraday.live import run_session
from quant_rsi.intraday.paper import EpisodeJournal, PaperRunner
from quant_rsi.intraday.simulator import simulate
from quant_rsi.intraday.storage import VALID, Block, SnapshotStore


# Paper: order safety -----------------------------------------------------------------------------
class _CrashingBackend:
    """Wraps the real backend; crashes around the broker send to simulate process death."""

    def __init__(self, backend, *, before_send):
        self.backend, self.before_send = backend, before_send

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def submit(self, intent):
        if self.before_send:
            raise RuntimeError("crash before send")
        self.backend.submit(intent)
        raise RuntimeError("crash after send, before the journal commit")


def _restart(tmp_path, world, broker):
    snapshot, model, _ = world
    bindings = {b.instrument: b for b in snapshot.contract.instruments}
    backend = IbkrPaperBackend(broker.session(), config(), bindings)
    return PaperRunner(
        EpisodeJournal(tmp_path / "paper.sqlite"),
        _plan(snapshot, model),
        model.model_text,
        snapshot,
        backend,
    )


def _placed(broker, ref):
    return [
        r
        for r in broker.transport.sent
        if r.kind == "place_order" and r.params["order"]["order_ref"] == ref
    ]


@pytest.mark.parametrize("before_send", [False, True])
def test_crash_around_submission_never_resends(tmp_path, world, before_send):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    runner.backend = _CrashingBackend(runner.backend, before_send=before_send)
    with pytest.raises(RuntimeError, match="crash"):
        _run_day(runner, world[2])
    (record,) = journal.intents("ko-p1", {"submitting"})
    ref = record["intent"].intent_id
    assert len(_placed(broker, ref)) == (0 if before_send else 1)
    restarted = _restart(tmp_path, world, broker)
    restarted.start_session(_today())
    crash_time = record["intent"].submit_at
    restarted.begin_recovery(crash_time)
    restarted.reconcile(crash_time)
    _run_day(
        restarted,
        world[2],
        start_session=False,
        start=crash_time,
        until=crash_time + timedelta(minutes=10),
    )
    assert len(_placed(broker, ref)) == (0 if before_send else 1)
    status = {r["intent"].intent_id: r["status"] for r in journal.intents("ko-p1")}[ref]
    assert status == ("abandoned_unsubmitted" if before_send else "filled")
    if not before_send:
        # Reconcile applies the broker execution first: the crashed order is recognised
        # as filled by orderRef, and nothing is sent again.
        fills = [f for f in journal.fills("ko-p1") if f["intent_id"] == ref]
        assert len(fills) == 1


def _pending_entry(journal):
    return [r for r in journal.intents("ko-p1", {"pending"}) if r["intent"].purpose == "entry"]


def _until_pending_entry(runner, world, journal):
    return _run_day(
        runner, world[2], step_seconds=5, stop_when=lambda now: bool(_pending_entry(journal))
    )


def test_risk_block_after_decision_abandons_queued_entry(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    now = _until_pending_entry(runner, world, journal)
    (record,) = _pending_entry(journal)
    runner.clock_skew = lambda: 5.0
    _run_day(
        runner,
        world[2],
        start_session=False,
        start=now,
        until=record["intent"].submit_deadline + timedelta(seconds=30),
    )
    ref = record["intent"].intent_id
    assert not _placed(broker, ref)
    reason = next(
        e["payload"]["reason"]
        for e in journal.events("ko-p1", "abandoned_unsubmitted")
        if e["payload"]["intent_id"] == ref
    )
    assert reason.startswith("blocked:") and "clock_skew" in reason


def test_stop_abandons_queued_entries(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    now = _until_pending_entry(runner, world, journal)
    (record,) = _pending_entry(journal)
    runner.stop("stop-early", now)
    _run_day(
        runner,
        world[2],
        start_session=False,
        start=now,
        until=record["intent"].submit_deadline + timedelta(seconds=30),
    )
    assert not _placed(broker, record["intent"].intent_id)
    reasons = {
        e["payload"]["intent_id"]: e["payload"]["reason"]
        for e in journal.events("ko-p1", "abandoned_unsubmitted")
    }
    assert reasons[record["intent"].intent_id] == "stopped"


def test_overnight_position_exits_at_next_open(tmp_path, world):
    broker = FakeBroker(fill_sides=("BUY",))
    runner, _, journal = _runner(tmp_path, world, broker=broker)
    _register(runner)
    _run_day(runner, world[2])
    assert journal.status("ko-p1") == "suspended" and runner.positions()
    broker.fill_sides = {"BUY", "SELL"}
    day2 = sessions(HISTORY + 2)[-1]
    rng = np.random.default_rng(5)
    live2 = {}
    for name, price in (("KO", 60.0), ("SPY", 500.0)):
        blocks, _ = blocks_for(name, day2, rng, price, 300)
        live2[name] = {b.grid: b for b in blocks}
    open2 = day2.open_at.astimezone(timezone.utc)
    _run_day(runner, live2, session=day2, until=open2 + timedelta(minutes=15))
    assert runner.positions() == {}
    reasons = [e["payload"]["reason"] for e in journal.events("ko-p1", "intent")]
    assert "overnight" in reasons
    day2_entries = [
        r
        for r in journal.intents("ko-p1")
        if r["intent"].session_id == day2.session_id and r["intent"].purpose == "entry"
    ]
    assert day2_entries == []  # suspended: exits only


# Paper: recovery and reporting ------------------------------------------------------------------
def test_restart_replay_is_marked_recovered_and_not_traded(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    restart_at = _today().open_at.astimezone(timezone.utc) + timedelta(hours=3)
    _run_day(runner, world[2], until=restart_at - timedelta(hours=1))
    before = {r["intent"].intent_id for r in journal.intents("ko-p1")}
    restarted = _restart(tmp_path, world, broker)
    restarted.start_session(_today())
    restarted.begin_recovery(restart_at)
    for name, grids in world[2].items():  # the re-subscription replays earlier bars now
        for grid in ("decision", "execution"):
            block = grids[grid]
            seconds = 300 if grid == "decision" else 60
            for i in np.flatnonzero(block.columns["status"] == VALID):
                if int(block.columns["start"][i]) + seconds < restart_at.timestamp():
                    restarted.on_update(name, grid, _payload(block, i), restart_at)
    for key in restarted.finalisers:
        restarted.on_backfill_end(*key)
    restarted.step(restart_at)
    replayed = [
        e["payload"]
        for e in journal.events("ko-p1", "bar")
        if datetime.fromisoformat(e["at"]) >= restart_at
    ]
    assert replayed and all(b["recovered"] and b["arrival_delay"] is None for b in replayed)
    assert not {r["intent"].intent_id for r in journal.intents("ko-p1")} - before
    assert journal.events("ko-p1", "decision_window_passed")
    assert restarted.availability()["violated"] is False


def test_shadow_report_uses_simulator_price_rule(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    _run_day(runner, world[2])
    report = runner.shadow_report()
    assert report and all(r["spread_policy"] == "mid_plus_half_spread" for r in report)
    assert all(r["spread_fallback"] for r in report)  # no live bid/ask grid is collected


def test_delayed_snapshot_cannot_feed_released_episode(tmp_path, world):
    snapshot, model, _ = world
    delayed = snapshot.__class__(
        snapshot.ref,
        snapshot.contract.model_copy(update={"data_entitlement": "delayed"}),
        snapshot.blocks,
    )
    broker = FakeBroker()
    backend = IbkrPaperBackend(
        broker.session(), config(), {b.instrument: b for b in snapshot.contract.instruments}
    )
    with pytest.raises(ValueError, match="delayed-data"):
        PaperRunner(
            EpisodeJournal(tmp_path / "j.sqlite"),
            _plan(snapshot, model, purpose="released_to_paper"),
            model.model_text,
            delayed,
            backend,
        )


def test_reregistering_an_episode_that_holds_a_position_is_idempotent(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world, broker=FakeBroker(fill_sides=("BUY",)))
    _register(runner)
    _run_day(runner, world[2], until=_today().open_at.astimezone(timezone.utc) + timedelta(hours=3))
    assert runner.positions()
    runner.register(_today().open_at)  # its own shares are not "unattributable"
    snapshot, model, _ = world
    changed = PaperRunner(
        journal,
        _plan(snapshot, model, capital="90000"),
        model.model_text,
        snapshot,
        runner.backend,
    )
    with pytest.raises(ValueError, match="different plan"):
        changed.register(_today().open_at)


# Live loop: disconnect is a pause --------------------------------------------------------------
def _loop_runner(tmp_path, world, orders):
    snapshot, model, _ = world
    backend = IbkrPaperBackend(
        orders.session(), config(), {b.instrument: b for b in snapshot.contract.instruments}
    )
    runner = PaperRunner(
        EpisodeJournal(tmp_path / "loop.sqlite"),
        _plan(snapshot, model),
        model.model_text,
        snapshot,
        backend,
    )
    start = _today().open_at.astimezone(timezone.utc) - timedelta(minutes=1)
    runner.register(start)
    runner.activate(start)
    return runner, start


def test_disconnect_reconnects_reconciles_and_continues(tmp_path, world):
    orders, data = FakeBroker(), FakeBroker()
    runner, start = _loop_runner(tmp_path, world, orders)
    data_session = data.session(client_id=11)
    ticks = iter(start + timedelta(minutes=m) for m in range(0, 500))

    def clock():
        now = next(ticks)
        if now == start + timedelta(minutes=30):
            data.transport.push(Message("connection_closed"))
        return now

    result = run_session(runner, data_session, _today(), clock=clock, sleep=lambda s: None)
    journal, episode = runner.journal, runner.episode
    assert journal.events(episode, "disconnected")
    assert len(journal.events(episode, "recovery_started")) == 2
    assert len(result["reconciliations_during_session"]) == 2
    assert journal.events(episode, "session_closed")
    assert journal.status(episode) == "active"
    assert len(data.transport.connections) == 2


def test_reconnect_failure_suspends(tmp_path, world):
    orders, data = FakeBroker(), FakeBroker()
    runner, start = _loop_runner(tmp_path, world, orders)
    data_session = data.session(client_id=11)
    ticks = iter(start + timedelta(minutes=m) for m in range(0, 500))

    def clock():
        now = next(ticks)
        if now == start + timedelta(minutes=30):
            data.transport.push(Message("connection_closed"))
            data.transport.refuse_connect = True
        return now

    result = run_session(
        runner, data_session, _today(), clock=clock, sleep=lambda s: None, reconnect_attempts=2
    )
    assert result["status"] == "suspended"
    assert runner.journal.status(runner.episode) == "suspended"


# Restart window and clock skew ------------------------------------------------------------------
def test_restart_window_uses_configured_local_time():
    cfg = config(restart_time="23:45", restart_guard_seconds=900)
    eastern_2340 = datetime(2026, 2, 10, 4, 40, tzinfo=timezone.utc)  # 23:40 EST
    assert in_restart_window(cfg, eastern_2340)
    assert in_restart_window(cfg, eastern_2340 + timedelta(minutes=19))  # 23:59 EST
    assert not in_restart_window(cfg, eastern_2340 + timedelta(hours=1))
    assert not in_restart_window(cfg, datetime(2026, 2, 10, 15, 0, tzinfo=timezone.utc))


def test_clock_skew_is_remeasured_on_interval():
    broker = FakeBroker()
    offset = [0.0]
    session = broker.session(clock=lambda: broker.now() + timedelta(seconds=offset[0]))
    ticks = [0.0]
    monitor = ClockSkewMonitor(session, interval_seconds=60, monotonic=lambda: ticks[0])
    assert monitor() == pytest.approx(0, abs=0.5)
    offset[0] = 4.0
    assert monitor() == pytest.approx(0, abs=0.5)  # cached inside the interval
    ticks[0] = 61.0
    assert monitor() == pytest.approx(4, abs=0.5)


# Features: early close ---------------------------------------------------------------------------
def test_volume_z_score_survives_an_early_close(tmp_path):
    store, ref = build_snapshot(tmp_path, count=30, early_close=10)
    snapshot = store.load(ref)
    calendar = snapshot.contract.calendar.sessions
    _, values = session_features(snapshot, calendar, 25, "KO", ("vol_z_20",))
    assert np.isfinite(values[60:, 0]).all()  # afternoon slots past the 1 pm close


def test_volume_z_score_needs_twenty_sessions_with_the_slot(tmp_path):
    store, ref = build_snapshot(tmp_path, count=22, early_close=10)
    snapshot = store.load(ref)
    calendar = snapshot.contract.calendar.sessions
    _, values = session_features(snapshot, calendar, 20, "KO", ("vol_z_20",))
    assert np.isfinite(values[:42, 0]).all() and np.isnan(values[60:, 0]).all()


# Simulator: delayed retried exit -----------------------------------------------------------------
def test_retried_exit_beyond_fill_window_is_flagged(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    snapshot = store.load(ref)
    session = snapshot.contract.calendar.sessions[0]
    # The horizon exit for the bar-0 entry targets the fill bar 36 minutes after the open;
    # five minutes of one-share volume give a zero participation cap and force retries.
    thin_from = int(session.open_at.timestamp()) + 36 * 60

    class ThinVolume:
        contract = snapshot.contract

        def block(self, instrument, grid, session_id):
            block = snapshot.block(instrument, grid, session_id)
            if grid != "execution" or session_id != session.session_id or block is None:
                return block
            columns = {k: np.array(v, copy=True) for k, v in block.columns.items()}
            thin = (columns["start"] >= thin_from) & (columns["start"] < thin_from + 300)
            columns["volume"][thin & (columns["status"] == VALID)] = 1.0
            return Block(block.instrument, block.grid, block.session_id, columns)

    predictions = {("KO", 0, t): 1.0 for t in range(78)}
    result = simulate(ThinVolume(), plan(), predictions, [0])
    assert "halt_delayed_exit:KO" in result.sessions[0].notes
    assert not result.sessions[0].unresolved


# Collection: resumable, pacing retries -----------------------------------------------------------
def _collect_fixture(tmp_path):
    store, ref = build_snapshot(tmp_path / "source", count=3)
    source = store.load(ref)
    broker = FakeBroker()
    from quant_rsi.brokers.ibkr.history import intraday_params

    for (instrument, grid, session_id), block in source.blocks.items():
        if grid == "decision":
            continue
        b = next(x for x in source.contract.instruments if x.instrument == instrument)
        market = next(s for s in source.contract.calendar.sessions if s.session_id == session_id)
        what = "BID_ASK" if grid == "execution_quotes" else "TRADES"
        params = intraday_params(b, market, 60, what)
        rows = []
        for i in np.flatnonzero(block.columns["status"] == VALID):
            if grid == "execution_quotes":
                bid, ask = block.columns["bid"][i], block.columns["ask"][i]
                rows.append(
                    {
                        "date": str(block.columns["start"][i]),
                        "open": bid,
                        "high": ask,
                        "low": bid,
                        "close": ask,
                        "volume": "-1",
                    }
                )
            else:
                rows.append(
                    {
                        "date": str(block.columns["start"][i]),
                        **{k: float(block.columns[k][i]) for k in ("open", "high", "low", "close")},
                        "volume": str(block.columns["volume"][i]),
                    }
                )
        broker.bars[(b.con_id, "1 min", what, params["end"])] = rows
    return source, broker


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_collection_resumes_from_stored_responses(tmp_path):
    source, broker = _collect_fixture(tmp_path)
    clock = _Clock()
    pacing = PacingLimiter(clock=clock, sleep=clock.sleep, per_contract_seconds=0)
    session = broker.session(client_id=11)
    out = SnapshotStore(tmp_path / "collected")
    after = datetime(2026, 2, 1, tzinfo=timezone.utc)
    original = broker.respond
    calls = {"n": 0}

    def flaky(request):
        if request.kind == "historical_data":
            calls["n"] += 1
            if calls["n"] == 7:
                return [Message("error", request.req_id, {"code": 321, "message": "boom"})]
        return original(request)

    broker.transport.responder = flaky
    with pytest.raises(Exception, match="321"):
        collect_intraday(session, pacing, out, source.contract, clock=lambda: after)
    sent_before = len([r for r in broker.transport.sent if r.kind == "historical_data"])
    broker.transport.responder = original
    events = []
    ref = collect_intraday(
        session, pacing, out, source.contract, clock=lambda: after, progress=events.append
    )
    sent_after = len([r for r in broker.transport.sent if r.kind == "historical_data"])
    assert sum(e["reused"] for e in events) == 6
    assert sent_after - sent_before == len(events) - 6
    collected = out.load(ref)
    for key, block in source.blocks.items():
        for name, values in block.columns.items():
            np.testing.assert_allclose(collected.blocks[key].columns[name], values)


def test_collection_retries_after_pacing_violation(tmp_path):
    source, broker = _collect_fixture(tmp_path)
    clock = _Clock()
    pacing = PacingLimiter(
        clock=clock, sleep=clock.sleep, per_contract_seconds=0, backoff_seconds=60
    )
    session = broker.session(client_id=11)
    original = broker.respond
    paced = {"done": False}

    def pace_once(request):
        if request.kind == "historical_data" and not paced["done"]:
            paced["done"] = True
            return [Message("error", request.req_id, {"code": 162, "message": "pacing"})]
        return original(request)

    broker.transport.responder = pace_once
    start = clock.now
    ref = collect_intraday(
        session,
        pacing,
        SnapshotStore(tmp_path / "c"),
        source.contract,
        clock=lambda: datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    assert ref and clock.now - start >= 60


def test_daily_collection_reuses_identical_requests(tmp_path):
    from quant_rsi.intraday.collect import collect_daily

    calendar = (
        build_snapshot(tmp_path / "s", count=3)[0]
        .load(build_snapshot(tmp_path / "s", count=3)[1])
        .contract.calendar
    )
    broker = FakeBroker()
    ko = binding("KO", 100)
    from quant_rsi.brokers.ibkr.history import daily_params

    params = daily_params(ko, calendar)
    broker.bars[(100, "1 day", "TRADES", params["end"])] = [
        {"date": "20260105", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": "10"}
    ]
    clock = _Clock()
    pacing = PacingLimiter(clock=clock, sleep=clock.sleep)
    store = SnapshotStore(tmp_path / "daily")
    session = broker.session(client_id=11)
    first = collect_daily(session, pacing, store, [ko], calendar)
    second = collect_daily(session, pacing, store, [ko], calendar)
    assert first == second
    assert len([r for r in broker.transport.sent if r.kind == "historical_data"]) == 1
    assert Decimal(first["instruments"]["KO"]["rows"][0]["close"]) == Decimal("1.5")
