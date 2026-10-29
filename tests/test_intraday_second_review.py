"""Regressions for the second (independent) review: broker error codes, cancel races,
ledger anomalies, recovery backfill, connectivity restoration, restart risk state, live
history continuity, stop exits, pre-submission reconciliation and sizing on missing bars."""

# `world` is a module-scoped pytest fixture imported from test_intraday_paper.
# ruff: noqa: F811

from datetime import timedelta, timezone
from decimal import Decimal

import numpy as np
from ibkr_fakes import FakeBroker, config
from intraday_helpers import plan, sessions
from test_intraday_paper import (  # noqa: F401
    HISTORY,
    _payload,
    _plan,
    _register,
    _run_day,
    _runner,
    _today,
    world,
)

from quant_rsi.brokers.ibkr.orders import IbkrPaperBackend
from quant_rsi.brokers.ibkr.transport import Message
from quant_rsi.intraday.features import build_features
from quant_rsi.intraday.live import run_session
from quant_rsi.intraday.model import fit, label_vector
from quant_rsi.intraday.paper import EpisodeJournal, PaperRunner
from quant_rsi.intraday.storage import MISSING, VALID, Block, SnapshotStore


def _requests(broker, kind):
    return [r for r in broker.transport.sent if r.kind == kind]


def _entries(journal):
    return [r for r in journal.intents("ko-p1") if r["intent"].purpose == "entry"]


# Finding 1: order-ID error codes ------------------------------------------------------------------
def test_cancel_acknowledgements_do_not_count_as_rejects(tmp_path, world):
    broker = FakeBroker(fill_sides=(), cancel_ack_error=True)
    runner, _, journal = _runner(tmp_path, world, broker=broker)
    _register(runner)
    outputs = _run_day(runner, world[2])
    assert not any("consecutive_rejects" in o.get("risk", []) for o in outputs)
    assert len(_entries(journal)) > 3  # trading continued after many acknowledged cancels
    assert not [r for r in journal.intents("ko-p1") if r["status"] == "rejected"]
    assert runner.consecutive_rejects == 0


def test_order_warning_neither_rejects_nor_skips_the_deadline_cancel(tmp_path, world):
    broker = FakeBroker(fill_sides=(), warn_on_place=True)
    runner, _, journal = _runner(tmp_path, world, broker=broker)
    _register(runner)
    _run_day(runner, world[2])
    placed = {r.params["order_id"] for r in _requests(broker, "place_order")}
    cancelled = {r.params["order_id"] for r in _requests(broker, "cancel_order")}
    assert placed and placed <= cancelled
    assert not [r for r in journal.intents("ko-p1") if r["status"] == "rejected"]


# Finding 2: cancel races and ledger anomalies -----------------------------------------------------
def test_fill_racing_a_cancel_never_doubles_the_order(tmp_path, world):
    broker = FakeBroker(fill_sides=("BUY",), race_cancel_fill=True)
    runner, _, journal = _runner(tmp_path, world, broker=broker)
    _register(runner)
    _run_day(runner, world[2])
    fills = journal.fills("ko-p1")
    bought = sum(f["quantity"] for f in fills if f["side"] == "buy")
    sold = sum(f["quantity"] for f in fills if f["side"] == "sell")
    assert bought > 0 and bought == sold
    assert runner.ledger_anomalies() == [] and runner.positions() == {}
    assert journal.status("ko-p1") == "active"


def test_oversell_suspends_instead_of_being_dropped(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    _run_day(runner, world[2])
    assert runner.positions() == {}  # flat after the close
    after = _today().close_at.astimezone(timezone.utc) + timedelta(minutes=2)
    sell_order = max(o for o, v in broker.orders.items() if v["action"] == "SELL")
    broker.transport.push(*broker.fill(sell_order))  # a second execution: broker is now short
    runner.step(after)
    assert runner.ledger_anomalies()
    assert journal.status("ko-p1") == "suspended"
    reason = journal.events("ko-p1", "status_suspended")[-1]["payload"]["reason"]
    assert reason.startswith("position_mismatch:oversell:KO")


# Finding 3: recovery backfill and connectivity restoration ---------------------------------------
def _restarted(tmp_path, world, broker, **kwargs):
    snapshot, model, _ = world
    backend = IbkrPaperBackend(
        broker.session(), config(), {b.instrument: b for b in snapshot.contract.instruments}
    )
    return PaperRunner(
        EpisodeJournal(tmp_path / "paper.sqlite"),
        _plan(snapshot, model),
        model.model_text,
        snapshot,
        backend,
        **kwargs,
    )


def _replay(runner, live, before, received):
    for name, grids in live.items():
        for grid in ("decision", "execution"):
            block = grids[grid]
            seconds = 300 if grid == "decision" else 60
            for i in np.flatnonzero(block.columns["status"] == VALID):
                if int(block.columns["start"][i]) + seconds < before.timestamp():
                    runner.on_update(name, grid, _payload(block, i), received)


def test_backfill_arriving_after_the_first_step_is_not_lost(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    restart = _today().open_at.astimezone(timezone.utc) + timedelta(hours=2)
    _run_day(runner, world[2], until=restart - timedelta(hours=1))
    restarted = _restarted(tmp_path, world, broker)
    restarted.start_session(_today())
    restarted.begin_recovery(restart)
    restarted.step(restart)  # TWS has not delivered the backfill yet
    assert all(f.next_final == 0 for f in restarted.finalisers.values())
    _replay(restarted, world[2], restart, restart + timedelta(seconds=3))
    for key in restarted.finalisers:
        restarted.on_backfill_end(*key)
    _run_day(restarted, world[2], start_session=False, start=restart + timedelta(seconds=5))
    decision = restarted.panel.live[("KO", "decision")]["status"]
    assert (decision == VALID).sum() == len(decision)  # nothing lost to STALE
    later = [r for r in _entries(journal) if r["intent"].submit_at > restart]
    assert later, "trading must resume after recovery"


def test_backfill_timeout_finalises_with_an_event(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    restart = _today().open_at.astimezone(timezone.utc) + timedelta(hours=2)
    runner.start_session(_today())
    runner.begin_recovery(restart)
    runner.step(restart + timedelta(seconds=100))
    assert all(f.next_final == 0 for f in runner.finalisers.values())
    runner.step(restart + timedelta(seconds=401))  # plan max_lag_seconds is 400
    assert journal.events("ko-p1", "feed_backfill_timeout")
    assert all(f.next_final > 0 for f in runner.finalisers.values())


def test_stale_bars_do_not_latch_a_trading_halt():
    from quant_rsi.intraday.storage import STALE
    from quant_rsi.intraday.timing import halted

    status = np.array([VALID] + [STALE] * 10 + [VALID] + [MISSING] * 2 + [VALID])
    assert not halted(status, 3).any()


class _LiveTws:
    """Streams today's bars to whichever keepUpToDate subscriptions are current.

    A fresh subscription first receives the backfill of completed bars, then its end marker,
    then live updates, as TWS does. Scripted connectivity errors can be injected by time.
    """

    def __init__(self, broker, snapshot, live):
        self.broker, self.live = broker, live
        self.by_con = {b.con_id: b.instrument for b in snapshot.contract.instruments}
        self.sent: dict[int, set] = {}
        broker.transport.responder = self.respond
        self.base = FakeBroker.respond.__get__(broker)

    def respond(self, request):
        if request.kind == "historical_data" and request.params.get("keep_up_to_date"):
            self.sent[request.req_id] = set()
            return []  # backfill is pushed by `tick`, with the clock
        return self.base(request)

    def tick(self, now):
        for req_id, done in self.sent.items():
            request = next(r for r in self.broker.transport.sent if r.req_id == req_id)
            name = self.by_con[request.params["contract"]["con_id"]]
            grid = "decision" if request.params["bar_size"] == "5 mins" else "execution"
            block, seconds = self.live[name][grid], 300 if grid == "decision" else 60
            first = not done
            for i in np.flatnonzero(block.columns["status"] == VALID):
                end = int(block.columns["start"][i]) + seconds
                if end - 1 <= now.timestamp() and i not in done:
                    done.add(i)
                    kind = "historical_bar" if first else "historical_update"
                    self.broker.transport.push(Message(kind, req_id, _payload(block, i)))
            if first:
                done.add(-1)
                self.broker.transport.push(Message("historical_end", req_id, {}))


def _loop(tmp_path, world, events):
    snapshot, model, live = world
    orders, data = FakeBroker(), FakeBroker()
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
    tws = _LiveTws(data, snapshot, live)
    data_session = data.session(client_id=11)
    ticks = iter(start + timedelta(seconds=30 * k) for k in range(0, 1000))

    def clock():
        now = next(ticks)
        for at, action in events:
            if now == start + at:
                action(data)
        tws.tick(now)
        return now

    result = run_session(runner, data_session, _today(), clock=clock, sleep=lambda s: None)
    return runner, data, result, start


def test_live_loop_trades_and_resumes_after_disconnect(tmp_path, world):
    drop = timedelta(hours=2)
    runner, data, result, start = _loop(
        tmp_path, world, [(drop, lambda b: b.transport.push(Message("connection_closed")))]
    )
    journal = runner.journal
    assert journal.events(runner.episode, "disconnected")
    resumed = [r for r in journal.intents(runner.episode) if r["intent"].submit_at > start + drop]
    assert resumed, "entries continue after the reconnect"
    statuses = [
        e["payload"]["status"]
        for e in journal.events(runner.episode, "bar")
        if e["payload"]["grid"] == "decision" and e["payload"]["instrument"] == "KO"
    ]
    assert statuses.count(VALID) == len(statuses)
    assert result["reconciliation"] == {"findings": []}


def test_connectivity_loss_blocks_entries_and_restore_resubscribes(tmp_path, world):
    lost, back = timedelta(hours=2), timedelta(hours=2, minutes=20)
    runner, data, result, start = _loop(
        tmp_path,
        world,
        [
            (
                lost,
                lambda b: b.transport.push(
                    Message("error", None, {"code": 1100, "message": "lost"})
                ),
            ),
            (
                back,
                lambda b: b.transport.push(
                    Message("error", None, {"code": 1101, "message": "restored, data lost"})
                ),
            ),
        ],
    )
    journal = runner.journal
    assert journal.events(runner.episode, "connectivity_restored")
    during = [
        r
        for r in journal.intents(runner.episode)
        if r["intent"].purpose == "entry" and start + lost < r["intent"].submit_at < start + back
    ]
    assert during == []
    subscriptions = [
        r
        for r in data.transport.sent
        if r.kind == "historical_data" and r.params.get("keep_up_to_date")
    ]
    assert len(subscriptions) == 8  # 4 at start, 4 renewed after 1101
    after = [r for r in journal.intents(runner.episode) if r["intent"].submit_at > start + back]
    assert after


# Finding 4: restart keeps daily risk state --------------------------------------------------------
def test_restart_keeps_the_daily_baseline_and_reject_streak(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    noon = _today().open_at.astimezone(timezone.utc) + timedelta(hours=3)
    _run_day(runner, world[2], until=noon)
    baseline = runner.session_start_equity
    journal.append("ko-p1", "rejected", noon, {"intent_id": "x"})
    journal.append("ko-p1", "rejected", noon, {"intent_id": "y"})
    restarted = _restarted(tmp_path, world, broker)
    restarted.start_session(_today())
    assert restarted.session_start_equity == baseline
    assert restarted.consecutive_rejects == 2
    assert len(journal.events("ko-p1", "session_started")) == 1


# Finding 5: live features see true previous sessions ----------------------------------------------
def test_live_panel_includes_archived_sessions_and_detects_gaps(tmp_path, world):
    archive = SnapshotStore(tmp_path / "archive")
    runner, broker, journal = _runner(tmp_path, world)
    runner.archive = archive
    _register(runner)
    _run_day(runner, world[2])
    assert journal.events("ko-p1", "session_archived")
    day2 = sessions(HISTORY + 2)[-1]
    runner.start_session(day2)
    day1 = _today().session_id
    assert runner.panel.continuous
    assert runner.panel.contract.calendar.sessions[-2].session_id == day1
    archived = runner.panel.block("KO", "decision", day1)
    np.testing.assert_array_equal(
        archived.columns["close"],
        runner.panel.archived_blocks[("KO", "decision", day1)].columns["close"],
    )
    far = sessions(HISTORY + 12)[-1]
    runner.start_session(far)
    assert not runner.panel.continuous
    assert runner.panel.block("KO", "decision", day1) is None
    assert journal.events("ko-p1", "history_discontinuity")


# Finding 6: stop exits positions; reconcile before submission -------------------------------------
def test_stop_exits_open_positions_at_the_next_fill_bar(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world, broker=FakeBroker(fill_sides=("BUY",)))
    _register(runner)
    now = _run_day(
        runner, world[2], step_seconds=10, stop_when=lambda now: bool(runner.positions())
    )
    broker.fill_sides = {"BUY", "SELL"}
    runner.stop("stop-1", now)
    _run_day(runner, world[2], start_session=False, start=now, until=now + timedelta(minutes=3))
    reasons = [e["payload"]["reason"] for e in journal.events("ko-p1", "intent")]
    assert "stop" in reasons and runner.positions() == {}


def test_reconcile_runs_before_every_submission(tmp_path, world):
    runner, broker, journal = _runner(tmp_path, world)
    _register(runner)
    _run_day(runner, world[2])
    assert len(_requests(broker, "open_orders")) >= len(_requests(broker, "place_order")) > 0


# Finding 7: sizing never uses a missing bar -------------------------------------------------------
def test_missing_decision_bar_does_not_crash_sizing(tmp_path, world):
    snapshot, _, live = world
    leaves = ("minutes_since_open",)
    development = plan(leaves=list(leaves))
    table = build_features(snapshot, leaves)
    model = fit(
        table.values, label_vector(snapshot, table, development), table.names, seed=1, min_rows=50
    )
    holes = {}
    for name, grids in live.items():
        holes[name] = dict(grids)
        block = grids["decision"]
        columns = {k: np.array(v, copy=True) for k, v in block.columns.items()}
        for i in range(20, 30):
            columns["status"][i] = MISSING
            for k in ("open", "high", "low", "close", "volume"):
                columns[k][i] = np.nan
        holes[name]["decision"] = Block(block.instrument, "decision", block.session_id, columns)
    broker = FakeBroker()
    backend = IbkrPaperBackend(
        broker.session(), config(), {b.instrument: b for b in snapshot.contract.instruments}
    )
    runner = PaperRunner(
        EpisodeJournal(tmp_path / "paper.sqlite"),
        _plan(
            snapshot,
            model,
            leaves=leaves,
            signal={**development.signal.model_dump(), "entry_threshold": -1.0},
        ),
        model.model_text,
        snapshot,
        backend,
    )
    _register(runner)
    _run_day(runner, holes)  # must not raise
    traded = {r["intent"].decision_index for r in _entries(runner.journal)}
    assert traded and not traded & set(range(20, 30))
    assert Decimal(runner.cash()) > 0
