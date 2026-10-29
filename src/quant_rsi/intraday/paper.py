"""Prospective intraday paper episodes against an IBKR paper account (design section 6.2).

The runner is driven by `step(now)`. Each step finalises bars by the clock, makes decisions
on newly final decision bars, submits intents only at their scheduled fill-bar start and
before their deadline, applies broker events, flattens before the close and evaluates risk.
Every fact is appended to an SQLite journal; positions and cash are derived from journaled
fills of this episode only (a virtual sub-account), never from the broker account balance.
"""

import json
import math
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import AwareDatetime, Field

from quant_rsi.brokers.contracts import BrokerEvent, OrderIntent
from quant_rsi.brokers.ibkr.connection import IbkrSession
from quant_rsi.brokers.ibkr.history import intraday_params
from quant_rsi.brokers.ibkr.stream import BarFinaliser, FeedEvent, FinalBar, availability_check
from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.intraday.contracts import (
    EXECUTION_BAR_SECONDS,
    CostModel,
    ExecutionTiming,
    SignalPolicy,
    validate_timing,
)
from quant_rsi.intraday.features import session_features, validate_leaves
from quant_rsi.intraday.model import predict
from quant_rsi.intraday.risk import RiskLimits, RiskState, evaluate
from quant_rsi.intraday.schedule import ForwardSessionSchedule
from quant_rsi.intraday.simulator import fill_price
from quant_rsi.intraday.storage import MISSING, VALID, Block, Snapshot, SnapshotStore
from quant_rsi.intraday.timing import (
    NO_INDEX,
    entry_allowed,
    fill_bar_start,
    halted,
    session_timing,
)
from quant_rsi.market.market_contracts import CalendarSpec, MarketSession, decimal_text, utc_text

ACCOUNT_HOLDING = ("registered", "active", "stopping", "suspended")
# "cancelling" stays active until the broker confirms the cancel or a fill completes it, so
# no replacement order can be sent while the old one might still fill.
ACTIVE_INTENT = frozenset({"pending", "submitting", "submitted", "cancelling"})
TERMINAL_INTENT = {"filled", "cancelled", "abandoned_unsubmitted", "rejected", "expired"}


class PaperPlan(StrictModel):
    schema_version: Literal[1]
    episode_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    purpose: Literal["observational_baseline_only", "released_to_paper"]
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    leaves: tuple[str, ...]
    context_missing_tolerance: int = Field(ge=0, le=12)
    timing: ExecutionTiming
    signal: SignalPolicy
    costs: CostModel
    capital: str
    limit_offset_bps: str
    risk: RiskLimits
    schedule: ForwardSessionSchedule
    expires_at: AwareDatetime
    max_lag_seconds: int = Field(gt=0, le=3600)


# Journal ---------------------------------------------------------------------------------------
class EpisodeJournal:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY, account TEXT NOT NULL, status TEXT NOT NULL,
                plan TEXT NOT NULL, plan_digest TEXT NOT NULL, registered_at TEXT NOT NULL,
                activated_at TEXT);
            CREATE TABLE IF NOT EXISTS events (
                episode_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
                at TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY (episode_id, seq));
            CREATE TABLE IF NOT EXISTS intents (
                intent_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, order_id INTEGER, filled INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS fills (
                exec_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, intent_id TEXT NOT NULL,
                instrument TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL,
                price TEXT NOT NULL, fee TEXT, at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
                episode_id TEXT NOT NULL, request_id TEXT NOT NULL, kind TEXT NOT NULL,
                PRIMARY KEY (episode_id, request_id));
            """
        )

    def transaction(self):
        return _Transaction(self.db)

    def register(self, plan: PaperPlan, account: str, at: datetime) -> None:
        text = canonical(plan.model_dump(mode="json")).decode()
        with self.transaction():
            existing = self.db.execute(
                "SELECT plan_digest FROM episodes WHERE episode_id = ?", (plan.episode_id,)
            ).fetchone()
            if existing:
                if existing[0] != digest(text.encode()):
                    raise ValueError("episode ID already registered with a different plan")
                return
            busy = self.db.execute(
                f"SELECT episode_id FROM episodes WHERE account = ? AND status IN "
                f"({','.join('?' * len(ACCOUNT_HOLDING))})",
                (account, *ACCOUNT_HOLDING),
            ).fetchone()
            if busy:
                raise ValueError(
                    f"paper account already has active episode {busy[0]}; stop and finish it "
                    "first or configure a second paper account"
                )
            self.db.execute(
                "INSERT INTO episodes VALUES (?, ?, 'registered', ?, ?, ?, NULL)",
                (plan.episode_id, account, text, digest(text.encode()), utc_text(at)),
            )
            self._append(plan.episode_id, "registered", at, {"account": account})

    def status(self, episode_id: str) -> str:
        row = self.db.execute(
            "SELECT status FROM episodes WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        return row[0]

    def set_status(self, episode_id: str, status: str, at: datetime, reason: str) -> None:
        with self.transaction():
            self.db.execute(
                "UPDATE episodes SET status = ? WHERE episode_id = ?", (status, episode_id)
            )
            if status == "active":
                self.db.execute(
                    "UPDATE episodes SET activated_at = ? WHERE episode_id = ?",
                    (utc_text(at), episode_id),
                )
            self._append(episode_id, f"status_{status}", at, {"reason": reason})

    def append(self, episode_id: str, kind: str, at: datetime, payload: dict) -> None:
        with self.transaction():
            self._append(episode_id, kind, at, payload)

    def _append(self, episode_id, kind, at, payload) -> None:
        (seq,) = self.db.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE episode_id = ?", (episode_id,)
        ).fetchone()
        self.db.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
            (episode_id, seq, kind, utc_text(at), json.dumps(payload, sort_keys=True)),
        )

    def events(self, episode_id: str, kind: str | None = None) -> list[dict]:
        query = "SELECT seq, kind, at, payload FROM events WHERE episode_id = ?"
        args: tuple = (episode_id,)
        if kind:
            query, args = query + " AND kind = ?", (episode_id, kind)
        return [
            {"seq": s, "kind": k, "at": a, "payload": json.loads(p)}
            for s, k, a, p in self.db.execute(query + " ORDER BY seq", args)
        ]

    def put_intent(self, intent: OrderIntent) -> bool:
        payload = canonical(intent.model_dump(mode="json")).decode()
        row = self.db.execute(
            "SELECT payload FROM intents WHERE intent_id = ?", (intent.intent_id,)
        ).fetchone()
        if row:
            if row[0] != payload:
                raise ValueError(f"intent {intent.intent_id} reused with different content")
            return False
        self.db.execute(
            "INSERT INTO intents VALUES (?, ?, 'pending', ?, NULL, 0)",
            (intent.intent_id, intent.episode_id, payload),
        )
        return True

    def intents(self, episode_id: str, statuses: Iterable[str] | None = None) -> list[dict]:
        rows = self.db.execute(
            "SELECT intent_id, status, payload, order_id, filled FROM intents "
            "WHERE episode_id = ? ORDER BY intent_id",
            (episode_id,),
        ).fetchall()
        wanted = None if statuses is None else set(statuses)
        return [
            {
                "intent": OrderIntent.model_validate_json(p),
                "status": s,
                "order_id": o,
                "filled": f,
            }
            for _, s, p, o, f in rows
            if wanted is None or s in wanted
        ]

    def update_intent(self, intent_id: str, **values) -> None:
        sets = ", ".join(f"{key} = ?" for key in values)
        self.db.execute(
            f"UPDATE intents SET {sets} WHERE intent_id = ?", (*values.values(), intent_id)
        )

    def add_fill(self, episode_id: str, intent: OrderIntent, event: BrokerEvent) -> bool:
        try:
            self.db.execute(
                "INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    event.exec_id,
                    episode_id,
                    intent.intent_id,
                    intent.instrument,
                    intent.side,
                    event.quantity,
                    event.price,
                    utc_text(event.at),
                ),
            )
        except sqlite3.IntegrityError:
            return False  # duplicate execution after reconnect
        return True

    def fills(self, episode_id: str) -> list[dict]:
        return [
            dict(
                zip(
                    (
                        "exec_id",
                        "intent_id",
                        "instrument",
                        "side",
                        "quantity",
                        "price",
                        "fee",
                        "at",
                    ),
                    row,
                    strict=True,
                )
            )
            for row in self.db.execute(
                "SELECT exec_id, intent_id, instrument, side, quantity, price, fee, at FROM fills "
                "WHERE episode_id = ? ORDER BY rowid",
                (episode_id,),
            )
        ]

    def claim_request(self, episode_id: str, request_id: str, kind: str) -> bool:
        try:
            self.db.execute("INSERT INTO requests VALUES (?, ?, ?)", (episode_id, request_id, kind))
        except sqlite3.IntegrityError:
            return False
        return True


class _Transaction:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.db.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, kind, value, traceback):
        self.db.execute("COMMIT" if kind is None else "ROLLBACK")
        return False


# Live panel ------------------------------------------------------------------------------------
class LivePanel:
    """History snapshot, archived earlier live sessions and today's session.

    Cross-session leaves (`gap`, `vol_z_20`) must see the true previous sessions. Earlier
    sessions of the episode are archived at each close and included here. If the most recent
    known session ended more than `max_session_gap_days` before today, the panel is not
    continuous and earlier sessions are hidden, so those leaves are missing, not stale.
    """

    def __init__(
        self,
        history: Snapshot,
        today: MarketSession,
        archived: Sequence[tuple[MarketSession, dict]] = (),
        *,
        max_session_gap_days: int = 5,
    ):
        known = {s.session_id: s for s in history.contract.calendar.sessions}
        self.archived_blocks: dict[tuple[str, str, str], Block] = {}
        for session, blocks in archived:
            known.setdefault(session.session_id, session)
            for (instrument, grid), block in blocks.items():
                self.archived_blocks[(instrument, grid, session.session_id)] = block
        sessions = tuple(
            sorted(
                (s for s in known.values() if s.close_at < today.open_at),
                key=lambda s: s.open_at,
            )
        )
        self.continuous = bool(sessions) and (
            today.open_at - sessions[-1].close_at <= timedelta(days=max_session_gap_days)
        )
        calendar = CalendarSpec.model_construct(
            calendar_id=history.contract.calendar.calendar_id,
            timezone=history.contract.calendar.timezone,
            sessions=(*sessions, today),
        )
        self.contract = history.contract.model_copy(update={"calendar": calendar})
        self.history, self.today = history, today
        self.live: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        bar_seconds = {
            "decision": self.contract.grid.bar_seconds,
            "execution": EXECUTION_BAR_SECONDS,
        }
        opened = int(today.open_at.timestamp())
        closed = int(today.close_at.timestamp())
        for binding in self.contract.instruments:
            for grid, seconds in bar_seconds.items():
                starts = np.arange(opened, closed, seconds, dtype=np.int64)
                n = len(starts)
                self.live[(binding.instrument, grid)] = {
                    "start": starts,
                    **{k: np.full(n, np.nan) for k in ("open", "high", "low", "close", "volume")},
                    "status": np.full(n, MISSING, np.uint8),
                }

    @property
    def index(self) -> int:
        return len(self.contract.calendar.sessions) - 1

    def record(self, bar: FinalBar) -> None:
        columns = self.live[(bar.instrument, bar.grid)]
        columns["status"][bar.index] = bar.status
        if bar.status == VALID:
            for name in ("open", "high", "low", "close", "volume"):
                columns[name][bar.index] = getattr(bar, name)

    def block(self, instrument: str, grid: str, session_id: str) -> Block | None:
        if session_id == self.today.session_id:
            columns = self.live.get((instrument, grid))
            return None if columns is None else Block(instrument, grid, session_id, columns)
        if not self.continuous:
            return None
        archived = self.archived_blocks.get((instrument, grid, session_id))
        return archived or self.history.block(instrument, grid, session_id)

    def today_blocks(self) -> list[Block]:
        return [
            Block(instrument, grid, self.today.session_id, columns)
            for (instrument, grid), columns in sorted(self.live.items())
        ]


# Live data feed --------------------------------------------------------------------------------
class LiveFeed:
    """keepUpToDate subscriptions on the data client; routes updates by request ID."""

    def __init__(self, session: IbkrSession, contract, today: MarketSession):
        self.session, self.routes = session, {}
        grids = {"decision": contract.grid.bar_seconds, "execution": EXECUTION_BAR_SECONDS}
        for binding in contract.instruments:
            for grid, seconds in grids.items():
                params = intraday_params(binding, today, seconds, "TRADES")
                params.update(keep_up_to_date=True, duration="1 D")
                req_id = session.next_request_id()
                session.send("historical_data", params, req_id=req_id)
                self.routes[req_id] = (binding.instrument, grid)

    def poll(self) -> list[tuple[str, str, str, dict]]:
        """(instrument, grid, kind, payload); kind is "update" or "backfill_end"."""
        updates = []
        for message in self.session.drain():
            route = self.routes.get(message.req_id)
            if route and message.kind in {"historical_update", "historical_bar"}:
                updates.append((*route, "update", dict(message.payload)))
            elif route and message.kind == "historical_end":
                updates.append((*route, "backfill_end", dict(message.payload)))
        return updates


# Runner ----------------------------------------------------------------------------------------
@dataclass
class _Position:
    quantity: int
    entry_bar: int
    entry_session: str
    cost: Decimal


class PaperRunner:
    def __init__(
        self,
        journal: EpisodeJournal,
        plan: PaperPlan,
        model_text: str,
        history: Snapshot,
        backend,
        *,
        restart_window: Callable[[datetime], bool] = lambda now: False,
        clock_skew: Callable[[], float] = lambda: 0.0,
        connectivity_lost: Callable[[], bool] = lambda: False,
        archive: SnapshotStore | None = None,
        max_session_gap_days: int = 5,
    ):
        if digest(model_text.encode()) != plan.model_sha256:
            raise ValueError("model text does not match the sealed model digest")
        if history.ref != plan.snapshot_ref:
            raise ValueError("history snapshot does not match the plan")
        if plan.purpose == "released_to_paper" and history.contract.data_entitlement != "realtime":
            raise ValueError("a delayed-data snapshot cannot feed a released paper episode")
        validate_timing(history.contract, plan.timing)
        validate_leaves(history.contract, plan.leaves)
        self.journal, self.plan, self.model_text = journal, plan, model_text
        self.history, self.backend = history, backend
        self.restart_window, self.clock_skew = restart_window, clock_skew
        self.connectivity_lost = connectivity_lost
        self.archive, self.max_session_gap_days = archive, max_session_gap_days
        self.episode = plan.episode_id
        self.targets = tuple(sorted(history.contract.targets))
        self.panel: LivePanel | None = None
        self.finalisers: dict[tuple[str, str], BarFinaliser] = {}
        self.timing = None
        self.decided: set[int] = set()
        self.flattened = False
        self.session_closed = False
        self.consecutive_rejects = 0
        self.stale = False
        self.session_start_equity = Decimal(plan.capital)
        self.now = datetime.now(timezone.utc)
        self.recovery_cutoff: datetime | None = None

    # Registration ------------------------------------------------------------------------------
    def register(self, now: datetime) -> None:
        try:
            self.journal.status(self.episode)
        except KeyError:
            pass
        else:
            # Already registered: only verify the plan is unchanged; never re-check positions
            # the episode itself may now hold.
            self.journal.register(self.plan, self.backend.account, now)
            return
        reconciliation = self.backend.reconcile(now - timedelta(days=1))
        con_ids = {
            b.con_id for b in self.history.contract.instruments if b.instrument in self.targets
        }
        held = {c: q for c, q in reconciliation.positions.items() if c in con_ids and q != 0}
        if held:
            raise ValueError(
                f"paper account already holds episode instruments {sorted(held)}; refusing to "
                "register because those shares cannot be attributed to the episode"
            )
        if self.backend.available_funds() < Decimal(self.plan.capital):
            raise ValueError("paper account available funds do not cover the episode capital")
        self.journal.register(self.plan, self.backend.account, now)

    def activate(self, now: datetime) -> None:
        if self.journal.status(self.episode) != "registered":
            raise ValueError("only a registered episode can be activated")
        self.journal.set_status(self.episode, "active", now, "activated")

    def stop(self, request_id: str, now: datetime) -> None:
        """Durable stop: cancel entries, exit positions at the next fill bar, then end flat."""
        if not self.journal.claim_request(self.episode, request_id, "stop"):
            return
        if self.journal.status(self.episode) in {"active", "registered"}:
            self.journal.set_status(self.episode, "stopping", now, f"stop:{request_id}")
        for record in self._active_intents("entry"):
            self._cancel_or_abandon(record, now, "stopped")

    # Session lifecycle -------------------------------------------------------------------------
    def start_session(self, session: MarketSession) -> None:
        self.panel = LivePanel(
            self.history,
            session,
            self._archived_sessions(session),
            max_session_gap_days=self.max_session_gap_days,
        )
        if not self.panel.continuous:
            self.journal.append(
                self.episode, "history_discontinuity", self.now, {"session_id": session.session_id}
            )
        contract = self.panel.contract
        self.finalisers = {
            (b.instrument, grid): BarFinaliser(
                b.instrument,
                grid,
                session,
                seconds,
                buffer_seconds=contract.availability_buffer_seconds,
                max_lag_seconds=self.plan.max_lag_seconds,
            )
            for b in contract.instruments
            for grid, seconds in (
                ("decision", contract.grid.bar_seconds),
                ("execution", EXECUTION_BAR_SECONDS),
            )
        }
        self.timing = session_timing(
            session,
            contract.grid.bar_seconds,
            self.plan.timing,
            self.plan.signal.flatten_offset_seconds,
        )
        self.decided, self.flattened, self.session_closed = set(), False, False
        # Risk state survives restarts: the daily baseline is recorded once per session and
        # the reject streak is derived from the journal, never reset by a restart.
        started = [
            e["payload"]
            for e in self.journal.events(self.episode, "session_started")
            if e["payload"]["session_id"] == session.session_id
        ]
        if started:
            self.session_start_equity = Decimal(started[0]["start_equity"])
        else:
            self.session_start_equity = self.cash() + self._position_value()
            self.journal.append(
                self.episode,
                "session_started",
                self.now,
                {
                    "session_id": session.session_id,
                    "start_equity": format(self.session_start_equity, "f"),
                },
            )
        self.consecutive_rejects = self._reject_streak()

    def _reject_streak(self) -> int:
        streak = 0
        for event in self.journal.events(self.episode):
            if event["kind"] == "rejected":
                streak += 1
            elif event["kind"] == "fill":
                streak = 0
        return streak

    def _archived_sessions(self, today: MarketSession) -> list[tuple[MarketSession, dict]]:
        if self.archive is None:
            return []
        result = []
        for event in self.journal.events(self.episode, "session_archived"):
            session = MarketSession.model_validate(event["payload"]["session"], strict=False)
            if session.session_id == today.session_id:
                continue
            blocks = {}
            for entry in event["payload"]["blocks"]:
                block = self.archive.load_block(entry)
                blocks[(block.instrument, block.grid)] = block
            result.append((session, blocks))
        return result

    def begin_recovery(self, now: datetime) -> None:
        """Mark a (re)start: bars already due before `now` are replays, not live receipts.

        Replayed bars still fill the panel so features work, but they are excluded from the
        arrival-delay statistics, and a decision whose submission window has already passed is
        skipped instead of producing intents that could only be abandoned.
        """
        self.recovery_cutoff = now
        for finaliser in self.finalisers.values():
            finaliser.await_backfill(now)
        self.journal.append(self.episode, "recovery_started", now, {})

    def on_backfill_end(self, instrument: str, grid: str) -> None:
        self.finalisers[(instrument, grid)].backfill_complete()

    def on_update(self, instrument: str, grid: str, payload: dict, received_at: datetime) -> None:
        for event in self.finalisers[(instrument, grid)].on_update(payload, received_at):
            self._feed_event(event)

    # Main step -----------------------------------------------------------------------------------
    def step(self, now: datetime) -> dict:
        if self.panel is None:
            raise RuntimeError("start_session first")
        status = self.journal.status(self.episode)
        if status in {"terminated", "registered"}:
            return {"status": status}
        self.now = now
        new_decisions = self._finalise(now)
        self._apply_broker_events(self.backend.poll())
        self._check_ledger(now)
        reasons = self.risk_reasons(now)
        for t in new_decisions:
            window_end = int(self.timing.fill_start[t]) + self.plan.timing.max_fill_delay_seconds
            if int(self.timing.fill_start[t]) != NO_INDEX and now.timestamp() >= window_end:
                self.journal.append(self.episode, "decision_window_passed", now, {"bar": t})
                continue
            self._decide(t, now, blocked=bool(reasons) or status != "active")
        self._exits_and_flatten(now)
        self._submit_and_expire(now)
        self._apply_broker_events(self.backend.poll())
        self._check_ledger(now)
        self._close_session(now)
        return {"status": self.journal.status(self.episode), "risk": reasons}

    def _check_ledger(self, now: datetime) -> None:
        anomalies = self.ledger_anomalies()
        if anomalies:
            self._suspend(now, "position_mismatch:" + ",".join(anomalies))

    def _finalise(self, now: datetime) -> list[int]:
        decisions = []
        for (instrument, grid), finaliser in self.finalisers.items():
            for item in finaliser.on_clock(now):
                if isinstance(item, FeedEvent):
                    self._feed_event(item)
                    continue
                self.panel.record(item)
                due = item.end + self.panel.contract.availability_buffer_seconds
                recovered = (
                    self.recovery_cutoff is not None and due < self.recovery_cutoff.timestamp()
                )
                self.journal.append(
                    self.episode,
                    "bar",
                    now,
                    {
                        "instrument": instrument,
                        "grid": grid,
                        "index": item.index,
                        "status": int(item.status),
                        "arrival_delay": None if recovered else item.arrival_delay_seconds,
                        "recovered": recovered,
                    },
                )
        bars = self.timing.bars
        for t in range(bars):
            if t in self.decided:
                continue
            ready = all(
                self.finalisers[(b.instrument, "decision")].next_final > t
                for b in self.panel.contract.instruments
            )
            if not ready:
                break
            self.decided.add(t)
            decisions.append(t)
        return decisions

    def _feed_event(self, event: FeedEvent) -> None:
        if event.kind == "stale_feed":
            self.stale = True
        elif event.kind == "feed_resumed":
            self.stale = False
        self.journal.append(
            self.episode,
            f"feed_{event.kind}",
            event.at,
            {
                "instrument": event.instrument,
                "detail": event.detail,
            },
        )

    # Decisions ---------------------------------------------------------------------------------
    def _score(self, target: str, t: int) -> tuple[float, tuple[str, ...]]:
        rows, values = session_features(
            self.panel,
            self.panel.contract.calendar.sessions,
            self.panel.index,
            target,
            self.plan.leaves,
            context_tolerance=self.plan.context_missing_tolerance,
            last_bar=t,
        )
        row = values[t : t + 1]
        if not np.isfinite(row).all():
            missing = tuple(
                n for n, v in zip(self.plan.leaves, row[0], strict=True) if not math.isfinite(v)
            )
            return math.nan, missing
        return float(predict(self.model_text, row, self.plan.leaves)[0]), ()

    def _decide(self, t: int, now: datetime, *, blocked: bool) -> None:
        signal = self.plan.signal
        positions = self.positions()
        for target in self.targets:
            score, missing = self._score(target, t)
            self.journal.append(
                self.episode,
                "prediction" if not missing else "no_prediction",
                now,
                {
                    "instrument": target,
                    "bar": t,
                    "score": None if missing else score,
                    "missing_leaves": list(missing),
                },
            )
            position = positions.get(target)
            if position is not None:
                overnight = position.entry_session != self.panel.today.session_id
                if overnight or self._active_intent(target, "exit"):
                    continue  # overnight positions exit in _exits_and_flatten
                if t >= position.entry_bar + signal.horizon_bars:
                    self._exit_intent(target, position, t, "horizon")
                elif math.isfinite(score) and score < signal.exit_threshold:
                    self._exit_intent(target, position, t, "signal")
                continue
            if blocked or self._active_intent(target, "entry"):
                continue
            decision = self.panel.block(target, "decision", self.panel.today.session_id)
            if halted(decision.columns["status"][: t + 1], signal.halt_missing_bars)[-1]:
                continue
            if not entry_allowed(self.timing, t, signal.horizon_bars) or not math.isfinite(score):
                continue
            occupied = len(positions) + len(self._active_intents("entry"))
            if score <= signal.entry_threshold or occupied >= signal.max_positions:
                continue
            if decision.columns["status"][t] != VALID:
                continue  # quantities come only from a valid decision-bar close
            close = Decimal(repr(round(float(decision.columns["close"][t]), 6)))
            budget = self.session_start_equity * Decimal(signal.notional_fraction)
            lots = (budget / close / signal.lot_size).to_integral_value(rounding=ROUND_FLOOR)
            quantity = int(lots) * signal.lot_size
            if quantity > 0:
                self._new_intent(
                    target, "buy", "entry", quantity, close, t, int(self.timing.fill_start[t])
                )

    def _exit_intent(self, target: str, position: _Position, t: int, reason: str) -> None:
        decision = self.panel.block(target, "decision", self.panel.today.session_id)
        reference = decision.columns["close"][t]
        price = (
            Decimal(repr(round(float(reference), 6)))
            if math.isfinite(reference)
            else (self._last_close(target))
        )
        start = int(self.timing.fill_start[t]) if t < self.timing.bars else NO_INDEX
        self._new_intent(target, "sell", "exit", position.quantity, price, t, start, reason=reason)

    def _new_intent(self, target, side, purpose, quantity, reference, t, start, *, reason=""):
        if start == NO_INDEX:
            self.journal.append(
                self.episode,
                "intent_unschedulable",
                self.now,
                {"instrument": target, "bar": t, "side": side},
            )
            return
        offset = Decimal(self.plan.limit_offset_bps) / Decimal(10000)
        limit = reference * (1 + offset if side == "buy" else 1 - offset)
        submit_at = datetime.fromtimestamp(start, timezone.utc)
        session_id = self.panel.today.session_id
        attempt = len(
            [
                i
                for i in self.journal.intents(self.episode)
                if i["intent"].session_id == session_id
                and i["intent"].decision_index == t
                and i["intent"].instrument == target
                and i["intent"].side == side
            ]
        )
        intent = OrderIntent(
            intent_id=f"{self.episode}:{session_id}:{t}:{target}:{side}:{attempt}",
            episode_id=self.episode,
            instrument=target,
            side=side,
            purpose=purpose,
            quantity=quantity,
            limit_price=decimal_text(format(limit.quantize(Decimal("0.01")), "f"), positive=True),
            decision_index=t,
            session_id=session_id,
            submit_at=submit_at,
            submit_deadline=submit_at + timedelta(seconds=self.plan.timing.max_fill_delay_seconds),
        )
        with self.journal.transaction():
            if self.journal.put_intent(intent):
                self.journal._append(
                    self.episode,
                    "intent",
                    submit_at,
                    {
                        "intent_id": intent.intent_id,
                        "reason": reason or purpose,
                    },
                )

    def _exits_and_flatten(self, now: datetime) -> None:
        today = self.panel.today.session_id
        for target, position in self.positions().items():
            # A position carried over a close (an overnight breach) exits at the first
            # opportunity, exactly as the replay simulator does.
            if position.entry_session != today and not self._active_intent(target, "exit"):
                start = fill_bar_start(self.panel.today, int(now.timestamp()))
                try:
                    reference = self._last_close(target)
                except ValueError:
                    continue  # no price yet today; retry on the next step
                self._new_intent(
                    target,
                    "sell",
                    "exit",
                    position.quantity,
                    reference,
                    self.timing.bars,
                    start,
                    reason="overnight",
                )
        if not self.flattened and now.timestamp() >= self.timing.flatten_at:
            self.flattened = True
            self.journal.append(self.episode, "flatten", now, {})
            for intent in self._active_intents("entry"):
                self._cancel_or_abandon(intent, now, "flatten")
        stopping = self.journal.status(self.episode) == "stopping"
        if not (self.flattened or stopping):
            return
        for target, position in self.positions().items():
            if self._active_intent(target, "exit"):
                continue
            start = fill_bar_start(self.panel.today, int(now.timestamp()))
            self._new_intent(
                target,
                "sell",
                "exit",
                position.quantity,
                self._last_close(target),
                self.timing.bars,
                start,
                reason="flatten" if self.flattened else "stop",
            )

    def _last_close(self, target: str) -> Decimal:
        columns = self.panel.live[(target, "decision")]
        valid = np.flatnonzero(columns["status"] == VALID)
        if not len(valid):
            raise ValueError(f"no valid price for {target} today")
        return Decimal(repr(round(float(columns["close"][valid[-1]]), 6)))

    # Submission window -------------------------------------------------------------------------
    def _submit_and_expire(self, now: datetime) -> None:
        due = [
            r
            for r in self.journal.intents(self.episode, {"pending"})
            if r["intent"].submit_at <= now < r["intent"].submit_deadline
        ]
        if due:
            # Design 6.3: reconcile open orders, executions and positions before submitting.
            self.reconcile(now)
        for record in self.journal.intents(self.episode, ACTIVE_INTENT):
            intent, status = record["intent"], record["status"]
            if status == "cancelling":
                continue  # resolved by the broker's cancel or fill, or by reconcile
            if status == "submitting":
                # Submission outcome unknown (crash between send and commit). Never re-send;
                # only broker records matched by orderRef in `reconcile` resolve it.
                if now >= intent.submit_deadline:
                    self._resolve_unknown(record, now)
                continue
            if now >= intent.submit_deadline:
                self._cancel_or_abandon(record, now, "deadline")
            elif status == "pending" and now >= intent.submit_at:
                if intent.purpose == "entry":
                    blocked = self._entry_block_reasons(now)
                    if blocked:
                        self._cancel_or_abandon(record, now, "blocked:" + ",".join(blocked))
                        continue
                with self.journal.transaction():
                    self.journal.update_intent(intent.intent_id, status="submitting")
                    self.journal._append(
                        self.episode, "submitting", now, {"intent_id": intent.intent_id}
                    )
                event = self.backend.submit(intent)
                with self.journal.transaction():
                    self.journal.update_intent(
                        intent.intent_id, status="submitted", order_id=event.order_id
                    )
                    self.journal._append(
                        self.episode,
                        "submitted",
                        now,
                        {
                            "intent_id": intent.intent_id,
                            "order_id": event.order_id,
                        },
                    )

    def _entry_block_reasons(self, now: datetime) -> list[str]:
        """Entries are re-checked at submission: a stop or risk block since the decision wins."""
        reasons = list(self.risk_reasons(now))
        status = self.journal.status(self.episode)
        if status != "active":
            reasons.append(f"episode_{status}")
        if self.flattened:
            reasons.append("after_flatten")
        return reasons

    def _resolve_unknown(self, record: dict, now: datetime) -> None:
        """A `submitting` intent past its deadline that the broker has never reported."""
        intent = record["intent"]
        with self.journal.transaction():
            self.journal.update_intent(intent.intent_id, status="abandoned_unsubmitted")
            self.journal._append(
                self.episode,
                "abandoned_unsubmitted",
                now,
                {"intent_id": intent.intent_id, "reason": "submission_unknown_no_broker_record"},
            )

    def _cancel_or_abandon(self, record: dict, now: datetime, reason: str) -> None:
        intent = record["intent"]
        if record["status"] == "submitting":
            return  # cannot cancel without a broker order ID; reconcile resolves it
        if record["status"] == "cancelling":
            return
        if record["status"] == "pending":
            new = "abandoned_unsubmitted"
        else:
            if record["order_id"] is not None:
                self.backend.cancel(record["order_id"])
            new = "cancelling"
        with self.journal.transaction():
            self.journal.update_intent(intent.intent_id, status=new)
            self.journal._append(
                self.episode,
                new,
                now,
                {
                    "intent_id": intent.intent_id,
                    "reason": reason,
                    "filled": record["filled"],
                },
            )

    # Broker events -----------------------------------------------------------------------------
    def _apply_broker_events(self, events: Sequence[BrokerEvent]) -> None:
        intents = {r["intent"].intent_id: r for r in self.journal.intents(self.episode)}
        for event in events:
            record = intents.get(event.intent_id or "")
            if event.kind == "fee":
                self.journal.db.execute(
                    "UPDATE fills SET fee = ? WHERE exec_id = ? AND episode_id = ?",
                    (event.amount, event.exec_id, self.episode),
                )
                continue
            if record is None:
                continue  # another episode's or a manual order: never attributed
            intent = record["intent"]
            if event.kind == "fill":
                with self.journal.transaction():
                    if self.journal.add_fill(self.episode, intent, event):
                        filled = record["filled"] + event.quantity
                        record["filled"] = filled
                        status = "filled" if filled >= intent.quantity else record["status"]
                        record["status"] = status
                        self.journal.update_intent(intent.intent_id, filled=filled, status=status)
                        self.journal._append(
                            self.episode,
                            "fill",
                            event.at,
                            {
                                "intent_id": intent.intent_id,
                                "exec_id": event.exec_id,
                                "quantity": event.quantity,
                                "price": event.price,
                            },
                        )
                self.consecutive_rejects = 0
            elif event.kind == "rejected":
                if record["status"] not in {"submitting", "submitted"}:
                    continue  # a late error never overwrites cancelling or terminal states
                record["status"] = "rejected"
                self.consecutive_rejects += 1
                with self.journal.transaction():
                    self.journal.update_intent(intent.intent_id, status="rejected")
                    self.journal._append(
                        self.episode,
                        "rejected",
                        event.at,
                        {
                            "intent_id": intent.intent_id,
                            "detail": event.detail,
                        },
                    )
            elif event.kind == "cancelled" and record["status"] not in TERMINAL_INTENT:
                record["status"] = "cancelled"
                with self.journal.transaction():
                    self.journal.update_intent(intent.intent_id, status="cancelled")
                    self.journal._append(
                        self.episode,
                        "cancelled",
                        event.at,
                        {
                            "intent_id": intent.intent_id,
                        },
                    )

    # Ledger --------------------------------------------------------------------------------------
    def positions(self) -> dict[str, _Position]:
        return self._ledger()[0]

    def ledger_anomalies(self) -> list[str]:
        return self._ledger()[1]

    def _ledger(self) -> tuple[dict[str, _Position], list[str]]:
        """Long-only positions from this episode's fills, in receipt order.

        A sell with no position, or larger than the position, is an anomaly (the broker is
        short or out of sync); it is reported, never silently dropped.
        """
        intents = {r["intent"].intent_id: r["intent"] for r in self.journal.intents(self.episode)}
        result: dict[str, _Position] = {}
        anomalies: list[str] = []
        for fill in self.journal.fills(self.episode):
            quantity, price = int(fill["quantity"]), Decimal(fill["price"])
            current = result.get(fill["instrument"])
            if fill["side"] == "buy":
                entry = intents[fill["intent_id"]]
                if current is None:
                    result[fill["instrument"]] = _Position(
                        quantity, entry.decision_index, entry.session_id, price * quantity
                    )
                else:
                    current.quantity += quantity
                    current.cost += price * quantity
            elif current is None or quantity > current.quantity:
                anomalies.append(f"oversell:{fill['instrument']}:{fill['exec_id']}")
                result.pop(fill["instrument"], None)
            else:
                current.cost -= current.cost * quantity / current.quantity
                current.quantity -= quantity
                if current.quantity == 0:
                    del result[fill["instrument"]]
        return result, anomalies

    def cash(self) -> Decimal:
        cash = Decimal(self.plan.capital)
        for fill in self.journal.fills(self.episode):
            notional = Decimal(fill["price"]) * int(fill["quantity"])
            cash += -notional if fill["side"] == "buy" else notional
            cash -= Decimal(fill["fee"] or 0)
        return cash

    def _position_value(self) -> Decimal:
        value = Decimal(0)
        for target, position in self.positions().items():
            try:
                value += self._last_close(target) * position.quantity
            except (ValueError, AttributeError, TypeError):
                value += position.cost
        return value

    def _active_intents(self, purpose: str) -> list[dict]:
        return [
            r
            for r in self.journal.intents(self.episode, ACTIVE_INTENT)
            if r["intent"].purpose == purpose
        ]

    def _active_intent(self, target: str, purpose: str) -> bool:
        return any(r["intent"].instrument == target for r in self._active_intents(purpose))

    # Risk and session end ------------------------------------------------------------------------
    def risk_reasons(self, now: datetime) -> list[str]:
        positions = self.positions()
        values = []
        for target, position in positions.items():
            try:
                values.append(self._last_close(target) * position.quantity)
            except ValueError:
                values.append(position.cost)
        recent = [
            r
            for r in self.journal.intents(self.episode)
            if r["status"] != "pending"
            and timedelta(0) <= now - r["intent"].submit_at <= timedelta(minutes=1)
        ]
        state = RiskState(
            gross_notional=sum(values, Decimal(0)),
            largest_position_notional=max(values, default=Decimal(0)),
            orders_last_minute=len(recent),
            session_pnl=self.cash() + sum(values, Decimal(0)) - self.session_start_equity,
            capital=Decimal(self.plan.capital),
            consecutive_rejects=self.consecutive_rejects,
            clock_skew_seconds=self.clock_skew(),
            stale_feed=self.stale,
            model_expired=now >= self.plan.expires_at,
            in_restart_window=self.restart_window(now),
            extra_reasons=["broker_connectivity_lost"] if self.connectivity_lost() else [],
        )
        return evaluate(state, self.plan.risk)

    def _close_session(self, now: datetime) -> None:
        if self.session_closed:
            return
        closed = self.panel.today.close_at + timedelta(
            seconds=self.panel.contract.availability_buffer_seconds
        )
        if now < closed:
            return
        self.session_closed = True
        positions = self.positions()
        purpose = self.plan.schedule.purpose(date.fromisoformat(self.panel.today.session_id))
        self.journal.append(
            self.episode,
            "session_closed",
            now,
            {
                "session_id": self.panel.today.session_id,
                "purpose": purpose,
                "equity": format(self.cash() + self._position_value(), "f"),
                "start_equity": format(self.session_start_equity, "f"),
                "open_positions": {k: v.quantity for k, v in positions.items()},
            },
        )
        if self.archive is not None:
            entries = [self.archive.put_block(b) for b in self.panel.today_blocks()]
            self.journal.append(
                self.episode,
                "session_archived",
                now,
                {"session": self.panel.today.model_dump(mode="json"), "blocks": entries},
            )
        if positions:
            self._suspend(now, "overnight_breach")
        elif self.journal.status(self.episode) == "stopping":
            self.journal.set_status(self.episode, "terminated", now, "stopped_flat")

    # Reconciliation ------------------------------------------------------------------------------
    def reconcile(self, now: datetime) -> dict:
        """Match broker records to intents by orderRef; run at start and after reconnects.

        A `submitting` intent the broker knows becomes `submitted`; one it does not know stays
        unresolved until its deadline and is never re-sent. Broker orders still working for
        an intent this episode already gave up on are cancelled.
        """
        today = self.panel.today.session_id if self.panel else None
        session_start = self.panel.today.open_at if self.panel else now - timedelta(days=1)
        broker = self.backend.reconcile(session_start)
        orders = {
            str(o.get("order_ref")): o
            for o in (*broker.completed_orders, *broker.open_orders)
            if o.get("order_ref")
        }
        working = {str(o.get("order_ref")): o for o in broker.open_orders if o.get("order_ref")}
        done = {str(o.get("order_ref")): o for o in broker.completed_orders if o.get("order_ref")}
        executed = {str(e.get("order_ref")) for e in broker.executions if e.get("order_ref")}
        # Executions first, so intents a fill completed are terminal before they are judged.
        events = [self.backend.translate_execution(e) for e in broker.executions]
        self._apply_broker_events([e for e in events if e is not None])
        findings = []
        for record in self.journal.intents(self.episode):
            intent, status = record["intent"], record["status"]
            ref = intent.intent_id
            if status in ACTIVE_INTENT and intent.session_id != today:
                with self.journal.transaction():
                    self.journal.update_intent(ref, status="expired")
                    self.journal._append(
                        self.episode, "unresolved_prior_day", now, {"intent_id": ref}
                    )
                findings.append("unresolved_prior_day")
            elif status == "submitting" and (ref in orders or ref in executed):
                order_id = orders.get(ref, {}).get("order_id")
                with self.journal.transaction():
                    self.journal.update_intent(ref, status="submitted", order_id=order_id)
                    self.journal._append(
                        self.episode,
                        "submission_confirmed",
                        now,
                        {"intent_id": ref, "order_id": order_id},
                    )
            elif status in {"pending", "submitting"} and now >= intent.submit_deadline:
                with self.journal.transaction():
                    self.journal.update_intent(ref, status="abandoned_unsubmitted")
                    self.journal._append(
                        self.episode,
                        "abandoned_unsubmitted",
                        now,
                        {"intent_id": ref, "reason": "reconcile"},
                    )
            elif status in {"submitted", "cancelling"} and ref not in working and ref in done:
                final = str(done[ref].get("status", ""))
                new = "rejected" if final == "Inactive" else "cancelled"
                with self.journal.transaction():
                    self.journal.update_intent(ref, status=new)
                    self.journal._append(
                        self.episode, f"{new}_by_reconcile", now, {"intent_id": ref}
                    )
            elif status == "cancelling" and ref not in working:
                # Not working and not completed in this session's records: no longer live.
                with self.journal.transaction():
                    self.journal.update_intent(ref, status="cancelled")
                    self.journal._append(
                        self.episode, "cancelled_by_reconcile", now, {"intent_id": ref}
                    )
            elif status in {"abandoned_unsubmitted", "expired", "cancelled"} and ref in working:
                order_id = working[ref].get("order_id")
                if order_id is not None:
                    self.backend.cancel(int(order_id))
                    self.journal.append(
                        self.episode,
                        "stray_order_cancelled",
                        now,
                        {"intent_id": ref, "order_id": order_id},
                    )
        con_ids = {b.instrument: b.con_id for b in self.history.contract.instruments}
        positions, anomalies = self._ledger()
        ledger = {k: v.quantity for k, v in positions.items()}
        for target in self.targets:
            broker_qty = int(broker.positions.get(con_ids[target], Decimal(0)))
            if broker_qty != ledger.get(target, 0):
                findings.append(f"position_mismatch:{target}")
        findings.extend(f"position_mismatch:{a}" for a in anomalies)
        if findings:
            self._suspend(now, ";".join(sorted(set(findings))))
        return {"findings": sorted(set(findings))}

    def _suspend(self, now: datetime, reason: str) -> None:
        if self.journal.status(self.episode) not in {"suspended", "terminated"}:
            self.journal.set_status(self.episode, "suspended", now, reason)

    # Reports -------------------------------------------------------------------------------------
    def availability(self) -> dict:
        delays = [
            e["payload"]["arrival_delay"]
            for e in self.journal.events(self.episode, "bar")
            if e["payload"]["grid"] == "decision"
        ]
        result = availability_check(delays, self.panel.contract.availability_buffer_seconds)
        if result["violated"]:
            self.journal.append(self.episode, "availability_assumption_violated", self.now, result)
        return result

    def shadow_report(self) -> list[dict]:
        """Paper fill price versus the simulator's price for the same intent and fill bar."""
        intents = {r["intent"].intent_id: r["intent"] for r in self.journal.intents(self.episode)}
        report = []
        for fill in self.journal.fills(self.episode):
            intent = intents[fill["intent_id"]]
            execution = self.panel.block(intent.instrument, "execution", intent.session_id)
            if execution is None:
                continue
            start = int(intent.submit_at.timestamp())
            starts, status = execution.columns["start"], execution.columns["status"]
            window = np.flatnonzero(
                (starts >= start)
                & (starts <= start + self.plan.timing.max_fill_delay_seconds)
                & (status == VALID)
            )
            if not len(window):
                continue
            # The same pricing function the trusted simulator uses. Live episodes collect no
            # bid/ask grid, so the midpoint policy falls back (flagged) exactly as replay would.
            quotes = self.panel.block(intent.instrument, "execution_quotes", intent.session_id)
            simulated, fallback = fill_price(
                intent.side, execution, quotes, int(window[0]), self.plan.costs
            )
            paper = Decimal(fill["price"])
            sign = 1 if intent.side == "buy" else -1
            report.append(
                {
                    "intent_id": intent.intent_id,
                    "exec_id": fill["exec_id"],
                    "paper_price": format(paper, "f"),
                    "simulated_price": format(simulated, "f"),
                    "spread_policy": self.plan.costs.spread_policy,
                    "spread_fallback": fallback,
                    "shortfall_bps": float(sign * (paper - simulated) / simulated * 10000),
                    "purpose": self.plan.schedule.purpose(date.fromisoformat(intent.session_id)),
                }
            )
        return report
