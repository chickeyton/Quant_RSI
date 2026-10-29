"""Deterministic intraday replay simulator: the trusted evaluator for intraday accounting.

Events are processed chronologically per session (decisions at `decision_time`, fills at the
execution bar start, flatten at `flatten_at`). Order quantities use only information
available at the decision; realised fill prices never change submitted quantities. Entries
that cannot fill inside their window are cancelled; exits are retried at later valid bars
and never given a fictional price. Positions left at the close are `unresolved`.
"""

import heapq
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

import numpy as np

from quant_rsi.intraday.contracts import CostModel, IntradayDevelopmentPlan
from quant_rsi.intraday.features import BarSource
from quant_rsi.intraday.storage import VALID, Block
from quant_rsi.intraday.timing import (
    NO_INDEX,
    SessionTiming,
    entry_allowed,
    fill_index,
    first_valid_at_or_after,
    halted,
    session_timing,
)

PriceKey = tuple[str, int, int]  # (target, session_index, decision bar)


def money(value: Decimal, costs: CostModel) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-costs.currency_precision), rounding=ROUND_HALF_EVEN)


def price_decimal(value: float) -> Decimal:
    return Decimal(repr(round(float(value), 6)))


def commission(quantity: int, notional: Decimal, costs: CostModel) -> Decimal:
    """IBKR fixed per-share: per-share rate, per-order minimum, capped at a notional fraction."""
    raw = Decimal(costs.commission_per_share) * quantity
    raw = max(raw, Decimal(costs.min_commission))
    raw = min(raw, notional * Decimal(costs.max_commission_fraction))
    return money(raw, costs)


def fill_price(
    side: str, execution: Block, quotes: Block | None, index: int, costs: CostModel
) -> tuple[Decimal, bool]:
    """Return (price, spread_fallback). Midpoint policy uses BID_ASK open=bid, close=ask."""
    sign = Decimal(1) if side == "buy" else Decimal(-1)
    if (
        costs.spread_policy == "mid_plus_half_spread"
        and quotes is not None
        and index < len(quotes)
        and quotes.columns["start"][index] == execution.columns["start"][index]
        and quotes.columns["status"][index] == VALID
    ):
        bid = price_decimal(quotes.columns["bid"][index])
        ask = price_decimal(quotes.columns["ask"][index])
        mid, half = (bid + ask) / 2, (ask - bid) / 2
        return mid + sign * half, False
    trade_open = price_decimal(execution.columns["open"][index])
    half = trade_open * Decimal(costs.assumed_spread_bps) / Decimal(20000)
    return trade_open + sign * half, costs.spread_policy == "mid_plus_half_spread"


@dataclass
class Position:
    instrument: str
    quantity: int
    entry_bar: int
    entry_session: int
    entry_cost: Decimal
    exit_proceeds: Decimal = Decimal(0)
    exit_pending: bool = False
    exit_reason: str | None = None
    exit_requested_at: int | None = None
    delayed_noted: bool = False


@dataclass
class SessionResult:
    session_id: str
    start_equity: Decimal
    end_equity: Decimal | None
    net_return: float | None
    round_trips: int
    unresolved: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    sessions: list[SessionResult]
    fills: list[dict]
    round_trips: list[dict]
    eligible_bars: int
    scored_bars: int
    spread_fallbacks: int
    cancelled_entries: int
    initial_capital: Decimal

    def metrics(self, sessions_per_year: int = 252) -> dict:
        returns = [s.net_return for s in self.sessions]
        resolved = [r for r in returns if r is not None]
        gross = sum((Decimal(f["notional"]) for f in self.fills), Decimal(0))
        fees = sum((Decimal(f["commission"]) for f in self.fills), Decimal(0))
        pnl = [Decimal(t["net_pnl"]) for t in self.round_trips]
        n_sessions = len(self.sessions)
        return {
            "sessions": n_sessions,
            "unresolved_sessions": sum(r is None for r in returns),
            "mean_net_return_bps": None if not resolved else 1e4 * float(np.mean(resolved)),
            "annualised_utility": (
                None
                if not resolved or len(resolved) != n_sessions
                else sessions_per_year * float(np.mean(resolved))
            ),
            "round_trips": len(self.round_trips),
            "hit_rate": None if not pnl else sum(p > 0 for p in pnl) / len(pnl),
            "turnover": (
                0.0 if not n_sessions else float(gross / (self.initial_capital * n_sessions))
            ),
            "cost_share": None if gross == 0 else float(fees / gross),
            "holding_bars": [t["holding_bars"] for t in self.round_trips],
            "prediction_coverage": (
                None if not self.eligible_bars else self.scored_bars / self.eligible_bars
            ),
            "spread_fallbacks": self.spread_fallbacks,
            "cancelled_entries": self.cancelled_entries,
        }


class _Session:
    """Mutable per-session event loop over immutable inputs."""

    def __init__(self, sim: "Simulator", index: int, cash: Decimal, carried: dict):
        self.sim, self.index = sim, index
        self.session = sim.sessions[index]
        self.timing: SessionTiming = session_timing(
            self.session,
            sim.bar_seconds,
            sim.plan.timing,
            sim.plan.signal.flatten_offset_seconds,
        )
        self.cash, self.start_cash = cash, cash
        self.positions: dict[str, Position] = carried
        self.pending_entries: set[str] = set()
        self.events: list = []
        self.counter = 0
        self.round_trips = 0
        self.notes: list[str] = []
        self.execution = {i: sim.block(i, "execution", self.session) for i in sim.targets}
        self.quotes = {i: sim.block(i, "execution_quotes", self.session) for i in sim.targets}
        self.decision = {i: sim.block(i, "decision", self.session) for i in sim.targets}
        self.halted = {
            i: (
                np.zeros(self.timing.bars, bool)
                if b is None
                else halted(b.columns["status"], sim.plan.signal.halt_missing_bars)
            )
            for i, b in self.decision.items()
        }

    def push(self, at: int, kind: str, payload: dict) -> None:
        self.counter += 1
        heapq.heappush(self.events, (at, self.counter, kind, payload))

    def run(self) -> SessionResult:
        for instrument, position in self.positions.items():
            self.notes.append(f"overnight_breach_exit:{instrument}")
            self.schedule_exit(position, int(self.session.open_at.timestamp()), "overnight")
        for t in range(self.timing.bars):
            self.push(int(self.timing.decision_time[t]), "decision", {"t": t})
        self.push(self.timing.flatten_at, "flatten", {})
        while self.events:
            at, _, kind, payload = heapq.heappop(self.events)
            getattr(self, f"on_{kind}")(at, **payload)
        unresolved = bool(self.positions) or any(n.startswith("overnight") for n in self.notes)
        end = None if unresolved else self.cash
        result = None if end is None else float((end - self.start_cash) / self.start_cash)
        if self.positions:
            self.notes.append("unresolved_position_at_close")
        return SessionResult(
            self.session.session_id,
            self.start_cash,
            end,
            result,
            self.round_trips,
            unresolved,
            self.notes,
        )

    # Decisions ---------------------------------------------------------------------------
    def on_decision(self, at: int, t: int) -> None:
        signal = self.sim.plan.signal
        for instrument in self.sim.targets:
            score = self.sim.predictions.get((instrument, self.index, t), math.nan)
            position = self.positions.get(instrument)
            if position is not None:
                if position.exit_pending or position.entry_session != self.index:
                    continue
                if t >= position.entry_bar + signal.horizon_bars:
                    self.schedule_exit(position, int(self.timing.fill_start[t]), "horizon")
                elif math.isfinite(score) and score < signal.exit_threshold:
                    self.schedule_exit(position, int(self.timing.fill_start[t]), "signal")
                continue
            if instrument in self.pending_entries:
                continue
            if self.halted[instrument][t] or not entry_allowed(self.timing, t, signal.horizon_bars):
                continue
            self.sim.eligible_bars += 1
            if not math.isfinite(score):
                continue
            self.sim.scored_bars += 1
            occupied = len(self.positions) + len(self.pending_entries)
            if score <= signal.entry_threshold or occupied >= signal.max_positions:
                continue
            self.schedule_entry(instrument, t)

    def schedule_entry(self, instrument: str, t: int) -> None:
        signal = self.sim.plan.signal
        decision = self.decision[instrument]
        if decision is None or decision.columns["status"][t] != VALID:
            return
        reference = price_decimal(decision.columns["close"][t])
        budget = self.start_cash * Decimal(signal.notional_fraction)
        lots = (budget / reference / signal.lot_size).to_integral_value(rounding=ROUND_FLOOR)
        quantity = int(lots) * signal.lot_size
        execution = self.execution[instrument]
        index = (
            NO_INDEX
            if execution is None
            else fill_index(
                execution,
                int(self.timing.fill_start[t]),
                self.sim.plan.timing.max_fill_delay_seconds,
            )
        )
        if quantity <= 0 or index == NO_INDEX:
            self.sim.cancelled_entries += 1
            return
        self.pending_entries.add(instrument)
        start = int(execution.columns["start"][index])
        self.push(
            start,
            "fill",
            {"instrument": instrument, "side": "buy", "quantity": quantity, "index": index, "t": t},
        )

    def schedule_exit(self, position: Position, start: int, reason: str) -> None:
        execution = self.execution[position.instrument]
        index = (
            NO_INDEX
            if execution is None or start == NO_INDEX
            else (first_valid_at_or_after(execution, start))
        )
        position.exit_pending = True
        position.exit_reason = position.exit_reason or reason
        if position.exit_requested_at is None and start != NO_INDEX:
            position.exit_requested_at = start
        if index == NO_INDEX:
            self.notes.append(f"exit_unfilled:{position.instrument}")
            return
        self.note_delay(position, int(execution.columns["start"][index]))
        self.push(
            int(execution.columns["start"][index]),
            "fill",
            {
                "instrument": position.instrument,
                "side": "sell",
                "quantity": position.quantity,
                "index": index,
                "t": None,
            },
        )

    def on_flatten(self, at: int) -> None:
        for position in list(self.positions.values()):
            if not position.exit_pending:
                self.schedule_exit(position, at, "flatten")

    # Fills -----------------------------------------------------------------------------------
    def on_fill(self, at: int, instrument: str, side: str, quantity: int, index: int, t) -> None:
        costs, signal = self.sim.plan.costs, self.sim.plan.signal
        execution = self.execution[instrument]
        cap_units = Decimal(costs.participation_cap) * Decimal(
            repr(float(execution.columns["volume"][index]))
        )
        cap = (
            int((cap_units / signal.lot_size).to_integral_value(rounding=ROUND_FLOOR))
            * signal.lot_size
        )
        filled = min(quantity, cap)
        price, fallback = fill_price(side, execution, self.quotes[instrument], index, costs)
        self.sim.spread_fallbacks += int(fallback)
        if side == "buy":
            self.pending_entries.discard(instrument)
            while filled > 0:
                notional = money(price * filled, costs)
                total = notional + commission(filled, notional, costs)
                if total <= self.cash:
                    break
                filled -= signal.lot_size
            if filled <= 0:
                self.sim.cancelled_entries += 1
                return
            notional = money(price * filled, costs)
            fee = commission(filled, notional, costs)
            self.cash -= notional + fee
            self.positions[instrument] = Position(instrument, filled, t, self.index, notional + fee)
            self.record(at, instrument, side, filled, price, notional, fee, fallback)
            return
        position = self.positions[instrument]
        if filled <= 0:
            self.retry_exit(position, index)
            return
        notional = money(price * filled, costs)
        fee = commission(filled, notional, costs)
        self.cash += notional - fee
        self.record(at, instrument, side, filled, price, notional, fee, fallback)
        share = Decimal(filled) / Decimal(position.quantity)
        cost_share = money(position.entry_cost * share, costs)
        position.entry_cost -= cost_share
        position.quantity -= filled
        net = notional - fee - cost_share
        if position.quantity == 0:
            del self.positions[instrument]
        self.round_trips += 1
        self.sim.round_trips.append(
            {
                "instrument": instrument,
                "session_id": self.session.session_id,
                "entry_bar": position.entry_bar,
                "holding_bars": (
                    None
                    if position.entry_bar is None or position.entry_session != self.index
                    else self.bar_at(at) - position.entry_bar
                ),
                "exit_reason": position.exit_reason,
                "quantity": filled,
                "net_pnl": format(net, "f"),
            }
        )
        if position.quantity > 0:
            self.retry_exit(position, index)

    def retry_exit(self, position: Position, index: int) -> None:
        execution = self.execution[position.instrument]
        starts = execution.columns["start"]
        later = np.flatnonzero(
            (np.arange(len(starts)) > index) & (execution.columns["status"] == VALID)
        )
        if not len(later):
            self.notes.append(f"exit_unfilled:{position.instrument}")
            return
        nxt = int(later[0])
        self.note_delay(position, int(starts[nxt]))
        self.push(
            int(starts[nxt]),
            "fill",
            {
                "instrument": position.instrument,
                "side": "sell",
                "quantity": position.quantity,
                "index": nxt,
                "t": None,
            },
        )

    def note_delay(self, position: Position, fill_start: int) -> None:
        """Flag an exit (first attempt or retry) delayed beyond the registered fill window."""
        requested = position.exit_requested_at
        limit = self.sim.plan.timing.max_fill_delay_seconds
        if requested is not None and fill_start - requested > limit and not position.delayed_noted:
            position.delayed_noted = True
            self.notes.append(f"halt_delayed_exit:{position.instrument}")

    def bar_at(self, at: int) -> int:
        """Decision bar whose decision time most recently preceded `at`."""
        earlier = np.flatnonzero(self.timing.decision_time <= at)
        return int(earlier[-1]) if len(earlier) else 0

    def record(self, at, instrument, side, quantity, price, notional, fee, fallback) -> None:
        self.sim.fills.append(
            {
                "session_id": self.session.session_id,
                "at": at,
                "instrument": instrument,
                "side": side,
                "quantity": quantity,
                "price": format(price, "f"),
                "notional": format(notional, "f"),
                "commission": format(fee, "f"),
                "spread_fallback": fallback,
            }
        )


class Simulator:
    def __init__(
        self,
        source: BarSource,
        plan: IntradayDevelopmentPlan,
        predictions: Mapping[PriceKey, float],
    ):
        self.source, self.plan, self.predictions = source, plan, predictions
        self.sessions = source.contract.calendar.sessions
        self.targets = tuple(sorted(source.contract.targets))
        self.bar_seconds = source.contract.grid.bar_seconds
        self.fills: list[dict] = []
        self.round_trips: list[dict] = []
        self.eligible_bars = self.scored_bars = 0
        self.spread_fallbacks = self.cancelled_entries = 0

    def block(self, instrument, grid, session):
        return self.source.block(instrument, grid, session.session_id)

    def run(self, session_indices: Sequence[int], capital: Decimal) -> SimulationResult:
        if list(session_indices) != sorted(set(session_indices)):
            raise ValueError("sessions must be unique and chronological")
        cash, carried, results = capital, {}, []
        for index in session_indices:
            session = _Session(self, index, cash, carried)
            results.append(session.run())
            cash, carried = session.cash, session.positions
            for position in carried.values():
                position.exit_pending = False
                position.exit_requested_at, position.delayed_noted = None, False
        return SimulationResult(
            results,
            self.fills,
            self.round_trips,
            self.eligible_bars,
            self.scored_bars,
            self.spread_fallbacks,
            self.cancelled_entries,
            capital,
        )


def simulate(
    source: BarSource,
    plan: IntradayDevelopmentPlan,
    predictions: Mapping[PriceKey, float],
    session_indices: Sequence[int],
    capital: Decimal | None = None,
) -> SimulationResult:
    start = Decimal(plan.initial_capital) if capital is None else capital
    return Simulator(source, plan, predictions).run(session_indices, start)
