"""Decision, execution and label timing for one session (design section 5.3).

A decision bar ends exactly when the next starts, so a decision after bar `t` closes cannot
trade at the open of bar `t+1`. Execution uses the 1-minute execution grid: the fill bar is
the first execution bar starting at or after `exec_time`, and the execution price is the
open of the first *valid* execution bar within `max_fill_delay_seconds` of that start.
Labels use exactly the same price rule as fills. All times are UTC epoch seconds.
"""

from dataclasses import dataclass

import numpy as np

from quant_rsi.intraday.contracts import EXECUTION_BAR_SECONDS, ExecutionTiming
from quant_rsi.intraday.storage import MISSING, STALE, VALID, Block
from quant_rsi.market.market_contracts import MarketSession

NO_INDEX = -1


def bounds(session: MarketSession) -> tuple[int, int]:
    return int(session.open_at.timestamp()), int(session.close_at.timestamp())


def slot_starts(session: MarketSession, bar_seconds: int) -> np.ndarray:
    opened, closed = bounds(session)
    return np.arange(opened, closed, bar_seconds, dtype=np.int64)


def slot_ends(session: MarketSession, bar_seconds: int) -> np.ndarray:
    _, closed = bounds(session)
    return np.minimum(slot_starts(session, bar_seconds) + bar_seconds, closed)


def fill_bar_start(session: MarketSession, exec_time: int) -> int:
    """Start of the first execution bar starting at or after `exec_time`; NO_INDEX if none."""
    opened, closed = bounds(session)
    offset = max(0, exec_time - opened)
    start = opened + -(-offset // EXECUTION_BAR_SECONDS) * EXECUTION_BAR_SECONDS
    return int(start) if start < closed else NO_INDEX


@dataclass(frozen=True)
class SessionTiming:
    session: MarketSession
    decision_end: np.ndarray
    decision_time: np.ndarray
    exec_time: np.ndarray
    fill_start: np.ndarray
    flatten_at: int

    @property
    def bars(self) -> int:
        return len(self.decision_end)


def session_timing(
    session: MarketSession, bar_seconds: int, timing: ExecutionTiming, flatten_offset: int
) -> SessionTiming:
    ends = slot_ends(session, bar_seconds)
    decision = ends + timing.decision_latency_seconds
    exec_time = decision + timing.order_latency_seconds
    fills = np.array([fill_bar_start(session, int(x)) for x in exec_time], dtype=np.int64)
    _, closed = bounds(session)
    return SessionTiming(session, ends, decision, exec_time, fills, closed - flatten_offset)


def fill_index(execution: Block, start: int, max_delay: int) -> int:
    """Index of the first valid execution bar starting in [start, start + max_delay]."""
    if start == NO_INDEX:
        return NO_INDEX
    starts = execution.columns["start"]
    status = execution.columns["status"]
    candidates = np.flatnonzero((starts >= start) & (starts <= start + max_delay) & (status == 0))
    return int(candidates[0]) if len(candidates) else NO_INDEX


def first_valid_at_or_after(execution: Block, at: int) -> int:
    starts = execution.columns["start"]
    candidates = np.flatnonzero((starts >= at) & (execution.columns["status"] == VALID))
    return int(candidates[0]) if len(candidates) else NO_INDEX


def labels(execution: Block, timing: SessionTiming, horizon: int, max_delay: int) -> np.ndarray:
    """y[t] = px(t+H) / px(t) - 1 within one session; NaN where either price is missing."""
    result = np.full(timing.bars, np.nan)
    opens = execution.columns["open"]
    for t in range(timing.bars - horizon):
        entry = fill_index(execution, int(timing.fill_start[t]), max_delay)
        leave = fill_index(execution, int(timing.fill_start[t + horizon]), max_delay)
        if entry != NO_INDEX and leave != NO_INDEX:
            result[t] = opens[leave] / opens[entry] - 1.0
    return result


def entry_allowed(timing: SessionTiming, t: int, horizon: int) -> bool:
    """An entry is allowed only if its planned exit fill bar is not after `flatten_at`."""
    if t + horizon >= timing.bars:
        return False
    exit_start = int(timing.fill_start[t + horizon])
    return exit_start != NO_INDEX and exit_start <= timing.flatten_at


def halted(decision_status: np.ndarray, halt_missing_bars: int) -> np.ndarray:
    """Causal: True from the bar at which a run of > N consecutive missing bars is observed.

    Only MISSING counts: a market halt shows as bars with no trades. STALE marks a feed
    problem on our side; it neither extends nor resets the run (stale feeds block entries
    through the risk rules instead, and must not latch a session-long halt).
    """
    result = np.zeros(len(decision_status), dtype=bool)
    run, tripped = 0, False
    for t, status in enumerate(decision_status):
        if status == MISSING:
            run += 1
        elif status != STALE:
            run = 0
        tripped = tripped or run > halt_missing_bars
        result[t] = tripped
    return result
