"""Causal intraday feature leaves (design section 5.4).

Features at decision bar `t` read only decision bars `<= t` of the current session and, for
the explicitly cross-session leaves (`gap`, `vol_z_20` and the `d_` daily leaves), earlier
sessions that closed before this one opened. Plan validation guarantees bar `t` itself is
available at the decision. A leaf whose window contains a missing bar is missing (NaN); it
is never computed over a shortened window.

Daily leaves (design 5.4) summarise each earlier regular session as one daily bar built from
its decision bars. That bar exists only when every decision bar of the session is valid, so
its close is the last regular-hours bar's close, not the official closing auction.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

import numpy as np

from quant_rsi.intraday.contracts import IntradayDataContract
from quant_rsi.intraday.storage import VALID, Block
from quant_rsi.market.market_contracts import MarketSession

TARGET_LEAVES = (
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "rv_12",
    "rv_24",
    "range",
    "vol_z_20",
    "vwap_dev",
    "or_high_dist",
    "or_low_dist",
    "minutes_since_open",
    "gap",
    "rsi_14",
)
CONTEXT_LEAVES = ("ret_1", "ret_6")
DAILY_LEAVES = (
    "d_ret_1",
    "d_ret_5",
    "d_ret_20",
    "d_rv_20",
    "d_atr_14",
    "d_move_atr",
    "d_ma_20_dist",
    "d_ma_50_dist",
    "d_prev_range",
    "d_prev_close_pos",
    "d_prev_high_dist",
    "d_prev_low_dist",
)
ATR_PERIOD = 14
# Longer than any market-holiday weekend: a wider gap means the calendar is missing sessions,
# so a daily window across it would silently span the wrong number of sessions.
MAX_SESSION_GAP_DAYS = 5
VOLUME_LOOKBACK_SESSIONS = 20
OPENING_RANGE_SECONDS = 1800
RSI_PERIOD = 14


class BarSource(Protocol):
    contract: IntradayDataContract

    def block(self, instrument: str, grid: str, session_id: str) -> Block | None: ...


def default_leaves(contract: IntradayDataContract) -> tuple[str, ...]:
    context = tuple(f"ctx:{name}:{leaf}" for name in contract.context for leaf in CONTEXT_LEAVES)
    return TARGET_LEAVES + context


def validate_leaves(contract: IntradayDataContract, leaves: Sequence[str]) -> None:
    for leaf in leaves:
        if leaf in TARGET_LEAVES or leaf in DAILY_LEAVES:
            continue
        parts = leaf.split(":")
        if (
            len(parts) != 3
            or parts[0] != "ctx"
            or parts[1] not in contract.context
            or parts[2] not in CONTEXT_LEAVES
        ):
            raise ValueError(f"unsupported feature leaf {leaf}")


@dataclass(frozen=True)
class FeatureTable:
    names: tuple[str, ...]
    rows: tuple[tuple[str, str, int, int], ...]  # (target, session_id, session_index, t)
    values: np.ndarray

    def complete(self) -> np.ndarray:
        return np.isfinite(self.values).all(axis=1) if len(self.rows) else np.zeros(0, bool)

    def missing_leaves(self, row: int) -> tuple[str, ...]:
        return tuple(
            n for n, v in zip(self.names, self.values[row], strict=True) if not np.isfinite(v)
        )


def _valid_window(status: np.ndarray, first: int, last: int, tolerance: int = 0) -> bool:
    if first < 0:
        return False
    window = status[first : last + 1]
    return status[first] == VALID and status[last] == VALID and (window != VALID).sum() <= tolerance


def _ret(block: Block, t: int, k: int, tolerance: int = 0) -> float:
    status, close = block.columns["status"], block.columns["close"]
    if t >= len(status) or not _valid_window(status, t - k, t, tolerance):
        return math.nan
    return math.log(close[t] / close[t - k])


def _rv(block: Block, t: int, n: int) -> float:
    status, close = block.columns["status"], block.columns["close"]
    if not _valid_window(status, t - n, t):
        return math.nan
    returns = np.diff(np.log(close[t - n : t + 1]))
    return float(np.std(returns))


def _rsi(block: Block, t: int) -> float:
    status, close = block.columns["status"], block.columns["close"]
    if t < RSI_PERIOD or not _valid_window(status, 0, t):
        return math.nan
    changes = np.diff(close[: t + 1])
    gains, losses = np.clip(changes, 0, None), np.clip(-changes, 0, None)
    gain, loss = gains[:RSI_PERIOD].mean(), losses[:RSI_PERIOD].mean()
    for up, down in zip(gains[RSI_PERIOD:], losses[RSI_PERIOD:], strict=True):
        gain = (gain * (RSI_PERIOD - 1) + up) / RSI_PERIOD
        loss = (loss * (RSI_PERIOD - 1) + down) / RSI_PERIOD
    rsi = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)
    return (rsi - 50.0) / 50.0


@dataclass(frozen=True)
class DailyBar:
    open: float
    high: float
    low: float
    close: float


def daily_bar(block: Block | None) -> DailyBar | None:
    """One regular session as a daily bar; None unless every decision bar is valid."""
    if block is None or not len(block) or not (block.columns["status"] == VALID).all():
        return None
    cols = block.columns
    return DailyBar(
        float(cols["open"][0]),
        float(cols["high"].max()),
        float(cols["low"].min()),
        float(cols["close"][-1]),
    )


class _Session:
    def __init__(self, source: BarSource, sessions: Sequence[MarketSession], index: int):
        self.source, self.sessions, self.index = source, sessions, index
        self.session = sessions[index]
        self.bar_seconds = source.contract.grid.bar_seconds
        self._daily: dict[tuple[str, int], DailyBar | None] = {}

    def block(self, instrument: str, index: int | None = None) -> Block | None:
        session = self.sessions[self.index if index is None else index]
        return self.source.block(instrument, "decision", session.session_id)

    def prior_daily(self, instrument: str, count: int) -> list[DailyBar] | None:
        """The `count` sessions before this one, oldest first; None if any is unavailable."""
        first = self.index - count
        if first < 0:
            return None
        window = self.sessions[first : self.index + 1]
        limit = timedelta(days=MAX_SESSION_GAP_DAYS)
        if any(
            later.open_at - earlier.close_at > limit
            for earlier, later in zip(window[:-1], window[1:], strict=True)
        ):
            return None
        bars = []
        for index in range(first, self.index):
            key = (instrument, index)
            if key not in self._daily:
                self._daily[key] = daily_bar(self.block(instrument, index))
            if self._daily[key] is None:
                return None
            bars.append(self._daily[key])
        return bars


def _atr(bars: Sequence[DailyBar]) -> float:
    """Mean true range of the last ATR_PERIOD bars; `bars` holds one more for the first close."""
    ranges = [
        max(bar.high, before.close) - min(bar.low, before.close)
        for before, bar in zip(bars[:-1], bars[1:], strict=True)
    ]
    return float(np.mean(ranges))


def _daily_leaf(name: str, ctx: _Session, target: Block, t: int) -> float:
    instrument, cols = target.instrument, target.columns
    now = float(cols["close"][t]) if cols["status"][t] == VALID else math.nan
    if name.startswith("d_ret_"):
        k = int(name[6:])
        bars = ctx.prior_daily(instrument, k + 1)
        return math.nan if bars is None else math.log(bars[-1].close / bars[0].close)
    if name == "d_rv_20":
        bars = ctx.prior_daily(instrument, 21)
        if bars is None:
            return math.nan
        return float(np.std(np.diff(np.log([bar.close for bar in bars]))))
    if name in {"d_atr_14", "d_move_atr"}:
        bars = ctx.prior_daily(instrument, ATR_PERIOD + 1)
        if bars is None:
            return math.nan
        atr = _atr(bars)
        if name == "d_atr_14":
            return atr / bars[-1].close
        return math.nan if atr == 0 else (now - bars[-1].close) / atr
    if name.startswith("d_ma_"):
        bars = ctx.prior_daily(instrument, int(name.split("_")[2]))
        return math.nan if bars is None else now / float(np.mean([b.close for b in bars])) - 1.0
    bars = ctx.prior_daily(instrument, 1)
    if bars is None:
        return math.nan
    prev = bars[0]
    if name == "d_prev_range":
        return (prev.high - prev.low) / prev.close
    if name == "d_prev_close_pos":
        span = prev.high - prev.low
        return math.nan if span == 0 else (prev.close - prev.low) / span
    if name == "d_prev_high_dist":
        return now / prev.high - 1.0
    if name == "d_prev_low_dist":
        return now / prev.low - 1.0
    raise ValueError(f"unsupported feature leaf {name}")


def _target_leaf(name: str, ctx: _Session, target: Block, t: int) -> float:
    status, cols = target.columns["status"], target.columns
    if name in DAILY_LEAVES:
        return _daily_leaf(name, ctx, target, t)
    if name.startswith("ret_"):
        return _ret(target, t, int(name[4:]))
    if name.startswith("rv_"):
        return _rv(target, t, int(name[3:]))
    if name == "range":
        return (
            math.nan
            if status[t] != VALID
            else (cols["high"][t] - cols["low"][t]) / cols["close"][t]
        )
    if name == "minutes_since_open":
        end = min(int(cols["start"][t]) + ctx.bar_seconds, int(ctx.session.close_at.timestamp()))
        return (end - int(ctx.session.open_at.timestamp())) / 60.0
    if name == "vwap_dev":
        if not _valid_window(status, 0, t):
            return math.nan
        typical = (cols["high"][: t + 1] + cols["low"][: t + 1] + cols["close"][: t + 1]) / 3
        volume = cols["volume"][: t + 1]
        if volume.sum() <= 0:
            return math.nan
        return cols["close"][t] / float((typical * volume).sum() / volume.sum()) - 1.0
    if name in {"or_high_dist", "or_low_dist"}:
        bars = OPENING_RANGE_SECONDS // ctx.bar_seconds
        if t < bars - 1 or not _valid_window(status, 0, bars - 1) or status[t] != VALID:
            return math.nan
        level = cols["high"][:bars].max() if name == "or_high_dist" else cols["low"][:bars].min()
        return cols["close"][t] / level - 1.0
    if name == "gap":
        if ctx.index == 0 or status[0] != VALID:
            return math.nan
        previous = ctx.block(target.instrument, ctx.index - 1)
        if previous is None or previous.columns["status"][-1] != VALID:
            return math.nan
        return cols["open"][0] / previous.columns["close"][-1] - 1.0
    if name == "vol_z_20":
        if status[t] != VALID:
            return math.nan
        # The most recent earlier sessions that have slot t; shorter (early-close) sessions
        # are skipped rather than blanking this slot for the following 20 sessions. A session
        # that has the slot but no valid bar there still makes the leaf missing.
        history = []
        for back in range(ctx.index - 1, -1, -1):
            earlier = ctx.block(target.instrument, back)
            if earlier is None:
                return math.nan  # an uncollected session is missing data, not a short session
            if t >= len(earlier):
                continue
            if earlier.columns["status"][t] != VALID:
                return math.nan
            history.append(earlier.columns["volume"][t])
            if len(history) == VOLUME_LOOKBACK_SESSIONS:
                break
        if len(history) < VOLUME_LOOKBACK_SESSIONS:
            return math.nan
        spread = float(np.std(history))
        return math.nan if spread == 0 else (cols["volume"][t] - float(np.mean(history))) / spread
    if name == "rsi_14":
        return _rsi(target, t)
    raise ValueError(f"unsupported feature leaf {name}")


def session_features(
    source: BarSource,
    sessions: Sequence[MarketSession],
    index: int,
    target: str,
    leaves: Sequence[str],
    *,
    context_tolerance: int = 0,
    last_bar: int | None = None,
) -> tuple[list[tuple[str, str, int, int]], np.ndarray]:
    """Rows for decision bars 0..last_bar of one session (all bars when None)."""
    ctx = _Session(source, sessions, index)
    block = ctx.block(target)
    if block is None:
        return [], np.zeros((0, len(leaves)))
    bars = len(block) if last_bar is None else min(len(block), last_bar + 1)
    rows, values = [], np.full((bars, len(leaves)), np.nan)
    for t in range(bars):
        rows.append((target, ctx.session.session_id, index, t))
        for column, leaf in enumerate(leaves):
            if leaf.startswith("ctx:"):
                _, instrument, base = leaf.split(":")
                other = ctx.block(instrument)
                values[t, column] = (
                    math.nan if other is None else _ret(other, t, int(base[4:]), context_tolerance)
                )
            else:
                values[t, column] = _target_leaf(leaf, ctx, block, t)
    return rows, values


def build_features(
    source: BarSource,
    leaves: Sequence[str],
    *,
    session_indices: Sequence[int] | None = None,
    context_tolerance: int = 0,
) -> FeatureTable:
    contract = source.contract
    validate_leaves(contract, leaves)
    sessions = contract.calendar.sessions
    indices = range(len(sessions)) if session_indices is None else session_indices
    rows, chunks = [], []
    for index in indices:
        for target in contract.targets:
            session_rows, values = session_features(
                source, sessions, index, target, leaves, context_tolerance=context_tolerance
            )
            rows.extend(session_rows)
            chunks.append(values)
    values = np.vstack(chunks) if chunks else np.zeros((0, len(leaves)))
    return FeatureTable(tuple(leaves), tuple(rows), values)
