"""Live bar finalisation by the clock (design section 4.6).

`keepUpToDate` updates carry the running values of the current bar. A bar becomes final at
`bar_end + availability_buffer_seconds` from the last update received, never when the next
bar's first update happens to arrive. No update by then gives `missing` while the stream is
current; a silent stream gives `stale` plus a `stale_feed` event. Updates for a bar that is
already final are revisions and never change a decision already made.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from quant_rsi.intraday.storage import MISSING, STALE, VALID
from quant_rsi.intraday.timing import slot_ends, slot_starts
from quant_rsi.market.market_contracts import MarketSession


@dataclass(frozen=True)
class FinalBar:
    instrument: str
    grid: str
    session_id: str
    index: int
    start: int
    end: int
    status: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    finalised_at: datetime
    last_update_at: datetime | None

    @property
    def arrival_delay_seconds(self) -> float | None:
        if self.last_update_at is None:
            return None
        end = datetime.fromtimestamp(self.end, timezone.utc)
        return (self.last_update_at - end).total_seconds()


@dataclass(frozen=True)
class FeedEvent:
    kind: str  # stale_feed, feed_resumed, revision, out_of_grid
    instrument: str
    at: datetime
    detail: str = ""


class BarFinaliser:
    def __init__(
        self,
        instrument: str,
        grid: str,
        session: MarketSession,
        bar_seconds: int,
        *,
        buffer_seconds: int,
        max_lag_seconds: int,
    ):
        self.instrument, self.grid, self.session = instrument, grid, session
        self.starts = slot_starts(session, bar_seconds)
        self.ends = slot_ends(session, bar_seconds)
        self.position = {int(s): i for i, s in enumerate(self.starts)}
        self.buffer, self.max_lag = buffer_seconds, max_lag_seconds
        self.latest: dict[int, tuple[Mapping[str, object], datetime]] = {}
        self.next_final = 0
        self.last_message_at: datetime | None = None
        self.stale = False
        self.backfill_since: datetime | None = None

    def await_backfill(self, now: datetime) -> None:
        """After a (re)subscription, hold finalisation until the backfill has arrived.

        Without this, past bars would be finalised as stale/missing before TWS delivers the
        `keepUpToDate` backfill, and the backfill would then be discarded as revisions.
        """
        self.backfill_since = now
        self.last_message_at = now

    def backfill_complete(self) -> None:
        self.backfill_since = None

    def on_update(self, payload: Mapping[str, object], received_at: datetime) -> list[FeedEvent]:
        events = []
        if self.stale:
            self.stale = False
            events.append(FeedEvent("feed_resumed", self.instrument, received_at))
        self.last_message_at = received_at
        start = int(str(payload["date"]).strip())
        index = self.position.get(start)
        if index is None:
            return events + [FeedEvent("out_of_grid", self.instrument, received_at, str(start))]
        if index < self.next_final:
            return events + [FeedEvent("revision", self.instrument, received_at, str(start))]
        self.latest[index] = (payload, received_at)
        return events

    def _stream_silent(self, now: datetime) -> bool:
        reference = self.last_message_at or self.session.open_at
        return (now - reference) > timedelta(seconds=self.max_lag)

    def on_clock(self, now: datetime) -> list[FinalBar | FeedEvent]:
        out: list[FinalBar | FeedEvent] = []
        if self.backfill_since is not None:
            if (now - self.backfill_since) <= timedelta(seconds=self.max_lag):
                return out
            self.backfill_since = None
            out.append(FeedEvent("backfill_timeout", self.instrument, now))
        if now >= self.session.open_at and self._stream_silent(now) and not self.stale:
            self.stale = True
            out.append(FeedEvent("stale_feed", self.instrument, now))
        while self.next_final < len(self.starts):
            index = self.next_final
            due = datetime.fromtimestamp(int(self.ends[index]) + self.buffer, timezone.utc)
            if due > now:
                break
            out.append(self._finalise(index, now))
            self.next_final += 1
        return out

    def _finalise(self, index: int, now: datetime) -> FinalBar | FeedEvent:
        start, end = int(self.starts[index]), int(self.ends[index])
        if index in self.latest:
            payload, received = self.latest.pop(index)
            values = [float(str(payload[k])) for k in ("open", "high", "low", "close")]
            volume = float(str(payload.get("volume", "nan")))
            if all(math.isfinite(v) and v > 0 for v in values):
                return FinalBar(
                    self.instrument,
                    self.grid,
                    self.session.session_id,
                    index,
                    start,
                    end,
                    VALID,
                    *values,
                    volume,
                    now,
                    received,
                )
        status = STALE if self.stale else MISSING
        nan = math.nan
        return FinalBar(
            self.instrument,
            self.grid,
            self.session.session_id,
            index,
            start,
            end,
            status,
            nan,
            nan,
            nan,
            nan,
            nan,
            now,
            None,
        )


def availability_check(
    delays: Sequence[float], buffer_seconds: int, *, quantile: float = 0.99
) -> dict:
    """Compare live arrival delays with the backfill's availability assumption."""
    finite = [d for d in delays if d is not None and math.isfinite(d)]
    if not finite:
        return {"observations": 0, "quantile_seconds": None, "violated": False}
    value = float(np.quantile(finite, quantile))
    return {
        "observations": len(finite),
        "quantile": quantile,
        "quantile_seconds": value,
        "buffer_seconds": buffer_seconds,
        "violated": value > buffer_seconds,
    }
