"""IBKR historical bars: one request per instrument, grid and session, normalised onto the
registered grid. Raw responses are stored before normalisation.

IBKR's intraday bar timestamp (formatDate=2) is the bar start in epoch seconds. A bar outside
the registered grid or a duplicate is rejected, never silently dropped. Intervals with no bar
are marked missing; nothing is forward-filled.
"""

from collections.abc import Mapping, Sequence
from datetime import date, timezone
from decimal import Decimal, InvalidOperation

import numpy as np

from quant_rsi.brokers.contracts import InstrumentBinding
from quant_rsi.brokers.ibkr.connection import PACING_VIOLATION, IbkrError, IbkrSession
from quant_rsi.brokers.ibkr.instruments import contract_params
from quant_rsi.brokers.ibkr.pacing import PacingLimiter
from quant_rsi.core.artifacts import canonical
from quant_rsi.intraday.storage import INVALID, MISSING, VALID, Block
from quant_rsi.intraday.timing import slot_starts
from quant_rsi.market.market_contracts import CalendarSpec, MarketSession

BAR_SIZES = {60: "1 min", 300: "5 mins"}


def ibkr_utc(moment) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y%m%d-%H:%M:%S")


def intraday_params(
    binding: InstrumentBinding, session: MarketSession, bar_seconds: int, what_to_show: str
) -> dict:
    if bar_seconds not in BAR_SIZES or what_to_show not in {"TRADES", "BID_ASK"}:
        raise ValueError("unsupported intraday request")
    seconds = int((session.close_at - session.open_at).total_seconds())
    return {
        "contract": contract_params(binding),
        "end": ibkr_utc(session.close_at),
        "duration": f"{seconds} S",
        "bar_size": BAR_SIZES[bar_seconds],
        "what_to_show": what_to_show,
        "use_rth": 1,
        "format_date": 2,
        "keep_up_to_date": False,
    }


def daily_params(binding: InstrumentBinding, calendar: CalendarSpec) -> dict:
    last = calendar.sessions[-1]
    return {
        "contract": contract_params(binding),
        "end": ibkr_utc(last.close_at),
        "duration": f"{len(calendar.sessions)} D",
        "bar_size": "1 day",
        "what_to_show": "TRADES",
        "use_rth": 1,
        "format_date": 1,
        "keep_up_to_date": False,
    }


def fetch_bars(
    session: IbkrSession,
    pacing: PacingLimiter,
    params: Mapping[str, object],
    *,
    timeout: float | None = None,
) -> list[dict]:
    contract = str(params["contract"]["con_id"])
    pacing.acquire(canonical(dict(params)).decode(), contract)
    try:
        messages = session.call(
            "historical_data",
            params,
            end_kinds={"historical_end"},
            collect_kinds={"historical_bar"},
            timeout=timeout,
        )
    except IbkrError as error:
        if error.code == PACING_VIOLATION:
            pacing.record_violation()
        raise
    return [dict(m.payload) for m in messages if m.kind == "historical_bar"]


def _number(value) -> float:
    try:
        number = float(Decimal(str(value)))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"unparseable bar value {value!r}") from error
    return number


def normalise_intraday(
    instrument: str,
    grid: str,
    session: MarketSession,
    bar_seconds: int,
    payloads: Sequence[Mapping[str, object]],
) -> Block:
    starts = slot_starts(session, bar_seconds)
    position = {int(s): i for i, s in enumerate(starts)}
    n = len(starts)
    status = np.full(n, MISSING, np.uint8)
    values = {name: np.full(n, np.nan) for name in ("open", "high", "low", "close", "volume")}
    for payload in payloads:
        start = int(str(payload["date"]).strip())
        if start not in position:
            raise ValueError(f"{instrument}: bar at {start} is outside the registered session grid")
        i = position[start]
        if status[i] != MISSING:
            raise ValueError(f"{instrument}: duplicate bar at {start}")
        row = {name: _number(payload[name]) for name in ("open", "high", "low", "close")}
        row["volume"] = _number(payload.get("volume", "0"))
        ok = all(np.isfinite(v) and v > 0 for k, v in row.items() if k != "volume")
        ok = (
            ok
            and row["low"]
            <= min(row["open"], row["close"])
            <= max(row["open"], row["close"])
            <= row["high"]
        )
        if grid == "execution_quotes":
            ok = row["open"] > 0 and row["close"] >= row["open"]  # open=bid, close=ask
        status[i] = VALID if ok else INVALID
        if ok:
            for name in values:
                values[name][i] = row[name]
    if grid == "execution_quotes":
        columns = {"start": starts, "bid": values["open"], "ask": values["close"], "status": status}
    else:
        columns = {"start": starts, **values, "status": status}
    return Block(instrument, grid, session.session_id, columns)


def normalise_daily(calendar: CalendarSpec, payloads: Sequence[Mapping[str, object]]) -> list[dict]:
    """Daily TRADES bars mapped to registered sessions; unknown dates are rejected."""
    by_day = {s.open_at.astimezone(timezone.utc).date().isoformat(): s for s in calendar.sessions}
    rows, seen = [], set()
    for payload in payloads:
        text = str(payload["date"]).strip()
        day = date(int(text[:4]), int(text[4:6]), int(text[6:8])).isoformat()
        if day not in by_day:
            raise ValueError(f"daily bar {text} is not a registered session")
        if day in seen:
            raise ValueError(f"duplicate daily bar {text}")
        seen.add(day)
        rows.append(
            {
                "session_id": by_day[day].session_id,
                "open": str(Decimal(str(payload["open"]))),
                "high": str(Decimal(str(payload["high"]))),
                "low": str(Decimal(str(payload["low"]))),
                "close": str(Decimal(str(payload["close"]))),
                "volume": str(Decimal(str(payload.get("volume", "0")))),
                "price_basis": "raw_unadjusted",
            }
        )
    return sorted(rows, key=lambda r: r["session_id"])
