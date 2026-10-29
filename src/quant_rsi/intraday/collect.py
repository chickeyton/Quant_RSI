"""Collect IBKR intraday (and daily) history into immutable snapshots.

Only completed sessions are collected: a session whose close plus the availability buffer is
still in the future is refused. Only the 1-minute execution grid (and optional bid/ask) is
requested; decision bars are aggregated locally, which saves a third of the requests.

Collection is resumable: every raw response is stored with its request and indexed by the
request's digest, so a re-run reuses stored responses (keeping their original first-seen
receipts) and requests only what is missing. Pacing violations back off and retry.
"""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from quant_rsi.brokers.contracts import InstrumentBinding
from quant_rsi.brokers.ibkr.connection import PACING_VIOLATION, IbkrError, IbkrSession, utc_now
from quant_rsi.brokers.ibkr.history import (
    daily_params,
    fetch_bars,
    intraday_params,
    normalise_daily,
    normalise_intraday,
)
from quant_rsi.brokers.ibkr.pacing import PacingLimiter
from quant_rsi.core.artifacts import canonical
from quant_rsi.intraday.contracts import EXECUTION_BAR_SECONDS, IntradayDataContract
from quant_rsi.intraday.storage import SnapshotStore, aggregate_execution
from quant_rsi.intraday.timing import slot_starts
from quant_rsi.market.market_contracts import CalendarSpec, utc_text

MAX_PACING_RETRIES = 5


def _grids(contract: IntradayDataContract) -> list[tuple[str, str]]:
    grids = [("execution", "TRADES")]
    if contract.quote_fields == "trades_and_bid_ask":
        grids.append(("execution_quotes", "BID_ASK"))
    return grids


def _fetch_once(
    session: IbkrSession,
    pacing: PacingLimiter,
    store: SnapshotStore,
    name: str,
    params: dict,
    clock: Callable[[], datetime],
) -> tuple[list, dict, bool]:
    """Stored payloads for an identical request, otherwise fetch with pacing retries."""
    cached = store.find_raw(params)
    if cached is not None:
        payloads, receipt = cached
        return payloads, receipt, True
    for attempt in range(MAX_PACING_RETRIES + 1):
        try:
            payloads = fetch_bars(session, pacing, params)
            break
        except IbkrError as error:
            if error.code != PACING_VIOLATION or attempt == MAX_PACING_RETRIES:
                raise
    raw = store.put_raw(name, canonical(payloads), params)
    receipt = {"raw_ref": raw, "first_seen_at": utc_text(clock())}
    store.remember_raw(params, receipt)
    return payloads, receipt, False


def collect_intraday(
    session: IbkrSession,
    pacing: PacingLimiter,
    store: SnapshotStore,
    contract: IntradayDataContract,
    *,
    clock: Callable[[], datetime] = utc_now,
    progress: Callable[[dict], None] = lambda event: None,
) -> str:
    now = clock()
    buffer = timedelta(seconds=contract.availability_buffer_seconds)
    incomplete = [s.session_id for s in contract.calendar.sessions if s.close_at + buffer > now]
    if incomplete:
        raise ValueError(f"sessions not yet complete: {incomplete[:3]}")
    blocks, raw_refs, receipts = [], [], []
    bar_seconds = contract.grid.bar_seconds
    for market_session in contract.calendar.sessions:
        for binding in contract.instruments:
            for grid, what in _grids(contract):
                params = intraday_params(binding, market_session, EXECUTION_BAR_SECONDS, what)
                name = f"ibkr_{binding.instrument}_{grid}_{market_session.session_id}"
                payloads, receipt, reused = _fetch_once(session, pacing, store, name, params, clock)
                raw_refs.append(receipt["raw_ref"])
                receipts.append(dict(receipt))
                block = normalise_intraday(
                    binding.instrument, grid, market_session, EXECUTION_BAR_SECONDS, payloads
                )
                blocks.append(block)
                if grid == "execution":
                    blocks.append(
                        aggregate_execution(
                            block, slot_starts(market_session, bar_seconds), bar_seconds
                        )
                    )
                progress(
                    {
                        "session_id": market_session.session_id,
                        "instrument": binding.instrument,
                        "grid": grid,
                        "reused": reused,
                    }
                )
    return store.commit(contract, blocks, raw_refs=raw_refs, receipts=receipts)


def collect_daily(
    session: IbkrSession,
    pacing: PacingLimiter,
    store: SnapshotStore,
    bindings: Sequence[InstrumentBinding],
    calendar: CalendarSpec,
    *,
    clock: Callable[[], datetime] = utc_now,
) -> dict:
    """Raw-price daily rows per instrument. Corporate-action coverage is declared unknown;
    the forecast envelope must attach a separate notice source or record an abstention."""
    result = {
        "kind": "IbkrDailyCollection",
        "price_basis": "raw_unadjusted",
        "corporate_action_coverage": "unknown",
        "instruments": {},
    }
    for binding in bindings:
        params = daily_params(binding, calendar)
        payloads, receipt, _ = _fetch_once(
            session, pacing, store, f"ibkr_daily_{binding.instrument}", params, clock
        )
        result["instruments"][binding.instrument] = {
            "con_id": binding.con_id,
            "raw_ref": receipt["raw_ref"],
            "first_seen_at": receipt["first_seen_at"],
            "rows": normalise_daily(calendar, payloads),
        }
    return result
