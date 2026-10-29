"""Real-time paper session loop: data client feeds bars, order client trades a paper account.

The loop reconciles with the broker before trading and after every reconnect. A disconnect
is a pause, not a crash: both sessions reconnect, the paper guard is re-checked, intents are
matched to broker records by orderRef, and the feed is re-subscribed in recovery mode. If
reconnection keeps failing the episode is suspended, preserving every committed fact.
"""

import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

from quant_rsi.brokers.ibkr.connection import IbkrDisconnected, IbkrError, IbkrSession, utc_now
from quant_rsi.intraday.paper import LiveFeed, PaperRunner
from quant_rsi.market.market_contracts import MarketSession


def _reconnect(
    sessions: Sequence[IbkrSession], attempts: int, wait: float, sleep: Callable[[float], None]
) -> bool:
    for _ in range(attempts):
        try:
            for session in sessions:
                session.reconnect()
            return True
        except (IbkrError, OSError):
            sleep(wait)
    return False


def run_session(
    runner: PaperRunner,
    data_session: IbkrSession,
    today: MarketSession,
    *,
    order_session: IbkrSession | None = None,
    clock: Callable[[], datetime] = utc_now,
    sleep: Callable[[float], None] = time.sleep,
    step_seconds: float = 1.0,
    reconnect_attempts: int = 5,
    reconnect_wait_seconds: float = 10.0,
) -> dict:
    """Run one regular session until close plus the availability buffer; resumable."""
    order_session = order_session or runner.backend.session
    runner.connectivity_lost = lambda: (
        data_session.connectivity_lost or order_session.connectivity_lost
    )
    runner.start_session(today)
    contract = runner.panel.contract
    started = clock()
    runner.begin_recovery(started)
    reconciliations = [runner.reconcile(started)]
    feed = LiveFeed(data_session, contract, today)
    end = today.close_at + timedelta(seconds=contract.availability_buffer_seconds + 30)
    result: dict = {}
    while True:
        now = clock()
        try:
            restored_data = data_session.poll_connectivity()
            restored_orders = order_session.poll_connectivity()
            if restored_data or restored_orders:
                # 1101/1102: TWS regained IBKR connectivity. Orders must be reconciled; 1101
                # on the data client means its subscriptions were lost and must be renewed.
                runner.journal.append(
                    runner.episode,
                    "connectivity_restored",
                    now,
                    {"data": restored_data, "orders": restored_orders},
                )
                runner.begin_recovery(now)
                reconciliations.append(runner.reconcile(now))
                if 1101 in restored_data:
                    feed = LiveFeed(data_session, contract, today)
                else:
                    for finaliser in runner.finalisers.values():
                        finaliser.backfill_complete()  # subscriptions were kept
            for instrument, grid, kind, payload in feed.poll():
                if kind == "backfill_end":
                    runner.on_backfill_end(instrument, grid)
                else:
                    runner.on_update(instrument, grid, payload, now)
            result = runner.step(now)
        except IbkrDisconnected as error:
            runner.journal.append(runner.episode, "disconnected", now, {"code": error.code})
            if not _reconnect(
                (data_session, order_session), reconnect_attempts, reconnect_wait_seconds, sleep
            ):
                runner.journal.set_status(runner.episode, "suspended", clock(), "disconnected")
                result = {"status": "suspended", "reason": "reconnect_failed"}
                break
            resumed = clock()
            runner.backend.verify()  # the paper guard is re-checked on the new connection
            runner.begin_recovery(resumed)
            reconciliations.append(runner.reconcile(resumed))
            feed = LiveFeed(data_session, contract, today)
            continue
        # Always step at least once past the end so the session close is recorded.
        if runner.session_closed or result.get("status") == "terminated" or now > end:
            break
        sleep(step_seconds)
    result["availability"] = runner.availability()
    result["reconciliation"] = runner.reconcile(clock())
    result["reconciliations_during_session"] = reconciliations
    return result
