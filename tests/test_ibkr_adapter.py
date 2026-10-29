"""IBKR adapter behaviour against a scripted fake transport; no TWS or ibapi needed."""

import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from ibkr_fakes import ACCOUNT, FakeBroker, config
from intraday_helpers import binding, build_snapshot, contract, sessions
from pydantic import ValidationError

from quant_rsi.brokers.contracts import InstrumentSpec, OrderIntent
from quant_rsi.brokers.ibkr.connection import IbkrDisconnected, IbkrError, IbkrSession, IbkrTimeout
from quant_rsi.brokers.ibkr.guard import LiveAccountRefused, check_paper
from quant_rsi.brokers.ibkr.history import (
    intraday_params,
    normalise_daily,
    normalise_intraday,
)
from quant_rsi.brokers.ibkr.instruments import AmbiguousInstrument, resolve
from quant_rsi.brokers.ibkr.orders import IbkrPaperBackend
from quant_rsi.brokers.ibkr.pacing import PacingLimiter
from quant_rsi.brokers.ibkr.stream import BarFinaliser, FeedEvent, FinalBar, availability_check
from quant_rsi.brokers.ibkr.transport import FakeIbkrTransport, Message, handshake_messages
from quant_rsi.intraday.collect import collect_intraday
from quant_rsi.intraday.storage import MISSING, STALE, VALID, SnapshotStore


# Packaging --------------------------------------------------------------------------------------
def test_adapter_imports_without_ibapi():
    """Only constructing the real transport needs the locally installed GPL ibapi."""
    code = "\n".join(
        [
            "import sys; sys.modules['ibapi'] = None",
            "import quant_rsi.brokers.ibkr.orders, quant_rsi.intraday.paper",
            "from quant_rsi.brokers.ibkr.ibapi_transport import RealIbapiTransport",
            "try:",
            "    RealIbapiTransport()",
            "except ImportError:",
            "    print('lazy-ok')",
        ]
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "lazy-ok", result.stderr


# Configuration and guard -------------------------------------------------------------------------
def test_config_rejects_live_ports_and_shared_client_ids():
    with pytest.raises(ValidationError):
        config(paper_ports=(7496,))
    with pytest.raises(ValidationError):
        config(orders_client_id=11)
    with pytest.raises(ValidationError):
        config(account_mode="live")
    with pytest.raises(ValidationError):
        config(expected_account="U1234567")


@pytest.mark.parametrize(
    "port, accounts, client, reason",
    [
        (7496, (ACCOUNT,), 21, "port"),
        (7497, ("U7654321",), 21, "individual paper"),
        (7497, (ACCOUNT, "U7654321"), 21, "individual paper"),
        (7497, ("DF1234",), 21, "individual paper"),
        (7497, ("DU9999999",), 21, "expected paper account"),
        (7497, (ACCOUNT,), 11, "order client"),
        (7497, (), 21, "no managed accounts"),
    ],
)
def test_paper_guard_refuses(port, accounts, client, reason):
    with pytest.raises(LiveAccountRefused, match=reason):
        check_paper(config(), port=port, accounts=accounts, client_id=client)


def test_paper_guard_accepts_paper_session():
    assert check_paper(config(), port=4002, accounts=(ACCOUNT,), client_id=21) == ACCOUNT


# Session -----------------------------------------------------------------------------------------
def test_handshake_records_accounts_order_ids_and_clock_skew():
    broker = FakeBroker()
    later = broker.now() + timedelta(seconds=3)
    session = broker.session(clock=lambda: later)
    assert session.info.accounts == (ACCOUNT,) and session.info.next_order_id == 100
    assert session.info.clock_skew_seconds == pytest.approx(3)
    assert session.next_order_id() == 100 and session.next_order_id() == 101


def test_handshake_timeout_without_accounts():
    transport = FakeIbkrTransport(handshake=[Message("next_valid_id", payload={"order_id": 1})])
    ticks = iter(range(0, 1000, 10))
    session = IbkrSession(
        transport, host="h", port=7497, client_id=21, monotonic=lambda: next(ticks), timeout=5
    )
    with pytest.raises(IbkrTimeout):
        session.connect()


def test_request_error_is_typed_and_notices_are_ignored():
    def respond(request):
        if request.kind == "current_time":
            return [Message("current_time", payload={"epoch": 0})]
        return [
            Message("error", None, {"code": 2104, "message": "farm ok"}),
            Message("error", request.req_id, {"code": 200, "message": "no security"}),
        ]

    transport = FakeIbkrTransport(respond, handshake=handshake_messages())
    session = IbkrSession(transport, host="h", port=7497, client_id=21)
    session.connect()
    with pytest.raises(IbkrError) as caught:
        session.call("contract_details", {}, end_kinds={"contract_details_end"})
    assert caught.value.code == 200
    assert [e.code for e in session.connection_events] == [2104]


def test_connection_loss_while_waiting_raises_disconnected():
    def respond(request):
        if request.kind == "current_time":
            return [Message("current_time", payload={"epoch": 0})]
        return [Message("error", None, {"code": 1100, "message": "connectivity lost"})]

    transport = FakeIbkrTransport(respond, handshake=handshake_messages())
    session = IbkrSession(transport, host="h", port=7497, client_id=21)
    session.connect()
    with pytest.raises(IbkrDisconnected):
        session.call("positions", {}, end_kinds={"position_end"}, req_id=None)
    assert session.connection_events[-1].kind == "connection_lost"


def test_unsolicited_order_messages_are_buffered_for_drain():
    broker = FakeBroker()
    session = broker.session()
    broker.transport.push(Message("order_status", payload={"order_id": 5, "status": "Filled"}))
    session.call("current_time", {}, end_kinds={"current_time"}, req_id=None)
    assert [m.kind for m in session.drain()] == ["order_status"]


# Pacing ------------------------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_pacing_identical_request_waits_15_seconds():
    clock = FakeClock()
    limiter = PacingLimiter(clock=clock, sleep=clock.sleep)
    limiter.acquire("same", "1")
    assert limiter.acquire("same", "1") == pytest.approx(15)


def test_pacing_window_survives_restart(tmp_path):
    clock = FakeClock()
    first = PacingLimiter(
        tmp_path / "pacing.sqlite",
        clock=clock,
        sleep=clock.sleep,
        max_requests=3,
        per_contract_seconds=0,
    )
    for number in range(3):
        first.acquire(f"r{number}", str(number))
    first.close()
    second = PacingLimiter(
        tmp_path / "pacing.sqlite",
        clock=clock,
        sleep=clock.sleep,
        max_requests=3,
        per_contract_seconds=0,
    )
    assert second.acquire("r9", "9") == pytest.approx(600)


def test_pacing_violation_backs_off():
    clock = FakeClock()
    limiter = PacingLimiter(clock=clock, sleep=clock.sleep, backoff_seconds=60)
    limiter.record_violation()
    assert limiter.acquire("x", "1") == pytest.approx(60)


# Instruments -------------------------------------------------------------------------------------
def _spec():
    return InstrumentSpec(
        instrument="KO",
        symbol="KO",
        sec_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NYSE",
    )


def _details(con_id):
    return {
        "con_id": con_id,
        "symbol": "KO",
        "sec_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NYSE",
        "time_zone_id": "US/Eastern",
        "trading_hours": "x",
        "liquid_hours": "y",
    }


def test_resolve_requires_exactly_one_contract():
    broker = FakeBroker()
    session = broker.session(client_id=11)
    broker.contract_details["KO"] = [_details(8894)]
    assert resolve(session, _spec()).con_id == 8894
    broker.contract_details["KO"] = [_details(1), _details(2)]
    with pytest.raises(AmbiguousInstrument):
        resolve(session, _spec())
    broker.contract_details["KO"] = []
    with pytest.raises(AmbiguousInstrument):
        resolve(session, _spec())


# History -----------------------------------------------------------------------------------------
def _payload(start, price=10.0):
    return {
        "date": str(start),
        "open": price,
        "high": price + 0.1,
        "low": price - 0.1,
        "close": price,
        "volume": "100",
        "wap": "10",
        "bar_count": 3,
    }


def test_normalise_marks_missing_and_rejects_outside_or_duplicate_bars():
    session = sessions(3)[0]
    opened = int(session.open_at.timestamp())
    block = normalise_intraday("KO", "decision", session, 300, [_payload(opened + 300)])
    assert block.columns["status"][0] == MISSING and block.columns["status"][1] == VALID
    assert np.isnan(block.columns["close"][0])
    with pytest.raises(ValueError, match="outside"):
        normalise_intraday("KO", "decision", session, 300, [_payload(opened + 30)])
    with pytest.raises(ValueError, match="duplicate"):
        normalise_intraday("KO", "decision", session, 300, [_payload(opened)] * 2)


def test_quote_bars_use_open_as_bid_and_close_as_ask():
    session = sessions(3)[0]
    opened = int(session.open_at.timestamp())
    row = {"date": str(opened), "open": 9.99, "high": 10.2, "low": 9.9, "close": 10.01}
    block = normalise_intraday("KO", "execution_quotes", session, 60, [row])
    assert block.columns["bid"][0] == 9.99 and block.columns["ask"][0] == 10.01


def test_daily_rows_are_raw_and_must_be_registered_sessions():
    calendar = contract(sessions(3)).calendar
    rows = normalise_daily(
        calendar,
        [{"date": "20260105", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": "10"}],
    )
    assert rows[0]["session_id"] == "2026-01-05" and rows[0]["price_basis"] == "raw_unadjusted"
    with pytest.raises(ValueError, match="not a registered session"):
        normalise_daily(
            calendar, [{"date": "20260104", "open": 1, "high": 2, "low": 0.5, "close": 1.5}]
        )


def test_collect_intraday_end_to_end_reproduces_fixture_blocks(tmp_path):
    store, ref = build_snapshot(tmp_path / "source", count=3)
    source = store.load(ref)
    broker = FakeBroker()
    grids = {"decision": "5 mins", "execution": "1 min", "execution_quotes": "1 min"}
    for (instrument, grid, session_id), block in source.blocks.items():
        b = next(x for x in source.contract.instruments if x.instrument == instrument)
        session = next(s for s in source.contract.calendar.sessions if s.session_id == session_id)
        what = "BID_ASK" if grid == "execution_quotes" else "TRADES"
        params = intraday_params(b, session, 300 if grid == "decision" else 60, what)
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
        broker.bars[(b.con_id, grids[grid], what, params["end"])] = rows
    clock = FakeClock()
    pacing = PacingLimiter(clock=clock, sleep=clock.sleep, per_contract_seconds=0)
    session = broker.session(client_id=11)
    after = datetime(2026, 2, 1, tzinfo=timezone.utc)
    out = SnapshotStore(tmp_path / "collected")
    new_ref = collect_intraday(session, pacing, out, source.contract, clock=lambda: after)
    collected = out.load(new_ref)
    for key, block in source.blocks.items():
        for name in block.columns:
            np.testing.assert_allclose(collected.blocks[key].columns[name], block.columns[name])
    # Decision bars are aggregated locally: only execution and quote grids are requested.
    requests = [r for r in broker.transport.sent if r.kind == "historical_data"]
    assert len(collected.receipts) == len(requests) == 2 * len(source.blocks) // 3
    assert {r.params["bar_size"] for r in requests} == {"1 min"}
    with pytest.raises(ValueError, match="not yet complete"):
        collect_intraday(
            session,
            pacing,
            out,
            source.contract,
            clock=lambda: datetime(2026, 1, 5, tzinfo=timezone.utc),
        )


# Streaming finalisation --------------------------------------------------------------------------
def _at(session, seconds):
    return session.open_at.astimezone(timezone.utc) + timedelta(seconds=seconds)


def test_bar_is_final_by_clock_with_last_update():
    session = sessions(3)[0]
    finaliser = BarFinaliser("KO", "decision", session, 300, buffer_seconds=5, max_lag_seconds=120)
    opened = int(session.open_at.timestamp())
    finaliser.on_update(_payload(opened, 10.0), _at(session, 100))
    finaliser.on_update(_payload(opened, 10.5), _at(session, 299))
    assert finaliser.on_clock(_at(session, 304)) == []
    (bar,) = finaliser.on_clock(_at(session, 305))
    assert isinstance(bar, FinalBar) and bar.status == VALID and bar.close == 10.5
    assert bar.arrival_delay_seconds == pytest.approx(-1)
    (event,) = finaliser.on_update(_payload(opened, 11.0), _at(session, 310))
    assert event.kind == "revision"


def test_no_update_is_missing_while_current_and_stale_when_silent():
    session = sessions(3)[0]
    finaliser = BarFinaliser("KO", "decision", session, 300, buffer_seconds=5, max_lag_seconds=400)
    opened = int(session.open_at.timestamp())
    finaliser.on_update(_payload(opened + 300, 10.0), _at(session, 290))
    (bar,) = finaliser.on_clock(_at(session, 305))
    assert bar.status == MISSING
    out = finaliser.on_clock(_at(session, 1200))
    assert isinstance(out[0], FeedEvent) and out[0].kind == "stale_feed"
    assert [b.status for b in out[1:] if isinstance(b, FinalBar)][-1] == STALE
    events = finaliser.on_update(_payload(opened + 1200, 10.0), _at(session, 1210))
    assert events[0].kind == "feed_resumed"


def test_availability_check_flags_slow_feed():
    assert availability_check([1, 2, 3], 5)["violated"] is False
    assert availability_check([1, 2, 30], 5)["violated"] is True


# Paper order backend -----------------------------------------------------------------------------
def _intent(side="buy"):
    now = datetime(2026, 2, 9, 15, 0, tzinfo=timezone.utc)
    return OrderIntent(
        intent_id=f"ep:2026-02-09:3:KO:{side}:0",
        episode_id="ep",
        instrument="KO",
        side=side,
        purpose="entry",
        quantity=10,
        limit_price="60.05",
        decision_index=3,
        session_id="2026-02-09",
        submit_at=now,
        submit_deadline=now + timedelta(seconds=120),
    )


def test_backend_submits_limit_order_with_order_ref_and_account():
    broker = FakeBroker()
    backend = IbkrPaperBackend(broker.session(), config(), {"KO": binding("KO", 100)})
    event = backend.submit(_intent())
    order = broker.orders[event.order_id]
    assert order["order_ref"] == _intent().intent_id and order["account"] == ACCOUNT
    events = backend.poll()
    fills = [e for e in events if e.kind == "fill"]
    fees = [e for e in events if e.kind == "fee"]
    assert fills[0].intent_id == _intent().intent_id and fills[0].quantity == 10
    assert fees[0].amount == "1.0"


def test_backend_refuses_live_session_before_any_order():
    broker = FakeBroker(accounts=("U7654321",))
    with pytest.raises(LiveAccountRefused):
        IbkrPaperBackend(broker.session(), config(), {"KO": binding("KO", 100)})
    assert not [r for r in broker.transport.sent if r.kind == "place_order"]


def test_backend_rechecks_guard_on_every_submit():
    broker = FakeBroker()
    session = broker.session()
    backend = IbkrPaperBackend(session, config(), {"KO": binding("KO", 100)})
    session.info = type(session.info)(**{**session.info.__dict__, "accounts": ("U1",)})
    with pytest.raises(LiveAccountRefused):
        backend.submit(_intent())


def test_backend_translates_cancel_reject_and_foreign_executions():
    broker = FakeBroker(fill_sides=())
    session = broker.session()
    backend = IbkrPaperBackend(session, config(), {"KO": binding("KO", 100)})
    event = backend.submit(_intent())
    backend.cancel(event.order_id)
    kinds = [e.kind for e in backend.poll()]
    assert "cancelled" in kinds
    foreign = {
        "exec_id": "x",
        "order_id": 1,
        "order_ref": "",
        "account": "DU0000001",
        "con_id": 100,
        "side": "BOT",
        "shares": "5",
        "price": 1.0,
    }
    assert backend.translate(Message("exec_details", None, foreign)) is None
    rejected = backend.translate(Message("error", event.order_id, {"code": 201, "message": "no"}))
    assert rejected.kind == "rejected" and rejected.intent_id == _intent().intent_id
