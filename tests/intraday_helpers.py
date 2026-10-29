"""Small deterministic synthetic intraday fixtures; engineering only, not market evidence."""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from quant_rsi.brokers.contracts import InstrumentBinding
from quant_rsi.intraday.contracts import IntradayDataContract, IntradayDevelopmentPlan
from quant_rsi.intraday.storage import MISSING, VALID, Block, SnapshotStore
from quant_rsi.market.market_contracts import CalendarSpec, MarketSession

EASTERN = ZoneInfo("America/New_York")


def sessions(count: int, *, first=date(2026, 1, 5), early_close: int | None = None):
    result, day = [], first
    while len(result) < count:
        if day.weekday() < 5:
            close_hour = 13 if len(result) == early_close else 16
            result.append(
                MarketSession(
                    session_id=day.isoformat(),
                    open_at=datetime(day.year, day.month, day.day, 9, 30, tzinfo=EASTERN),
                    close_at=datetime(day.year, day.month, day.day, close_hour, 0, tzinfo=EASTERN),
                )
            )
        day += timedelta(days=1)
    return tuple(result)


def binding(name: str, con_id: int) -> InstrumentBinding:
    return InstrumentBinding(
        instrument=name,
        symbol=name,
        sec_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NYSE" if name == "KO" else "ARCA",
        con_id=con_id,
        time_zone_id="US/Eastern",
        trading_hours="",
        liquid_hours="",
    )


def contract(session_list, *, targets=("KO",), context=("SPY",), bar_seconds=300, buffer=5):
    names = list(targets) + list(context)
    return IntradayDataContract(
        schema_version=1,
        source_id="fixture-intraday",
        provider="engineering_fixture",
        provider_statement="synthetic engineering fixture",
        frequency="intraday_equities",
        grid={
            "bar_seconds": bar_seconds,
            "execution_bar_seconds": 60,
            "session_scope": "regular_trading_hours",
            "bar_timestamp": "bar_start",
            "final_bar_policy": "clip_to_session_close",
            "missing_bar_policy": "explicit_missing_no_fill_forward",
        },
        calendar=CalendarSpec(
            calendar_id="fixture-nyse", timezone="America/New_York", sessions=session_list
        ),
        instruments=tuple(binding(n, 100 + i) for i, n in enumerate(names)),
        targets=tuple(targets),
        knowledge_basis="assumed_historical_availability",
        availability_buffer_seconds=buffer,
        data_entitlement="realtime",
        quote_fields="trades_and_bid_ask",
        corporate_action_coverage="unknown",
    )


def minute_path(rng, count, start_price):
    returns = rng.normal(0, 0.0008, count)
    closes = start_price * np.exp(np.cumsum(returns))
    opens = np.concatenate([[start_price], closes[:-1]]) * (1 + rng.normal(0, 0.0001, count))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.0003, count)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.0003, count)))
    volume = rng.integers(2_000, 20_000, count).astype(float)
    return opens, highs, lows, closes, volume


def blocks_for(name, session, rng, price, bar_seconds, missing_minutes=()):
    opened, closed = int(session.open_at.timestamp()), int(session.close_at.timestamp())
    minutes = (closed - opened) // 60
    o, h, lo, c, v = minute_path(rng, minutes, price)
    status = np.full(minutes, VALID, np.uint8)
    for index in missing_minutes:
        status[index] = MISSING
    nan = ~(status == VALID)
    for column in (o, h, lo, c, v):
        column[nan] = np.nan
    starts = opened + 60 * np.arange(minutes, dtype=np.int64)
    execution = Block(
        name,
        "execution",
        session.session_id,
        {
            "start": starts,
            "open": o,
            "high": h,
            "low": lo,
            "close": c,
            "volume": v,
            "status": status,
        },
    )
    spread = np.where(nan, np.nan, o * 0.0002)
    quotes = Block(
        name,
        "execution_quotes",
        session.session_id,
        {
            "start": starts,
            "bid": o - spread / 2,
            "ask": o + spread / 2,
            "status": status.copy(),
        },
    )
    per = bar_seconds // 60
    bars = -(-minutes // per)
    agg = {k: np.full(bars, np.nan) for k in ("open", "high", "low", "close", "volume")}
    agg_status = np.full(bars, MISSING, np.uint8)
    for b in range(bars):
        part = slice(b * per, min((b + 1) * per, minutes))
        ok = status[part] == VALID
        if ok.any():
            idx = np.arange(minutes)[part][ok]
            agg["open"][b], agg["close"][b] = o[idx[0]], c[idx[-1]]
            agg["high"][b], agg["low"][b] = h[idx].max(), lo[idx].min()
            agg["volume"][b] = v[idx].sum()
            agg_status[b] = VALID
    decision = Block(
        name,
        "decision",
        session.session_id,
        {
            "start": opened + bar_seconds * np.arange(bars, dtype=np.int64),
            **agg,
            "status": agg_status,
        },
    )
    last = c[status == VALID][-1] if (status == VALID).any() else price
    return [decision, execution, quotes], float(last)


def build_snapshot(tmp_path, count=40, *, seed=7, missing=None, early_close=None, bar_seconds=300):
    session_list = sessions(count, early_close=early_close)
    spec = contract(session_list, bar_seconds=bar_seconds)
    rng = np.random.default_rng(seed)
    prices = {"KO": 60.0, "SPY": 500.0}
    blocks = []
    for session in session_list:
        for name in ("KO", "SPY"):
            gaps = (missing or {}).get((name, session.session_id), ())
            made, prices[name] = blocks_for(name, session, rng, prices[name], bar_seconds, gaps)
            blocks.extend(made)
    store = SnapshotStore(tmp_path / "store")
    ref = store.commit(spec, blocks, receipts=[{"source": "fixture", "first_seen_at": "fixture"}])
    return store, ref


def plan(**overrides) -> IntradayDevelopmentPlan:
    value = {
        "schema_version": 1,
        "plan_type": "intraday_development",
        "training_mode": "M0",
        "timing": {
            "decision_latency_seconds": 10,
            "order_latency_seconds": 1,
            "max_fill_delay_seconds": 120,
        },
        "signal": {
            "kind": "threshold_long_only",
            "entry_threshold": -1.0,
            "exit_threshold": -2.0,
            "horizon_bars": 6,
            "notional_fraction": "0.2",
            "max_positions": 1,
            "lot_size": 1,
            "flatten_offset_seconds": 900,
            "halt_missing_bars": 3,
            "overnight_policy": "flat",
        },
        "costs": {
            "policy": "ibkr_fixed_per_share",
            "commission_per_share": "0.005",
            "min_commission": "1",
            "max_commission_fraction": "0.01",
            "spread_policy": "mid_plus_half_spread",
            "assumed_spread_bps": "2",
            "participation_cap": "0.1",
            "currency": "USD",
            "currency_precision": 2,
        },
        "leaves": [
            "ret_1",
            "ret_3",
            "ret_6",
            "rv_12",
            "range",
            "minutes_since_open",
            "rsi_14",
            "ctx:SPY:ret_1",
        ],
        "folds": {"first_test_session": 24, "test_sessions": 4, "folds": 2, "embargo_sessions": 1},
        "seed": 11,
        "initial_capital": "100000",
        "min_training_rows": 50,
        "min_trades_per_fold": 1,
        "min_prediction_coverage": 0.5,
        "context_missing_tolerance": 0,
        "allow_cross_session_lookback": False,
        "sessions_per_year": 252,
    }
    for key, item in overrides.items():
        if isinstance(item, dict) and isinstance(value.get(key), dict):
            value[key] = {**value[key], **item}
        else:
            value[key] = item
    return IntradayDevelopmentPlan.model_validate(value, strict=False)
