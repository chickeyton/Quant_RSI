from copy import deepcopy

import numpy as np
import pytest

from quant_rsi.accounting import evaluate, validate_predictions
from quant_rsi.data import MarketData
from quant_rsi.schemas import PortfolioSpec


def quotes(opens, closes=None):
    values = np.asarray(opens, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return MarketData(
        tuple(f"2020-01-{i + 1:02d}" for i in range(len(values))),
        tuple(f"A{i}" for i in range(values.shape[1])),
        values,
        values.copy() if closes is None else np.asarray(closes, dtype=float),
    )


def predictions(data, scores=None):
    return [
        {
            "decision": t,
            "decision_time": data.timestamp(t, "close"),
            "instrument": name,
            "score": 1.0 if scores is None else scores[t - 1][i],
        }
        for t in range(1, len(data.dates) - 2)
        for i, name in enumerate(data.instruments)
    ]


def portfolio(**overrides):
    return PortfolioSpec(
        **(
            {
                "kind": "long_only_topk",
                "initial_cash": 100.0,
                "max_positions": 1,
                "allocation": 1.0,
                "fee_bps": 0.0,
                "slippage_bps": 0.0,
                "max_drawdown": 1.0,
                "benchmark": "cash_zero_interest",
                "terminal_policy": "liquidate_at_final_open",
            }
            | overrides
        )
    )


def test_cash_limited_entry_and_terminal_fee_golden():
    data = quotes([10, 10, 10, 11])
    report = evaluate(data, 1, predictions(data), portfolio(fee_bps=100.0), 252)
    # Independent arithmetic: buy 9 at 10 + .90 fee; sell 9 at 11 - .99 fee.
    assert [o["quantity"] for o in report["orders"]] == [10, -9]
    assert [f["quantity"] for f in report["fills"]] == [9, -9]
    assert report["metrics"]["final_cash"] == pytest.approx(107.11)
    assert report["metrics"]["fees"] == pytest.approx(1.89)
    assert report["net_returns"] == pytest.approx([0.0711])
    assert report["metrics"]["annualized_mean_net_active_return"] == pytest.approx(17.9172)
    assert report["snapshots"][-1]["holdings"] == [0]


def test_future_open_changes_fill_but_never_submitted_order():
    ordinary = quotes([10, 10, 10, 11])
    gap = quotes([10, 10, 20, 11])
    a = evaluate(ordinary, 1, predictions(ordinary), portfolio(), 252)
    b = evaluate(gap, 1, predictions(gap), portfolio(), 252)
    assert a["orders"][0] == b["orders"][0]
    assert a["fills"][0]["quantity"] == 10
    assert b["fills"][0]["quantity"] == 5
    assert b["metrics"]["final_cash"] == 55.0
    assert b["metrics"]["max_drawdown"] == pytest.approx(-0.45)


def test_slippage_and_losing_terminal_inventory_are_charged():
    data = quotes([10, 10, 10, 8])
    report = evaluate(data, 1, predictions(data), portfolio(slippage_bps=100.0), 252)
    # 100 - 9*10.10 + 9*7.92 = 80.38, with no open position left behind.
    assert report["metrics"]["final_cash"] == pytest.approx(80.38)
    assert report["metrics"]["total_return"] == pytest.approx(-0.1962)
    assert report["fills"][-1]["terminal"] is True


def test_sell_before_buy_and_interval_returns_golden():
    data = quotes([[10, 20], [10, 20], [10, 20], [12, 20], [12, 22]])
    rows = predictions(data, [[2.0, 1.0], [1.0, 2.0]])
    report = evaluate(data, 1, rows, portfolio(), 252)
    # Buy 10 A0. Next close targets 5 A1 using known NAV=100.
    # Sell A0 for 120, buy 5 A1 for 100, then exit for 110 + cash 20.
    assert [f["quantity"] for f in report["fills"]] == [10, -10, 5, -5]
    assert [s["equity"] for s in report["snapshots"]] == [120.0, 130.0]
    assert report["net_returns"] == pytest.approx([0.2, 1 / 12])
    assert report["metrics"]["final_cash"] == 130.0
    assert report["metrics"]["annualized_mean_net_active_return"] == pytest.approx(35.7)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "timestamp", "nan", "extra"])
def test_predictions_must_match_controller_grid(mutation):
    data = quotes([10, 10, 10, 11])
    rows = deepcopy(predictions(data))
    if mutation == "missing":
        rows.clear()
    elif mutation == "duplicate":
        rows *= 2
    elif mutation == "timestamp":
        rows[0]["decision_time"] = data.timestamp(2, "close")
    elif mutation == "nan":
        rows[0]["score"] = float("nan")
    else:
        rows[0]["instrument"] = "UNKNOWN"
    with pytest.raises(ValueError):
        validate_predictions(data, 1, rows)
