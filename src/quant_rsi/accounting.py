"""Trusted causal order construction, deterministic synthetic fills, and net accounting."""

import math

import numpy as np

from quant_rsi.data import MarketData, decision_indices
from quant_rsi.schemas import PortfolioSpec


def equity_return(previous, equity, *, registered_zero_equity=False):
    if not all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (previous, equity)):
        raise ValueError("missing or invalid valuation/denominator")
    if previous == 0:
        if equity != 0 or not registered_zero_equity:
            raise ValueError("unregistered reset after zero equity")
        return 0.0
    if equity == 0 and not registered_zero_equity:
        raise ValueError("unregistered zero equity")
    return equity / previous - 1


def returns_from_equities(initial, equities, *, registered_zero_equity=False):
    if type(initial) not in (float, int) or not math.isfinite(initial) or initial <= 0:
        raise ValueError("invalid initial denominator")
    previous, returns = initial, []
    for equity in equities:
        returns.append(
            equity_return(previous, equity, registered_zero_equity=registered_zero_equity)
        )
        previous = equity
    return returns


def validate_predictions(data: MarketData, cutoff: int, rows: list[dict]) -> np.ndarray:
    if not isinstance(rows, list):
        raise ValueError("predictions must be a list")
    expected = [(t, name) for t in decision_indices(data, cutoff) for name in data.instruments]
    values = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "decision",
            "decision_time",
            "instrument",
            "score",
        }:
            raise ValueError("invalid prediction schema")
        t, name, score = row["decision"], row["instrument"], row["score"]
        if not isinstance(name, str):
            raise ValueError("invalid prediction instrument")
        if type(t) is not int or not 0 <= t < len(data.dates):
            raise ValueError("invalid prediction decision")
        if row["decision_time"] != data.timestamp(t, "close"):
            raise ValueError("prediction timestamp mismatch")
        if type(score) not in (float, int) or not math.isfinite(score):
            raise ValueError("non-finite prediction")
        key = (t, name)
        if key in values:
            raise ValueError("duplicate prediction")
        values[key] = float(score)
    if set(values) != set(expected):
        raise ValueError("prediction coverage differs from controller-bound grid")
    return np.asarray([values[k] for k in expected]).reshape(-1, len(data.instruments))


def target_orders(
    cash: float,
    holdings: np.ndarray,
    closes: np.ndarray,
    scores: np.ndarray,
    spec: PortfolioSpec,
) -> np.ndarray:
    nav = cash + float(holdings @ closes)
    ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[: spec.max_positions]
    targets = np.zeros(len(scores), dtype=int)
    for i in ranking:
        targets[i] = math.floor(nav * spec.allocation / spec.max_positions / closes[i])
    return targets - holdings


def evaluate(
    data: MarketData,
    cutoff: int,
    predictions: list[dict],
    spec: PortfolioSpec,
    annualization: int,
    *,
    total_loss_at: int | None = None,
) -> dict:
    # Explicit accounting-fixture event, never inferred from prices or missing data.
    # Production C5 registers no total-loss events: positive quotes and long-only
    # cash-constrained execution cannot generate exact bankruptcy in this source.
    if total_loss_at is not None and (
        type(total_loss_at) is not int
        or total_loss_at not in [t + 2 for t in decision_indices(data, cutoff)]
    ):
        raise ValueError("total loss must bind an explicit valuation session")
    scores = validate_predictions(data, cutoff, predictions)
    cash = spec.initial_cash
    holdings = np.zeros(len(data.instruments), dtype=int)
    fee_rate, slip = spec.fee_bps / 10_000, spec.slippage_bps / 10_000
    fills, orders, snapshots, returns = [], [], [], []
    previous_nav = cash
    loss_events = []

    def execute(t: int, quantities: np.ndarray, decision: int, terminal: bool = False):
        nonlocal cash
        # Registered priority: sells before buys, instrument order within each side.
        priority = sorted(range(len(quantities)), key=lambda i: (quantities[i] >= 0, i))
        for i in priority:
            submitted = int(quantities[i])
            if not submitted:
                continue
            orders.append(
                {
                    "decision_time": data.timestamp(decision, "close"),
                    "execution_time": data.timestamp(t, "open"),
                    "instrument": data.instruments[i],
                    "quantity": submitted,
                    "terminal": terminal,
                }
            )
            price = float(data.opens[t, i]) * (1 + slip if submitted > 0 else 1 - slip)
            if submitted > 0:
                # Exchange cash constraint affects fills, never submitted quantities.
                quantity = min(submitted, max(0, math.floor(cash / (price * (1 + fee_rate)))))
            else:
                quantity = -min(-submitted, int(holdings[i]))
            fee = abs(quantity) * price * fee_rate
            cash -= quantity * price + fee
            if -1e-8 < cash < 0:
                cash = 0.0
            holdings[i] += quantity
            if cash < 0 or (holdings < 0).any():
                raise ValueError("cash or inventory conservation failed")
            fills.append(
                {
                    "execution_time": data.timestamp(t, "open"),
                    "instrument": data.instruments[i],
                    "submitted": submitted,
                    "quantity": quantity,
                    "price": price,
                    "fee": fee,
                    "cash_after": cash,
                    "terminal": terminal,
                }
            )

    decisions = list(decision_indices(data, cutoff))
    for row, t in enumerate(decisions):
        quantities = target_orders(cash, holdings, data.closes[t], scores[row], spec)
        execute(t + 1, quantities, t)
        if row == len(decisions) - 1:
            execute(t + 2, -holdings.copy(), t + 1, terminal=True)
        nav = cash + float(holdings @ data.opens[t + 2])
        if not math.isfinite(nav) or nav < 0:
            raise ValueError("invalid portfolio valuation")
        if t + 2 == total_loss_at:
            loss_events.append(
                {
                    "kind": "registered_total_loss",
                    "time": data.timestamp(t + 2, "open"),
                    "equity_destroyed": nav,
                    "cash_destroyed": cash,
                    "positions_destroyed": holdings.tolist(),
                }
            )
            cash, nav = 0.0, 0.0
            holdings[:] = 0
        returns.append(equity_return(previous_nav, nav, registered_zero_equity=bool(loss_events)))
        snapshots.append(
            {
                "valuation_time": data.timestamp(t + 2, "open"),
                "cash": cash,
                "holdings": holdings.tolist(),
                "equity": nav,
            }
        )
        previous_nav = nav
    navs = np.asarray([spec.initial_cash] + [s["equity"] for s in snapshots])
    report = {
        "orders": orders,
        "fills": fills,
        "snapshots": snapshots,
        "net_returns": returns,
        "metrics": {
            "annualized_mean_net_active_return": annualization * float(np.mean(returns)),
            "total_return": previous_nav / spec.initial_cash - 1,
            "max_drawdown": float(np.min(navs / np.maximum.accumulate(navs) - 1)),
            "fees": sum(f["fee"] for f in fills),
            "filled_trades": sum(f["quantity"] != 0 for f in fills),
            "final_cash": cash,
        },
        "benchmark": "cash_zero_interest",
        "capacity_status": "unverified",
        "evidence_status": "development_only",
        "promotion_decision": "not_requested",
    }
    if loss_events:
        report["registered_total_loss_events"] = loss_events
    return report
