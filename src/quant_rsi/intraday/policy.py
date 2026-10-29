"""Trading policy rules shared by the replay simulator and the paper runner (design 5.7).

The model predicts the H-bar forward return. The policy turns a finite prediction into an
entry quantity using only information available at the decision: the score, the decision
bar's close and the session's starting equity. Both evaluators call `entry_quantity`, so a
backtest and a paper episode with the same `SignalPolicy` take the same entries.

The cost hurdle is an estimate for the full order at the decision bar's close. A fill
reduced by the participation cap pays a higher commission share than estimated.
"""

import math
from decimal import ROUND_FLOOR, Decimal

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.intraday.contracts import CostModel, SignalPolicy
from quant_rsi.intraday.costs import commission, money


def signal_digest(signal: SignalPolicy) -> str:
    """SHA-256 of the canonical policy JSON; `fit` prints it and paper plans must match it."""
    return digest(canonical(signal.model_dump(mode="json")))


def lots(budget: Decimal, reference: Decimal, lot_size: int) -> int:
    return int((budget / reference / lot_size).to_integral_value(rounding=ROUND_FLOOR)) * lot_size


def round_trip_cost(costs: CostModel, reference: Decimal, quantity: int) -> float:
    """Estimated entry plus exit cost as a return: two commissions and one full spread."""
    notional = money(reference * quantity, costs)
    if notional <= 0:
        return math.inf
    fee = commission(quantity, notional, costs)
    return float(2 * fee / notional + Decimal(costs.assumed_spread_bps) / Decimal(10000))


def entry_hurdle(
    signal: SignalPolicy, costs: CostModel, reference: Decimal, quantity: int
) -> float:
    if signal.entry_rule == "fixed_threshold":
        return signal.entry_threshold
    return signal.entry_threshold + round_trip_cost(costs, reference, quantity)


def entry_quantity(
    signal: SignalPolicy, costs: CostModel, score: float, reference: Decimal, equity: Decimal
) -> int | None:
    """Shares to buy for a finite score; None when the policy declines the entry.

    0 means the full budget buys less than one lot, which callers count as a cancelled
    entry, as before policies were learnable.
    """
    budget = equity * Decimal(signal.notional_fraction)
    full = lots(budget, reference, signal.lot_size)
    if full <= 0:
        return 0
    hurdle = entry_hurdle(signal, costs, reference, full)
    if score <= hurdle:
        return None
    if signal.sizing == "fixed_fraction":
        return full
    scale = min(1.0, (score - hurdle) / signal.full_size_edge)
    quantity = lots(budget * Decimal(repr(scale)), reference, signal.lot_size)
    # A smaller order pays a larger commission share, so the hurdle is checked again.
    if quantity <= 0 or score <= entry_hurdle(signal, costs, reference, quantity):
        return None
    return quantity
