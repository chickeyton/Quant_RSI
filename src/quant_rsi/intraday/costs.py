"""Money rounding and the IBKR fixed per-share commission, shared by the simulator and policy."""

from decimal import ROUND_HALF_EVEN, Decimal

from quant_rsi.intraday.contracts import CostModel


def money(value: Decimal, costs: CostModel) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-costs.currency_precision), rounding=ROUND_HALF_EVEN)


def price_decimal(value: float) -> Decimal:
    return Decimal(repr(round(float(value), 6)))


def commission(quantity: int, notional: Decimal, costs: CostModel) -> Decimal:
    """IBKR fixed per-share: per-share rate, per-order minimum, capped at a notional fraction."""
    raw = Decimal(costs.commission_per_share) * quantity
    raw = max(raw, Decimal(costs.min_commission))
    raw = min(raw, notional * Decimal(costs.max_commission_fraction))
    return money(raw, costs)
