"""Bounded, full-window close-price adapters for source-pinned survey candidates.

These local implementations specify complete finite positive inputs. They do not
claim equivalence to Qlib's missing-data or partial-window operator behavior.
"""

import math
from typing import Literal

from pydantic import Field

from quant_rsi.core.contracts_base import StrictModel

QLIB_REVISION = "54355232463878d2eebb91fe0ee5fa7fa1f5976c"


class CandidatePriceRecipe(StrictModel):
    """Versioned semantics separate from the legacy two-operator FactorSpec."""

    op: Literal["qlib_roc_v1", "qlib_ma_v1"]
    window: int = Field(ge=1, le=30)
    upstream_revision: Literal["54355232463878d2eebb91fe0ee5fa7fa1f5976c"] = QLIB_REVISION

    @property
    def observations(self) -> int:
        return self.window + 1 if self.op == "qlib_roc_v1" else self.window


def evaluate_price_adapter(recipe: CandidatePriceRecipe, prices: list[float]) -> float | None:
    """Evaluate exactly one complete chronological window; never pad or impute.

    Both outputs are dimensionless levels (no subtraction of one). Invalid inputs
    or an unrepresentable output remain missing, including a zero denominator.
    """
    if len(prices) != recipe.observations:
        return None
    if any(not math.isfinite(price) or price <= 0 for price in prices):
        return None
    if recipe.op == "qlib_roc_v1":
        value = prices[0] / prices[-1]
    else:
        # Scale before summation to avoid overflowing an otherwise finite mean.
        scale = max(prices)
        mean = (math.fsum(price / scale for price in prices) / len(prices)) * scale
        value = mean / prices[-1]
    return value if math.isfinite(value) and value > 0 else None
