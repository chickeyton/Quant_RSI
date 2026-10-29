"""Finite price recipes and descriptive metrics for the optional research study."""

import numpy as np
from pydantic import TypeAdapter

from quant_rsi.core.data import features
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.factors.price_adapters import CandidatePriceRecipe, evaluate_price_adapter

PRICE_RECIPE = TypeAdapter(FactorSpec | CandidatePriceRecipe)


def price_feature(data, recipe):
    """Use existing executors; incomplete candidate windows remain unavailable."""
    recipe = PRICE_RECIPE.validate_python(recipe)
    if isinstance(recipe, FactorSpec):
        return features(data, recipe)[:, 0, 0]
    prices = data.closes[:, 0]
    result = np.full(len(prices), np.nan)
    for index in range(recipe.observations - 1, len(prices)):
        value = evaluate_price_adapter(
            recipe, prices[index + 1 - recipe.observations : index + 1].tolist()
        )
        if value is not None:
            result[index] = value
    return result


def price_metrics(rows):
    """Score identical matured origins; zero predicted change is its own direction."""
    reference = np.array([row["reference_price"] for row in rows])
    predicted = np.array([row["predicted_price"] for row in rows])
    observed = np.array([row["observed_price"] for row in rows])
    error = predicted - observed
    return {
        "mae_price": float(np.abs(error).mean()),
        "rmse_price": float(np.sqrt(np.square(error).mean())),
        "mape_percent": float((np.abs(error) / observed).mean() * 100),
        "direction_hit_rate": float(
            (np.sign(predicted - reference) == np.sign(observed - reference)).mean()
        ),
        "sample_count": len(rows),
    }


def compact_development(development):
    """Feedback retains metrics and bounds, while detailed rows stay in the report."""
    return {
        key: value for key, value in development.items() if key not in {"rows", "baselines"}
    } | {
        "baselines": {
            name: {key: value for key, value in baseline.items() if key != "rows"}
            for name, baseline in development["baselines"].items()
        }
    }
