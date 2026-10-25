import numpy as np
import pytest
from pydantic import ValidationError

from quant_rsi.artifacts import canonical
from quant_rsi.data import MarketData, features, synthetic_data
from quant_rsi.model import fit_predict
from quant_rsi.schemas import DevelopmentPlan, FactorSpec


@pytest.mark.parametrize("op", ["momentum", "mean_return"])
def test_future_quotes_cannot_change_earlier_features_or_predictions(plan, op):
    data = synthetic_data(plan.data)
    boundary = plan.data.train_cutoff
    altered = MarketData(data.dates, data.instruments, data.opens.copy(), data.closes.copy())
    altered.opens[boundary + 1 :] *= 1.7
    altered.closes[boundary + 1 :] *= 0.6
    factor = FactorSpec(op=op, window=5)
    np.testing.assert_allclose(
        features(data, factor)[: boundary + 1],
        features(altered, factor)[: boundary + 1],
        equal_nan=True,
    )
    original = fit_predict(data, plan, factor)
    changed = fit_predict(altered, plan, factor)
    assert original["model_digest"] == changed["model_digest"]
    assert [r for r in original["predictions"] if r["decision"] <= boundary] == [
        r for r in changed["predictions"] if r["decision"] <= boundary
    ]
    assert original["last_training_label_available"] <= original["training_information_cutoff"]
    assert original["training_rows"] == (plan.data.train_cutoff - 1 - 5) * plan.data.instruments


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_mode", "M1"),
        ("plan_type", "protected"),
        ("unknown", True),
        ("annualization", "252"),
        ("min_development_sessions", 200),
    ],
)
def test_unsupported_or_coerced_plans_are_rejected(plan, field, value):
    document = plan.model_dump(mode="json") | {field: value}
    with pytest.raises(ValidationError):
        DevelopmentPlan.model_validate_json(canonical(document))


def test_duplicate_candidates_are_rejected(plan):
    document = plan.model_dump(mode="json")
    document["proposals"][1]["name"] = document["baseline"]
    with pytest.raises(ValidationError):
        DevelopmentPlan.model_validate_json(canonical(document))
