"""Small volatility oracles and real study fits; synthetic data proves no market edge."""

from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pytest
from pydantic import ValidationError

from quant_rsi.research.demo import load_catalog, select_factors
from quant_rsi.research.demo_features import PRICE_RECIPE, price_feature
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.demo_proposals import FactorProposal, extend_library

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def test_log_return_volatility_uses_full_trailing_window_and_sample_deviation():
    data = SimpleNamespace(closes=np.exp([0.0, 0.1, 0.3, 0.2, 0.4])[:, None])
    recipe = {"op": "log_return_volatility", "window": 2}
    values = price_feature(data, recipe)
    assert np.isnan(values[:2]).all()
    # First two daily returns are 0.1 and 0.2: sample deviation is sqrt(0.005).
    assert values[2] == pytest.approx(np.sqrt(0.005))
    assert values[3] == pytest.approx(np.sqrt(0.045))
    data.closes[3:, 0] *= 100
    np.testing.assert_array_equal(price_feature(data, recipe)[:3], values[:3])
    data.closes[:] = 10
    np.testing.assert_array_equal(price_feature(data, recipe)[2:], 0)


@pytest.mark.parametrize("invalid_price", [0, -1, float("nan"), float("inf")])
def test_invalid_price_does_not_create_zero_or_bridge_a_missing_window(invalid_price):
    data = SimpleNamespace(closes=np.array([10, 11, invalid_price, 12, 13, 14])[:, None])
    values = price_feature(data, {"op": "log_return_volatility", "window": 2})
    assert np.isnan(values[:5]).all()
    assert np.isfinite(values[5])


@pytest.mark.parametrize("window", [1, 31, 2.5, True])
def test_proposal_and_executor_reject_invalid_volatility_windows(window):
    recipe = {"op": "log_return_volatility", "window": window}
    with pytest.raises(ValidationError):
        FactorProposal.model_validate(recipe | {"description": "Sample daily volatility"})
    with pytest.raises(ValidationError):
        PRICE_RECIPE.validate_python(recipe)


def test_volatility_proposal_reaches_real_training_and_preserves_development_rows(monkeypatch):
    fits = []
    train = lgb.train

    def record_train(params, dataset, **kwargs):
        fits.append(dataset.data.copy())
        return train(params, dataset, **kwargs)

    monkeypatch.setattr(lgb, "train", record_train)
    library = load_catalog(CATALOG)
    recipe = {"op": "log_return_volatility", "window": 30}
    proposal = {
        "factor_ids": ["research.log_return_volatility_30"],
        "proposed_factors": [recipe | {"description": "Sample daily volatility, not annualized"}],
    }
    extended = extend_library(library, proposal)
    assert extended != library
    assert "research.log_return_volatility_30" not in {
        item["factor_id"] for item in library["definitions"]
    }
    model = LightGBMDemo(extended, evaluation_sessions=20)
    context = {"symbol": "KO", **model.context("KO")}
    first = model.predict(context, model.library["definitions"][:1], {"event_signal": 0})
    second = model.predict(context, select_factors(extended, proposal), {"event_signal": 0})
    assert len(fits) == 2
    expected = price_feature(model.data, recipe)[model.train]
    np.testing.assert_allclose(fits[1][:, 0], expected)
    assert fits[1].shape == (46, 1)
    assert first["development"]["baselines"] == second["development"]["baselines"]
    assert second["development"]["metrics"]["sample_count"] == 20
