"""Offline development metrics and actual two-fit execution, not predictive evidence."""

from copy import deepcopy
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from quant_rsi.core.data import features
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.research import demo
from quant_rsi.research.demo_features import price_feature, price_metrics
from quant_rsi.research.demo_model import LightGBMDemo

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def proposed(op, window):
    return {
        "factor_id": f"research.{op}_{window}",
        "description": "A finite price recipe proposed during the development revision",
        "kind": "numeric",
        "input_kind": "positive_price_series",
        "recipe": {"op": op, "window": window},
    }


@pytest.mark.parametrize("op", ["momentum", "mean_return", "qlib_roc_v1", "qlib_ma_v1"])
def test_selected_price_recipes_use_existing_executors(op):
    model = LightGBMDemo(demo.load_catalog(CATALOG))
    prices = model.data.closes[:, 0]
    values = price_feature(model.data, {"op": op, "window": 5})
    if op in {"momentum", "mean_return"}:
        expected = features(model.data, FactorSpec(op=op, window=5))[:, 0, 0]
        np.testing.assert_array_equal(values, expected)
    elif op == "qlib_roc_v1":
        assert np.isnan(values[:5]).all()
        assert values[100] == pytest.approx(prices[95] / prices[100])
    else:
        assert np.isnan(values[:4]).all()
        assert values[4] == pytest.approx(prices[:5].mean() / prices[4])
        assert values[100] == pytest.approx(prices[96:101].mean() / prices[100])


def test_metric_units_and_zero_direction_are_explicit():
    rows = [
        {"reference_price": 100.0, "predicted_price": 110.0, "observed_price": 120.0},
        {"reference_price": 100.0, "predicted_price": 90.0, "observed_price": 80.0},
    ]
    assert price_metrics(rows) == pytest.approx(
        {
            "mae_price": 10.0,
            "rmse_price": 10.0,
            "mape_percent": (10 / 120 + 10 / 80) * 50,
            "direction_hit_rate": 1.0,
            "sample_count": 2,
        }
    )
    assert (
        price_metrics([row | {"predicted_price": 100.0} for row in rows])["direction_hit_rate"]
        == 0.0
    )


def test_two_real_fits_share_purged_training_and_report_twenty_matured_origins(monkeypatch):
    training = []
    actual_train = lgb.train

    def train(params, data, **kwargs):
        training.append((data.data.copy(), data.label.copy()))
        return actual_train(params, data, **kwargs)

    monkeypatch.setattr(lgb, "train", train)
    model = LightGBMDemo(demo.load_catalog(CATALOG), evaluation_sessions=20)
    context = {"symbol": "KO", **model.context("KO")}
    first = model.predict(context, model.library["definitions"][:1], {"event_signal": -1})
    observed = model.observe(context, first)
    # Newly proposed windows were absent during initialization; the same fixed
    # training rows support every registered safe window through thirty.
    second = model.predict(
        context,
        [proposed("qlib_ma_v1", 30), proposed("mean_return", 12)],
        {"event_signal": 1},
    )
    assert len(training) == 2
    assert [x.shape for x, _ in training] == [(46, 1), (46, 2)]
    np.testing.assert_array_equal(training[0][1], training[1][1])
    assert first["last_training_label_session"] == model.data.dates[80]
    assert first["last_training_label_session"] < first["development"]["first_evaluation_session"]
    assert first["training_rows"] == second["training_rows"] == 46
    assert "retrospective" in context["model_note"] and "not independent" in context["model_note"]
    assert "retrospective" in first["development"]["evaluation_note"]
    assert first["development"]["baselines"] == second["development"]["baselines"]
    for prediction in (first, second):
        development = prediction["development"]
        rows = development["rows"]
        assert len(rows) == 20
        assert [row["origin_session"] for row in rows] == list(model.data.dates[81:101])
        assert [row["target_session"] for row in rows] == list(model.data.dates[86:106])
        assert prediction["predicted_price"] == rows[-1]["predicted_price"]
        assert prediction["predicted_log_return"] == rows[-1]["predicted_log_return"]
        expected_mae = sum(abs(row["predicted_price"] - row["observed_price"]) for row in rows) / 20
        assert development["metrics"]["mae_price"] == pytest.approx(expected_mae)
        assert development["metrics"]["sample_count"] == 20
        assert development["metric_units"]["mape_percent"] == "percent"
        assert development["metric_units"]["direction_hit_rate"] == "fraction"
        for name, baseline in development["baselines"].items():
            assert baseline["metrics"]["sample_count"] == 20
            assert [row["origin_session"] for row in baseline["rows"]] == [
                row["origin_session"] for row in rows
            ]
            expected_return = 0.0 if name == "unchanged_price" else training[0][1].mean()
            assert all(row["predicted_log_return"] == expected_return for row in baseline["rows"])
            for row in baseline["rows"]:
                assert row["predicted_price"] == pytest.approx(
                    row["reference_price"] * np.exp(expected_return)
                )
    feedback = observed | {"error": first["predicted_price"] - observed["observed_price"]}
    assert "rows" not in feedback["development"]
    assert all("rows" not in b for b in feedback["development"]["baselines"].values())
    assert feedback["development"]["metrics"] == first["development"]["metrics"]
    summary = model.summarize(context, first, second, feedback)
    assert summary["development"]["first_metrics"] == first["development"]["metrics"]
    assert summary["development"]["revised_metrics"] == second["development"]["metrics"]
    assert summary["improvement"] == "not_evaluated"
    assert "not independent validation" in summary["text"]
    assert "Overlapping" in summary["text"]
    # Feedback is detached from the complete immutable trace payload.
    observed["development"]["metrics"]["sample_count"] = -1
    assert first["development"]["metrics"]["sample_count"] == 20


def test_future_outcomes_change_metrics_without_changing_any_fitted_prediction():
    model = LightGBMDemo(demo.load_catalog(CATALOG), evaluation_sessions=20)
    selected = [proposed("qlib_roc_v1", 30), proposed("momentum", 5)]
    first = model.predict({}, selected, {"event_signal": -1})
    model.data.closes[101:] *= 1.25
    second = model.predict({}, selected, {"event_signal": 1})
    assert first["feature_values"] == second["feature_values"]
    for before, after in zip(
        first["development"]["rows"], second["development"]["rows"], strict=True
    ):
        assert before["predicted_price"] == after["predicted_price"]
        assert before["predicted_log_return"] == after["predicted_log_return"]
    assert first["development"]["metrics"] != second["development"]["metrics"]


def test_retained_prices_use_last_twenty_matured_origins_and_unobserved_latest_forecast():
    fixture = LightGBMDemo(demo.load_catalog(CATALOG))
    data = deepcopy(fixture.data)
    model = LightGBMDemo(
        demo.load_catalog(CATALOG),
        prices=(data, {"currency": "TEST", "latest_target_session": "2030-01-01"}),
        evaluation_sessions=20,
    )
    assert model.origin == len(data.dates) - 6
    prediction = model.predict({}, [proposed("qlib_ma_v1", 5)], {})
    rows = prediction["development"]["rows"]
    assert rows[-1]["target_session"] == data.dates[-1]
    assert rows[0]["origin_session"] == data.dates[-25]
    assert prediction["last_training_label_session"] < rows[0]["origin_session"]
    assert prediction["latest_forecast"]["outcome"] == "not_observed"
    assert (
        prediction["latest_forecast"]["training_information_cutoff"]
        == prediction["last_training_label_session"]
    )
    assert prediction["development"]["currency"] == "TEST"


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 21])
def test_study_rejects_invalid_evaluation_counts(count):
    with pytest.raises(ValueError, match="evaluation_sessions"):
        LightGBMDemo(demo.load_catalog(CATALOG), evaluation_sessions=count)


def test_default_demo_retains_old_window_and_output_shape():
    model = LightGBMDemo(demo.load_catalog(CATALOG))
    prediction = model.predict({}, model.library["definitions"][:1], {})
    assert prediction["training_rows"] == 91
    assert prediction["last_training_label_session"] == prediction["origin_session"]
    assert "development" not in prediction
    assert "development" not in model.observe({}, prediction)
    assert "evaluation_sessions" not in model.context("KO")
