"""Numerical and connected ridge checks; synthetic results do not establish alpha."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from quant_rsi.cli import main
from quant_rsi.research import study
from quant_rsi.research.demo import load_catalog
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_estimators import StandardizedRidge

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def command(root, *extra):
    return [
        "research-study",
        "KO",
        "--output",
        str(root),
        "--llm",
        "simulated",
        "--news",
        "simulated",
        "--related",
        "simulated",
        "--prices",
        "synthetic",
        *extra,
    ]


def test_ridge_matches_hand_solution_with_unpenalized_intercept():
    # x mean=2, population std=sqrt(2/3), y mean=5. The standardized
    # slope is (4/sqrt(2/3))/(3+1), giving predictions [3.5,5,6.5].
    model = StandardizedRidge([[1], [2], [3]], [3, 5, 7], 1.0)
    np.testing.assert_allclose(model.predict([[1], [2], [3]]), [3.5, 5, 6.5])
    assert model.state()["intercept"] == 5
    np.testing.assert_allclose(model.state()["mean"], [2])
    np.testing.assert_allclose(model.state()["scale"], [np.sqrt(2 / 3)])
    assert model.predict([[4]])[0] == pytest.approx(8.0)


def test_duplicate_and_constant_columns_have_finite_stable_solution():
    model = StandardizedRidge([[1, 1, 8], [2, 2, 8], [3, 3, 8]], [3, 5, 7], 1.0)
    # Two identical standardized columns split the slope equally, reducing
    # their joint effective penalty to 1/2. The constant column contributes zero.
    np.testing.assert_allclose(model.predict([[1, 1, 8], [3, 3, 8]]), [23 / 7, 47 / 7])
    assert model.coefficients[0] == pytest.approx(model.coefficients[1])
    assert model.coefficients[2] == pytest.approx(0)
    assert model.scale[2] == 1


def test_decimal_constant_column_cannot_gain_slope_from_roundoff():
    model = StandardizedRidge(np.full((46, 1), 0.1), np.arange(46), 1.0)
    assert model.mean[0] == 0.1
    assert model.scale[0] == 1
    assert model.coefficients[0] == 0
    np.testing.assert_array_equal(model.predict([[0.1], [0.2], [1000]]), [22.5] * 3)


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf"), True])
def test_invalid_penalty_is_rejected(alpha):
    with pytest.raises(ValueError, match="finite and positive"):
        StandardizedRidge([[1], [2]], [3, 4], alpha)


@pytest.mark.parametrize("x,y", [([[1], [float("nan")]], [3, 4]), ([[1], [2]], [3]), ([], [])])
def test_invalid_training_inputs_are_rejected(x, y):
    with pytest.raises(ValueError, match="finite nonempty training"):
        StandardizedRidge(x, y, 1)


def test_evaluation_changes_cannot_refit_scaler_or_coefficients():
    model = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20, estimator="ridge")
    factors = model.library["definitions"][:2]
    before = model.predict({}, factors, {})
    # Every training label has matured before the first evaluation origin.
    # Change evaluation inputs/outcomes aggressively while retaining training history.
    evaluation = model.data.closes[model.origins[0] :]
    evaluation *= np.linspace(1.5, 2, len(evaluation)).reshape(-1, 1)
    after = model.predict({}, factors, {})
    assert before["fitted_model"] == after["fitted_model"]
    assert before["feature_values"] != after["feature_values"]
    assert before["development"]["metrics"] != after["development"]["metrics"]


@pytest.mark.parametrize(
    "options", [("--model", "ridge", "--ridge-alpha", "nan"), ("--ridge-alpha", "1")]
)
def test_invalid_configuration_stops_before_data_or_provider_calls(tmp_path, monkeypatch, options):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid model settings must be rejected before providers")

    monkeypatch.setattr(study, "load_daily_prices", forbidden)
    monkeypatch.setattr(study, "LLMHypothesizer", forbidden)
    root = tmp_path / "invalid"
    assert main(command(root, *options, "--prices", "daily", "--llm", "codex")) == 1
    assert not root.exists()


def test_connected_ridge_study_uses_two_fits_news_selection_accounts_and_report(
    tmp_path, monkeypatch
):
    fits = []
    original = StandardizedRidge.__init__

    def counted(self, x, y, alpha):
        fits.append((x.copy(), y.copy(), alpha))
        original(self, x, y, alpha)

    monkeypatch.setattr(StandardizedRidge, "__init__", counted)
    root = tmp_path / "ridge"
    assert main(command(root, "--model", "ridge", "--ridge-alpha", "2.5")) == 0
    result = json.loads((root / "result.json").read_text())
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    assert len(fits) == 2 and all(alpha == 2.5 for _, _, alpha in fits)
    np.testing.assert_array_equal(fits[0][1], fits[1][1])
    assert result["summary"]["factors_changed"]
    assert result["mode"] == "mixed" and "real Ridge training" in result["notice"]
    configuration = result["configuration"]["model_configuration"]
    assert configuration["name"] == "ridge" and configuration["schema_version"] == 1
    assert configuration == json.loads((root / "config.json").read_text())["model_configuration"]
    for index, key in enumerate(("first_prediction", "second_prediction")):
        prediction = steps[key]
        assert prediction["provider"] == "ridge" and "boost_rounds" not in prediction
        assert prediction["model_configuration"] == configuration
        assert prediction["training_feature_columns"] == 3
        assert prediction["news_feature_inputs"]
        assert prediction["development"]["metrics"]["sample_count"] == 20
        assert prediction["strategy"]["buy_and_hold"]["fills"]
        assert "sharpe_ratio" in prediction["strategy"]["strategy"]["metrics"]
        np.testing.assert_allclose(prediction["fitted_model"]["mean"], fits[index][0].mean(axis=0))
    first, second = (
        steps[key]["development"]["metrics"]["mae_price"]
        for key in ("first_prediction", "second_prediction")
    )
    expected = "revised" if second < first else "initial"
    assert result["study"]["selection"]["selected_attempt"] == expected
    assert result["runtime"]["counts"]["model_stages_completed"] == 2
    for name in ("report.html", "report.md"):
        report = (root / name).read_text()
        assert "Model configuration" in report and "ridge" in report and "2.5" in report
        assert "Sharpe" in report and "Buy-and-hold" in report


def test_latest_ridge_forecast_uses_same_fitted_coefficients_without_refit(monkeypatch):
    fixture = LightGBMDemo(load_catalog(CATALOG))
    data = deepcopy(fixture.data)
    model = LightGBMDemo(
        load_catalog(CATALOG),
        prices=(data, {"currency": "TEST", "latest_target_session": "2030-01-01"}),
        evaluation_sessions=20,
        estimator="ridge",
    )
    calls = []
    predict = StandardizedRidge.predict

    def observed(self, rows):
        calls.append((id(self), len(rows)))
        return predict(self, rows)

    monkeypatch.setattr(StandardizedRidge, "predict", observed)
    result = model.predict({}, model.library["definitions"][:1], {})
    assert [count for _, count in calls] == [20, 1]
    assert calls[0][0] == calls[1][0]
    assert result["latest_forecast"]["outcome"] == "not_observed"
    assert (
        result["latest_forecast"]["training_information_cutoff"]
        == result["last_training_label_session"]
    )
