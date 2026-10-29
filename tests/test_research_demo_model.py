"""Actual LightGBM fits on synthetic prices, with offline LLM and news fixtures."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from quant_rsi.cli import main
from quant_rsi.research import demo, llm
from quant_rsi.research.demo_model import SyntheticLightGBMDemo

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def test_one_command_fits_selected_columns_then_reuses_feedback(monkeypatch, capsys):
    fit_columns, calls = [], []
    train = lgb.train

    def actual_train(params, data, **kwargs):
        fit_columns.append(data.data.shape[1])
        return train(params, data, **kwargs)

    def codex(self, prompt, schema):
        data = json.loads(prompt.split("\nINPUT_JSON:\n")[1])
        calls.append(data)
        assert data["context"]["model_mode"] == "lightgbm"
        assert "synthetic prices" in data["context"]["model_note"]
        assert "equity.announced_events" in data["catalog"]["excluded_factor_ids"]
        available = schema["properties"]["factor_ids"]["items"]["enum"]
        assert "equity.announced_events" not in available
        return json.dumps(
            {
                "hypothesis": "Explore numeric prices, then revise after the synthetic error.",
                "factor_ids": available if data["feedback"] else available[:1],
                "evidence_news_ids": [data["context"]["news"]["items"][0]["id"]],
                "event_signal": -0.5,
            }
        )

    monkeypatch.setattr(lgb, "train", actual_train)
    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", codex)
    assert (
        main(
            [
                "research-demo",
                "KO",
                "--catalog",
                str(CATALOG),
                "--model",
                "lightgbm",
                "--llm",
                "codex",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    assert len(calls) == 2 and fit_columns == [1, 2]
    assert calls[1]["feedback"] == steps["feedback"]
    assert calls[1]["previous"] == steps["initial_hypothesis"]
    first, second = steps["first_prediction"], steps["second_prediction"]
    assert first["provider"] == second["provider"] == "lightgbm"
    assert first["origin_session"] == second["origin_session"]
    assert first["training_rows"] == second["training_rows"] == 91
    assert first["last_training_label_session"] <= first["origin_session"]
    assert steps["feedback"]["observed_session"] > first["origin_session"]
    # Extra/revised columns need not change the fitted estimate or improve it.
    assert np.isfinite([first["predicted_price"], second["predicted_price"]]).all()
    assert list(second["feature_values"]) == second["factor_ids"]
    assert result["mode"] == "mixed" and "Real LightGBM" in result["notice"]
    assert result["summary"]["second_outcome"] == "reused_development_observation"
    assert result["summary"]["improvement"] == "not_evaluated"


def test_future_prices_and_news_signal_cannot_change_fitted_prediction():
    model = SyntheticLightGBMDemo(demo.load_catalog(CATALOG))
    selected = model.library["definitions"]
    prediction = model.predict({}, selected, {"event_signal": -1})
    prices = model.data.closes[:, 0]
    expected_momentum = prices[100] / prices[95] - 1
    expected_mean = np.mean(prices[96:101] / prices[95:100] - 1)
    assert prediction["feature_values"][selected[0]["factor_id"]] == expected_momentum
    assert prediction["feature_values"][selected[1]["factor_id"]] == expected_mean
    observation = model.observe({}, prediction)
    assert observation["observed_price"] == prices[105]
    model.data.closes[101:] *= 1.25
    assert model.predict({}, selected, {"event_signal": 1}) == prediction
    assert model.observe({}, prediction)["observed_price"] != observation["observed_price"]


def test_offline_hypothesis_uses_executable_catalog_with_real_model(capsys):
    assert (
        main(
            [
                "research-demo",
                "KO",
                "--catalog",
                str(CATALOG),
                "--model",
                "lightgbm",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["factors_changed"]
    assert "simulated LLM" in result["notice"]
    assert "synthetic" in result["notice"]


def test_model_rejects_text_inputs_and_catalog_without_revision_choices():
    library = demo.load_catalog(CATALOG)
    model = SyntheticLightGBMDemo(library)
    with pytest.raises(ValueError, match="only selected numeric"):
        model.predict({}, library["definitions"], {})
    with pytest.raises(ValueError, match="at least two numeric"):
        SyntheticLightGBMDemo({"definitions": library["definitions"][:1]})
