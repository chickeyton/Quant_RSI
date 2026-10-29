"""Product wiring tests: fake providers, real catalog metadata, no market-quality claims."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from quant_rsi.research import demo

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "instruments/shared/equities.json"


def catalog():
    return demo.load_catalog(CATALOG)


def outputs(result):
    return {event["stage"]: event["output"] for event in result["trace"]}


def test_one_command_shows_complete_simulated_loop_without_workspace(tmp_path):
    env = dict(os.environ)
    env.pop("QUANT_RSI_WORKSPACE", None)
    command = [sys.executable, "-m", "quant_rsi", "research-demo", "ko", "--catalog", str(CATALOG)]
    run = subprocess.run(
        command + ["--json"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["target"] == "KO" and result["mode"] == "simulation"
    assert [event["stage"] for event in result["trace"]] == [
        "target",
        "related_stocks",
        "news",
        "factor_library",
        "initial_hypothesis",
        "initial_factors",
        "first_prediction",
        "feedback",
        "revised_hypothesis",
        "revised_factors",
        "second_prediction",
        "summary",
    ]
    assert result["summary"]["factors_changed"]
    assert result["summary"]["improvement"] == "not_evaluated"
    readable = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert readable.returncode == 0
    assert "SIMULATION" in readable.stdout and "second_prediction" in readable.stdout
    assert not list(tmp_path.iterdir())


def test_news_feedback_and_modified_factors_reach_the_next_provider_calls():
    hypothesis_calls, prediction_calls = [], []

    def hypothesize(context, library, **kwargs):
        hypothesis_calls.append(deepcopy((context, library, kwargs)))
        return demo.simulated_hypothesis(context, library, **kwargs)

    def predict(context, factors, hypothesis):
        prediction_calls.append(deepcopy((context, factors, hypothesis)))
        return demo.simulated_predict(context, factors, hypothesis)

    result = demo.run_demo("CALM", catalog=catalog, hypothesize=hypothesize, predict=predict)
    steps = outputs(result)
    assert hypothesis_calls[0][0]["symbol"] == "CALM"
    assert hypothesis_calls[0][0]["news"] == steps["news"]
    assert hypothesis_calls[0][1] == steps["factor_library"]
    revised_input = hypothesis_calls[1][2]
    assert revised_input["previous"] == steps["initial_hypothesis"]
    assert revised_input["feedback"] == steps["feedback"]
    assert revised_input["feedback"]["error"] == 4
    assert steps["revised_hypothesis"]["evidence_news_ids"] == [steps["news"]["items"][0]["id"]]
    assert prediction_calls[1][1] == steps["revised_factors"]
    assert prediction_calls[1][1] != prediction_calls[0][1]
    assert steps["second_prediction"]["factor_ids"] == [
        factor["factor_id"] for factor in prediction_calls[1][1]
    ]
    assert steps["first_prediction"]["predicted_price"] == 102
    assert steps["second_prediction"]["predicted_price"] == 99.5


def test_replacing_one_feedback_or_news_stub_changes_the_loop():
    original = outputs(demo.run_demo("KO", catalog=catalog))
    other_feedback = outputs(
        demo.run_demo(
            "KO",
            catalog=catalog,
            observe=lambda context, prediction: {
                "provider": "test_observation",
                "observed_price": 106,
            },
        )
    )
    assert other_feedback["feedback"]["error"] == -4
    assert other_feedback["revised_factors"] != original["revised_factors"]
    assert other_feedback["second_prediction"]["predicted_price"] == 102.5

    def news(context):
        items = demo.simulated_news(context)
        items["items"][0]["text"] = "SIMULATED: stronger demand"
        return items

    other_news = outputs(demo.run_demo("KO", catalog=catalog, news=news))
    assert other_news["first_prediction"] == original["first_prediction"]
    assert other_news["revised_hypothesis"]["event_signal"] == 1
    assert other_news["second_prediction"]["predicted_price"] == 101.5


@pytest.mark.parametrize("ids", [["unknown.factor"], ["equity.price_momentum_5"] * 2])
def test_invalid_llm_factor_choices_are_not_silently_substituted(ids):
    with pytest.raises(ValueError, match="unknown factors|duplicate"):
        demo.run_demo("KO", catalog=catalog, hypothesize=lambda *args: {"factor_ids": ids})
