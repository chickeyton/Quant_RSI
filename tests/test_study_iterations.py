"""Budgeted real fits and controlled development scores; no predictive-quality claim."""

import json
from copy import deepcopy

import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_ui import read_state


@pytest.mark.parametrize("scores,chosen", [([5, 2, 3, 4], 2), ([5, 4, 1, 2], 3)])
def test_all_rounds_run_and_forecast_selection_retains_best_not_last(
    tmp_path, monkeypatch, scores, chosen
):
    predict, hypothesize = LightGBMDemo.predict, study.simulated_hypothesis
    predictions, calls = [], []

    def controlled(self, context, factors, hypothesis):
        result = predict(self, context, factors, hypothesis)
        result["development"]["metrics"]["mae_price"] = scores[len(predictions)]
        predictions.append(deepcopy(result))
        return result

    def hypotheses(context, library, **kwargs):
        calls.append(deepcopy(kwargs))
        result = hypothesize(context, library, **kwargs)
        result["hypothesis"] = f"Hypothesis round {len(calls)}"
        return result

    monkeypatch.setattr(LightGBMDemo, "predict", controlled)
    monkeypatch.setattr(study, "simulated_hypothesis", hypotheses)
    root = tmp_path / "four-rounds"
    assert main(command(root, "--llm", "simulated", "--rounds", "4", "--no-ui")) == 0
    result = json.loads((root / "result.json").read_text())
    loop = result["iterations"]
    assert len(predictions) == len(calls) == loop["completed_fits"] == 4
    assert loop["stop_reason"] == "round_limit" and loop["selected_round"] == chosen
    assert result["summary"]["selection"]["selected_round"] == chosen
    assert calls[3]["previous"]["hypothesis"] == f"Hypothesis round {chosen}"
    assert len(calls[3]["feedback"]["research_history"]) == 3
    assert calls[3]["feedback"]["incumbent_round"] == chosen
    assert (
        len({tuple(row["origin_session"] for row in p["development"]["rows"]) for p in predictions})
        == 1
    )
    assert len({p["training_rows"] for p in predictions}) == 1
    assert result["runtime"]["counts"]["model_stages_started"] == 4
    assert (root / "trace.jsonl").read_text().count('"stage": "round_4_prediction"') == 1
    for path in (root / "report.html", root / "report.md"):
        report = path.read_text()
        assert "Research rounds" in report and f"selected round: {chosen}" in report
        assert f"selected round_{chosen} fitted model" in report
    state = read_state(root)
    assert state["config"]["research_rounds"] == state["config"]["fit_budget"] == 4
    assert state["iterations"] == {
        "completed_fits": 4,
        "selected_round": chosen,
        "stop_reason": "round_limit",
    }
    views = {event["stage"]: event["view"] for event in state["events"]}
    assert views["round_3_prediction"]["kind"] == "prediction"
    assert views["round_4_decision"]["incumbent_round"] == chosen
    assert len(views["summary"]["rounds"]) == 4
    if chosen == 3:
        assert "Selected" in [account["label"] for account in views["summary"]["accounts"]]


@pytest.mark.parametrize(
    "extra,count,reason",
    [
        (["--max-fits", "3"], 3, "fit_budget_exhausted"),
        (["--patience", "1"], 2, "no_progress_limit"),
    ],
)
def test_stop_before_dispatching_extra_fit(tmp_path, monkeypatch, extra, count, reason):
    predict = LightGBMDemo.predict
    fitted = []

    def tied(self, *args):
        result = predict(self, *args)
        result["development"]["metrics"]["mae_price"] = 1.0
        fitted.append(result)
        return result

    monkeypatch.setattr(LightGBMDemo, "predict", tied)
    root = tmp_path / "stopped"
    assert main(command(root, "--llm", "simulated", "--rounds", "5", *extra)) == 0
    result = json.loads((root / "result.json").read_text())
    assert len(fitted) == result["iterations"]["completed_fits"] == count
    assert result["iterations"]["stop_reason"] == reason
    assert result["iterations"]["selected_round"] == 1
    state = read_state(root)
    assert state["iterations"]["completed_fits"] == count
    assert state["config"]["research_rounds"] == 5
    assert state["config"]["fit_budget"] == (3 if reason == "fit_budget_exhausted" else 5)


def test_failure_keeps_completed_rounds_and_stops(tmp_path, monkeypatch):
    predict = LightGBMDemo.predict
    calls = []

    def fail(self, *args):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError("third fit failed")
        return predict(self, *args)

    monkeypatch.setattr(LightGBMDemo, "predict", fail)
    root = tmp_path / "failed"
    assert main(command(root, "--llm", "simulated", "--rounds", "5")) == 1
    assert len(calls) == 3 and not (root / "result.json").exists()
    assert json.loads((root / "failure.json").read_text())["error"] == "third fit failed"
    events = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["stage"].endswith("_decision")]) == 2


def test_strict_improvement_resets_patience(tmp_path, monkeypatch):
    predict = LightGBMDemo.predict
    scores = iter([5, 6, 4, 5, 6])

    def controlled(self, *args):
        result = predict(self, *args)
        result["development"]["metrics"]["mae_price"] = next(scores)
        return result

    monkeypatch.setattr(LightGBMDemo, "predict", controlled)
    root = tmp_path / "reset-patience"
    assert main(command(root, "--llm", "simulated", "--rounds", "7", "--patience", "2")) == 0
    loop = json.loads((root / "result.json").read_text())["iterations"]
    assert loop["completed_fits"] == 5 and loop["selected_round"] == 3
    assert loop["stop_reason"] == "no_progress_limit"
