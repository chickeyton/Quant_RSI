"""Reserved terminal observations exercise real models, with offline provider fixtures."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from quant_rsi.cli import main
from quant_rsi.market import market_contracts
from quant_rsi.research import study
from quant_rsi.research.demo import load_catalog, simulated_hypothesis
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_estimators import StandardizedRidge
from quant_rsi.research.study_news import NewsHistory, simulated_history
from quant_rsi.research.study_ui import read_state

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def command(root, *options):
    return [
        "research-study",
        "KO",
        "--llm",
        "simulated",
        "--news",
        "simulated",
        "--prices",
        "synthetic",
        "--related",
        "simulated",
        "--no-ui",
        "--quiet",
        "--output",
        str(root),
        *options,
    ]


def factor(op, window):
    return {
        "factor_id": f"test.{op}_{window}",
        "kind": "numeric",
        "recipe": {"op": op, "window": window},
    }


@pytest.mark.parametrize("estimator", ["lightgbm", "ridge"])
def test_terminal_uses_exact_middle_model_once_without_refitting(estimator, monkeypatch):
    cls = lgb.Booster if estimator == "lightgbm" else StandardizedRidge
    original = cls.predict
    calls = []

    def observed(self, rows, **kwargs):
        values = original(self, rows, **kwargs)
        calls.append((self, rows.copy(), values.copy()))
        return values

    monkeypatch.setattr(cls, "predict", observed)
    model = LightGBMDemo(
        load_catalog(CATALOG), evaluation_sessions=20, terminal_sessions=10, estimator=estimator
    )
    plan = model.terminal_plan()
    for op, window in [("momentum", 5), ("mean_return", 10), ("qlib_ma_v1", 10)]:
        model.predict(model.context("KO"), [factor(op, window)], {})
    fitted = [call[0] for call in calls]
    assert len(fitted) == 3

    def forbidden(*args, **kwargs):
        raise AssertionError("Terminal evaluation must not fit another model")

    monkeypatch.setattr(lgb, "train", forbidden)
    monkeypatch.setattr(StandardizedRidge, "__init__", forbidden)
    terminal = model.evaluate_terminal(2)
    assert calls[-1][0] is fitted[1]
    assert len(calls) == 4
    assert terminal["factor_ids"] == ["test.mean_return_10"]
    assert terminal["refitted"] is False
    assert terminal["status"] == "completed"
    assert terminal["evaluation"]["metrics"]["sample_count"] == 10
    assert [r["predicted_log_return"] for r in terminal["evaluation"]["rows"]] == pytest.approx(
        calls[-1][2]
    )
    assert model.terminal_plan() == plan
    assert model.train[-1] + model.horizon < model.origins[0]
    assert model.origins[-1] + model.horizon < model.terminal_origins[0]
    with pytest.raises(ValueError, match="only run once"):
        model.evaluate_terminal(2)
    with pytest.raises(ValueError, match="fitting is closed"):
        model.predict({}, [factor("momentum", 5)], {})


@pytest.mark.parametrize("estimator", ["lightgbm", "ridge"])
def test_terminal_changes_do_not_change_research_prompts_selection_or_accounts(
    tmp_path, monkeypatch, estimator
):
    # Keep the explicit recipe-admission input fixed; perturb only terminal data.
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-11-01T10:00:00Z")
    prompts, runs = [], []
    changed = False

    def model_factory(*args, **kwargs):
        model = LightGBMDemo(*args, **kwargs)
        if changed:
            first = model.terminal_origins[0]
            # All development labels and account prices precede this perturbation.
            model.data.closes[first:, 0] *= np.linspace(1.2, 1.5, len(model.data.dates) - first)
            model.data.opens[first:, 0] *= np.linspace(1.2, 1.5, len(model.data.dates) - first)
        return model

    def hypothesis(*args, **kwargs):
        prompts.append(deepcopy((args, kwargs)))
        return simulated_hypothesis(*args, **kwargs)

    monkeypatch.setattr(study, "LightGBMDemo", model_factory)
    monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    for changed in (False, True):
        root = tmp_path / str(changed)
        assert main(command(root, "--model", estimator, "--rounds", "3")) == 0
        runs.append(json.loads((root / "result.json").read_text()))
    assert prompts[:3] == prompts[3:]
    assert runs[0]["iterations"] == runs[1]["iterations"]
    first_terminal = runs[0]["terminal_evaluation"]
    second_terminal = runs[1]["terminal_evaluation"]
    assert first_terminal["evaluation"]["metrics"] != second_terminal["evaluation"]["metrics"]
    for run in runs:
        terminal = run["terminal_evaluation"]
        assert terminal["selected_round"] == run["iterations"]["selected_round"]
        assert [e["stage"] for e in run["trace"]][-2:] == ["terminal_evaluation", "summary"]
        for trial in run["iterations"]["attempts"]:
            assert trial["prediction"]["strategy"]["period"]["end"] < terminal["origin_sessions"][0]
        assert run["runtime"]["counts"]["model_stages_completed"] == 3


def test_cli_saves_selection_before_evaluation_and_exposes_terminal_tables(tmp_path, monkeypatch):
    root = tmp_path / "study"
    original = LightGBMDemo.evaluate_terminal

    def evaluate(self, chosen):
        receipt = json.loads((root / "selected_attempt.json").read_text())
        plan = json.loads((root / "evaluation_plan.json").read_text())
        assert receipt["selected_round"] == chosen
        assert plan == self.terminal_plan()
        assert not (root / "terminal_evaluation.json").exists()
        return original(self, chosen)

    monkeypatch.setattr(LightGBMDemo, "evaluate_terminal", evaluate)
    assert main(command(root)) == 0
    result = json.loads((root / "result.json").read_text())
    assert [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()] == result[
        "trace"
    ]
    terminal = result["terminal_evaluation"]
    assert json.loads((root / "terminal_evaluation.json").read_text()) == terminal
    assert terminal["strategy"]["period"]["start"] == terminal["origin_sessions"][0]
    assert terminal["strategy"]["period"]["end"] == terminal["target_sessions"][-1]
    html = (root / "report.html").read_text()
    assert "Terminal forecast" in html and "Terminal account" in html
    state = read_state(root)
    views = {event["stage"]: event["view"] for event in state["events"]}
    assert views["terminal_evaluation"]["kind"] == "terminal"
    assert len(views["terminal_evaluation"]["rows"]) == 10
    assert views["summary"]["terminal"]["metrics"] == terminal["evaluation"]["metrics"]


@pytest.mark.parametrize("source", ["news", "peer"])
def test_missing_terminal_features_do_not_drop_rows_or_select_another_model(source):
    model = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20, terminal_sessions=10)
    first = int(model.terminal_origins[0])
    if source == "news":
        snapshots = simulated_history(model.data, "KO")[:first]
        cutoffs = [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
        model.configure_news(NewsHistory(snapshots), cutoffs, "KO")
        selected = {
            "factor_id": "test.news",
            "kind": "news",
            "recipe": {"op": "headline_keyword_count", "keywords": ["earnings"], "window_days": 1},
        }
    else:
        selected = {
            "factor_id": "test.peer",
            "kind": "numeric",
            "runnable": True,
            "recipe": {"op": "peer_return", "symbol": "PEP", "window": 5},
        }
        column = np.full(len(model.data.dates), 0.01)
        column[first:] = np.nan
        model.configure_peers(
            {
                "definitions": [selected],
                "columns": {"test.peer": column},
                "support": [],
                "inputs": {},
            }
        )
    before = model.origins.copy(), model.train.copy()
    model.predict(model.context("KO"), [selected], {})
    result = model.evaluate_terminal(1)
    assert result["status"] == "unavailable"
    assert result["sample_count"] == 10
    assert result["missing_inputs"]
    assert "evaluation" not in result
    np.testing.assert_array_equal(model.origins, before[0])
    np.testing.assert_array_equal(model.train, before[1])


def test_explicit_development_only_mode_does_not_evaluate_terminal(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled terminal evaluation must not be dispatched")

    monkeypatch.setattr(LightGBMDemo, "evaluate_terminal", forbidden)
    root = tmp_path / "development"
    assert main(command(root, "--terminal-sessions", "0")) == 0
    result = json.loads((root / "result.json").read_text())
    assert "terminal_evaluation" not in result
    assert json.loads((root / "evaluation_plan.json").read_text())["status"] == "disabled"
    assert not (root / "selected_attempt.json").exists()


def test_terminal_news_missingness_does_not_change_research_support():
    contexts, libraries = [], []
    for missing in (False, True):
        model = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20, terminal_sessions=10)
        snapshots = simulated_history(model.data, "KO")
        if missing:
            snapshots = snapshots[: model.terminal_origins[0]]
        cutoffs = [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
        model.configure_news(NewsHistory(snapshots), cutoffs, "KO")
        contexts.append(model.context("KO"))
        libraries.append(model.library)
    assert contexts[0] == contexts[1]
    assert libraries[0] == libraries[1]


@pytest.mark.parametrize("estimator", ["lightgbm", "ridge"])
def test_retained_daily_shape_keeps_latest_forecast_separate_from_terminal(estimator):
    fixture = LightGBMDemo(load_catalog(CATALOG))
    data = replace(fixture.data, instruments=("KO",))
    info = {"currency": "USD", "latest_target_session": "2020-07-03"}
    model = LightGBMDemo(
        load_catalog(CATALOG),
        prices=(data, info),
        evaluation_sessions=20,
        terminal_sessions=10,
        estimator=estimator,
    )
    prediction = model.predict(model.context("KO"), [factor("momentum", 5)], {})
    terminal = model.evaluate_terminal(1)
    latest = prediction["latest_forecast"]
    assert latest["outcome"] == "not_observed"
    assert latest["origin_session"] == data.dates[-1]
    assert terminal["target_sessions"][-1] == data.dates[-1]
    assert terminal["origin_sessions"][-1] < latest["origin_session"]
    assert (
        latest["training_information_cutoff"]
        < prediction["development"]["first_evaluation_session"]
    )
    assert "latest_forecast" not in model.observe(model.context("KO"), prediction)


def test_terminal_failure_keeps_frozen_selection_without_success_report(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("terminal prediction failed")

    monkeypatch.setattr(LightGBMDemo, "evaluate_terminal", failed)
    root = tmp_path / "failure"
    assert main(command(root)) != 0
    assert (root / "selected_attempt.json").exists()
    assert not (root / "result.json").exists()
    assert not (root / "report.html").exists()
    runtime = json.loads((root / "runtime.json").read_text())
    assert runtime["status"] == "failed"
    assert runtime["stages"][-1]["stage"] == "terminal_evaluation"
    assert runtime["counts"]["model_stages_completed"] == 2
