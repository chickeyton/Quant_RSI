"""Chronological diagnostic behavior; fixtures do not establish predictive quality."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_research_study_model import CATALOG, proposed
from test_study_horizon import command

from quant_rsi.cli import main
from quant_rsi.research import demo, study_walk_forward
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_queue import study_arguments, study_options
from quant_rsi.research.study_ui import read_state
from quant_rsi.research.study_ui_views import visual_output


@pytest.mark.parametrize("horizon", [5, 20, 60])
@pytest.mark.parametrize("estimator", ["ridge", "lightgbm"])
def test_selected_recipe_is_refitted_on_three_disjoint_windows(tmp_path, horizon, estimator):
    root = tmp_path / "study"
    assert main(command(root, horizon, model=estimator) + ["--walk-forward-folds", "3"]) == 0
    result = json.loads((root / "result.json").read_text())
    diagnostic = result["study"]["walk_forward"]
    plan = json.loads((root / "walk_forward_plan.json").read_text())
    assert diagnostic["status"] == "completed"
    assert diagnostic["extra_fits"] == diagnostic["fit_budget"] == 3
    assert result["configuration"]["walk_forward_fit_budget"] == 3
    assert len(result["iterations"]["attempts"]) == 2
    chosen = diagnostic["selected_round"]
    selected = result["iterations"]["attempts"][chosen - 1]
    assert diagnostic["factor_ids"] == selected["prediction"]["factor_ids"]
    assert (
        diagnostic["model_configuration"]["parameters"]
        == selected["prediction"]["model_configuration"]["parameters"]
    )
    assert diagnostic["model_configuration"]["training_policy"].startswith("expanding_")
    prior_end = None
    for fold, frozen in zip(diagnostic["folds"], plan["folds"], strict=True):
        assert fold["first_origin"] == frozen["first_origin"]
        assert fold["last_training_label_session"] < fold["first_origin"]
        assert len(fold["training_sessions"]) >= 12
        assert fold["evaluation"]["metrics"]["sample_count"] == 20
        assert fold["strategy"]["period"]["end"] == fold["last_target"]
        assert fold["strategy"]["strategy"]["metrics"]["fees"]["commission"] is not None
        if prior_end:
            assert prior_end < fold["first_origin"]
        prior_end = fold["last_target"]
    assert prior_end < result["terminal_evaluation"]["evaluation"]["rows"][0]["origin_session"]
    assert diagnostic["pooled_forecast_metrics"]["model"]["sample_count"] == 60
    assert "Frozen-recipe time-window diagnostic" in (root / "report.html").read_text()
    view = visual_output("walk_forward", diagnostic, {}, "KO")
    assert view["kind"] == "engine_comparison" and len(view["rows"]) == 6
    assert view["extra_fits"] == 3
    assert read_state(root)["config"]["walk_forward_fit_budget"] == 3


def test_missing_window_is_visible_and_cannot_improve_pooled_metrics(tmp_path, monkeypatch):
    original = study_walk_forward.evaluate_windows

    def missing(model, chosen, policy):
        matrix = model._fits[chosen - 1][1]
        index = model.walk_forward_plan["folds"][0]["origin_indices"][0]
        saved = matrix[index].copy()
        matrix[index] = np.nan
        try:
            return original(model, chosen, policy)
        finally:
            matrix[index] = saved

    monkeypatch.setattr(study_walk_forward, "evaluate_windows", missing)
    root = tmp_path / "missing"
    assert main(command(root, 20) + ["--walk-forward-folds", "3"]) == 0
    result = json.loads((root / "result.json").read_text())
    diagnostic = result["study"]["walk_forward"]
    assert diagnostic["status"] == "incomplete"
    assert diagnostic["extra_fits"] == 2
    assert diagnostic["folds"][0]["status"] == "unavailable"
    assert diagnostic["folds"][0]["reason"] == "missing_evaluation_inputs"
    assert diagnostic["folds"][0]["missing_evaluation_sessions"]
    assert diagnostic["pooled_forecast_metrics"] is None
    assert result["terminal_evaluation"]["status"] == "completed"


def test_diagnostic_failure_resume_reuses_finished_fits(tmp_path, monkeypatch):
    import lightgbm as lgb

    original, calls = lgb.train, []

    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 4:  # Two research fits, one completed diagnostic, then failure.
            raise RuntimeError("deliberate diagnostic fit failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(lgb, "train", fail_once)
    root = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="deliberate diagnostic"):
        main(command(root, 5, model="lightgbm") + ["--walk-forward-folds", "3"])
    assert len(calls) == 4
    first = (root / "checkpoints/walk_forward_fit_1.json").read_bytes()
    assert main(["research-resume", str(root)]) == 0
    assert len(calls) == 6
    assert (root / "checkpoints/walk_forward_fit_1.json").read_bytes() == first
    result = json.loads((root / "result.json").read_text())
    assert result["study"]["walk_forward"]["extra_fits"] == 3


def test_later_outcomes_cannot_change_earlier_fold_prediction(tmp_path, monkeypatch):
    original = study_walk_forward.evaluate_windows
    compared = []

    def check(model, chosen, policy):
        first = original(model, chosen, policy)
        saved = model.data.closes.copy()
        cutoff = model.walk_forward_plan["folds"][0]["origin_indices"][-1] + model.horizon
        try:
            model.data.closes[cutoff + 1 :] *= 1.5
            # Avoid reusing cached fits so this tests training boundaries, not cache hits.
            cache = model.fit_cache
            model.fit_cache = None
            later = original(model, chosen, policy)
        finally:
            model.data.closes[:] = saved
            model.fit_cache = cache
        assert first["folds"][0]["evaluation"] == later["folds"][0]["evaluation"]
        compared.append(True)
        return first

    monkeypatch.setattr(study_walk_forward, "evaluate_windows", check)
    assert main(command(tmp_path / "future", 5) + ["--walk-forward-folds", "2"]) == 0
    assert compared == [True]


@pytest.mark.parametrize("folds", [True, 1, 4, "3"])
def test_invalid_budget_rejected_by_workbench(folds):
    with pytest.raises(ValueError, match="Walk-forward"):
        study_options({"walk_forward_folds": folds})


def test_workbench_forwards_explicit_fit_budget(tmp_path):
    options = study_options({"walk_forward_folds": 3})
    args = study_arguments(tmp_path / "task", tmp_path, "KO", options)
    assert args[args.index("--walk-forward-folds") + 1] == "3"
    assert study_options({})["walk_forward_folds"] == 0


def test_extra_diagnostic_history_does_not_change_research_fit():
    library = demo.load_catalog(CATALOG)
    h, folds = 20, 3
    fixture = LightGBMDemo(
        library, horizon_sessions=h, evaluation_sessions=20, walk_forward_folds=folds
    )
    extra = study_walk_forward.extra_history(folds, 20, h)
    longer = fixture.data
    shorter = replace(
        longer,
        dates=longer.dates[extra:],
        opens=longer.opens[extra:].copy(),
        closes=longer.closes[extra:].copy(),
    )
    info = {"horizon_sessions": h, "currency": "TEST", "latest_target_session": "2030-01-01"}
    predictions = []
    for data, count in ((shorter, 0), (longer, folds)):
        model = LightGBMDemo(
            library,
            prices=(data, info),
            horizon_sessions=h,
            evaluation_sessions=20,
            walk_forward_folds=count,
            estimator="ridge",
        )
        predictions.append(model.predict({}, [proposed("momentum", 10)], {}))
    assert predictions[0]["development"] == predictions[1]["development"]
    assert predictions[0]["latest_forecast"] == predictions[1]["latest_forecast"]
