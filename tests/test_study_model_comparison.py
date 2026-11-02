"""Paired estimator fits preserve selection, common samples, costs and recovery."""

import json
from copy import deepcopy

import numpy as np
import pytest
from test_research_study import command
from test_study_queue import OFFLINE, finish_queue

from quant_rsi.cli import main
from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import study_model_comparison, study_walk_forward
from quant_rsi.research.study_estimators import StandardizedRidge
from quant_rsi.research.study_queue import study_arguments, study_options
from quant_rsi.research.study_ui import read_state
from quant_rsi.research.study_workbench import Workbench


def run(root, model="lightgbm", *extra):
    return main(
        command(
            root,
            "--llm",
            "simulated",
            "--model",
            model,
            "--no-history",
            "--no-ui",
            "--quiet",
            *extra,
        )
    )


@pytest.mark.parametrize("model", ["lightgbm", "ridge"])
def test_common_input_model_comparison_preserves_selected_predictions(tmp_path, model):
    baseline, paired = tmp_path / "baseline", tmp_path / "paired"
    assert run(baseline, model) == run(paired, model, "--compare-models") == 0
    before = json.loads((baseline / "result.json").read_text())
    result = json.loads((paired / "result.json").read_text())
    selected = result["iterations"]["selected_round"]
    assert before["iterations"]["selected_round"] == selected
    native = result["iterations"]["attempts"][selected - 1]["prediction"]
    assert native == before["iterations"]["attempts"][selected - 1]["prediction"]
    assert result["terminal_evaluation"] == before["terminal_evaluation"]
    comparison = result["study"]["model_comparison"]
    assert comparison["selected_model"] == model
    assert set(comparison["models"]) == {"lightgbm", "ridge"}
    assert comparison["inputs_sha256"] == digest(
        canonical(json.loads((paired / "model_comparison_inputs.json").read_text()))
    )
    assert result["configuration"]["comparison_fit_budget"] == comparison["comparison_fits"] == 1
    assert result["runtime"]["counts"]["model_stages_completed"] == 2
    assert result["runtime"]["counts"]["comparison_model_stages_completed"] == 1
    inputs = json.loads((paired / "model_comparison_inputs.json").read_text())
    assert inputs["factor_ids"] == native["factor_ids"]
    for name, segment in comparison["segments"].items():
        assert segment["status"] == "completed"
        assert segment["sessions"] == inputs["segments"][name]["sessions"]
        accounts = [value["strategy"] for value in segment["models"].values()]
        accounts += [value["strategy"] for value in segment["baselines"].values()]
        assert all(account["account_rules"] == accounts[0]["account_rules"] for account in accounts)
        assert all(account["buy_and_hold"] == segment["buy_and_hold"] for account in accounts)
        assert all(account["period"] == accounts[0]["period"] for account in accounts)
    view = next(e["view"] for e in read_state(paired)["events"] if e["stage"] == "model_comparison")
    assert view["kind"] == "model_comparison" and len(view["rows"]) == 10
    for row in view["rows"]:
        assert row["daily_return_count"] > 0
        if row["model"] == "buy and hold":
            assert row["sample_count"] is None and row["mae"] is None
        else:
            assert row["sample_count"] == comparison["segments"][row["window"]]["sample_count"]
    assert "Same-input model comparison" in (paired / "report.html").read_text()


def test_alternate_fit_reads_only_training_rows_and_keeps_missing_terminal(tmp_path, monkeypatch):
    captured, fitted = {}, []
    original = study_model_comparison.compare_models
    initialize = StandardizedRidge.__init__

    def counted(self, x, labels, alpha):
        fitted.append((x.copy(), labels.copy(), alpha))
        initialize(self, x, labels, alpha)

    def incomplete(bundle, data, selected, terminal, *, policy):
        bundle = deepcopy(bundle)
        captured.update(bundle)
        bundle["segments"]["terminal"]["features"][0][0] = None
        bundle["segments"]["terminal"]["available"] = False
        return original(bundle, data, selected, terminal, policy=policy)

    monkeypatch.setattr(StandardizedRidge, "__init__", counted)
    monkeypatch.setattr(study_model_comparison, "compare_models", incomplete)
    assert run(tmp_path / "study", "lightgbm", "--compare-models") == 0
    assert len(fitted) == 1
    x, labels, alpha = fitted[0]
    np.testing.assert_array_equal(x, captured["segments"]["train"]["features"])
    np.testing.assert_array_equal(labels, captured["segments"]["train"]["labels"])
    assert alpha == 1
    assert "labels" not in captured["segments"]["terminal"]
    result = json.loads((tmp_path / "study/model_comparison.json").read_text())
    assert result["segments"]["terminal"]["status"] == "unavailable"
    assert result["segments"]["development"]["status"] == "completed"


def test_resume_reuses_completed_comparison_fit(tmp_path, monkeypatch):
    root = tmp_path / "study"
    evaluate = study_walk_forward.evaluate_windows
    monkeypatch.setattr(
        study_walk_forward,
        "evaluate_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("disk interrupted")),
    )
    assert run(root, "lightgbm", "--compare-models", "--walk-forward-folds", "2") == 1
    receipt = (root / "checkpoints/model_comparison.json").read_bytes()
    monkeypatch.setattr(study_walk_forward, "evaluate_windows", evaluate)

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Completed alternate model must not be refitted")

    monkeypatch.setattr(study_model_comparison, "compare_models", unexpected_fit)
    assert main(["research-resume", str(root)]) == 0
    assert (root / "checkpoints/model_comparison.json").read_bytes() == receipt
    result = json.loads((root / "result.json").read_text())
    assert result["runtime"]["counts"]["comparison_model_stages_completed"] == 0


def test_workbench_validates_and_passes_the_comparison_option(tmp_path):
    options = study_options({"compare_models": True, "llm": "simulated", "data": "synthetic"})
    assert "--compare-models" in study_arguments(tmp_path / "task", tmp_path, "KO", options)
    assert study_options({})["compare_models"] is False
    with pytest.raises(ValueError, match="compare_models must be boolean"):
        study_options({"compare_models": "yes"})


def test_enabling_comparison_runs_once_even_with_unchanged_evidence(tmp_path):
    workbench = Workbench(tmp_path)
    options = OFFLINE | {"only_new_evidence": True}
    workbench.batch(["KO"], options)
    finish_queue(workbench)
    paired = workbench.batch(["KO"], options | {"compare_models": True})
    finish_queue(workbench)
    task = workbench.task(paired["tasks"][-1]["id"])
    assert workbench.item(task)["status"] == "completed"
    assert (task / "study/model_comparison.json").is_file()
    repeated = workbench.batch(["KO"], options | {"compare_models": True})
    finish_queue(workbench)
    assert workbench.item(workbench.task(repeated["tasks"][-1]["id"]))["status"] == "skipped"
