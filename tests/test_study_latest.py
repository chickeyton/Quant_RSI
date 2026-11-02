"""Publication refits use mature labels without rewriting research evidence."""

import json
from copy import deepcopy

import numpy as np
import pytest
from test_research_study_model import CATALOG, proposed
from test_study_history import command, prices, result_for

from quant_rsi.cli import main
from quant_rsi.research import demo, study_history, study_report
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_latest import publication_prediction, refit_latest
from quant_rsi.research.study_queue import study_arguments, study_options
from quant_rsi.research.study_ui import read_state


@pytest.mark.parametrize("estimator", ["ridge", "lightgbm"])
@pytest.mark.parametrize("horizon", [5, 20, 60])
def test_publication_uses_mature_labels_and_leaves_research_fits_frozen(estimator, horizon):
    model = LightGBMDemo(
        demo.load_catalog(CATALOG),
        evaluation_sessions=20,
        terminal_sessions=10,
        estimator=estimator,
        horizon_sessions=horizon,
    )
    selected = model.predict({}, [proposed("momentum", 10)], {})
    model.price_info = {"latest_target_session": "future-session"}
    original = deepcopy(selected)
    terminal = model.evaluate_terminal(1)
    predict, matrix, labels, ids = model._fits[0]
    before = predict(matrix[-1:]).copy()
    result = refit_latest(model, 1, selected)
    assert result["status"] == "completed" and result["extra_fits"] == 1
    assert result["factor_ids"] == ids
    assert result["training_sessions"] == list(model.data.dates[30:-horizon])
    forecast = result["latest_forecast"]
    assert forecast["training_information_cutoff"] == model.data.dates[-1]
    assert forecast["training_rows"] > selected["training_rows"]
    assert forecast["origin_session"] == model.data.dates[-1]
    assert forecast["target_session"] == "future-session"
    assert forecast["outcome"] == "not_observed"
    assert "development" not in result and "accuracy" not in result
    assert selected == original and len(model._fits) == 1
    np.testing.assert_array_equal(predict(matrix[-1:]), before)
    np.testing.assert_array_equal(model._fits[0][2], labels)
    assert terminal["refitted"] is False


@pytest.mark.parametrize("gap", ["training", "latest", "too_few"])
def test_missing_rows_are_disclosed_without_stale_forecast_fallback(gap):
    model = LightGBMDemo(demo.load_catalog(CATALOG), evaluation_sessions=20, estimator="ridge")
    selected = model.predict({}, [proposed("momentum", 10)], {})
    model.price_info = {"latest_target_session": "future-session"}
    matrix = model._fits[0][1]
    if gap == "training":
        matrix[40] = np.nan
    elif gap == "latest":
        matrix[-1] = np.nan
    else:
        matrix[30 : -model.horizon] = np.nan
    output = refit_latest(model, 1, selected)
    if gap == "training":
        assert output["status"] == "completed"
        assert output["excluded_training_sessions"] == [model.data.dates[40]]
        assert model.data.dates[40] not in output["training_sessions"]
    else:
        assert output["status"] == "unavailable" and output["extra_fits"] == 0
        assert "latest_forecast" not in publication_prediction({"study": {"latest_fit": output}})


@pytest.mark.parametrize("mode", ["refit", "selected"])
def test_cli_report_history_and_ui_publish_the_same_forecast(tmp_path, monkeypatch, mode):
    root, history = tmp_path / "study", tmp_path / "history"
    collection, _ = prices(tmp_path, monkeypatch, "2026-10-23")
    args = command(root, history, collection) + ["--compare-models"]
    if mode == "selected":
        args += ["--latest-fit", mode]
    assert main(args) == 0
    result = json.loads((root / "result.json").read_bytes())
    published = publication_prediction(result)["latest_forecast"]
    assert result["summary"]["latest_forecast"] == published
    trace = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    assert trace == result["trace"]
    assert next(e["output"] for e in trace if e["stage"] == "summary") == result["summary"]
    assert study_history.history_view(history)["entries"][0]["forecast"] == published
    assert study_report._number(published["predicted_price"]) in (root / "report.html").read_text()
    assert result["configuration"]["publication_fit_budget"] == int(mode == "refit")
    assert result["iterations"]["completed_fits"] == 2
    assert result["study"]["model_comparison"]["comparison_fits"] == 1
    if mode == "refit":
        state = read_state(root)
        event = next(e for e in state["events"] if e["stage"] == "latest_fit")
        assert event["view"]["facts"]["additional_fits"] == 1
        assert event["view"]["facts"]["training_information_cutoff"] == published["origin_session"]
        assert state["config"]["publication_fit_budget"] == 1
        chosen = result["iterations"]["selected_round"]
        research = result["iterations"]["attempts"][chosen - 1]["prediction"]
        assert (
            research["latest_forecast"]["training_information_cutoff"] < published["origin_session"]
        )
        assert result["terminal_evaluation"]["refitted"] is False


@pytest.mark.parametrize("failure_stage", ["summary", "report"])
def test_resume_after_publication_reuses_its_model(tmp_path, monkeypatch, failure_stage):
    import lightgbm as lgb

    train, calls = lgb.train, []
    owner, name = (
        (study_report, "render_study") if failure_stage == "report" else (LightGBMDemo, "summarize")
    )
    original = getattr(owner, name)

    def count(*args, **kwargs):
        calls.append(1)
        return train(*args, **kwargs)

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate post-publication failure")

    monkeypatch.setattr(lgb, "train", count)
    monkeypatch.setattr(owner, name, fail)
    collection, _ = prices(tmp_path, monkeypatch, "2026-10-23")
    root = tmp_path / "study"
    with pytest.raises(RuntimeError, match="deliberate post-publication"):
        main(command(root, tmp_path / "history", collection))
    assert len(calls) == 3
    frozen = (root / "checkpoints/latest_fit.json").read_bytes()
    monkeypatch.setattr(owner, name, original)
    assert main(["research-resume", str(root)]) == 0
    assert len(calls) == 3
    assert (root / "checkpoints/latest_fit.json").read_bytes() == frozen


def test_workbench_preserves_publication_mode(tmp_path):
    options = study_options({"latest_fit": "selected"})
    args = study_arguments(tmp_path / "task", tmp_path, "KO", options)
    assert args[args.index("--latest-fit") + 1] == "selected"
    assert study_options({})["latest_fit"] == "refit"
    with pytest.raises(ValueError, match="Unsupported"):
        study_options({"latest_fit": "unknown"})


def test_unavailable_publication_does_not_register_the_old_research_prediction(
    tmp_path, monkeypatch
):
    _, snapshot = prices(tmp_path, monkeypatch, "2026-10-23")
    result = result_for(snapshot)
    result["study"] = {"latest_fit": {"status": "unavailable"}}
    entry = study_history.register_study(tmp_path / "history", tmp_path / "run", result, snapshot)
    assert entry["status"] == "unavailable"
    assert entry["forecast"] is None and entry["publication"] is None
