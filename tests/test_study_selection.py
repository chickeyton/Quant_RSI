"""Development adoption decisions and the selected forecast, not investment-quality claims."""

import json
from copy import deepcopy

import pytest

from quant_rsi.cli import main
from quant_rsi.research import study_report
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_selection import select_attempt


def attempt(mae, sharpe, factor):
    return {
        "factor_ids": [factor],
        "development": {"metrics": {"mae_price": mae}},
        "strategy": {"strategy": {"metrics": {"sharpe_ratio": sharpe}}},
    }


@pytest.mark.parametrize(
    "metric,first,second,selected",
    [
        ("mae", (1, 1), (2, 10), "initial"),
        ("mae", (2, 10), (1, 1), "revised"),
        ("mae", (1, 1), (1, 2), "initial"),
        ("sharpe", (1, 1), (2, 2), "revised"),
        ("sharpe", (2, 2), (1, 1), "initial"),
        ("sharpe", (2, 1), (1, 1), "initial"),
        ("sharpe", (2, None), (1, 1), "initial"),
        ("sharpe", (2, 1), (1, None), "initial"),
        ("sharpe", (2, float("nan")), (1, 1), "initial"),
        ("mae", (2, 1), (float("inf"), 2), "initial"),
    ],
)
def test_only_preselected_metric_decides_adoption(metric, first, second, selected):
    before, after = attempt(*first, "initial_factor"), attempt(*second, "revised_factor")
    snapshot = deepcopy(after)
    decision = select_attempt(before, after, metric)
    assert decision["selected_attempt"] == selected
    assert decision["revision_accepted"] == (selected == "revised")
    assert decision["selected_factor_ids"] == [selected + "_factor"]
    assert "not_independent_validation" in decision["evidence_scope"]
    assert after == snapshot
    json.dumps(decision, allow_nan=False)


def command(root, *extra):
    return [
        "research-study",
        "KO",
        "--output",
        str(root),
        "--llm",
        "simulated",
        "--related",
        "simulated",
        "--news",
        "simulated",
        "--prices",
        "synthetic",
        *extra,
    ]


@pytest.mark.parametrize("winner", ["initial", "revised", "initial_unavailable"])
def test_latest_forecast_and_report_follow_selected_attempt_without_refit(
    tmp_path, monkeypatch, winner
):
    predict = LightGBMDemo.predict
    calls = []

    def controlled(self, *args):
        result = predict(self, *args)
        index = len(calls)
        # Deliberate test scores isolate adoption from any stochastic model behavior.
        result["development"]["metrics"]["mae_price"] = (
            (2, 1)[index] if winner == "revised" else (1, 2)[index]
        )
        if winner == "initial_unavailable" and index == 0:
            result["latest_forecast_unavailable"] = {
                "reasons": ["controlled_missing_news"],
                "missing_factor_ids": ["external.eia_wti_5obs_return"],
            }
        else:
            result["latest_forecast"] = {
                "origin_session": "2026-10-28",
                "target_session": "2026-11-04",
                "reference_price": 100.0,
                "predicted_price": 111.1234 if index == 0 else 222.5678,
                "training_information_cutoff": "2026-09-01",
                "outcome": "not_observed",
            }
        calls.append(deepcopy(result))
        return result

    monkeypatch.setattr(LightGBMDemo, "predict", controlled)
    root = tmp_path / "study"
    assert main(command(root)) == 0
    assert len(calls) == 2
    result = json.loads((root / "result.json").read_text())
    selection = result["study"]["selection"]
    selected = "revised" if winner == "revised" else "initial"
    assert selection == result["summary"]["selection"]
    assert selection["selected_attempt"] == selected
    assert result["configuration"]["selection_metric"] == "mae"
    steps = {row["stage"]: row["output"] for row in result["trace"]}
    assert steps["summary"] == result["summary"]
    html = (root / "report.html").read_text()
    assert "Attempt selection" in html and f"selected {selected} fitted model" in html
    if winner == "initial_unavailable":
        assert "latest_forecast" not in result["summary"]
        assert "controlled_missing_news" in html
        assert "No latest forecast was published" in html
        assert "This forecast reuses" not in html
        assert "Unavailable selected factor" in html
        assert "external.eia_wti_5obs_return" in html
        assert "222.5678" not in html
    else:
        forecast = calls[1 if winner == "revised" else 0]["latest_forecast"]
        assert result["summary"]["latest_forecast"] == forecast
        assert str(forecast["predicted_price"]) in html
        assert (
            str(calls[0 if winner == "revised" else 1]["latest_forecast"]["predicted_price"])
            not in html
        )
    # Both attempted forecasts stay available in the original stage evidence.
    assert steps["second_prediction"]["latest_forecast"] == calls[1]["latest_forecast"]


@pytest.mark.parametrize("quiet", [False, True])
def test_json_stdout_stays_parseable_and_runtime_is_saved(tmp_path, capsys, quiet):
    root = tmp_path / "study"
    args = command(root, "--json", "--selection-metric", "sharpe")
    if quiet:
        args.append("--quiet")
    assert main(args) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert (not captured.err) if quiet else "first_prediction" in captured.err
    assert result["study"]["selection"]["metric"] == "sharpe"
    runtime = json.loads((root / "runtime.json").read_text())
    assert runtime == result["runtime"]
    assert runtime["status"] == "completed"
    assert runtime["counts"]["model_stages_completed"] == 2
    assert runtime["counts"]["llm_requests_started"] == 0
    assert runtime["token_usage"]["input_tokens"] is None
    assert all(stage["duration_seconds"] >= 0 for stage in runtime["stages"])


def test_report_failure_does_not_announce_a_completed_run(tmp_path, monkeypatch, capsys):
    def broken_report(result):
        raise ValueError("Controlled render failure")

    monkeypatch.setattr(study_report, "render_study", broken_report)
    root = tmp_path / "failed_report"
    assert main(command(root)) == 1
    captured = capsys.readouterr()
    assert "[quant-rsi] completed in " not in captured.err
    assert not (root / "report.html").exists()
    assert json.loads((root / "runtime.json").read_text())["status"] == "failed"
