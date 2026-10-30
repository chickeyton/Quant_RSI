"""Connected study tests use real features/fits/accounts with offline LLM/news providers."""

import io
import json
from pathlib import Path

import pytest

from quant_rsi.cli import main
from quant_rsi.research import llm, study
from quant_rsi.research.demo import load_catalog
from quant_rsi.research.demo_proposals import FactorProposal, extend_library

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def command(root, *extra):
    return [
        "research-study",
        "KO",
        "--catalog",
        str(CATALOG),
        "--prices",
        "synthetic",
        "--related",
        "simulated",
        "--news",
        "simulated",
        "--output",
        str(root),
        *extra,
    ]


def test_offline_entry_saves_report_and_complete_account_loop(tmp_path):
    root = tmp_path / "study"
    assert main(command(root, "--llm", "simulated")) == 0
    result = json.loads((root / "result.json").read_text())
    assert result["configuration"]["model"] == "lightgbm"
    assert result["summary"]["factors_changed"]
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    for key in ("first_prediction", "second_prediction"):
        prediction = steps[key]
        assert prediction["training_feature_columns"] == 2
        assert prediction["news_feature_support"]["runnable"]
        assert {row["value"] for row in prediction["news_feature_inputs"]} == {0.0, 1.0}
        assert prediction["development"]["metrics"]["sample_count"] == 20
        assert prediction["strategy"]["strategy"]["valuations"]
        assert prediction["strategy"]["buy_and_hold"]["fills"]
    assert "research.momentum_10" in result["study"]["factor_changes"]["deactivated"]
    assert "research.qlib_ma_v1_10" in result["study"]["factor_changes"]["activated"]
    assert len((root / "trace.jsonl").read_text().splitlines()) == len(result["trace"])
    assert {
        "report.html",
        "report.md",
        "result.json",
        "config.json",
        "inputs.json",
        "news_history.json",
        "trace.jsonl",
    } <= {path.name for path in root.iterdir()}
    html = (root / "report.html").read_text()
    assert "<svg" in html and "Price MAE" in html and "No independent holdout" in html
    for report in (html, (root / "report.md").read_text()):
        assert "Sharpe (ann.)" in report and "Volatility (ann.)" in report
        assert "Daily N" in report and "0% risk-free rate" in report
        assert "252 sessions/year" in report and "ddof=1" in report
    assert "Annualized Sharpe:" in result["study"]["conclusion"]
    for key in ("first_prediction", "second_prediction"):
        accounts = steps[key]["strategy"]
        for name in ("strategy", "buy_and_hold"):
            metrics = accounts[name]["metrics"]
            assert metrics["risk_statistics"]["daily_return_count"] == 24
            assert metrics["annualized_volatility"] >= 0
            value = metrics["sharpe_ratio"]
            assert (f"{value:.4f}" if value is not None else "N/A") in html
    assert (
        steps["first_prediction"]["strategy"]["buy_and_hold"]
        == steps["second_prediction"]["strategy"]["buy_and_hold"]
    )
    before = (root / "result.json").read_bytes()
    assert main(command(root, "--llm", "simulated")) == 1
    assert (root / "result.json").read_bytes() == before


@pytest.mark.parametrize("use_news", [True, False])
def test_codex_proposals_execute_and_feedback_includes_predictions_and_strategy(
    tmp_path,
    monkeypatch,
    use_news,
):
    calls = []

    def codex(self, prompt, schema):
        data = json.loads(prompt.split("\nINPUT_JSON:\n")[1])
        calls.append(data)
        assert "proposed_factors" in schema["required"]
        revision = data["feedback"] is not None
        if revision:
            assert "strategy_diagnostics" in data["feedback"]
            for name in ("strategy", "buy_and_hold"):
                metrics = data["feedback"]["strategy_diagnostics"][name]
                assert "sharpe_ratio" in metrics and "annualized_volatility" in metrics
                assert metrics["risk_statistics"]["daily_return_count"] == 24
                assert metrics["risk_statistics"]["annual_risk_free_rate"] == 0
            assert data["feedback"]["development"]["metrics"]["sample_count"] == 20
            assert "rows" not in data["feedback"]["development"]
            assert "research.momentum_7" in {x["factor_id"] for x in data["catalog"]["definitions"]}
        op, window = ("qlib_roc_v1", 13) if revision else ("momentum", 7)
        return json.dumps(
            {
                "hypothesis": "<script>alert('untrusted')</script> test supplied news hypothesis",
                "factor_ids": [f"research.{op}_{window}"] + (["news.events_7"] if use_news else []),
                "evidence_news_ids": [data["context"]["news"]["items"][0]["id"]],
                "event_signal": 0.2,
                "proposed_factors": [{"op": op, "window": window, "description": "Bounded test"}],
                "proposed_news_factors": [
                    {
                        "name": "events",
                        "keywords": ["earnings"],
                        "window_days": 7,
                        "description": "Headline count",
                    }
                ],
            }
        )

    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", codex)
    root = tmp_path / "connected"
    assert main(command(root, "--llm", "codex")) == 0
    assert len(calls) == 2 and calls[0]["context"]["news"] == calls[1]["context"]["news"]
    result = json.loads((root / "result.json").read_text())
    assert result["study"]["factor_changes"]["activated"] == ["research.qlib_roc_v1_13"]
    assert result["study"]["news_factors"]["support"]["runnable"]
    assert result["study"]["news_factors"]["executed_initial"] == (
        ["news.events_7"] if use_news else []
    )
    if not use_news:
        assert "no news column was fitted" in result["study"]["news_use"]
        assert "pending" not in result["study"]["news_use"]
    html = (root / "report.html").read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_failed_revision_keeps_first_fit_and_feedback_without_success_report(tmp_path, monkeypatch):
    def hypothesis(context, library, *, previous=None, feedback=None):
        if previous:
            raise ValueError("Controlled second-provider failure")
        return {
            "provider": "controlled_test",
            "hypothesis": "initial test",
            "factor_ids": ["research.momentum_9"],
            "proposed_factors": [{"op": "momentum", "window": 9, "description": "Test"}],
            "evidence_news_ids": [context["news"]["items"][0]["id"]],
            "event_signal": 0.0,
        }

    monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    root = tmp_path / "failed"
    assert main(command(root, "--llm", "simulated")) == 1
    failure = json.loads((root / "failure.json").read_text())
    assert failure["last_completed_stage"] == "feedback"
    stages = [json.loads(line)["stage"] for line in (root / "trace.jsonl").read_text().splitlines()]
    assert "first_prediction" in stages
    assert not (root / "result.json").exists() and not (root / "report.html").exists()


@pytest.mark.parametrize(
    "op, window", [("eval", 5), ("momentum", 0), ("mean_return", 31), ("qlib_roc_v1", True)]
)
def test_generated_recipe_cannot_escape_operator_or_window_bounds(op, window):
    with pytest.raises(ValueError):
        FactorProposal(op=op, window=window, description="Invalid test")


def test_proposals_are_run_local_must_be_selected_and_cannot_redefine_existing_recipe():
    library = load_catalog(CATALOG)
    proposal = {"op": "momentum", "window": 7, "description": "Test"}
    value = {"factor_ids": ["research.momentum_7"], "proposed_factors": [proposal]}
    extended = extend_library(library, value)
    assert len(extended["definitions"]) == len(library["definitions"]) + 1
    with pytest.raises(ValueError, match="Select at least"):
        extend_library(library, value | {"factor_ids": ["equity.price_momentum_5"]})
    extended["definitions"][-1]["recipe"]["window"] = 8
    with pytest.raises(ValueError, match="conflicts"):
        extend_library(extended, value)


def test_api_study_uses_strict_proposal_schema_and_keeps_key_out_of_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "study-api-test-secret")
    calls = []

    def urlopen(request, *, timeout):
        assert request.get_header("Authorization") == "Bearer study-api-test-secret"
        body = json.loads(request.data)
        schema = body["text"]["format"]["schema"]
        assert body["store"] is False and body["model"] == "test-model"
        assert "proposed_factors" in schema["required"]
        assert set(schema["$defs"]["FactorProposal"]["required"]) == {"op", "window", "description"}
        calls.append(body)
        window = 9 if len(calls) == 1 else 15
        reply = {
            "hypothesis": "Test API research",
            "factor_ids": [f"research.momentum_{window}", "news.api_events_7"],
            "evidence_news_ids": ["demo-demand"],
            "event_signal": 0.0,
            "proposed_factors": [
                {"op": "momentum", "window": window, "description": "Offline transport test"}
            ],
            "proposed_news_factors": [
                {
                    "name": "api_events",
                    "keywords": ["demand"],
                    "window_days": 7,
                    "description": "API headline count",
                }
            ],
        }
        return io.BytesIO(
            json.dumps(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": json.dumps(reply)}],
                        }
                    ],
                }
            ).encode()
        )

    monkeypatch.setattr(llm, "urlopen", urlopen)
    root = tmp_path / "api"
    assert main(command(root, "--llm", "api", "--llm-model", "test-model")) == 0
    assert len(calls) == 2
    assert all("study-api-test-secret" not in path.read_text() for path in root.iterdir())
