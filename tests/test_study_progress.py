"""Deterministic progress and usage checks; no live providers or wall-clock sleeps."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_rsi.research import demo, llm, study_progress
from quant_rsi.research.study_progress import StudyProgress


def test_stage_progress_is_visible_before_completion_and_stdout_stays_clean(
    tmp_path, monkeypatch, capsys
):
    now = [10.0]
    monkeypatch.setattr(study_progress.time, "monotonic", lambda: now[0])
    progress = StudyProgress(tmp_path, "codex")
    progress.start("initial_hypothesis")
    current = json.loads((tmp_path / "runtime.json").read_text())
    assert current["current_stage"] == "initial_hypothesis"
    assert current["counts"]["llm_requests_started"] == 1
    assert current["counts"]["llm_requests_completed"] == 0
    assert current["token_usage"]["input_tokens"] is None
    assert "running initial_hypothesis" in capsys.readouterr().err
    now[0] = 12.5
    progress.usage(
        {"mode": "codex", "input_tokens": 20, "cached_input_tokens": 10, "output_tokens": 3}
    )
    progress.finish("initial_hypothesis")
    now[0] = 13.0
    result = progress.complete()
    assert result["stages"] == [
        {"stage": "initial_hypothesis", "status": "completed", "duration_seconds": 2.5}
    ]
    assert result["total_elapsed_seconds"] == 3
    assert result["counts"]["llm_requests_completed"] == 1
    assert result["token_usage"]["status"] == "complete"
    assert result["token_usage"]["input_tokens"] == 20
    assert not capsys.readouterr().out


def test_failed_stage_retains_elapsed_attempt_and_unknown_usage(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(study_progress.time, "monotonic", lambda: now[0])
    progress = StudyProgress(tmp_path, "api", quiet=True)
    progress.start("news_search_1")
    progress.usage({"mode": "api", "input_tokens": 8, "cached_input_tokens": 0, "output_tokens": 2})
    progress.finish("news_search_1")
    progress.start("news_review_1")
    now[0] = 4
    result = progress.fail()
    assert result["status"] == "failed" and result["current_stage"] is None
    assert result["stages"][-1]["duration_seconds"] == 4
    assert result["counts"]["llm_requests_started"] == 2
    assert result["counts"]["llm_requests_completed"] == 1
    assert result["counts"]["search_requests_completed"] == 1
    assert result["token_usage"]["status"] == "partial"
    assert json.loads((tmp_path / "runtime.json").read_text()) == result


def test_simulation_counts_stages_but_never_claims_llm_usage(tmp_path, capsys):
    progress = StudyProgress(tmp_path, "simulated", quiet=True)
    for name in ("news_search_1", "news_review_1", "initial_hypothesis", "first_prediction"):
        progress.start(name)
        progress.finish(name)
    result = progress.complete()
    assert result["counts"]["search_requests_completed"] == 1
    assert result["counts"]["model_stages_completed"] == 1
    assert result["counts"]["llm_requests_started"] == 0
    assert result["token_usage"]["status"] == "unavailable"
    assert result["token_usage"]["input_tokens"] is None
    assert not capsys.readouterr().err


def test_one_reported_token_field_is_partial_not_unavailable(tmp_path):
    progress = StudyProgress(tmp_path, "api", quiet=True)
    progress.start("initial_hypothesis")
    progress.usage({"mode": "api", "input_tokens": 50})
    progress.finish("initial_hypothesis")
    usage = progress.complete()["token_usage"]
    assert usage["status"] == "partial"
    assert usage["input_tokens"] == 50
    assert usage["output_tokens"] is None


def test_demo_start_hook_precedes_provider_calls_and_preserves_the_trace():
    order = []

    def observe(context, prediction):
        assert order[-1] == ("start", "feedback")
        return demo.simulated_observe(context, prediction)

    result = demo.run_demo(
        "KO",
        observe=observe,
        on_start=lambda stage: order.append(("start", stage)),
        on_stage=lambda event: order.append(("complete", event["stage"])),
    )
    assert order == [
        pair
        for event in result["trace"]
        for pair in (("start", event["stage"]), ("complete", event["stage"]))
    ]
    assert result == demo.run_demo("KO")


def test_codex_plain_request_reports_tokens_without_enabling_search(monkeypatch):
    events = []

    def run(command, **kwargs):
        assert "--json" in command and 'web_search="disabled"' in command
        Path(command[command.index("--output-last-message") + 1]).write_text("{}")
        for event in (
            {"type": "item.completed", "item": {"type": "reasoning", "text": "private text"}},
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 100,
                    "cached_input_tokens": 80,
                    "output_tokens": 5,
                    "secret": "not retained",
                },
            },
        ):
            kwargs["stdout"].write(json.dumps(event) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(llm, "run_provider", run)
    assert llm.LLMHypothesizer("codex", on_usage=events.append).structured("prompt", {}) == "{}"
    assert events == [
        {"mode": "codex", "input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 5}
    ]


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {
            "input_tokens": True,
            "output_tokens": -2,
            "input_tokens_details": {"cached_tokens": "12"},
        },
    ],
)
def test_api_absent_or_malformed_usage_is_unknown_without_secret_fields(monkeypatch, usage):
    events = []
    monkeypatch.setenv("OPENAI_API_KEY", "offline-secret")
    payload = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
        "usage": usage,
    }
    monkeypatch.setattr(llm, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    assert (
        llm.LLMHypothesizer("api", "test-model", on_usage=events.append).structured("prompt", {})
        == "{}"
    )
    assert events == [
        {"mode": "api", "input_tokens": None, "cached_input_tokens": None, "output_tokens": None}
    ]


def test_api_incomplete_response_still_records_reported_usage(monkeypatch):
    events = []
    monkeypatch.setenv("OPENAI_API_KEY", "offline-secret")
    payload = {
        "status": "incomplete",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens": 2,
        },
    }
    monkeypatch.setattr(llm, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="did not complete"):
        llm.LLMHypothesizer("api", "test-model", on_usage=events.append).structured("prompt", {})
    assert events == [
        {"mode": "api", "input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2}
    ]
