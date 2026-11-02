"""Offline transport and wiring tests; live LLM calls are explicit manual checks."""

import io
import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from quant_rsi.cli import main
from quant_rsi.research import demo, llm

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def reply(revised=False):
    return {
        "hypothesis": "Weaker demand may reduce returns; revise after the positive error."
        if revised
        else "Explore momentum against the supplied demand news.",
        "factor_ids": ["equity.price_mean_return_5", "equity.announced_events"]
        if revised
        else ["equity.price_momentum_5"],
        "evidence_news_ids": ["demo-demand"],
        "event_signal": -0.5,
    }


def inputs(prompt):
    return json.loads(prompt.split("\nINPUT_JSON:\n", 1)[1])


def test_codex_cli_runs_both_hypotheses_with_feedback_and_visible_mixed_mode(monkeypatch, capsys):
    calls = []

    def run(command, **kwargs):
        data = inputs(kwargs["input"])
        calls.append(data)
        assert command[-1] == "-"
        assert command[command.index("--model") + 1] == "test-model"
        assert "--ephemeral" in command and "--ignore-user-config" in command
        assert command[command.index("--sandbox") + 1] == "read-only"
        assert kwargs["timeout"] == 42
        schema = json.loads(Path(command[command.index("--output-schema") + 1]).read_text())
        assert schema["properties"]["factor_ids"]["items"]["enum"] == [
            item["factor_id"] for item in data["catalog"]["definitions"]
        ]
        Path(command[command.index("--output-last-message") + 1]).write_text(
            json.dumps(reply(data["feedback"] is not None))
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(llm, "run_provider", run)
    assert (
        main(
            [
                "research-demo",
                "KO",
                "--catalog",
                str(CATALOG),
                "--llm",
                "codex",
                "--llm-model",
                "test-model",
                "--llm-timeout",
                "42",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    stages = {item["stage"]: item["output"] for item in result["trace"]}
    assert len(calls) == 2
    assert calls[0]["context"]["news"] == stages["news"]
    assert calls[1]["previous"] == stages["initial_hypothesis"]
    assert calls[1]["feedback"] == stages["feedback"]
    assert stages["revised_hypothesis"]["feedback_used"]["error"] == 4
    assert stages["second_prediction"]["predicted_price"] == 100
    assert result["mode"] == "mixed" and "real LLM" in result["notice"]
    assert result["summary"]["factors_changed"]
    assert result["summary"]["improvement"] == "not_evaluated"
    assert "MIXED / SIMULATION" in demo.render_demo(result)


def test_api_key_transport_completes_same_loop_without_exposing_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    calls = []

    def urlopen(request, *, timeout):
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert request.get_header("Authorization") == "Bearer test-secret"
        assert timeout == 180
        body = json.loads(request.data)
        calls.append(inputs(body["input"]))
        assert body["store"] is False and body["model"] == "test-model"
        assert body["text"]["format"]["strict"] is True
        return io.BytesIO(
            json.dumps(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(reply(len(calls) == 2)),
                                }
                            ],
                        }
                    ],
                }
            ).encode()
        )

    monkeypatch.setattr(llm, "urlopen", urlopen)
    result = demo.run_demo(
        "KO",
        catalog=lambda: demo.load_catalog(CATALOG),
        hypothesize=llm.LLMHypothesizer("api", "test-model"),
    )
    assert len(calls) == 2 and calls[1]["feedback"]["error"] == 4
    assert result["mode"] == "mixed"
    assert "test-secret" not in json.dumps(result)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"factor_ids": ["invented"]}, "unknown factors"),
        ({"evidence_news_ids": ["invented"]}, "outside the supplied context"),
        ({"event_signal": float("nan")}, "invalid hypothesis"),
        ({"factor_ids": ["equity.price_momentum_5"] * 2}, "duplicate factor"),
    ],
)
def test_untrusted_hypothesis_is_rejected(monkeypatch, change, message):
    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", lambda *_: json.dumps(reply() | change))
    with pytest.raises(ValueError, match=message):
        demo.run_demo(
            "KO",
            catalog=lambda: demo.load_catalog(CATALOG),
            hypothesize=llm.LLMHypothesizer("codex"),
        )


def test_unchanged_factor_set_stops_instead_of_claiming_revision(monkeypatch):
    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", lambda *_: json.dumps(reply()))
    with pytest.raises(ValueError, match="did not revise"):
        demo.run_demo(
            "KO",
            catalog=lambda: demo.load_catalog(CATALOG),
            hypothesize=llm.LLMHypothesizer("codex"),
        )


@pytest.mark.parametrize(
    "failure, message",
    [
        (FileNotFoundError(), "CLI not found"),
        (subprocess.TimeoutExpired("codex", 1), "timed out"),
    ],
)
def test_codex_failures_stop_cli_without_simulated_fallback(monkeypatch, capsys, failure, message):
    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(llm, "run_provider", run)
    assert main(["research-demo", "--catalog", str(CATALOG), "--llm", "codex"]) == 1
    captured = capsys.readouterr()
    assert message in captured.err and not captured.out


def test_api_configuration_and_http_errors_are_actionable_without_leaking_key(monkeypatch):
    with pytest.raises(ValueError, match="requires --llm-model"):
        llm.LLMHypothesizer("api")
    with pytest.raises(ValueError, match="positive finite"):
        llm.LLMHypothesizer("codex", timeout=float("nan"))
    provider = llm.LLMHypothesizer("api", "test-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires OPENAI_API_KEY"):
        provider._api("test", {})
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")

    def urlopen(*args, **kwargs):
        raise HTTPError("https://api.openai.com", 401, "test-secret", {}, None)

    monkeypatch.setattr(llm, "urlopen", urlopen)
    with pytest.raises(ValueError, match=r"^OpenAI API request failed \(HTTP 401\)$"):
        provider._api("test", {})


def test_deepseek_transport_uses_json_mode_without_exposing_the_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    seen = {}

    def urlopen(request, *, timeout):
        assert request.full_url == "https://api.deepseek.com/chat/completions"
        assert request.get_header("Authorization") == "Bearer deepseek-secret"
        assert timeout == 12
        body = json.loads(request.data)
        seen.update(body)
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"content": '{"hypothesis": "x"}'}}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "prompt_cache_hit_tokens": 1,
                    },
                }
            ).encode()
        )

    usage = []
    monkeypatch.setattr(llm, "urlopen", urlopen)
    provider = llm.LLMHypothesizer("deepseek", timeout=12, on_usage=usage.append)
    assert provider._deepseek("Return JSON", {"type": "object"}) == '{"hypothesis": "x"}'
    assert seen["model"] == "deepseek-chat"
    assert seen["response_format"] == {"type": "json_object"}
    assert "JSON_SCHEMA:" in seen["messages"][0]["content"]
    assert usage == [
        {
            "mode": "deepseek",
            "input_tokens": 3,
            "cached_input_tokens": 1,
            "output_tokens": 4,
        }
    ]
    assert "deepseek-secret" not in json.dumps(usage)


def test_deepseek_and_cursor_do_not_invent_a_web_search():
    with pytest.raises(ValueError, match="codex or --llm api"):
        llm.LLMHypothesizer("deepseek").search("prompt", {})
    with pytest.raises(ValueError, match="codex or --llm api"):
        llm.LLMHypothesizer("cursor", handoff_dir=Path("/tmp")).search("prompt", {})


def test_cursor_handoff_reads_a_replaced_response(tmp_path):
    provider = llm.LLMHypothesizer("cursor", timeout=5, handoff_dir=tmp_path)

    def write():
        request = tmp_path / "round-1.request.json"
        while not request.is_file():
            threading.Event().wait(0.05)
        (tmp_path / "round-1.response.json").write_text('{"factor_ids": ["a"]}', encoding="utf-8")

    thread = threading.Thread(target=write)
    thread.start()
    assert provider._cursor("PROMPT", {"type": "object"}) == '{"factor_ids": ["a"]}'
    thread.join()
    saved = json.loads((tmp_path / "round-1.request.json").read_text(encoding="utf-8"))
    assert saved["prompt"] == "PROMPT" and saved["round"] == 1


def test_cursor_handoff_timeout_names_the_response_path(tmp_path):
    provider = llm.LLMHypothesizer("cursor", timeout=0.35, handoff_dir=tmp_path)
    with pytest.raises(ValueError, match="round-1.response.json"):
        provider._cursor("prompt", {})


def test_cli_rejects_incompatible_llm_choices(tmp_path, capsys):
    assert main(["research-demo", "--llm", "cursor"]) == 1
    assert "requires --cursor-dir" in capsys.readouterr().err
    assert main(["research-demo", "--llm", "simulated", "--cursor-dir", str(tmp_path)]) == 1
    assert "only used with --llm cursor" in capsys.readouterr().err
    assert (
        main(
            [
                "research-study",
                "KO",
                "--output",
                str(tmp_path / "study"),
                "--llm",
                "deepseek",
                "--news-research",
                "--no-ui",
            ]
        )
        == 1
    )
    assert "web search requires --llm codex or --llm api" in capsys.readouterr().err


def test_deepseek_missing_key_stops_the_demo(monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert main(["research-demo", "--catalog", str(CATALOG), "--llm", "deepseek"]) == 1
    error = capsys.readouterr().err
    assert "DEEPSEEK_API_KEY" in error and "deepseek-secret" not in error


def test_workbench_can_select_cursor_without_news_search(tmp_path):
    from quant_rsi.research.study_queue import study_arguments, study_options

    with pytest.raises(ValueError, match="web search"):
        study_options({"llm": "deepseek"})
    options = study_options({"llm": "cursor", "news_research": False, "data": "synthetic"})
    arguments = study_arguments(tmp_path / "task", tmp_path, "KO", options)
    assert arguments[arguments.index("--llm") + 1] == "cursor"
    assert arguments[arguments.index("--cursor-dir") + 1].endswith("task/study/cursor-llm")
