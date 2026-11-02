"""A research focus survives the existing loop without changing its measured objective."""

import json

import pytest
from test_active_news import ResearchFixture, article
from test_research_study import command
from test_study_queue import OFFLINE, finish_queue

from quant_rsi.cli import main
from quant_rsi.research import active_news, study
from quant_rsi.research.demo import research_question
from quant_rsi.research.llm import LLMHypothesizer
from quant_rsi.research.study_history import history_view, mature_feedback, recent_factor_recipes
from quant_rsi.research.study_queue import study_arguments, study_options
from quant_rsi.research.study_ui import read_state
from quant_rsi.research.study_workbench import Workbench


@pytest.mark.parametrize("mode", ["codex", "api"])
@pytest.mark.parametrize("binding", ["research.momentum_5", "invented.input", "omitted"])
def test_question_map_requires_current_selected_bindings(monkeypatch, mode, binding):
    response = {
        "hypothesis": "Examine a price control while inventory measurements are unavailable.",
        "factor_ids": ["research.momentum_5"],
        "evidence_news_ids": ["source"],
        "event_signal": 0,
        "proposed_factors": [{"op": "momentum", "window": 5, "description": "Price control"}],
        "question_map": [
            {
                "claim": "Recent price direction may matter",
                "factor_ids": [binding],
                "rationale": "A trailing price control",
                "limitations": ["Does not measure inventories"],
            },
            {
                "claim": "Inventory builds could signal weaker demand",
                "factor_ids": [],
                "rationale": "No inventory measurement was supplied",
                "limitations": ["Missing industry inventory and demand measurements"],
            },
        ],
    }
    if binding == "omitted":
        response.pop("question_map")

    def transport(self, prompt, schema):
        assert schema["properties"]["question_map"]["minItems"] == 1
        assert "question_map" in schema["required"]
        return json.dumps(response)

    monkeypatch.setattr(LLMHypothesizer, "_" + mode, transport)
    hypothesize = LLMHypothesizer(mode, model="offline-fixture")
    context = {
        "allow_factor_proposals": True,
        "research_question": "Could inventories explain returns?",
        "news": {"items": [{"id": "source"}]},
    }
    if binding == "research.momentum_5":
        result = hypothesize(context, {"definitions": []})
        assert result["question_map"] == response["question_map"]
    else:
        with pytest.raises(ValueError, match="[Qq]uestion map"):
            hypothesize(context, {"definitions": []})


@pytest.mark.parametrize("invalid", [None, True, 3, [], "x" * 1001])
def test_invalid_question_is_rejected_at_both_input_boundaries(invalid):
    with pytest.raises(ValueError, match="Research question"):
        research_question(invalid)
    with pytest.raises(ValueError, match="Research question"):
        study_options({"question": invalid})


def test_question_reaches_news_both_hypotheses_report_and_scoped_history(tmp_path, monkeypatch):
    original = study.simulated_hypothesis
    contexts = []

    def hypothesize(context, library, **kwargs):
        contexts.append(context)
        return original(context, library, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", hypothesize)
    history = tmp_path / "history"
    question = "Could inventory <script>grow</script> faster than demand?"
    for number, focus in enumerate((question, "", question)):
        root = tmp_path / str(number)
        assert (
            main(
                command(
                    root,
                    "--llm",
                    "simulated",
                    "--news-research",
                    "--question",
                    "  " + focus + "  ",
                    "--history-dir",
                    str(history),
                )
            )
            == 0
        )
        result = json.loads((root / "result.json").read_text())
        assert result["configuration"]["question"] == focus
        assert result["configuration"]["selection_metric"] == "mae"
        assert result["configuration"]["fit_budget"] == 2
        assert all(c["research_question"] == focus for c in contexts[-2:])
        assert json.loads((root / "news_research.json").read_text())["question"] == focus
        assert read_state(root)["config"]["question"] == focus
        report = (root / "report.html").read_text()
        if focus:
            assert "Research question" in report and "Selected research hypothesis" in report
            assert "&lt;script&gt;grow&lt;/script&gt;" in report
            assert "<script>grow</script>" not in report
    assert len(contexts[2]["factor_memory"]["entries"]) == 0  # General study is a different focus.
    assert contexts[4]["factor_memory"]["entries"]  # Repeating the question restores its recipes.
    entries = history_view(history, "KO")["entries"]
    assert [e["question"] for e in entries] == [question, "", question]
    assert len(recent_factor_recipes(history, "KO", question=question)) == 2
    assert len(recent_factor_recipes(history, "KO")) == 1
    assert len(mature_feedback(history, "KO", question=question)["history"]["entries"]) == 2


def test_real_provider_search_and_review_prompts_preserve_question():
    fixture = ResearchFixture()
    question = "Does inventory growth reflect stronger demand?"
    result = active_news.research_news(
        "KO", {}, fixture, on_stage=lambda _: None, reader=article, question=question
    )
    assert len(fixture.search_inputs) == len(fixture.review_inputs) == 2
    assert all(
        d["research_question"] == question for d in fixture.search_inputs + fixture.review_inputs
    )
    assert result["question"] == question
    assert result["limits"]["search_requests"] == 2


def test_queue_template_question_changes_trigger_and_survives_restart(tmp_path):
    workbench = Workbench(tmp_path)
    question = "--inventory $literal and <tags>"
    plan = workbench.save_plan(
        {
            "name": "Inventory",
            "symbols": ["KO"],
            "interval_hours": 24,
            "enabled": False,
            "options": OFFLINE | {"question": question, "only_new_evidence": True},
        }
    )
    options = Workbench(tmp_path).plans()[0]["options"]
    assert options["question"] == question
    arguments = study_arguments(tmp_path / "task", tmp_path, "KO", study_options(options))
    assert "--question=" + question in arguments
    for focus, expected in (
        (question, "completed"),
        (question, "skipped"),
        ("Different question", "completed"),
    ):
        batch = workbench.batch(["KO"], options | {"question": focus})
        finish_queue(workbench)
        task = workbench.task(batch["tasks"][-1]["id"])
        assert workbench.item(task)["status"] == expected
        if expected == "completed":
            assert json.loads((task / "study/config.json").read_text())["question"] == focus
    assert any(row["question"] == question for row in Workbench(tmp_path).overview()["tasks"])
    assert (
        plan["options"]["question"] == question
    )  # Queuing another focus does not edit the template.


def test_overlong_cli_question_fails_before_creating_study(tmp_path):
    root = tmp_path / "invalid"
    assert main(command(root, "--question", "x" * 1001)) != 0
    assert not root.exists()


def test_resume_keeps_question_and_completed_first_fit(tmp_path, monkeypatch):
    original = study.simulated_hypothesis
    questions = []
    fail = True

    def hypothesize(context, library, **kwargs):
        questions.append(context["research_question"])
        if kwargs.get("previous") and fail:
            raise ValueError("Temporary provider outage")
        return original(context, library, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", hypothesize)
    root = tmp_path / "resume"
    question = "Do inventory changes explain demand risk?"
    assert main(command(root, "--llm", "simulated", "--question", question)) == 1
    checkpoint = (root / "checkpoints/fit_1.json").read_bytes()
    fail = False
    assert main(["research-resume", str(root)]) == 0
    assert questions == [question, question, question]
    assert (root / "checkpoints/fit_1.json").read_bytes() == checkpoint
    assert read_state(root)["config"]["question"] == question
