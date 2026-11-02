"""Event-recipe admission deadlines do not rewrite historical feature timestamps."""

import json
from copy import deepcopy

import pytest
from test_research_study import command
from test_study_factor_memory import memory

from quant_rsi.cli import main
from quant_rsi.market import market_contracts
from quant_rsi.research import llm, study
from quant_rsi.research.demo import select_factors, simulated_hypothesis
from quant_rsi.research.demo_proposals import extend_library
from quant_rsi.research.study_factor_memory import reuse_recipes
from quant_rsi.research.study_history import recent_factor_recipes

START = "2026-11-01T10:00:00+00:00"
END = "2026-11-02T10:00:00+00:00"


def library(as_of=START):
    return {"definitions": [], "recipe_as_of": as_of, "news_feature_support": {"runnable": True}}


def hypothesis(hours=24):
    return {
        "factor_ids": ["research.momentum_10", "news.release_7"],
        "evidence_news_ids": ["retained-release"],
        "proposed_factors": [{"op": "momentum", "window": 10, "description": "Price"}],
        "proposed_news_factors": [
            {
                "name": "release",
                "keywords": ["earnings"],
                "window_days": 7,
                "description": "Specific release hypothesis",
                "reuse_for_hours": hours,
            }
        ],
    }


def event():
    return extend_library(library(), hypothesis())["definitions"][-1]


def test_expiry_boundary_blocks_selection_and_reproposal_never_renews():
    original = event()
    assert market_contracts.timestamp(original["reuse_lifetime"]["expires_at"]) == (
        market_contracts.timestamp(END)
    )
    before, report = reuse_recipes(library("2026-11-02T09:59:59Z"), [memory(original)])
    assert report["entries"][0]["status"] == "restored_candidate"
    repeated = extend_library(before, hypothesis())
    assert repeated["definitions"][0]["reuse_lifetime"] == original["reuse_lifetime"]
    at, report = reuse_recipes(library(END), [memory(original)])
    assert report["entries"][0]["status"] == "expired_event_recipe"
    repeated = extend_library(at, hypothesis())
    with pytest.raises(ValueError, match="expired"):
        select_factors(repeated, hypothesis())
    with pytest.raises(ValueError, match="reuse lifetime"):
        extend_library(at, hypothesis(None))
    with pytest.raises(ValueError, match="reuse lifetime"):
        extend_library(at, hypothesis(48))
    # A catalog containing the same restored definition cannot bypass the clock check.
    before["recipe_as_of"] = END
    at, report = reuse_recipes(before, [memory(original)])
    assert report["entries"][0]["status"] == "expired_event_recipe"
    assert at["definitions"][0]["runnable"] is False
    assert original["runnable"] is True


@pytest.mark.parametrize("hours", [0, 721, True])
def test_lifetime_bounds(hours):
    with pytest.raises(ValueError):
        extend_library(library(), hypothesis(hours))


def test_unbound_and_corrupt_event_recipes_cannot_be_admitted():
    without_clock = library()
    without_clock.pop("recipe_as_of")
    with pytest.raises(ValueError, match="clock"):
        extend_library(without_clock, hypothesis())
    with pytest.raises(ValueError, match="evidence IDs"):
        extend_library(library(), hypothesis() | {"evidence_news_ids": []})
    for change in (
        {"expires_at": "2026-11-03T10:00:00Z"},
        {"evidence_news_ids": []},
        {"hours": True},
    ):
        old = event()
        old["reuse_lifetime"].update(change)
        current, report = reuse_recipes(library(), [memory(old)])
        assert current["definitions"] == []
        assert report["entries"][0]["status"] == "unsupported_or_invalid_recipe"
    current, report = reuse_recipes(library("2026-10-31T00:00:00Z"), [memory(event())])
    assert current["definitions"] == []
    assert report["entries"][0]["status"] == "unsupported_or_invalid_recipe"
    old = event()
    old["reuse_lifetime"] = "invalid"
    current, report = reuse_recipes(library(), [memory(old)])
    assert current["definitions"] == []
    assert report["entries"][0]["reuse_lifetime"] is None
    assert report["entries"][0]["status"] == "unsupported_or_invalid_recipe"


def test_reusable_and_legacy_news_recipes_are_not_given_invented_deadlines():
    reusable = extend_library(library(), hypothesis(None))["definitions"][-1]
    assert "reuse_lifetime" not in reusable
    current, report = reuse_recipes(library("2027-01-01T00:00:00Z"), [memory(reusable)])
    assert current["definitions"][0]["runnable"] is True
    assert "reuse_lifetime" not in current["definitions"][0]
    assert report["entries"][0]["status"] == "restored_candidate"
    reusable["reuse_lifetime"] = None
    current, report = reuse_recipes(library(), [memory(reusable)])
    assert "reuse_lifetime" not in current["definitions"][0]


def test_llm_schema_requires_explicit_lifetime_choice_and_binds_supplied_news(monkeypatch):
    def respond(self, prompt, schema):
        news_schema = schema["$defs"]["NewsFactorProposal"]
        assert "reuse_for_hours" in news_schema["required"]
        assert "default" not in news_schema["properties"]["reuse_for_hours"]
        assert "expired recipes cannot be selected or renewed" in prompt
        return json.dumps(
            hypothesis() | {"hypothesis": "Release-specific trial", "event_signal": 0.1}
        )

    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", respond)
    context = {
        "allow_factor_proposals": True,
        "news_feature_support": {"runnable": True},
        "news": {"items": [{"id": "retained-release"}]},
    }
    proposal = llm.LLMHypothesizer("codex")(context, library())
    assert proposal["proposed_news_factors"][0]["reuse_for_hours"] == 24
    definition = extend_library(library(), proposal)["definitions"][-1]
    assert definition["reuse_lifetime"]["evidence_news_ids"] == ["retained-release"]


def test_resume_preserves_admission_clock_and_completed_fit_past_deadline(tmp_path, monkeypatch):
    now, fail = START, True
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: now)

    def researcher(context, catalog, **kwargs):
        if fail and kwargs.get("previous") is not None:
            raise ValueError("Temporary provider failure")
        proposal = simulated_hypothesis(context, catalog, **kwargs)
        proposal["proposed_news_factors"] = deepcopy(hypothesis()["proposed_news_factors"])
        proposal["factor_ids"] = [
            key for key in proposal["factor_ids"] if not key.startswith("news.")
        ] + ["news.release_7"]
        return proposal

    monkeypatch.setattr(study, "simulated_hypothesis", researcher)
    root = tmp_path / "resume"
    assert main(command(root, "--llm", "simulated", "--model", "ridge", "--quiet", "--no-ui")) == 1
    fit = (root / "checkpoints/fit_1.json").read_bytes()
    clock = (root / "checkpoints/recipe_as_of.json").read_bytes()
    now, fail = END, False
    assert main(["research-resume", str(root)]) == 0
    assert (root / "checkpoints/recipe_as_of.json").read_bytes() == clock
    assert (root / "checkpoints/fit_1.json").read_bytes() == fit
    result = json.loads((root / "result.json").read_bytes())
    definitions = result["study"]["news_factors"]["definitions"]
    assert all(d["reuse_lifetime"]["created_at"] == START for d in definitions)
    assert all(
        market_contracts.timestamp(d["reuse_lifetime"]["expires_at"])
        == market_contracts.timestamp(END)
        for d in definitions
    )


def test_complete_studies_stop_using_expired_recipe_without_rewriting_old_result(
    tmp_path, monkeypatch
):
    history = tmp_path / "history"
    now = START
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: now)
    expired = False

    def researcher(context, catalog, **kwargs):
        proposal = simulated_hypothesis(context, catalog, **kwargs)
        if not expired:
            proposal["proposed_news_factors"] = deepcopy(hypothesis()["proposed_news_factors"])
            proposal["factor_ids"] = [
                key for key in proposal["factor_ids"] if not key.startswith("news.")
            ] + ["news.release_7"]
        else:
            old = next(d for d in catalog["definitions"] if d["factor_id"] == "news.release_7")
            assert old["runnable"] is False
            assert old["availability_reason"] == "event_recipe_expired"
        return proposal

    monkeypatch.setattr(study, "simulated_hypothesis", researcher)
    options = [
        "--llm",
        "simulated",
        "--model",
        "ridge",
        "--no-ui",
        "--quiet",
        "--history-dir",
        str(history),
    ]
    first = tmp_path / "first"
    assert main(command(first, *options)) == 0
    original = (first / "result.json").read_bytes()
    first_result = json.loads(original)
    assert all(
        "news.release_7" in a["prediction"]["feature_values"]
        for a in first_result["iterations"]["attempts"]
    )
    binding = next(
        d["reuse_lifetime"]
        for d in recent_factor_recipes(history, "KO")[0]["definitions"]
        if d["factor_id"] == "news.release_7"
    )
    now, expired = END, True
    second = tmp_path / "second"
    assert main(command(second, *options)) == 0
    result = json.loads((second / "result.json").read_bytes())
    assert all(
        "news.release_7" not in a["prediction"]["feature_values"]
        for a in result["iterations"]["attempts"]
    )
    entry = next(
        e for e in result["study"]["factor_memory"]["entries"] if e["factor_id"] == "news.release_7"
    )
    assert entry["status"] == "expired_event_recipe" and not entry["selected_in_this_study"]
    assert entry["reuse_lifetime"] == binding
    assert (first / "result.json").read_bytes() == original
    assert "expired event recipe" in (second / "report.html").read_text()
    assert "Reuse until" in (first / "report.html").read_text()
