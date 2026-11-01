"""Recipes survive studies; their past availability and outcomes do not become current truth."""

import json
from copy import deepcopy

from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study, study_history
from quant_rsi.research.demo import simulated_hypothesis
from quant_rsi.research.demo_proposals import FactorProposal, NewsFactorProposal
from quant_rsi.research.study_factor_memory import reuse_recipes, selected_definitions
from quant_rsi.research.study_history import recent_factor_recipes


def memory(*definitions, **changes):
    return {
        "study_id": "previous",
        "result_ref": "retained-result",
        "issued_at": "2026-01-01",
        "outcome_status": "pending",
        "input_mode": "daily",
        "definitions": list(definitions),
    } | changes


def price(window):
    return FactorProposal(
        op="momentum", window=window, description="Past research recipe"
    ).definition()


def test_reuse_revalidates_news_and_never_imports_old_flags_or_values():
    news = NewsFactorProposal(
        name="earnings",
        keywords=["earnings"],
        window_days=7,
        description="Past earnings hypothesis",
    ).definition(True)
    news["column"] = [999]
    library = {"definitions": [], "news_feature_support": {"runnable": False}}
    result, report = reuse_recipes(library, [memory(price(23), news)])
    assert library["definitions"] == []
    definitions = {d["factor_id"]: d for d in result["definitions"]}
    assert definitions["research.momentum_23"]["recipe"] == {"op": "momentum", "window": 23}
    assert definitions["news.earnings_7"]["runnable"] is False
    assert "column" not in definitions["news.earnings_7"]
    assert [e["status"] for e in report["entries"]] == [
        "restored_candidate",
        "unavailable_current_inputs",
    ]
    assert report["entries"][0]["source_outcome_status"] == "pending"


def test_current_catalog_wins_conflicts_and_unsupported_bound_inputs_stay_unavailable():
    current, old = price(23), price(23)
    old["recipe"]["window"] = 24
    unsupported = {
        "factor_id": "financial.revenue_yoy",
        "kind": "numeric",
        "recipe": {"op": "financial", "metric": "revenue_yoy"},
    }
    invalid = price(23) | {"factor_id": "research.momentum_100"}
    invalid["recipe"] = {"op": "momentum", "window": 100}
    library, report = reuse_recipes({"definitions": [current]}, [memory(old, unsupported, invalid)])
    assert library["definitions"] == [current]
    assert [e["status"] for e in report["entries"]] == [
        "definition_conflict",
        "requires_current_binding",
        "unsupported_or_invalid_recipe",
    ]


def test_latest_selected_recipe_wins_without_using_a_historical_score():
    old = NewsFactorProposal(
        name="business",
        keywords=["earnings"],
        window_days=7,
        description="Older definition",
    ).definition(True)
    new = NewsFactorProposal(
        name="business",
        keywords=["demand"],
        window_days=7,
        description="Latest selected definition",
    ).definition(False)
    library, report = reuse_recipes(
        {"definitions": [], "news_feature_support": {"runnable": True}},
        [memory(new, study_id="newer"), memory(old, study_id="older")],
    )
    assert len(library["definitions"]) == 1
    assert library["definitions"][0]["recipe"]["keywords"] == ["demand"]
    assert library["definitions"][0]["runnable"] is True
    assert len(report["entries"]) == 1
    assert report["entries"][0]["source_study_id"] == "newer"


def test_extracts_only_selected_factors_even_when_final_trial_is_rejected():
    result = {
        "iterations": {
            "selected_round": 1,
            "attempts": [
                {"prediction": {"factor_ids": ["research.momentum_23"]}},
                {"prediction": {"factor_ids": ["research.momentum_11"]}},
            ],
        },
        "trace": [
            {"stage": "initial_factor_library", "output": {"definitions": [price(23)]}},
            {"stage": "revised_factor_library", "output": {"definitions": [price(23), price(11)]}},
        ],
    }
    assert selected_definitions(result) == [price(23)]


def test_next_study_selects_and_fits_a_remembered_recipe(tmp_path, monkeypatch):
    history = tmp_path / "history"
    first, second = tmp_path / "first", tmp_path / "second"
    phase, contexts = 1, []

    def researcher(context, library, **kwargs):
        proposal = simulated_hypothesis(context, library, **kwargs)
        revised = kwargs.get("feedback") is not None
        if phase == 1:
            windows = [23, 11] if revised else [23]
            proposal["proposed_factors"] = [
                {"op": "momentum", "window": w, "description": "A repeatable research hypothesis"}
                for w in windows
            ]
            proposal["factor_ids"] = [f"research.momentum_{w}" for w in windows]
        else:
            contexts.append(deepcopy(context))
            restored = next(
                d for d in library["definitions"] if d["factor_id"] == "research.momentum_23"
            )
            assert restored["recipe"] == {"op": "momentum", "window": 23}
            window = 19 if revised else 17
            proposal["proposed_factors"] = [
                {
                    "op": "momentum",
                    "window": window,
                    "description": "New proposal alongside the remembered factor",
                }
            ]
            proposal["factor_ids"] = ["research.momentum_23", f"research.momentum_{window}"]
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
    assert main(command(first, *options)) == 0
    original = (first / "result.json").read_bytes()
    phase = 2
    assert main(command(second, *options)) == 0
    assert (first / "result.json").read_bytes() == original
    result = json.loads((second / "result.json").read_bytes())
    assert all(
        "research.momentum_23" in a["prediction"]["feature_values"]
        for a in result["iterations"]["attempts"]
    )
    remembered = next(
        e
        for e in result["study"]["factor_memory"]["entries"]
        if e["factor_id"] == "research.momentum_23"
    )
    assert remembered["status"] == "restored_candidate" and remembered["selected_in_this_study"]
    assert remembered["source_input_mode"] == "synthetic"
    assert all(c["prior_outcomes"]["entries"] == [] for c in contexts)
    assert "Recipes from previous studies" in (second / "report.html").read_text()

    # The ledger retains the result itself: removing the mutable workspace cannot change memory.
    (first / "result.json").unlink()
    entries = recent_factor_recipes(history, "KO")
    assert any(d["factor_id"] == "research.momentum_23" for e in entries for d in e["definitions"])
    assert recent_factor_recipes(history, "AAPL") == []

    # Older journals need no migration or rewriting to recover their existing frozen catalog.
    legacy = json.loads(original)
    legacy_root = tmp_path / "legacy"
    with study_history._ledger(legacy_root) as (journal, store):
        ref = store.put_json(legacy)
        study_history._append(
            journal,
            store,
            "study_registered",
            {
                "study_id": "legacy-study",
                "symbol": "KO",
                "result_ref": ref,
                "issued_at": "2026-01-01",
                "status": "unavailable",
            },
        )
    restored = recent_factor_recipes(legacy_root, "KO")
    assert restored[0]["definitions"] == selected_definitions(legacy)
    assert restored[0]["input_mode"] == "synthetic"
