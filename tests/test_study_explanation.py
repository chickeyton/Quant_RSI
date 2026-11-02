"""Explanations must distinguish cited context, actual fitting and unavailable inputs."""

from quant_rsi.research.study_explanation import (
    continuity,
    evidence_path,
    explain_prediction,
    question_coverage,
)
from quant_rsi.research.study_history import mature_feedback
from quant_rsi.research.study_ui_views import visual_output


def test_citing_news_does_not_claim_news_was_fitted_and_unavailable_reasons_survive():
    path = explain_prediction(
        {"factor_ids": ["price"], "news_feature_support": {"reasons": ["late_receipt"]}},
        {"hypothesis": "Demand may matter", "evidence_news_ids": ["article"]},
        [{"factor_id": "price", "kind": "numeric"}],
        {
            "definitions": [
                {"factor_id": "guidance", "kind": "news", "runnable": False},
                {"factor_id": "count", "kind": "news", "runnable": True},
                {
                    "factor_id": "inventory",
                    "kind": "numeric",
                    "runnable": False,
                    "reasons": ["insufficient_training_coverage"],
                },
            ]
        },
        {"items": [{"id": "article", "title": "Demand grew", "url": "https://issuer.test"}]},
    )
    assert path["news_columns_fitted"] == 0
    assert path["citations"][0]["role"] == "hypothesis_context"
    factors = {f["factor_id"]: f for f in path["factors"]}
    assert factors["price"]["status"] == "fitted"
    assert factors["guidance"]["status"] == "unavailable"
    assert factors["guidance"]["reason"] == "late_receipt"
    assert factors["count"]["status"] == "not_selected"
    assert factors["inventory"]["status"] == "unavailable"
    assert factors["inventory"]["reason"] == "insufficient_training_coverage"


def test_question_execution_is_derived_from_actual_fit_and_missing_inputs_stay_unmeasured():
    hypothesis = {
        "question_map": [
            {
                "claim": "A proposed association",
                "factor_ids": ids,
                "rationale": "Researcher's rationale",
                "limitations": ["Not causal evidence"],
            }
            for ids in (["price"], ["business"], [])
        ]
    }
    assert [row["status"] for row in question_coverage(hypothesis)] == [
        "proposed_inputs",
        "proposed_inputs",
        "no_measured_input",
    ]
    fitted = question_coverage(hypothesis, ["price"])
    assert [row["status"] for row in fitted] == [
        "inputs_fitted",
        "inputs_not_fitted",
        "no_measured_input",
    ]
    assert all("status" not in claim for claim in hypothesis["question_map"])
    assert question_coverage({}) == []  # Old reports remain readable, with no invented mapping.


def test_paths_keep_each_round_and_its_actual_inputs_including_rejected_news_trial():
    steps = {}
    for number in (1, 2, 3):
        prefix = {1: "initial", 2: "revised", 3: "round_3"}[number]
        stage = {1: "first_prediction", 2: "second_prediction", 3: "round_3_prediction"}[number]
        identity = "news" if number == 2 else "price"
        steps[f"{prefix}_factors"] = [
            {"factor_id": identity, "kind": "news" if number == 2 else "numeric"}
        ]
        steps[f"{prefix}_hypothesis"] = {"hypothesis": f"Attempt {number}"}
        steps[stage] = {"factor_ids": [identity]}
    paths = evidence_path(steps)
    assert [p["news_columns_fitted"] for p in paths] == [0, 1, 0]
    assert [p["round"] for p in paths] == [1, 2, 3]
    view = visual_output("round_3_prediction", steps["round_3_prediction"], steps, "KO")
    assert view["evidence_path"]["hypothesis"] == "Attempt 3"
    assert view["evidence_path"]["factors"][0]["factor_id"] == "price"


def test_continuity_only_marks_supplied_mature_feedback_and_compares_selected_factors():
    entries = [
        {
            "study_id": name,
            "status": status,
            "reason": status,
            "factor_ids": ["previous"],
            "forecast": {"target_session": "2026-11-06", "predicted_price": 10},
            "metrics": {"actual_price": 11, "price_error": -1} if status == "settled" else None,
        }
        for name, status in [("mature", "settled"), ("latest", "pending")]
    ]
    prior = {"entries": [entries[0]], "history": {"entries": entries}}
    view = continuity(prior, ["adopted"])
    assert [e["provided_as_feedback"] for e in view["entries"]] == [True, False]
    assert view["entries"][1]["price_error"] is None
    assert view["previous_study_id"] == "latest"
    assert view["added"] == ["adopted"] and view["removed"] == ["previous"]
    assert continuity(None) is None
    assert continuity({"entries": [], "history": {"entries": []}})["entries"] == []
    stage = visual_output("observe_prior_forecasts", {"prior_outcomes": prior}, {}, "KO")
    assert stage["kind"] == "continuity"
    summary = visual_output(
        "summary",
        {"selection": {"selected_factor_ids": ["adopted"]}},
        {"target": {"prior_outcomes": prior}},
        "KO",
    )
    assert summary["continuity"] == view


def test_news_failure_and_recovery_are_visible_without_opening_json():
    view = visual_output(
        "article_reads_1",
        [{"id": "failed", "reason": "http_403", "fetch_attempted": True}],
        {},
        "KO",
    )
    assert view["items"][0]["reason"] == "http_403"
    view = visual_output(
        "news_review_1",
        {"recovery_query": "KO official release", "recovery_reason": "No readable source"},
        {},
        "KO",
    )
    assert view["recovery_query"] == "KO official release"


def test_history_context_does_not_expand_the_twenty_outcome_feedback_limit(monkeypatch):
    entries = [
        {
            "study_id": str(i),
            "factor_ids": ["price"],
            "forecast": None,
            "status": "settled",
            "reason": "observed",
            "metrics": {"price_error": i},
            "strategy": {"net_return": 0},
        }
        for i in range(25)
    ]
    monkeypatch.setattr(
        "quant_rsi.research.study_history.history_view",
        lambda *_: {"entries": entries, "notice": "fixture"},
    )
    prior = mature_feedback("unused", "KO")
    assert len(prior["entries"]) == 20
    assert len(prior["history"]["entries"]) == 25
    assert all("metrics" not in e and "strategy" not in e for e in prior["history"]["entries"])
    assert [e["provided_as_feedback"] for e in continuity(prior)["entries"]] == [False] * 5 + [
        True
    ] * 20
