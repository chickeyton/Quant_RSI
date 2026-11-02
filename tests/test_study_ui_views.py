"""Visual projections must preserve the meaning of retained research evidence."""

import json

from quant_rsi.research.study_ui import page, read_state
from quant_rsi.research.study_ui_views import factor_changes, prediction_rows, visual_output


def factor(name, window, op="momentum"):
    return {"factor_id": name, "kind": "numeric", "recipe": {"op": op, "window": window}}


def test_factor_changes_distinguish_identity_parameters_and_ambiguous_replacements():
    before = [factor("same", 5), factor("changed", 5), factor("old", 5, "volatility")]
    after = [factor("same", 5), factor("changed", 10), factor("new", 10, "volatility")]
    changes = factor_changes(before, after)
    assert {c["after"]["factor_id"]: c["type"] for c in changes} == {
        "same": "retained",
        "changed": "changed",
        "new": "replaced",
    }
    assert changes[-1]["before"]["recipe"]["window"] == 5
    ambiguous = factor_changes([factor("a", 5), factor("b", 10)], [factor("c", 20)])
    assert [c["type"] for c in ambiguous] == ["removed", "removed", "added"]


def prediction(rows):
    return {"development": {"rows": rows}}


def row(origin, observed, predicted):
    return {
        "origin_session": origin,
        "target_session": origin + "-target",
        "observed_price": observed,
        "predicted_price": predicted,
        "reference_price": 10,
    }


def test_predictions_align_dates_and_reject_conflicting_outcomes():
    first = prediction([row("a", 11, 12), row("b", 12, 13), row("c", 13, 14)])
    second = prediction([row("b", 12, 15), row("a", 99, 16), row("d", None, None)])
    rows = prediction_rows(second, first)
    assert [r["initial"] for r in rows] == [13, None, None]
    assert rows[-1]["observed"] is None
    assert rows[-1]["prediction"] is None
    assert rows[0]["baseline"] == 10


def test_feedback_ranks_absolute_errors_and_skips_missing_observations():
    first = prediction([row("a", 10, 8), row("b", 10, 15), row("c", None, 7)])
    view = visual_output("feedback", {}, {"first_prediction": first}, "KO")
    assert [r["error"] for r in view["errors"]] == [5, -2]


def test_summary_preserves_missing_sharpe_and_incomplete_nav():
    account = {
        "metrics": {"sharpe_ratio": None, "risk_statistics": {"status": "zero_volatility"}},
        "valuations": [
            {"session_id": "a", "status": "complete", "nav": "100"},
            {"session_id": "b", "status": "incomplete", "nav": "99"},
        ],
    }
    view = visual_output(
        "summary", {}, {"first_prediction": {"strategy": {"strategy": account}}}, "KO"
    )
    assert view["accounts"][0]["sharpe_ratio"] is None
    assert view["accounts"][0]["risk_statistics"]["status"] == "zero_volatility"
    assert view["nav"] == [{"date": "a", "Initial": "100"}, {"date": "b", "Initial": None}]


def test_structured_views_survive_raw_preview_truncation_and_resolve_evidence(tmp_path):
    hostile = '</script><img src=x onerror="alert(1)">'
    article = {"id": "n1", "title": hostile, "text": "x" * 20000, "url": "javascript:alert(1)"}
    events = [
        {"stage": "news", "output": {"items": [article]}},
        {
            "stage": "initial_hypothesis",
            "output": {"hypothesis": "Test", "evidence_news_ids": ["n1", "missing"]},
        },
    ]
    (tmp_path / "trace.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    state = read_state(tmp_path)
    assert state["events"][0]["truncated"]
    evidence = state["events"][1]["view"]["evidence"]
    assert evidence[0]["title"] == hostile
    assert evidence[0]["excerpt_shortened"]
    assert len(evidence[0]["excerpt"]) == 600
    assert evidence[1]["id"] == "missing"
    html = page(state)
    assert hostile not in html
    assert "__STUDY_CARDS__" not in html
    assert "Inspect raw JSON" in html


def test_unknown_stage_remains_available_as_raw_data():
    assert visual_output("future_stage", {}, {}, "KO") == {"kind": "unknown"}


def test_peer_availability_is_distinct_from_executed_input_usage():
    availability = [{"symbol": "PEER", "status": "available"}]
    view = visual_output(
        "load_related_prices", {"related": {"price_features": availability}}, {}, "KO"
    )
    assert view["kind"] == "relationships"
    assert view["price_features"] == availability
    usage = [{"source": "PEER", "status": "not_selected", "factor_ids": []}]
    view = visual_output("first_prediction", {"input_usage": usage}, {}, "KO")
    assert view["input_usage"] == usage
