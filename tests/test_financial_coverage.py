"""Explain real missing financial inputs without changing factor admission or values."""

import json
from copy import deepcopy

import pytest
from test_cashflow_factors import cash_snapshot
from test_inventory_factors import inventory_snapshot
from test_research_study import command
from test_study_financials import snapshot

from quant_rsi.cli import main
from quant_rsi.research import study
from quant_rsi.research import study_financials as financials
from quant_rsi.research.study_ui import read_state


def test_gaps_count_only_missing_required_rows_and_keep_actual_cutoffs():
    cutoffs = [
        "2023-04-26T04:59:59+00:00",
        "2023-04-26T05:00:00+00:00",
        "2024-04-26T05:00:00+00:00",
        "2025-02-01T20:00:00+00:00",  # Stale, but outside the required fit/evaluation rows.
    ]
    bundle = financials.prepare_financials(inventory_snapshot(), cutoffs, [0, 1, 2])
    assert bundle["columns"]["financial.inventory_yoy"] == [None, None, 0.25, None]
    inventory = [g for g in bundle["coverage_gaps"] if g["factor_id"] == "financial.inventory_yoy"]
    assert [(g["reason"], g["missing_required_rows"], g["example_cutoff"]) for g in inventory] == [
        ("no_visible_inventory_balance", 1, cutoffs[0]),
        ("no_unique_comparable_inventory_balance", 1, cutoffs[1]),
    ]
    assert "350–380" in inventory[1]["next_step"]
    for definition in bundle["definitions"]:
        gaps = [g for g in bundle["coverage_gaps"] if g["factor_id"] == definition["factor_id"]]
        assert sum(g["missing_required_rows"] for g in gaps) == definition["missing_required_rows"]
        assert definition["runnable"] == (not gaps)
    only_complete = financials.prepare_financials(inventory_snapshot(), cutoffs, [2])
    assert not any(
        g["factor_id"] == "financial.inventory_yoy" for g in only_complete["coverage_gaps"]
    )
    assert only_complete["columns"] == bundle["columns"]


@pytest.mark.parametrize(
    "case", ["prior_quarter", "margin_filing", "zero_cash_base", "inventory_base"]
)
def test_guidance_matches_the_unavailable_measurement(case):
    data = snapshot()
    cutoff = "2024-04-27T20:00:00+00:00"
    if case == "prior_quarter":
        data["companyfacts"]["facts"]["us-gaap"]["Revenues"]["units"]["USD"].pop(0)
        factor, reason = "revenue_yoy", "no_comparable_income_quarter"
    elif case == "margin_filing":
        data["companyfacts"]["facts"]["us-gaap"]["OperatingIncomeLoss"]["units"]["USD"][-1][
            "accn"
        ] = "different"
        factor, reason = "operating_margin", "missing_or_incompatible_margin_facts"
    elif case == "zero_cash_base":
        data, cutoff = cash_snapshot(), "2024-10-26T05:00:00+00:00"
        data["companyfacts"]["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"][
            "units"
        ]["USD"][0]["val"] = 0
        factor, reason = "operating_cashflow_yoy", "missing_or_incompatible_cashflow_values"
    else:
        data = inventory_snapshot()
        data["companyfacts"]["facts"]["us-gaap"]["InventoryNet"]["units"]["USD"][0]["val"] = 0
        factor, reason = "inventory_yoy", "ambiguous_or_nonpositive_prior_inventory"
    bundle = financials.prepare_financials(data, [cutoff], [0])
    assert bundle["columns"]["financial." + factor] == [None]
    gap = next(g for g in bundle["coverage_gaps"] if g["factor_id"] == "financial." + factor)
    assert gap["reason"] == reason
    assert gap["missing_required_rows"] == 1
    assert gap["description"] and gap["next_step"]


def test_partial_coverage_reaches_preflight_model_context_live_cards_and_report(
    tmp_path, monkeypatch
):
    original_prepare = financials.prepare_financials
    original_hypothesis = study.simulated_hypothesis
    contexts = []

    def without_inventory(data, cutoffs, required):
        data = deepcopy(data)
        data["companyfacts"]["facts"]["us-gaap"].pop("InventoryNet")
        return original_prepare(data, cutoffs, required)

    def hypothesize(context, library, **kwargs):
        contexts.append(context)
        return original_hypothesis(context, library, **kwargs)

    monkeypatch.setattr(financials, "prepare_financials", without_inventory)
    monkeypatch.setattr(study, "simulated_hypothesis", hypothesize)
    root = tmp_path / "partial"
    assert main(command(root, "--llm", "simulated", "--financials", "--no-ui")) == 0
    result = json.loads((root / "result.json").read_text())
    data = result["study"]["financial_data"]
    assert len(data["coverage_gaps"]) == 1
    assert data["coverage_gaps"][0]["reason"] == "no_visible_inventory_balance"
    for context in contexts:
        assert context["financial_features"]["coverage_gaps"] == data["coverage_gaps"]
        assert context["financial_features"]["sample_coverage"] == data["sample_coverage"]
    for event in read_state(root)["events"]:
        if event["stage"] in {"load_financials", "check_inputs"}:
            assert event["view"]["coverage_gaps"] == data["coverage_gaps"]
            assert event["view"]["sample_coverage"] == data["sample_coverage"]
    for name in ("input_report.html", "report.html", "report.md"):
        text = (root / name).read_text()
        assert "Why financial factors are unavailable" in text
        assert "No reported inventory balance was visible" in text
        assert "do not fill missing values" in text
        assert "Financial variation on research rows" in text
        assert "Distinct training values" in text
    assert "Available, not selected" in (root / "report.html").read_text()
    assert result["iterations"]["completed_fits"] == 2
    assert result["configuration"]["comparison_fit_budget"] == 1
    assert result["study"]["financial_comparison"]["status"] == "completed"


def test_sample_coverage_excludes_terminal_values_and_does_not_count_missing_as_zero():
    bundle = {
        "definitions": [{"factor_id": "financial.inventory_yoy"}],
        "columns": {"financial.inventory_yoy": [None, 0.0, 0.0, 2.0, 3.0, 4.0]},
    }
    original = deepcopy(bundle)
    assert financials.sample_coverage(bundle, [0, 1, 2], [3, 4]) == [
        {
            "factor_id": "financial.inventory_yoy",
            "training_rows": 3,
            "training_available_rows": 2,
            "training_distinct_values": 1,
            "development_rows": 2,
            "development_available_rows": 2,
            "development_distinct_values": 2,
        }
    ]
    assert bundle == original
