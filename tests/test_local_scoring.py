"""Regression from the portfolio supervisor's independent Decimal loss oracle."""

import json
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from quant_rsi.core.artifacts import digest
from quant_rsi.evaluation.local_scoring import evaluate_consumed


@pytest.fixture
def oracle():
    root = Path(__file__).parent / "fixtures"
    fixture = json.loads((root / "local_score_oracle.json").read_text())
    model = (root / "local_score_model.txt").read_text()
    assert digest(model.encode()) == fixture["selection"]["model_ref"]
    assert digest(model.encode()) == fixture["provenance"]["model_sha256"]
    return fixture["evidence"], fixture["selection"], model


def test_flat_terminal_does_not_require_unregistered_prediction_or_quote(oracle):
    evidence, selection, model = oracle
    complete = evaluate_consumed(evidence, selection, model, 252)
    missing = deepcopy(evidence)
    missing["rows"][-1]["close_quotes"] = {}
    observed = evaluate_consumed(missing, selection, model, 252)
    assert complete["eligible"] and observed["eligible"]
    assert observed["accounting"]["account"] == complete["accounting"]["account"]
    assert observed["utility"] == complete["utility"]
    assert [row["session_id"] for row in observed["predictions"]] == ["s1", "s2", "s3"]
    assert observed["return_count"] == 4
    assert Decimal(observed["accounting"]["account"]["cash"]) == Decimal("714.70")
    assert Decimal(observed["accounting"]["metrics"]["costs"]) == Decimal("15.30")
    assert observed["accounting"]["account"]["holdings"]["A"]["quantity"] == 0


def test_unregistered_terminal_feature_is_not_consumed(oracle):
    evidence, selection, model = oracle
    evidence["rows"][-1]["feature_view"] = deepcopy(evidence["rows"][-2]["feature_view"])
    with pytest.raises(ValueError, match="outside the registered decision grid"):
        evaluate_consumed(evidence, selection, model, 252)


@pytest.mark.parametrize("failure", ["missing_exit_open", "missing_held_close"])
def test_terminal_trading_or_valuation_failure_keeps_inventory_and_fixed_denominator(
    oracle, failure
):
    evidence, selection, model = oracle
    if failure == "missing_exit_open":
        evidence["rows"][-1]["open_quotes"]["s4"] = {}
    else:
        selection["portfolio"]["termination"] = "preserve_inventory"
        evidence["grid"]["terminal_policy"] = "preserve_inventory"
        evidence["grid"]["terminal_exit_decision"] = None
        evidence["rows"][-1]["close_quotes"] = {}
    result = evaluate_consumed(evidence, selection, model, 252)
    assert result["status"] == "risk_failure"
    assert result["utility"] is None
    assert result["return_count"] == 4
    assert result["accounting"]["valid_return_count"] == 3
    assert result["accounting"]["net_returns"][-1] is None
    assert result["accounting"]["account"]["holdings"]["A"]["quantity"] > 0
    assert result["accounting"]["account"]["valuations"][-1]["status"] == "unresolved"
