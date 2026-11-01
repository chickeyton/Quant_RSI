"""Business hypotheses use real arithmetic/fits with offline news and LLM fixtures."""

import json
from copy import deepcopy

import numpy as np
import pytest
from test_research_study import command
from test_study_factor_memory import memory

from quant_rsi.cli import main
from quant_rsi.core.artifacts import canonical
from quant_rsi.research.demo import select_factors
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.demo_proposals import extend_library
from quant_rsi.research.llm import LLMHypothesizer
from quant_rsi.research.study_business import BusinessFactorProposal, business_column
from quant_rsi.research.study_factor_memory import reuse_recipes


def proposal(revised=False):
    return {
        "name": "margin_gap" if revised else "inventory_gap",
        "description": "Offline business hypothesis, not a causal conclusion",
        "inputs": [
            {
                "name": "left",
                "factor_id": "financial.operating_margin" if revised else "financial.inventory_yoy",
            },
            {"name": "right", "factor_id": "financial.revenue_yoy"},
        ],
        "expression": {
            "schema_version": 1,
            "nodes": [
                {"name": "a", "op": "leaf", "inputs": [], "leaf": "left"},
                {"name": "b", "op": "leaf", "inputs": [], "leaf": "right"},
                {"name": "gap", "op": "subtract", "inputs": ["a", "b"], "leaf": None},
            ],
        },
    }


def library():
    return {
        "definitions": [
            {
                "factor_id": f"financial.{metric}",
                "kind": "numeric",
                "recipe": {"op": "financial", "metric": metric},
                "runnable": True,
            }
            for metric in ("inventory_yoy", "revenue_yoy", "operating_margin")
        ]
    }


def hypothesis(business):
    return {
        "factor_ids": ["research.momentum_5", f"business.{business['name']}"],
        "proposed_factors": [{"op": "momentum", "window": 5, "description": "price control"}],
        "proposed_business_factors": [business],
    }


def test_business_arithmetic_propagates_missing_and_checks_retained_inputs():
    definitions = library()["definitions"]
    definition = BusinessFactorProposal.model_validate_json(canonical(proposal())).definition(
        definitions
    )
    bundle = {
        "definitions": definitions,
        "columns": {
            "financial.inventory_yoy": np.array([0.3, np.nan, 0.2]),
            "financial.revenue_yoy": np.array([0.1, 0.4, 0.5]),
        },
    }
    values, inputs = business_column(definition, bundle)
    np.testing.assert_allclose(values, [0.2, np.nan, -0.3])
    assert [item["factor_id"] for item in inputs] == definition["input_factor_ids"]
    bundle["definitions"][0]["runnable"] = False
    with pytest.raises(ValueError, match="exact usable"):
        business_column(definition, bundle)


def test_business_admission_is_local_and_revisions_cannot_overwrite_old_recipe():
    original = library()
    first = extend_library(original, hypothesis(proposal()))
    assert len(original["definitions"]) == 3
    select_factors(first, hypothesis(proposal()))
    changed = proposal(True) | {"name": "inventory_gap"}
    with pytest.raises(ValueError, match="new business factor name"):
        extend_library(first, hypothesis(changed))
    second = extend_library(first, hypothesis(proposal(True)))
    assert {item["factor_id"] for item in second["definitions"]} >= {
        "business.inventory_gap",
        "business.margin_gap",
    }
    original["definitions"][0]["runnable"] = False
    pending = extend_library(original, hypothesis(proposal()))
    with pytest.raises(ValueError, match="pending"):
        select_factors(pending, hypothesis(proposal()))


@pytest.mark.parametrize("invalid", ["unknown", "unbound", "cycle", "code", "unused"])
def test_business_proposals_reject_unexecutable_hypotheses(invalid):
    item = proposal()
    if invalid == "unknown":
        item["inputs"][0]["factor_id"] = "financial.invented_demand"
    elif invalid == "unbound":
        item["inputs"][0]["name"] = "missing"
    elif invalid == "cycle":
        item["expression"]["nodes"][-1]["inputs"] = ["a", "gap"]
    elif invalid == "code":
        item["expression"]["nodes"][-1]["op"] = "eval"
    else:
        item["expression"]["nodes"].insert(
            0, {"name": "unused", "op": "leaf", "inputs": [], "leaf": "left"}
        )
    with pytest.raises(ValueError):
        extend_library(library(), hypothesis(item))


def test_business_memory_rebinds_current_availability_without_old_values():
    first = extend_library(library(), hypothesis(proposal()))
    old = first["definitions"][-1]
    old["column"] = [999]
    current = library()
    current["definitions"][0]["runnable"] = False
    restored, report = reuse_recipes(current, [memory(old)])
    assert restored["definitions"][-1]["runnable"] is False
    assert "column" not in restored["definitions"][-1]
    assert report["entries"][0]["status"] == "unavailable_current_inputs"
    restored, report = reuse_recipes(library(), [memory(old)])
    assert restored["definitions"][-1]["runnable"] is True
    assert report["entries"][0]["status"] == "restored_candidate"


@pytest.mark.parametrize("financial", [False, True])
def test_industry_composition_missingness_and_current_source_rebinding(financial):
    external_ids = ["external.eia_commercial_crude_wow", "external.eia_distillate_wow"]
    external = {
        "definitions": [
            {
                "factor_id": identity,
                "kind": "numeric",
                "recipe": {"op": "external", "metric": identity.split(".")[1]},
                "runnable": True,
            }
            for identity in external_ids
        ],
        "columns": {external_ids[0]: [0.3, None, 0.2], external_ids[1]: [0.1, 0.4, None]},
    }
    business = proposal()
    business["inputs"][0]["factor_id"] = external_ids[0]
    business["inputs"][1]["factor_id"] = "financial.revenue_yoy" if financial else external_ids[1]
    bundle = (
        {
            "definitions": library()["definitions"],
            "columns": {"financial.revenue_yoy": [0.1, 0.4, None]},
        }
        if financial
        else None
    )
    current = {"definitions": (bundle or {}).get("definitions", []) + external["definitions"]}
    definition = BusinessFactorProposal.model_validate_json(canonical(business)).definition(
        current["definitions"]
    )
    assert definition["recipe"]["op"] == "business_expression"
    values, sources = business_column(definition, bundle, external)
    np.testing.assert_allclose(values, [0.2, np.nan, np.nan])
    assert [item["factor_id"] for item in sources] == definition["input_factor_ids"]
    # Reuse recipes with fresh source bindings, never old feature columns.
    old = deepcopy(definition) | {"column": [999, 999, 999]}
    restored, report = reuse_recipes(current, [memory(old)])
    assert restored["definitions"][-1] == definition
    assert report["entries"][0]["status"] == "restored_candidate"
    external["definitions"][0]["runnable"] = False
    restored, report = reuse_recipes(current, [memory(old)])
    assert restored["definitions"][-1]["runnable"] is False
    assert report["entries"][0]["status"] == "unavailable_current_inputs"
    with pytest.raises(ValueError, match="exact usable"):
        business_column(definition, bundle, external)
    # Merely spelling an industry ID cannot bind a price proxy or an invented source.
    external["definitions"][0]["recipe"]["op"] = "momentum"
    with pytest.raises(ValueError, match="supplied"):
        BusinessFactorProposal.model_validate_json(canonical(business)).definition(
            current["definitions"]
        )


@pytest.mark.parametrize("model", ["ridge", "lightgbm"])
@pytest.mark.parametrize("mixed", [False, True])
def test_llm_business_revision_reaches_model_report_and_financial_ablation(
    tmp_path, monkeypatch, model, mixed
):
    calls = []
    matrices = []
    original_predict = LightGBMDemo.predict

    def predict(self, context, factors, hypothesis):
        result = original_predict(self, context, factors, hypothesis)
        matrix = self._fits[-1][1]
        sources = factors[1]["input_factor_ids"]
        columns = self.financial_features["columns"] | (self.external_features or {}).get(
            "columns", {}
        )
        expected = np.asarray(columns[sources[0]], dtype=float) - np.asarray(
            columns[sources[1]], dtype=float
        )
        np.testing.assert_allclose(matrix[:, 1], expected)
        matrices.append(matrix.copy())
        return result

    monkeypatch.setattr(LightGBMDemo, "predict", predict)

    def codex(self, prompt, schema):
        payload = json.loads(prompt.split("\nINPUT_JSON:\n")[1])
        revised = payload["feedback"] is not None
        calls.append(deepcopy(payload))
        assert "proposed_business_factors" in schema["required"]
        assert (
            "financial.inventory_yoy"
            in schema["$defs"]["BusinessInput"]["properties"]["factor_id"]["enum"]
        )
        if revised:
            assert payload["feedback"]["development"]["metrics"]["sample_count"] == 20
        business = proposal(revised)
        if mixed:
            external_id = (
                "external.eia_distillate_wow" if revised else "external.eia_commercial_crude_wow"
            )
            business["inputs"][0]["factor_id"] = external_id
            assert (
                external_id in schema["$defs"]["BusinessInput"]["properties"]["factor_id"]["enum"]
            )
            assert "weekly US industry inventory changes" in prompt
        return json.dumps(
            hypothesis(business)
            | {
                "hypothesis": "Compare reported business growth rates",
                "event_signal": 0,
                "evidence_news_ids": [payload["context"]["news"]["items"][0]["id"]],
                "proposed_news_factors": [
                    {
                        "name": "earnings",
                        "keywords": ["earnings"],
                        "window_days": 7,
                        "description": "Unselected news control",
                        "reuse_for_hours": None,
                    }
                ],
            }
        )

    monkeypatch.setattr(LLMHypothesizer, "_codex", codex)
    root = tmp_path / "study"
    flags = ["--external-drivers"] if mixed else []
    assert main(command(root, "--llm", "codex", "--financials", "--model", model, *flags)) == 0
    result = json.loads((root / "result.json").read_bytes())
    assert len(calls) == 2
    assert len(matrices) == 2
    assert not np.allclose(matrices[0][:, 1], matrices[1][:, 1], equal_nan=True)
    attempts = result["iterations"]["attempts"]
    for attempt, name in zip(attempts, ("inventory_gap", "margin_gap"), strict=True):
        prediction = attempt["prediction"]
        assert f"business.{name}" in prediction["feature_values"]
        sources = prediction["bound_factor_inputs"][0]["inputs"]
        assert len(sources) == 2
        assert {item["recipe"]["op"] for item in sources} == (
            {"financial", "external"} if mixed else {"financial"}
        )
        for source in sources:
            assert any(
                usage["status"] == "used" and source["factor_id"] in usage["factor_ids"]
                for usage in prediction["input_usage"][1:]
            )
        assert not any(
            identity.startswith("business.")
            for identity in prediction["input_usage"][0]["factor_ids"]
        )
    comparison = result["study"]["financial_comparison"]
    assert comparison["status"] == "completed"
    assert len(comparison["removed_factor_ids"]) == 1
    assert comparison["removed_factor_ids"][0].startswith("business.")
    assert comparison["kept_factor_ids"] == ["research.momentum_5"]
    assert comparison["without_financials"]["forecast"]["sample_count"] == 20
    html = (root / "report.html").read_text()
    assert "Business factor hypotheses" in html and "Input alias" in html
    assert "business.inventory_gap" in html and "business.margin_gap" in html
    if mixed:
        external = result["study"]["external_comparison"]
        assert external["status"] == "completed" and external["extra_fits"] == 1
        assert external["removed_factor_ids"] == comparison["removed_factor_ids"]
        assert external["kept_factor_ids"] == comparison["kept_factor_ids"]
        assert (
            external["without_external_drivers"]["forecast"]
            == comparison["without_financials"]["forecast"]
        )
        assert "used by selected expression" in html
