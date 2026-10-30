"""Independent arithmetic and point-in-time/worker integration for price adapters."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_forecast_features import prepare_inputs as inputs_fixture

from quant_rsi.core.artifacts import ArtifactStore, canonical, decode
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.factors import discovery
from quant_rsi.factors.contracts import NumericFactorDefinition
from quant_rsi.factors.discovery_graph import FactorLeaf, evaluate, leaf_expression
from quant_rsi.factors.price_adapters import (
    QLIB_REVISION,
    CandidatePriceRecipe,
    evaluate_price_adapter,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture
def prepare_inputs(tmp_path, monkeypatch):
    return inputs_fixture.__wrapped__(tmp_path, monkeypatch)


def leaf(op, window, role="target"):
    return FactorLeaf(
        name=op,
        role=role,
        definition=NumericFactorDefinition(
            factor_id="equity." + op,
            description="Source-pinned test definition",
            kind="numeric",
            input_kind="positive_price_series",
            recipe=CandidatePriceRecipe(op=op, window=window),
        ),
    )


def result(inputs, definition, index):
    return evaluate(
        leaf_expression(definition.name),
        (definition,),
        inputs,
        inputs.target[0].dates[index],
        inputs.target[0].decision_at(index),
    )


@pytest.mark.parametrize(
    "op,window,prices,expected",
    [
        ("qlib_roc_v1", 2, [10.0, 20.0, 40.0], 0.25),
        ("qlib_roc_v1", 1, [40.0, 10.0], 4.0),
        ("qlib_roc_v1", 2, [10.0, 10.0, 10.0], 1.0),
        ("qlib_ma_v1", 3, [10.0, 20.0, 30.0], 2 / 3),
        ("qlib_ma_v1", 3, [30.0, 20.0, 10.0], 2.0),
        ("qlib_ma_v1", 1, [19.0], 1.0),
        ("qlib_ma_v1", 2, [1e308, 1e308], 1.0),
    ],
)
def test_independent_arithmetic(op, window, prices, expected):
    recipe = CandidatePriceRecipe(op=op, window=window)
    assert evaluate_price_adapter(recipe, prices) == pytest.approx(expected)


@pytest.mark.parametrize("op,window", [("qlib_roc_v1", 1), ("qlib_ma_v1", 2)])
@pytest.mark.parametrize(
    "prices",
    [
        [],
        [10.0],
        [1.0, 2.0, 3.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [float("nan"), 1.0],
        [1.0, float("inf")],
        [1e308, 1e-308],
    ],
)
def test_incomplete_invalid_and_unrepresentable_values_are_missing(op, window, prices):
    assert evaluate_price_adapter(CandidatePriceRecipe(op=op, window=window), prices) is None


def test_positive_inverse_ratio_underflow_remains_missing():
    recipe = CandidatePriceRecipe(op="qlib_roc_v1", window=1)
    assert evaluate_price_adapter(recipe, [1e-308, 1e308]) is None


@pytest.mark.parametrize(
    "change",
    [
        {"window": 0},
        {"window": 31},
        {"window": True},
        {"op": "qlib_rsi_v1"},
        {"upstream_revision": "latest"},
    ],
)
def test_unregistered_semantics_reject(change):
    with pytest.raises(ValidationError):
        CandidatePriceRecipe.model_validate(dict(op="qlib_ma_v1", window=5) | change)


def test_legacy_recipe_serialization_and_survey_pin():
    old = dict(op="momentum", window=5)
    definition = leaf("qlib_roc_v1", 5).definition.model_dump(mode="json")
    definition["recipe"] = old
    validated = NumericFactorDefinition.model_validate_json(canonical(definition))
    assert isinstance(validated.recipe, FactorSpec)
    assert validated.model_dump(mode="json")["recipe"] == old
    candidates = json.loads(
        (ROOT / "experimental/candidate_factors/definitions/qlib.json").read_text()
    )
    selected = [c for c in candidates["candidates"] if c["id"] in {"qlib.roc", "qlib.ma"}]
    assert len(selected) == 2
    assert all(QLIB_REVISION in c["source"]["url"] for c in selected)


def test_full_window_boundaries_and_price_units(prepare_inputs):
    inputs, _ = prepare_inputs()
    ma, roc = leaf("qlib_ma_v1", 3), leaf("qlib_roc_v1", 3)
    assert result(inputs, ma, 1)["value"] is None
    assert result(inputs, ma, 2)["value"] == pytest.approx((100 + 110 + 121) / 3 / 121)
    assert result(inputs, roc, 2)["value"] is None
    assert result(inputs, roc, 3)["value"] == pytest.approx(100 / 122)


@pytest.mark.parametrize("op", ["qlib_roc_v1", "qlib_ma_v1"])
def test_asof_future_mutation_missing_and_late_context(prepare_inputs, op):
    inputs, _ = prepare_inputs()
    future, _ = prepare_inputs(future_scale=17)
    item = leaf(op, 5, role="market")
    first, changed = result(inputs, item, 70), result(future, item, 70)
    assert first["value"] == changed["value"]
    assert first["source_refs"] and not first["reasons"]
    missing, _ = prepare_inputs(missing_market=70)
    late, _ = prepare_inputs(later_market_availability=True)
    for unavailable in (missing, late):
        value = result(unavailable, item, 70)
        assert value["value"] is None and value["reasons"]


@pytest.mark.parametrize("first", ["qlib_roc", "qlib_ma"])
def test_catalog_adapter_in_actual_worker_and_sealed_replay(prepare_inputs, first):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    other = "qlib_ma" if first == "qlib_roc" else "qlib_roc"
    envelope, leaves = discovery.bind_catalog(
        envelope, ROOT / "instruments", "US", "KO", {first: "target", other: "target"}
    )
    plan = discovery.DiscoveryPlan.model_validate_json(
        canonical(
            dict(
                campaign_id="adapter-test",
                instrument="KO",
                exposure="engineering_fixture",
                max_rounds=1,
                fold_origins=[[80], [90]],
                leaves=[item.model_dump(mode="json") for item in leaves],
            )
        )
    )
    discovery.initialize(envelope, plan)
    partial = discovery.resume("adapter-test", max_fits=1)
    assert partial["fits_attempted"] == 1
    report = discovery.resume("adapter-test", max_fits=4)
    assert report["status"] == "completed"
    assert report["fits_attempted"] == 5
    assert report["history"][0]["expression_ref"] == leaf_expression(first).ref
    assert report["history"][0]["improvement_pp"] is not None
    store = ArtifactStore(discovery.campaign_root("adapter-test") / "artifacts")
    binding = store.json(report["history"][0]["binding_ref"])
    assert binding["leaves"][0]["definition"]["recipe"]["upstream_revision"] == QLIB_REVISION
    assert discovery.inspect("adapter-test", replay=True) == report
    assert discovery.resume("adapter-test", max_fits=5) == report


def test_preparation_example_selects_adapters_explicitly(prepare_inputs, tmp_path):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    inputs = tmp_path / "inputs.json"
    inputs.write_bytes(canonical(envelope))
    output = tmp_path / "prepared"
    command = [
        sys.executable,
        str(ROOT / "examples/prepare_factor_discovery.py"),
        str(inputs),
        "--output",
        str(output),
        "--campaign-id",
        "prepared-adapters",
        "--folds",
        "[[80],[90]]",
        "--exposure",
        "engineering_fixture",
        "--factor-set",
        "candidate-prices",
    ]
    subprocess.run(command, check=True, capture_output=True)
    plan = decode((output / "plan.json").read_bytes())
    assert [item["name"] for item in plan["leaves"]] == ["qlib_roc", "qlib_ma"]
    assert all(item["binding_ref"] for item in plan["leaves"])
