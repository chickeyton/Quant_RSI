"""Profile ownership, source fidelity and compositions enter real study fits."""

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from quant_rsi.cli import main
from quant_rsi.factors.catalog import resolve_instrument_factors
from quant_rsi.research import study
from quant_rsi.research.demo import load_catalog, select_factors, simulated_hypothesis
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_factor_profiles import load_study_catalog
from quant_rsi.research.study_peers import prepare_related_prices
from quant_rsi.research.study_ui import read_state

ROOT = Path(__file__).resolve().parents[1] / "instruments"


@pytest.fixture
def profiles(tmp_path):
    root = tmp_path / "instruments"
    shutil.copytree(ROOT, root)
    path = root / "US/KO/factors.json"
    profile = json.loads(path.read_text())
    profile["definitions"].append(
        {
            "factor_id": "ko.local_test_return",
            "kind": "numeric",
            "input_kind": "positive_price_series",
            "description": "Local fixture hypothesis.",
            "recipe": {"op": "mean_return", "window": 3},
        }
    )
    profile["bindings"].append(
        {
            "binding_id": "local_return",
            "definition_id": "ko.local_test_return",
            "source": {"source_id": "target_daily_prices", "entity": "KO", "field": "close"},
        }
    )
    path.write_text(json.dumps(profile))
    return root


def configured(root=ROOT, *, symbol="KO", peer="KOF", estimator="ridge", missing_terminal=False):
    model = LightGBMDemo(
        load_catalog(ROOT / "shared/equities.json"),
        evaluation_sessions=20,
        terminal_sessions=10,
        estimator=estimator,
    )
    model.data = replace(model.data, instruments=(symbol,))
    cutoffs = [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
    peers = prepare_related_prices(
        model.data,
        cutoffs,
        list(model.train) + list(model.origins),
        {"stocks": [peer]},
        mode="synthetic",
        directory=None,
    )
    if missing_terminal:
        for values in peers["columns"].values():
            values[model.terminal_origins[0] :] = np.nan
    model.configure_peers(peers)
    model.configure_profile(resolve_instrument_factors(root, "US", symbol))
    return model


def test_xom_profile_executes_partner_composition_without_ko_bindings():
    model = configured(symbol="XOM", peer="CVX")
    columns = model.profile_features["columns"]
    np.testing.assert_allclose(
        columns["xom.relative_partner_momentum"],
        columns["bound.price_momentum"] - model.peer_features["columns"]["peer.CVX.peer_return_5"],
        equal_nan=True,
    )
    factors = select_factors(model.library, {"factor_ids": ["xom.relative_partner_momentum"]})
    prediction = model.predict(model.context("XOM"), factors, {})
    assert prediction["training_feature_columns"] == 1
    inputs = prediction["bound_factor_inputs"][0]["inputs"]
    assert [item["source"]["entity"] for item in inputs] == ["XOM", "CVX"]
    assert "ko.relative_bottler_momentum" not in columns


def test_shared_local_and_composed_features_use_exact_declared_columns(profiles):
    model = configured(profiles)
    definitions = {d["factor_id"]: d for d in model.profile_features["definitions"]}
    own, peer = definitions["bound.price_momentum"], definitions["bound.bottler_momentum"]
    assert own["definition_id"] == peer["definition_id"] == "equity.price_momentum_5"
    assert own["source"]["entity"] == "KO" and peer["source"]["entity"] == "KOF"
    assert own["definition_scope"] == peer["definition_scope"] == "shared"
    local = definitions["bound.local_return"]
    assert local["definition_scope"] == "instrument" and local["definition_owner"] == "US/KO"
    columns = model.profile_features["columns"]
    np.testing.assert_allclose(
        columns["bound.bottler_momentum"],
        model.peer_features["columns"]["peer.KOF.peer_return_5"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        columns["ko.relative_bottler_momentum"],
        columns["bound.price_momentum"] - columns["bound.bottler_momentum"],
        equal_nan=True,
    )
    factors = select_factors(
        model.library, {"factor_ids": [local["factor_id"], "ko.relative_bottler_momentum"]}
    )
    prediction = model.predict(model.context("KO"), factors, {})
    assert prediction["training_feature_columns"] == 2
    assert prediction["feature_values"]["ko.relative_bottler_momentum"] == pytest.approx(
        columns["ko.relative_bottler_momentum"][model.origin]
    )
    usage = {entry["source"]: entry for entry in prediction["input_usage"]}
    assert usage["KOF"]["status"] == "used"
    assert usage["KOF"]["factor_ids"] == ["peer.KOF.peer_return_5"]
    assert prediction["bound_factor_inputs"][1]["inputs"][1]["source"]["entity"] == "KOF"


@pytest.mark.parametrize(
    "source",
    [
        {"source_id": "usda_ams_egg_reports", "entity": "US eggs", "field": "wholesale_price"},
        {"source_id": "target_daily_prices", "entity": "CALM", "field": "close"},
        {"source_id": "target_daily_prices", "entity": "KO", "field": "open"},
    ],
)
def test_unsupported_binding_never_substitutes_target_prices(profiles, source):
    path = profiles / "US/KO/factors.json"
    profile = json.loads(path.read_text())
    profile["bindings"][0]["source"] = source
    path.write_text(json.dumps(profile))
    model = configured(profiles)
    definitions = {d["factor_id"]: d for d in model.profile_features["definitions"]}
    assert definitions["bound.price_momentum"]["source"] == source
    for identity in ("bound.price_momentum", "ko.relative_bottler_momentum"):
        assert definitions[identity]["runnable"] is False
        with pytest.raises(ValueError, match="pending"):
            select_factors(model.library, {"factor_ids": [identity]})
        assert np.isnan(model.profile_features["columns"][identity]).all()


def test_calm_business_sources_remain_pending_and_local_to_calm():
    calm, ko = configured(symbol="CALM"), configured()
    definitions = {d["factor_id"]: d for d in calm.profile_features["definitions"]}
    assert definitions["bound.price_momentum"]["runnable"]
    for identity in ("bound.egg_price", "bound.confirmed_layer_culls"):
        assert definitions[identity]["definition_owner"] == "US/CALM"
        assert definitions[identity]["runnable"] is False
        assert identity not in {d["factor_id"] for d in ko.library["definitions"]}


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda p: p["compositions"][0]["expression"]["nodes"][0].update(leaf="not_bound"),
            "unknown",
        ),
        (lambda p: p["compositions"].append(deepcopy(p["compositions"][0])), "duplicate"),
        (lambda p: p["compositions"][0].update(factor_id="equity.price_momentum_5"), "shadowed"),
        (
            lambda p: p["compositions"][0]["expression"]["nodes"][2].update(
                inputs=["spread", "bottler"]
            ),
            "topologically",
        ),
    ],
)
def test_invalid_composition_rejected_before_execution(profiles, change, match):
    path = profiles / "US/KO/factors.json"
    profile = json.loads(path.read_text())
    change(profile)
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match=match):
        load_study_catalog("KO", None, profiles)


def test_explicit_catalog_and_missing_profile_preserve_standalone_path(profiles):
    explicit = load_study_catalog("KO", profiles / "shared/equities.json", profiles)
    fallback = load_study_catalog("UNKNOWN", None, profiles)
    assert "instrument_profile" not in explicit
    assert fallback["definitions"] == explicit["definitions"]
    assert "No US/UNKNOWN" in fallback["profile_note"]


@pytest.mark.parametrize("estimator", ["lightgbm", "ridge"])
def test_profile_selection_revision_terminal_and_report_are_one_flow(
    tmp_path, profiles, monkeypatch, estimator
):
    def hypothesis(context, library, **kwargs):
        result = simulated_hypothesis(context, library, **kwargs)
        proposal = result["proposed_factors"][0]
        generated = f"research.{proposal['op']}_{proposal['window']}"
        result["factor_ids"] = (
            [generated, "bound.local_return"]
            if kwargs.get("previous")
            else [generated, "bound.price_momentum", "ko.relative_bottler_momentum"]
        )
        return result

    monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    root = tmp_path / "study"
    assert (
        main(
            [
                "research-study",
                "KO",
                "--factor-root",
                str(profiles),
                "--prices",
                "synthetic",
                "--news",
                "simulated",
                "--related",
                "profile",
                "--llm",
                "simulated",
                "--model",
                estimator,
                "--no-ui",
                "--quiet",
                "--output",
                str(root),
            ]
        )
        == 0
    )
    result = json.loads((root / "result.json").read_text())
    first, second = [a["prediction"] for a in result["iterations"]["attempts"]]
    assert first["training_feature_columns"] == 3 and second["training_feature_columns"] == 2
    assert second["factor_ids"][-1] == "bound.local_return"
    assert result["terminal_evaluation"]["status"] == "completed"
    assert (
        result["study"]["factor_profile"]
        == json.loads((root / "factor_profile.json").read_text())["execution"]
    )
    assert [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()] == result[
        "trace"
    ]
    html = (root / "report.html").read_text()
    assert "Instrument factor profile" in html and "related_daily_prices / KOF / close" in html
    state = read_state(root)
    event = next(e for e in state["events"] if e["stage"] == "initial_factors")
    composition = next(
        d for d in event["view"]["factors"] if d["factor_id"] == "ko.relative_bottler_momentum"
    )
    assert composition["definition_owner"] == "US/KO"
    assert composition["input_factor_ids"] == ["bound.price_momentum", "bound.bottler_momentum"]


def test_missing_terminal_peer_preserves_profile_eligibility_and_exposes_missing_forecast():
    model = configured(missing_terminal=True)
    model.price_info = {"latest_target_session": "2020-07-03", "currency": "USD"}
    factors = select_factors(model.library, {"factor_ids": ["ko.relative_bottler_momentum"]})
    assert factors[0]["runnable"]
    prediction = model.predict(model.context("KO"), factors, {})
    assert prediction["latest_forecast_unavailable"]["missing_factor_ids"] == [
        "ko.relative_bottler_momentum"
    ]
    terminal = model.evaluate_terminal(1)
    assert terminal["status"] == "unavailable" and terminal["sample_count"] == 10
    assert model.profile_features["definitions"][-1]["runnable"]


def test_bound_recipe_source_cannot_be_redefined_after_resolution():
    model = configured()
    factors = select_factors(model.library, {"factor_ids": ["bound.bottler_momentum"]})
    factors[0]["source"]["entity"] = "KO"
    with pytest.raises(ValueError, match="exact usable"):
        model.predict(model.context("KO"), factors, {})


def test_composition_change_changes_actual_model_input_and_prediction(profiles):
    first = configured(profiles)
    path = profiles / "US/KO/factors.json"
    profile = json.loads(path.read_text())
    profile["compositions"][0]["expression"]["nodes"][-1]["op"] = "multiply"
    path.write_text(json.dumps(profile))
    second = configured(profiles)
    predictions = [
        m.predict(
            m.context("KO"),
            select_factors(m.library, {"factor_ids": ["ko.relative_bottler_momentum"]}),
            {},
        )
        for m in (first, second)
    ]
    assert predictions[0]["feature_values"] != predictions[1]["feature_values"]
    assert predictions[0]["predicted_price"] != predictions[1]["predicted_price"]
    np.testing.assert_array_equal(first.origins, second.origins)
    np.testing.assert_array_equal(first.train, second.train)
