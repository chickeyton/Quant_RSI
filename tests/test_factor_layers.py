import json
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_rsi.cli import main
from quant_rsi.core.artifacts import canonical
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.factors.catalog import resolve_instrument_factors
from quant_rsi.factors.contracts import FactorCatalog, InstrumentFactors

REPO = Path(__file__).parents[1]


@pytest.fixture
def root(tmp_path):
    destination = tmp_path / "instruments"
    shutil.copytree(REPO / "instruments", destination)
    return destination


def edit(path, change):
    value = json.loads(path.read_bytes())
    change(value)
    path.write_bytes(canonical(value))


def resolved(root, symbol="CALM"):
    return resolve_instrument_factors(root, "US", symbol)


def test_shared_definition_is_reused_with_distinct_instrument_bindings(root):
    calm, ko = resolved(root), resolved(root, "KO")
    a, b = calm["factors"][0], ko["factors"][0]
    assert a["definition"] == b["definition"]
    assert a["definition_scope"] == b["definition_scope"] == "shared"
    assert a["definition_owner"] == b["definition_owner"] == "equities"
    assert a["source"]["entity"] == "CALM"
    assert b["source"]["entity"] == "KO"
    assert {item["definition_scope"] for item in calm["factors"]} == {"shared", "instrument"}
    assert {item["definition_scope"] for item in ko["factors"]} == {"shared"}
    assert calm["factors"][2]["definition_owner"] == "US/CALM"
    assert {item["definition"]["kind"] for item in calm["factors"]} == {"numeric", "text"}
    assert all(
        item["definition"]["factor_id"] != "equity.price_mean_return_5" for item in calm["factors"]
    )
    a["source"]["entity"] = "changed"
    a["definition"]["recipe"]["window"] = 20
    assert resolved(root)["factors"][0]["definition"]["recipe"]["window"] == 5
    assert resolved(root, "KO")["factors"][0] == b


def test_numeric_recipe_preserves_the_legacy_worker_interface(root):
    recipe = resolved(root)["factors"][0]["definition"]["recipe"]
    assert canonical(recipe) == canonical(FactorSpec(op="momentum", window=5).model_dump())
    assert FactorSpec.model_validate_json(canonical(recipe)) == FactorSpec(op="momentum", window=5)


def test_local_definitions_are_not_visible_to_another_instrument(root):
    edit(
        root / "US/KO/factors.json",
        lambda config: config["bindings"][0].update(definition_id="calm.egg_price_momentum_5"),
    )
    with pytest.raises(ValueError, match="unknown factor definition"):
        resolved(root, "KO")


def test_local_definition_cannot_shadow_a_shared_definition(root):
    shared = json.loads((root / "shared/equities.json").read_bytes())["definitions"][0]
    edit(root / "US/CALM/factors.json", lambda config: config["definitions"].append(shared))
    with pytest.raises(ValueError, match="duplicate or shadowed"):
        resolved(root)


@pytest.mark.parametrize("location", ["shared/equities.json", "US/CALM/factors.json"])
def test_duplicate_definitions_reject_even_when_unselected(root, location):
    edit(root / location, lambda config: config["definitions"].append(config["definitions"][0]))
    with pytest.raises(ValueError, match="duplicate or shadowed"):
        resolved(root)


@pytest.mark.parametrize("field", ["catalogs", "bindings"])
def test_duplicate_catalogs_or_binding_ids_reject(root, field):
    edit(root / "US/CALM/factors.json", lambda config: config[field].append(config[field][0]))
    with pytest.raises(ValidationError, match="must be unique"):
        resolved(root)


def test_multiple_explicit_catalogs_compose_without_overrides(root):
    extra = json.loads((root / "shared/equities.json").read_bytes())
    extra["catalog_id"] = "industry"
    extra["definitions"] = [extra["definitions"][0]]
    extra["definitions"][0]["factor_id"] = "industry.input_price"
    (root / "shared/industry.json").write_bytes(canonical(extra))

    def select(config):
        config["catalogs"].append("industry")
        config["bindings"][0]["definition_id"] = "industry.input_price"

    edit(root / "US/KO/factors.json", select)
    assert resolved(root, "KO")["factors"][0]["definition_owner"] == "industry"
    edit(
        root / "shared/industry.json",
        lambda config: config["definitions"][0].update(factor_id="equity.price_momentum_5"),
    )
    with pytest.raises(ValueError, match="duplicate or shadowed"):
        resolved(root, "KO")


@pytest.mark.parametrize(
    "index,change",
    [
        (0, {"kind": "text"}),
        (0, {"recipe": {"op": "arbitrary_code", "window": 5}}),
        (0, {"recipe": {"op": "momentum", "window": 31}}),
        (0, {"recipe": {"op": "momentum", "window": True}}),
        (0, {"input_kind": "arbitrary_numbers"}),
        (1, {"kind": "numeric"}),
        (1, {"output_fields": []}),
        (1, {"output_fields": ["count", "count"]}),
        (1, {"extraction_instruction": ""}),
        (1, {"execute": True}),
    ],
)
def test_numeric_and_text_variants_reject_invalid_processing_specs(root, index, change):
    edit(root / "US/CALM/factors.json", lambda config: config["definitions"][index].update(change))
    with pytest.raises(ValidationError):
        resolved(root)


@pytest.mark.parametrize(
    "location,change,message",
    [
        ("US/CALM/factors.json", {"instrument": "KO"}, "does not match its directory"),
        ("US/CALM/factors.json", {"market": "CA"}, "does not match its directory"),
        ("shared/equities.json", {"catalog_id": "other"}, "does not match its filename"),
        ("shared/equities.json", {"catalogs": ["calm"]}, "Extra inputs"),
        ("US/CALM/factors.json", {"catalogs": ["../../outside"]}, "String should match"),
    ],
)
def test_identity_and_layer_boundaries(root, location, change, message):
    edit(root / location, lambda config: config.update(change))
    with pytest.raises(ValueError, match=message):
        resolved(root)


@pytest.mark.parametrize("market,symbol", [("../US", "CALM"), ("US", "../CALM")])
def test_path_components_cannot_escape_root(root, market, symbol):
    with pytest.raises(ValidationError):
        resolve_instrument_factors(root, market, symbol)


def test_catalog_symlink_cannot_escape_root(root, tmp_path):
    catalog = root / "shared/equities.json"
    outside = tmp_path / "outside.json"
    catalog.rename(outside)
    catalog.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the instruments root"):
        resolved(root)


def test_cli_is_read_only_and_reports_specification_status(root, monkeypatch, capsys):
    def prohibited(*args, **kwargs):
        raise AssertionError("validation must not launch computation")

    monkeypatch.setattr(subprocess, "Popen", prohibited)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert main(["factors", "validate", "US", "CALM", "--root", str(root)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == resolved(root)
    assert report["status"] == "specification_only" and report["runnable"] is False
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert main(["factors", "validate", "US", "MISSING", "--root", str(root)]) == 1
    output = capsys.readouterr()
    assert not output.out and "quant-rsi:" in output.err


def test_committed_catalog_and_configs_round_trip_without_coercion(root):
    catalog = FactorCatalog.model_validate_json((root / "shared/equities.json").read_bytes())
    assert FactorCatalog.model_validate_json(catalog.model_dump_json()) == catalog
    for symbol in ("CALM", "KO"):
        config = InstrumentFactors.model_validate_json(
            (root / "US" / symbol / "factors.json").read_bytes()
        )
        assert InstrumentFactors.model_validate_json(config.model_dump_json()) == config
