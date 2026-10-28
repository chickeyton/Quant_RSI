"""Offline market preparation/registration checks; no worker or live-source acceptance."""

import json
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_enhanced_forecast_workflow import NOW, audit_package
from test_forecast_recursive_market import prepared_inputs

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, canonical, decode, digest
from quant_rsi.forecast import forecast_context_collection as collection
from quant_rsi.forecast import forecast_context_data as context
from quant_rsi.forecast.forecast_workflow_config import ForecastWorkflowConfig
from quant_rsi.forecast.recursive import forecast_recursive_campaign as campaign
from quant_rsi.forecast.recursive import forecast_recursive_preparation as preparation
from quant_rsi.forecast.recursive.forecast_recursive_market import project_market_target
from quant_rsi.forecast.recursive.forecast_recursive_registration import verify_root_manifest
from quant_rsi.market import market_contracts
from quant_rsi.research import research
from quant_rsi.research.researcher import ResearchPlan


def fingerprint(root):
    return {
        str(path.relative_to(root)): digest(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_named_market_profile_and_frozen_closure_require_original_full_inputs(tmp_path):
    inputs, receipt = prepared_inputs(756)
    projection = project_market_target(inputs, receipt)
    plan = preparation.market_research_plan(projection)
    assert plan.development.data.train_cutoff == 721
    assert plan.development.data.seed == 0
    assert plan.development.data.instruments == 1
    assert plan.development.min_development_sessions == 6
    assert plan.development.evaluation_stride == 5
    assert plan.development.objective.model_dump(mode="json") == inputs["target"]["objective"]
    assert plan.neighbor_domain.caps == (3, 4, 5)
    store = ArtifactStore(tmp_path / "objects")
    preparation.store_market_projection(store, projection)
    grant = dict(
        exposure_mode="historical_market_exploration",
        market_evidence_ref=projection["market_evidence_ref"],
        research_plan=plan.model_dump(mode="json"),
        workspace=decode(receipt)["workspace"],
    )
    verified = preparation.verify_market_registration(
        grant, store, data_ref=projection["target_data_ref"]
    )
    assert canonical(verified["inputs"]) == canonical(inputs)
    assert verified["preparation_receipt"] == receipt
    assert set(verified["inputs"]["contexts"]) == {"industry", "market"}
    with pytest.raises(ValueError, match="source data differs"):
        preparation.verify_market_registration(grant, store, data_ref="f" * 64)
    with pytest.raises(ValueError):
        preparation.verify_market_registration(grant | {"workspace": "/another"}, store)
    changed = deepcopy(grant)
    changed["research_plan"]["policy"]["min_improvement"] = 1.0
    with pytest.raises(ValueError):
        preparation.verify_market_registration(changed, store)
    (store.root / projection["inputs_ref"]).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        preparation.verify_market_registration(grant, store)


def test_short_engineering_projection_is_not_a_production_market_profile():
    projection = project_market_target(*prepared_inputs(120))
    with pytest.raises(ValueError, match="756"):
        preparation.market_research_plan(projection)


@pytest.fixture
def offline_preparation(tmp_path, monkeypatch):
    roles = {"target": "SOXL", "industry": "XSD", "market": "SPY"}
    audit = audit_package(tmp_path / "source-audit", roles=roles, history=756, cash=False)
    calls = {"collect": 0, "ingest": 0}
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: NOW)

    def collect(output, *, roles):
        assert roles == {"target": "SOXL", "industry": "XSD", "market": "SPY"}
        calls["collect"] += 1
        shutil.copytree(audit, output, dirs_exist_ok=True)
        return Path(output)

    def ingest_child(command, **kwargs):
        assert command[-2].startswith("import sys;"), "unexpected model dispatch"
        calls["ingest"] += 1
        with monkeypatch.context() as child:
            child.setenv("QUANT_RSI_WORKSPACE", kwargs["env"]["QUANT_RSI_WORKSPACE"])
            context._ingest_prepared(command[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(collection, "collect_context", collect)
    monkeypatch.setattr(context.subprocess, "run", ingest_child)
    profile = ForecastWorkflowConfig.model_validate(
        dict(schema_version=1, kind="ForecastWorkflowConfig", **roles)
    ).model_dump(mode="json")
    report = preparation.prepare_market_campaign(
        profile,
        tmp_path / "delivery",
        campaign_id="recursive-soxl",
        source_campaign_id="recursive-soxl-source",
    )
    return profile, report, calls


def test_actual_export_initialization_and_frozen_root_binding_without_a_fit(
    offline_preparation, monkeypatch
):
    _, prepared, calls = offline_preparation
    workspace = Path(prepared["workspace"])
    assert prepared["grant_registered"] is False and prepared["model_training_performed"] is False
    assert not (workspace / ".quant_rsi/forecast_campaigns").exists()
    assert calls == {"collect": 1, "ingest": 1}
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    value = json.loads(Path(prepared["campaign_plan"]).read_text())
    grant_report = campaign.init_recursive_campaign(
        value, market_preparation=prepared["preparation"]
    )
    assert grant_report["grant"]["schema_version"] == 4
    assert grant_report["exposure"]["permitted_evidence"] == "historical_market_development"
    assert grant_report["exposure"]["predictive_validity_established"] is False
    plan = ResearchPlan.model_validate_json(canonical(value["research_plan"]))
    source = Path(value["source_directory"])
    initialized = research.create_research(
        plan, source, campaign_id=value["source_campaign_id"], initialize_only=True
    )
    assert initialized["initialized"] is True
    manifest = json.loads((source / "manifest.json").read_text())
    store = ArtifactStore(source / "artifacts")
    market_evidence = store.json(value["market_evidence_ref"])
    assert manifest["data_ref"] == market_evidence["target_data_ref"]
    assert store.json(market_evidence["inputs_ref"])["contexts"].keys() == {"industry", "market"}
    verify_root_manifest(plan, store, manifest)
    with pytest.raises(ValueError, match="source data differs"):
        verify_root_manifest(plan, store, manifest | {"data_ref": "f" * 64})

    def forbidden(*args, **kwargs):
        raise AssertionError("initialization replay cannot collect, import, or launch a worker")

    monkeypatch.setattr(collection, "collect_context", forbidden)
    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    before = fingerprint(workspace)
    result = research.inspect_research(source, replay=True)
    assert result["attempts_reserved"] == result["kernel_attempts"] == 0
    assert result["worker_seconds_charged"] == 0 and result["results"] == []
    assert fingerprint(workspace) == before

    original_inputs = store.root / market_evidence["inputs_ref"]
    retained = original_inputs.read_bytes()
    original_inputs.unlink()
    missing = fingerprint(workspace)
    with pytest.raises(FileNotFoundError):
        research.resume_research(source, steps=1)
    assert fingerprint(workspace) == missing
    original_inputs.write_bytes(retained)
    assert fingerprint(workspace) == before


def test_wrong_actual_export_rejects_before_campaign_authority(offline_preparation, monkeypatch):
    from quant_rsi.evaluation import local_research

    _, prepared, _ = offline_preparation
    workspace = Path(prepared["workspace"])
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    value = json.loads(Path(prepared["campaign_plan"]).read_text())
    export = local_research.export_local_data

    def wrong(spec):
        result = export(spec)
        result["specification"]["seed"] = 9
        return result

    monkeypatch.setattr(local_research, "export_local_data", wrong)
    before = fingerprint(workspace)
    with pytest.raises(ValueError, match="actual market source export"):
        campaign.init_recursive_campaign(value, market_preparation=prepared["preparation"])
    assert fingerprint(workspace) == before
    assert not (workspace / ".quant_rsi/forecast_campaigns").exists()


def test_preparation_profile_change_never_recollects(offline_preparation):
    profile, prepared, calls = offline_preparation
    output = Path(prepared["preparation"]).parent
    before = fingerprint(output)
    changed = profile | {"industry": "XLP"}
    with pytest.raises(ValueError, match="another configuration"):
        preparation.prepare_market_campaign(
            changed,
            output,
            campaign_id="recursive-soxl",
            source_campaign_id="recursive-soxl-source",
        )
    assert calls == {"collect": 1, "ingest": 1}
    assert fingerprint(output) == before
