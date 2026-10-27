"""Campaign primitives: real exports/protocol artifacts, no HTTP or fitted models."""

import fcntl
import os
import shutil
from types import SimpleNamespace

import pytest
from test_forecast_context_data import audit_package

from quant_rsi import controller, market_contracts
from quant_rsi import forecast_campaign as campaign
from quant_rsi import forecast_context_data as context
from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.forecast_inputs import ForecastInputs
from quant_rsi.forecast_protocol import freeze_protocol
from quant_rsi.journal import Journal

NOW = "2026-10-26T21:00:00+00:00"


def config(target="SOXL", industry="XSD"):
    return dict(
        kind="ForecastWorkflowConfig",
        schema_version=1,
        target=target,
        industry=industry,
        market="SPY",
    )


def plan(*, two=False):
    runs = [dict(instrument_id="US/SOXL", run_id="soxl-one", config=config())]
    if two:
        runs.append(dict(instrument_id="US/KO", run_id="ko-one", config=config("KO", "XLP")))
    return dict(
        kind="ForecastCampaign",
        schema_version=1,
        campaign_id="research-one",
        budget=dict(max_fit_reservations=10 * len(runs), max_worker_seconds=1800 * len(runs)),
        runs=runs,
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    path = tmp_path / "workspace"
    path.mkdir()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(path))
    return path


def files(path):
    return {
        str(p.relative_to(path)): digest(p.read_bytes()) for p in path.rglob("*") if p.is_file()
    }


@pytest.fixture(scope="module")
def source_bundle(tmp_path_factory):
    """Generate actual adapter exports and a default frozen protocol exactly once."""
    base = tmp_path_factory.mktemp("campaign-source")
    audit = audit_package(base / "audit-source", history=756, cash=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(market_contracts, "trusted_now", lambda: NOW)

        def ingest_child(command, **kwargs):
            assert command[-2].startswith("import sys;"), "model subprocess is forbidden"
            with pytest.MonkeyPatch.context() as child:
                child.setenv("QUANT_RSI_WORKSPACE", kwargs["env"]["QUANT_RSI_WORKSPACE"])
                context._ingest_prepared(command[-1])
            return SimpleNamespace(returncode=0)

        patch.setattr(context.subprocess, "run", ingest_child)
        prepared = context.prepare_context_data(audit, base / "prepared")
    inputs = ForecastInputs(prepared["inputs"])
    store = ArtifactStore(base / "artifacts")
    assert store.put_json(inputs.value) == inputs.ref
    frozen = freeze_protocol(inputs, store, publication_asof=NOW)
    plan_ref = store.put_json(frozen.model_dump(mode="json"))
    return dict(
        base=base, audit=audit, prepared=prepared, inputs=inputs, plan=frozen, plan_ref=plan_ref
    )


def initialize_children(workspace, scope, bundle):
    """Build zero-dispatch child identities with the real frozen source/plan."""
    base = workspace / ".quant_rsi"
    workflow = base / "enhanced_forecast_workflows" / scope.binding["run_id"]
    workflow.mkdir(parents=True)
    manifest = dict(
        kind="EnhancedForecastWorkflow",
        schema_version=1,
        workflow_id=scope.binding["run_id"],
        workspace=str(workspace),
        config=scope.config,
        config_ref=digest(canonical(scope.config)),
        runtime=controller.runtime_identity(),
        created_at=NOW,
        managed_run=scope.binding,
    )
    atomic_write(workflow / "manifest.json", canonical(manifest))
    journal = Journal(workflow, digest(canonical(manifest)))
    journal.close()
    campaign.bind_workflow(scope, manifest)
    prepared = bundle["base"] / "prepared"
    frozen_inputs = dict(
        input_ref=bundle["inputs"].ref,
        preparation_refs={
            name: digest((prepared / name).read_bytes())
            for name in (
                "prepared.json",
                "intent.json",
                "registry.identity.json",
                "prepared.binding.json",
            )
        },
        receipt_ref=digest((prepared / "preparation.receipt.json").read_bytes()),
    )
    phases = [
        ("collection_started", {"config_ref": manifest["config_ref"]}),
        (
            "collection_committed",
            {"manifest_ref": digest((bundle["audit"] / "manifest.json").read_bytes())},
        ),
        (
            "preparation_started",
            {"manifest_ref": digest((bundle["audit"] / "manifest.json").read_bytes())},
        ),
        ("inputs_committed", frozen_inputs),
    ]
    journal = Journal(workflow)
    with journal.connection:
        for kind, payload in phases:
            journal.event(None, kind, payload)
    journal.close()
    experiment = base / "forecast_experiments" / scope.binding["run_id"]
    experiment.mkdir(parents=True)
    shutil.copytree(bundle["base"] / "artifacts", experiment / "artifacts")
    store = ArtifactStore(experiment / "artifacts")
    manifest = dict(
        kind="ForecastComparisonExperiment",
        schema_version=1,
        experiment_id=scope.binding["run_id"],
        data_ref=bundle["inputs"].ref,
        plan_ref=bundle["plan_ref"],
        protocol_ref=bundle["plan"].protocol_ref,
        runtime_ref=store.put_json(controller.runtime_identity()),
        managed_run=scope.binding,
    )
    atomic_write(experiment / "manifest.json", canonical(manifest))
    Journal(experiment, digest(canonical(manifest))).close()
    return experiment, store, manifest, bundle["plan"]


def test_atomic_roster_idempotency_and_derived_zero_cost(workspace):
    report = campaign.init_campaign(plan(two=True))
    assert report["allocated"] == {"max_attempts": 20, "max_worker_seconds": 3600}
    assert report["actual"] == {
        "fit_reservations": 0,
        "reserved_worker_seconds": 0,
        "charged_fits": 0,
    }
    assert {r["instrument_id"] for r in report["runs"]} == {"US/SOXL", "US/KO"}
    assert report["exposure"] == campaign.EXPOSURE
    before = files(workspace)
    assert campaign.init_campaign(plan(two=True)) == report
    assert campaign.inspect_campaign("research-one") == report
    assert files(workspace) == before
    journal = Journal(workspace / ".quant_rsi/forecast_campaigns")
    try:
        assert not journal.attempts()
        assert [e["kind"] for e in journal.events()] == ["run_created", "campaign_registered"]
    finally:
        journal.close()


@pytest.mark.parametrize("change", ["budget", "roster", "config"])
def test_registered_campaign_change_rejects_without_mutation(workspace, change):
    campaign.init_campaign(plan())
    changed = plan()
    if change == "budget":
        changed["budget"]["max_worker_seconds"] += 1800
    elif change == "roster":
        changed["runs"][0]["run_id"] = "replacement"
    else:
        changed["runs"][0]["config"]["industry"] = "SMH"
    before = files(workspace)
    with pytest.raises(ValueError, match="immutable plan"):
        campaign.init_campaign(changed)
    assert files(workspace) == before


def test_global_run_id_and_insufficient_grant(workspace):
    insufficient = plan(two=True)
    insufficient["budget"]["max_fit_reservations"] = 19
    with pytest.raises(ValueError, match="complete child roster"):
        campaign.init_campaign(insufficient)
    assert not (workspace / ".quant_rsi").exists()
    campaign.init_campaign(plan())
    changed = plan()
    changed["campaign_id"] = "second-grant"
    before = files(workspace)
    with pytest.raises(ValueError, match="already allocated"):
        campaign.init_campaign(changed)
    assert files(workspace) == before


@pytest.mark.parametrize("directory", ["forecast_experiments", "enhanced_forecast_workflows"])
def test_bootstrap_does_not_adopt_existing_legacy_children(workspace, directory):
    (workspace / ".quant_rsi" / directory).mkdir(parents=True)
    with pytest.raises(ValueError, match="preexisting"):
        campaign.init_campaign(plan())
    assert not (workspace / ".quant_rsi/forecast_campaign_identity.json").exists()


def test_legacy_unknown_managed_and_required_boundaries(workspace):
    with campaign.run_scope("legacy") as scope:
        assert scope is None
    with pytest.raises(ValueError, match="requires a registered"):
        with campaign.run_scope("legacy", required=True):
            pass
    campaign.init_campaign(plan())
    with pytest.raises(ValueError, match="not registered"):
        with campaign.run_scope("unallocated"):
            pass
    with campaign.run_scope("soxl-one", required=True) as scope:
        with campaign.run_scope("soxl-one") as nested:
            assert nested == scope
        with pytest.raises(ValueError, match="managed_run"):
            with campaign.run_scope("soxl-one", manifest={"kind": "unbound"}):
                pass


def test_ready_anchor_rejects_lost_parent_and_relocation(workspace, monkeypatch, tmp_path):
    report = campaign.init_campaign(plan())
    binding = report["runs"][0]["binding"]
    cloned = tmp_path / "cloned"
    shutil.copytree(workspace, cloned)
    with monkeypatch.context() as shifted:
        shifted.setenv("QUANT_RSI_WORKSPACE", str(cloned))
        with pytest.raises(ValueError, match="identity mismatch"):
            with campaign.run_scope("soxl-one"):
                pass
    shutil.rmtree(workspace / ".quant_rsi/forecast_campaigns")
    with pytest.raises(ValueError, match="missing"):
        with campaign.run_scope("soxl-one", manifest={"managed_run": binding}):
            pass
    with pytest.raises(ValueError):
        campaign.init_campaign(plan())


def test_managed_child_cannot_fall_back_after_parent_loss(workspace):
    fake = workspace / ".quant_rsi/forecast_experiments/copied"
    fake.mkdir(parents=True)
    atomic_write(fake / "manifest.json", canonical({"managed_run": {"authority_ref": "old"}}))
    with pytest.raises(ValueError, match="missing its original parent"):
        with campaign.run_scope("copied"):
            pass


def test_scope_rejects_changed_workspace_and_wrong_fd(workspace, monkeypatch, tmp_path):
    campaign.init_campaign(plan())
    other = tmp_path / "other"
    other.mkdir()
    with campaign.run_scope("soxl-one") as scope:
        with monkeypatch.context() as moved:
            moved.setenv("QUANT_RSI_WORKSPACE", str(other))
            with pytest.raises(ValueError, match="workspace or process"):
                with campaign.run_scope("soxl-one"):
                    pass
        path = workspace / ".quant_rsi/.forecast_campaign.lock"
        fd = os.open(path, os.O_RDWR)
        try:
            alternate = campaign.ManagedRun(scope.binding, scope.config, scope.allocation, fd)
            with pytest.raises(ValueError, match="owner differs"):
                campaign._scope_record(alternate)
        finally:
            os.close(fd)


def test_original_ofd_witness_rejects_close_and_recycle(workspace):
    campaign.init_campaign(plan())
    recycled = None
    with campaign.run_scope("soxl-one") as scope:
        old_fd = scope.owner_fd
        os.close(old_fd)
        with pytest.raises(ValueError, match="owner descriptor"):
            with campaign.run_scope("soxl-one"):
                pass
        recycled = os.open(workspace / ".quant_rsi/.forecast_campaign.lock", os.O_RDWR)
        assert recycled == old_fd
        with pytest.raises(BlockingIOError):
            fcntl.flock(recycled, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="actual owner"):
            with campaign.run_scope("soxl-one"):
                pass
    os.close(recycled)


def test_initializing_anchor_recovers_only_zero_bootstrap(workspace, monkeypatch):
    original = campaign.atomic_write
    tripped = False

    def stop(path, body):
        nonlocal tripped
        if path.name == "identity.json" and not tripped:
            tripped = True
            raise RuntimeError("before parent identity publication")
        original(path, body)

    monkeypatch.setattr(campaign, "atomic_write", stop)
    with pytest.raises(RuntimeError):
        campaign.init_campaign(plan())
    anchor = decode((workspace / ".quant_rsi/forecast_campaign_identity.json").read_bytes())
    assert anchor["phase"] == "initializing"
    with pytest.raises(ValueError):
        with campaign.run_scope("soxl-one"):
            pass
    monkeypatch.setattr(campaign, "atomic_write", original)
    report = campaign.init_campaign(plan())
    ready = decode((workspace / ".quant_rsi/forecast_campaign_identity.json").read_bytes())
    assert ready["identity"] == anchor["identity"] and ready["phase"] == "ready"
    assert report["actual"]["fit_reservations"] == 0


def test_real_export_bindings_ack_before_dispatch_and_original_costs(workspace, source_bundle):
    campaign.init_campaign(plan())
    with campaign.run_scope("soxl-one") as scope:
        root, store, manifest, frozen = initialize_children(workspace, scope, source_bundle)
        campaign.verify_experiment(scope, manifest, frozen)
        with pytest.raises(ValueError, match="parent experiment acknowledgement"):
            campaign.verify_experiment(scope, manifest, frozen, for_dispatch=True)
        campaign.bind_experiment(scope, manifest, frozen)
        campaign.verify_experiment(scope, manifest, frozen, for_dispatch=True)
        before = files(workspace)
        campaign.bind_experiment(scope, manifest, frozen)
        assert files(workspace) == before
        journal = Journal(root)
        try:
            proposal = frozen.proposals[0]
            journal.reserve(0, proposal.name, store.put_json(proposal.model_dump(mode="json")), 10)
            journal.start(0, 180, 1800)
            journal.finish(
                0,
                status="interrupted",
                result_ref=None,
                error="unit interruption",
                elapsed=0.0,
            )
        finally:
            journal.close()
    report = campaign.inspect_campaign("research-one")
    assert report["actual"] == {
        "fit_reservations": 1,
        "reserved_worker_seconds": 180,
        "charged_fits": 1,
    }
    assert report["allocated"] == {"max_attempts": 10, "max_worker_seconds": 1800}
    before = files(workspace)
    assert campaign.inspect_campaign("research-one") == report
    assert files(workspace) == before


def test_dispatched_unacknowledged_child_cannot_be_adopted(workspace, source_bundle):
    campaign.init_campaign(plan())
    with campaign.run_scope("soxl-one") as scope:
        root, store, manifest, frozen = initialize_children(workspace, scope, source_bundle)
        journal = Journal(root)
        try:
            proposal = frozen.proposals[0]
            journal.reserve(0, proposal.name, store.put_json(proposal.model_dump(mode="json")), 10)
            journal.start(0, 180, 1800)
        finally:
            journal.close()
        with pytest.raises(ValueError, match="unacknowledged experiment has"):
            campaign.bind_experiment(scope, manifest, frozen)
    with pytest.raises(ValueError, match="unacknowledged experiment has"):
        campaign.inspect_campaign("research-one")


def test_missing_acknowledged_child_is_not_repaired(workspace, source_bundle):
    campaign.init_campaign(plan())
    with campaign.run_scope("soxl-one") as scope:
        root, _, manifest, frozen = initialize_children(workspace, scope, source_bundle)
        campaign.bind_experiment(scope, manifest, frozen)
    (root / "journal.sqlite").unlink()
    before = files(workspace)
    with pytest.raises(ValueError, match="no initialized journal"):
        campaign.inspect_campaign("research-one")
    assert files(workspace) == before


def test_new_grant_cannot_adopt_an_existing_unregistered_directory(workspace):
    campaign.init_campaign(plan())
    rogue = workspace / ".quant_rsi/forecast_experiments/renamed-copy"
    rogue.mkdir(parents=True)
    changed = plan()
    changed["campaign_id"] = "second-grant"
    changed["runs"][0]["run_id"] = "renamed-copy"
    before = files(workspace)
    with pytest.raises(ValueError, match="unregistered child directory"):
        campaign.init_campaign(changed)
    assert files(workspace) == before


def test_workflow_ack_cannot_adopt_collection_activity(workspace):
    campaign.init_campaign(plan())
    with campaign.run_scope("soxl-one") as scope:
        root = workspace / ".quant_rsi/enhanced_forecast_workflows/soxl-one"
        root.mkdir(parents=True)
        manifest = dict(
            kind="EnhancedForecastWorkflow",
            schema_version=1,
            workflow_id="soxl-one",
            workspace=str(workspace),
            config=scope.config,
            config_ref=digest(canonical(scope.config)),
            runtime=controller.runtime_identity(),
            created_at=NOW,
            managed_run=scope.binding,
        )
        atomic_write(root / "manifest.json", canonical(manifest))
        Journal(root, digest(canonical(manifest))).close()
        (root / "collection").mkdir()
        with pytest.raises(ValueError, match="unacknowledged workflow has"):
            campaign.bind_workflow(scope, manifest)


@pytest.mark.parametrize(
    "member",
    [
        "forecast_campaign_identity.json",
        "forecast_campaigns/identity.json",
        "forecast_campaigns/journal.sqlite",
    ],
)
def test_nested_scope_rechecks_persisted_parent_authority(workspace, member):
    campaign.init_campaign(plan())
    with campaign.run_scope("soxl-one"):
        (workspace / ".quant_rsi" / member).unlink()
        with pytest.raises(ValueError):
            with campaign.run_scope("soxl-one"):
                pass


def test_legacy_nested_scope_retains_lock_and_cannot_initialize(workspace):
    with campaign.run_scope("legacy") as scope:
        assert scope is None
        owner = campaign._ACTIVE.get()
        with campaign.run_scope("legacy") as nested:
            assert nested is None
            assert campaign._ACTIVE.get() is owner
        with (workspace / ".quant_rsi/.forecast_campaign.lock").open("r+") as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active legacy scope"):
            campaign.init_campaign(plan())
        os.close(owner.owner_fd)
        with pytest.raises(ValueError, match="owner descriptor"):
            with campaign.run_scope("legacy"):
                pass
