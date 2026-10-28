"""Version and admission regressions without market ingestion or worker execution."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from test_task_definition import recursive_value

from quant_rsi import package_experiment
from quant_rsi.artifacts import ArtifactStore, canonical, digest
from quant_rsi.researcher import ResearchPlan
from quant_rsi.task_definition import verify_domain
from quant_rsi.trajectory_state import TrajectoryPlan


def trajectory_value():
    return dict(
        name="type-only recursive integration",
        source_directory="/unexecuted/research",
        source_checkpoint_ref="a" * 64,
        state_policy="reset",
        max_proposal_stages=2,
        budget=dict(worker_attempts=36, registered_worker_seconds=6480),
    )


def test_legacy_trajectory_keeps_exact_serialized_contract():
    value = trajectory_value()
    plan = TrajectoryPlan.model_validate_json(canonical(value))
    assert plan.model_dump(mode="json") == value
    for fields in (dict(schema_version=1), dict(max_generations=1), dict(schema_version=2)):
        with pytest.raises(ValidationError):
            TrajectoryPlan.model_validate_json(canonical(value | fields))


def test_recursive_trajectory_requires_explicit_generation_bound():
    value = trajectory_value() | dict(schema_version=3)
    with pytest.raises(ValidationError, match="explicit generations"):
        TrajectoryPlan.model_validate_json(canonical(value))
    for count in (1, 2):
        plan = TrajectoryPlan.model_validate_json(canonical(value | dict(max_generations=count)))
        assert plan.model_dump(mode="json") == value | dict(max_generations=count)
    with pytest.raises(ValidationError):
        TrajectoryPlan.model_validate_json(canonical(value | dict(max_generations=3)))


def test_task_requires_original_domain_and_forecast_profile_objects(tmp_path):
    plan = ResearchPlan.model_validate_json(canonical(recursive_value()))
    store = ArtifactStore(tmp_path / "artifacts")
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_domain(plan, store)
    store.put_json(plan.neighbor_domain.model_dump(mode="json"))
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_domain(plan, store)
    store.put_json(plan.forecast_profile.model_dump(mode="json"))
    verify_domain(plan, store)
    ref = digest(canonical(plan.forecast_profile.model_dump(mode="json")))
    (store.root / ref).write_bytes(b"{}")
    with pytest.raises(ValueError):
        verify_domain(plan, store)


def test_unmanaged_recursive_package_rejects_before_creating_child(tmp_path):
    plan = ResearchPlan.model_validate_json(canonical(recursive_value()))
    source = tmp_path / "source"
    store = ArtifactStore(source / "artifacts")
    plan_ref = store.put_json(plan.model_dump(mode="json"))
    (source / "manifest.json").write_bytes(canonical(dict(research_plan_ref=plan_ref)))
    child = tmp_path / "uncreated-package"
    with pytest.raises(ValueError, match="actual trajectory owner"):
        package_experiment.create_package(source, child, state_policy="reset")
    assert not child.exists()


def test_recursive_source_requires_shared_registration_before_input_access(tmp_path, monkeypatch):
    from quant_rsi import local_research, research

    workspace = tmp_path / "unregistered-workspace"
    workspace.mkdir()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    plan = ResearchPlan.model_validate_json(canonical(recursive_value()))
    root = workspace / "uncreated-source"

    def forbidden(*args, **kwargs):
        pytest.fail("unregistered root accessed data or dispatched work")

    monkeypatch.setattr(local_research, "export_local_data", forbidden)
    monkeypatch.setattr(research.controller, "execute_attempt", forbidden)
    with pytest.raises(ValueError, match="recursive|initialized|authority"):
        research.create_research(plan, root)
    assert not root.exists()


def test_root_allowance_preserves_full_paired_development_template(tmp_path, monkeypatch):
    from test_forecast_recursive_campaign import plan as campaign_value

    from quant_rsi.forecast_recursive_campaign import init_recursive_campaign
    from quant_rsi.forecast_recursive_registration import (
        effective_budget,
        require_root_identity,
        root_execution,
        verify_root_manifest,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    value = campaign_value(workspace)
    registration = init_recursive_campaign(value)
    plan = ResearchPlan.model_validate_json(canonical(value["research_plan"]))
    original = plan.model_dump(mode="json")
    source = workspace / "source"
    with root_execution(plan, source, None, None, create=True) as (campaign_id, fd):
        identity = require_root_identity(plan, source, campaign_id, fd)
        assert identity["binding"] == registration["binding"]
    store = ArtifactStore(tmp_path / "frozen")
    manifest = dict(
        campaign_id=value["source_campaign_id"],
        root_registration_identity_ref=store.put_json(identity),
    )
    verify_root_manifest(plan, store, manifest)
    budget = effective_budget(plan, store, manifest)
    assert budget["max_attempts"] == 4
    assert budget["max_worker_seconds"] == 720
    assert plan.development.budget.max_attempts == 8
    assert plan.model_dump(mode="json") == original
    assert effective_budget(plan, store, manifest | dict(launch_context_ref="managed")) == (
        plan.development.budget.model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="shared grant"):
        verify_root_manifest(plan, store, manifest | dict(campaign_id="copied-source"))


def test_c5_forecast_views_remain_rejected(tmp_path):
    from quant_rsi.research import authorize_managed

    plan = ResearchPlan.model_validate_json(canonical(recursive_value()))
    with pytest.raises(ValueError, match="C4/C5"):
        authorize_managed(
            tmp_path,
            plan,
            "source",
            {"trajectory_authorization_ref": "a" * 64},
            {"untrusted_bound_view": True},
            None,
            None,
        )


@pytest.mark.parametrize("cap", [3, 4])
def test_fixed_context_feedback_intervention_reaches_real_proposer_and_response(
    cap, tmp_path, monkeypatch
):
    """A test-only intervention after envelope validation, never an admitted history."""
    from test_forecast_recursive_changes import packet, plan_value

    from quant_rsi import changes
    from quant_rsi import forecast_recursive_changes as proposer
    from quant_rsi.researcher import package_definition

    plan = plan_value(cap=cap)
    args = packet(plan, tuple(float(x) for x in range(-cap, -1)))
    request_ref, request, evidence = args
    fixed_context = canonical(
        dict(
            request=request,
            evidence=evidence,
            plan=plan.model_dump(mode="json"),
            final_checkpoint=evidence["final_checkpoint"],
            runtime_source=proposer.proposer_definition(),
        )
    )
    store = ArtifactStore(tmp_path / "allowed-objects")
    manifest = dict(
        campaign_id="00000000000000000000000000000001",
        request_ref=store.put_json(request),
        evidence_ref=store.put_json(evidence),
    )
    original_predicate = proposer._supports_final_boundary
    original_validator = proposer.validate_registered_evidence
    validations, boundary_contexts = [], []

    def validate_original(request_value, evidence_value):
        validations.append((canonical(request_value), canonical(evidence_value)))
        return original_validator(request_value, evidence_value)

    monkeypatch.setattr(proposer, "validate_registered_evidence", validate_original)
    outcomes = []
    for intervention in ("support", "no_support"):

        def at_boundary(current_plan, feedback, intervention=intervention):
            assert package_definition(current_plan) == package_definition(plan)
            if (
                feedback.candidate is not None
                and feedback.selected == feedback.candidate
                and feedback.parent is not None
            ):
                # Keep the actual authenticated envelope/state immutable. Intervene only on
                # the incoming typed semantic projection, as a causal fixture in this test.
                value = feedback.model_dump(mode="json")
                boundary_contexts.append(canonical(value["parent"]))
                if intervention == "no_support":
                    value["candidate"]["score"] = value["parent"]["score"] - 1.0
                    value["selected"] = value["parent"]
                feedback = proposer._FinalFeedback.model_validate_json(canonical(value))
            return original_predicate(current_plan, feedback)

        monkeypatch.setattr(proposer, "_supports_final_boundary", at_boundary)
        response = changes.propose_change(*args)
        outcome, error = changes.validate_response(response, *args)
        assert error is None
        _, decision, _ = package_experiment.decision_content(store, manifest, plan, response)
        assert decision["outcome"] == outcome
        assert len(decision["grants"]) == (2 if outcome == "patch" else 0)
        outcomes.append(outcome)
        assert fixed_context == canonical(
            dict(
                request=request,
                evidence=evidence,
                plan=plan.model_dump(mode="json"),
                final_checkpoint=evidence["final_checkpoint"],
                runtime_source=proposer.proposer_definition(),
            )
        )
    assert outcomes == ["patch", "no_patch"]
    assert len(set(boundary_contexts)) == 1
    assert validations and all(
        item == (canonical(request), canonical(evidence)) for item in validations
    )


@pytest.mark.parametrize("cap", [3, 4])
def test_full_proposer_response_and_grants_ignore_protected_only_changes(
    cap, tmp_path, monkeypatch
):
    from test_forecast_recursive_changes import packet, plan_value

    from quant_rsi import changes
    from quant_rsi import forecast_recursive_changes as proposer

    plan = plan_value(cap=cap)
    args = packet(plan, tuple(float(x) for x in range(-cap, -1)))
    store = ArtifactStore(tmp_path / "allowed")
    manifest = dict(
        campaign_id="00000000000000000000000000000001",
        request_ref=store.put_json(args[1]),
        evidence_ref=store.put_json(args[2]),
    )
    response = changes.propose_change(*args)
    expected_decision = package_experiment.decision_content(store, manifest, plan, response)
    source_directory = Path(proposer.__file__).parent
    original_read = Path.read_bytes
    reads = []

    def permitted_read(path):
        reads.append(path)
        if path.parent not in (source_directory, store.root):
            pytest.fail("full decision path read protected or unrelated inputs")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", permitted_read)
    protected = tmp_path / "protected-only.txt"
    for value in ("held-out-score=-100", "held-out-score=100"):
        protected.write_text(value)
        actual = changes.propose_change(*args)
        assert actual == response
        assert changes.validate_response(actual, *args) == ("patch", None)
        assert (
            package_experiment.decision_content(store, manifest, plan, actual) == expected_decision
        )
    assert reads and protected not in reads


def test_two_policy_advances_preserve_complete_immediate_parent_closure(tmp_path):
    from test_forecast_recursive_changes import packet, plan_value

    from quant_rsi import changes
    from quant_rsi.researcher import package_definition

    store = ArtifactStore(tmp_path / "allowed")
    plan = plan_value()
    original_template = plan.development.model_dump(mode="json")
    store.put_json(plan.neighbor_domain.model_dump(mode="json"))
    store.put_json(plan.forecast_profile.model_dump(mode="json"))
    root_ref = store.put_json(package_definition(plan))
    parents, evidence_refs = [], []
    for cap in (3, 4):
        args = packet(plan, tuple(float(x) for x in range(-cap, -1)))
        response = changes.propose_change(*args)
        assert response["outcome"] == "patch"
        parent_ref = store.put_json(package_definition(plan))
        parents.append(parent_ref)
        store.put_json(changes.proposer_definition(plan))
        store.put_json(args[1])
        evidence_refs.append(store.put_json(args[2]))
        patch_ref = store.put_json(response["patch"])
        plan = changes.experimental_plan(plan, patch_ref, args[0], args[1])
        changes.validate_lineage(plan, store)
        assert plan.development.model_dump(mode="json") == original_template
    assert plan.policy.window_max == 5
    assert len(set(parents)) == 2
    assert plan.package_lineage.base_package_ref == parents[-1] != root_ref
    closure = changes.lineage_artifact_refs(plan, store)
    assert set(parents + evidence_refs).issubset(closure)
    stale = plan.model_dump(mode="json")
    stale["package_lineage"]["base_package_ref"] = root_ref
    with pytest.raises(ValueError, match="parent|lineage"):
        changes.validate_lineage(ResearchPlan.model_validate_json(canonical(stale)), store)
    (store.root / evidence_refs[0]).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        changes.validate_lineage(plan, store)
