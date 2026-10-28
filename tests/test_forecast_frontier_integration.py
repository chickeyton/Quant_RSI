"""No-fit policy dispatch and immutable grant integration checks."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_forecast_frontier_policy import bind, frontier_packet
from test_forecast_recursive_campaign import plan as legacy_plan
from test_forecast_recursive_changes import packet, plan_value

from quant_rsi import changes, package_experiment
from quant_rsi import forecast_frontier_policy as frontier
from quant_rsi import forecast_recursive_campaign as campaign
from quant_rsi import forecast_recursive_changes as legacy
from quant_rsi.artifacts import ArtifactStore, canonical, digest
from quant_rsi.forecast_recursive_registration import identity_definition
from quant_rsi.forecast_recursive_trajectory import authorization_definition
from quant_rsi.researcher import ResearchPlan, package_definition


def plan4(workspace):
    return legacy_plan(workspace) | dict(
        schema_version=4,
        proposer_policy=frontier.POLICY,
        proposer_definition=frontier.proposer_definition(),
        exposure_mode="engineering_fixture",
        market_evidence_ref=None,
    )


@pytest.mark.parametrize(
    "cap,scores",
    [
        (3, (-4.0, -3.0, -2.0, -3.5)),
        (4, (-5.0, -4.0, -3.0, -2.0, -3.5)),
    ],
)
def test_fixed_context_typed_feedback_controls_real_proposer_action_and_response(
    cap, scores, tmp_path, monkeypatch
):
    """Intervene only on the typed projection, never on the admitted evidence/receipts."""
    plan = plan_value(cap=cap)
    args = frontier_packet(plan, scores)
    request_ref, request, evidence = args
    fixed = canonical(
        dict(
            request=request,
            evidence=evidence,
            plan=plan.model_dump(mode="json"),
            checkpoint=evidence["final_checkpoint"],
            runtime=frontier.proposer_definition(),
        )
    )
    store = ArtifactStore(tmp_path / "objects")
    manifest = dict(
        campaign_id="00000000000000000000000000000001",
        request_ref=store.put_json(request),
        evidence_ref=store.put_json(evidence),
    )
    predicate = frontier._supports_terminal_frontier
    validator = frontier.validate_registered_evidence
    validations, observations, outcomes = [], [], []

    def original_validator(current_request, current_evidence):
        validations.append((canonical(current_request), canonical(current_evidence)))
        return validator(current_request, current_evidence)

    monkeypatch.setattr(frontier, "validate_registered_evidence", original_validator)
    for intervention in ("support", "no_support"):

        def at_boundary(current_plan, feedback, intervention=intervention):
            assert package_definition(current_plan) == package_definition(plan)
            value = feedback.model_dump(mode="json")
            observations.append(canonical(value))
            if intervention == "no_support":
                value["comparison"]["score"] = value["selected"]["score"]
            changed = frontier._FrontierFeedback.model_validate_json(canonical(value))
            return predicate(current_plan, changed)

        monkeypatch.setattr(frontier, "_supports_terminal_frontier", at_boundary)
        response = changes.propose_change(*args)
        outcome, error = changes.validate_response(response, *args)
        assert error is None
        _, decision, _ = package_experiment.decision_content(store, manifest, plan, response)
        assert decision["outcome"] == outcome
        assert len(decision["grants"]) == (2 if outcome == "patch" else 0)
        outcomes.append(outcome)
        assert fixed == canonical(
            dict(
                request=request,
                evidence=evidence,
                plan=plan.model_dump(mode="json"),
                checkpoint=evidence["final_checkpoint"],
                runtime=frontier.proposer_definition(),
            )
        )
    assert outcomes == ["patch", "no_patch"]
    assert len(set(observations)) == 1
    assert validations and all(
        item == (canonical(request), canonical(evidence)) for item in validations
    )


@pytest.mark.parametrize(
    "cap,scores",
    [
        (3, (-4.0, -3.0, -2.0, -3.5)),
        (4, (-5.0, -4.0, -3.0, -2.0, -3.5)),
    ],
)
def test_protected_only_changes_do_not_enter_either_transition(cap, scores, tmp_path, monkeypatch):
    args = frontier_packet(plan_value(cap=cap), scores)
    protected = tmp_path / "protected.json"
    protected.write_text('{"score":100}')
    expected = changes.propose_change(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure proposal must not read protected/artifact evidence")

    monkeypatch.setattr(ArtifactStore, "get", forbidden)
    protected.write_text('{"score":-100,"selected":"different"}')
    assert changes.propose_change(*args) == expected
    assert changes.validate_response(expected, *args) == ("patch", None)


def test_explicit_frontier_dispatch_preserves_legacy_decision():
    plan = plan_value()
    old = packet(plan, (-4.0, -3.0, -2.0, -3.5))
    before = canonical(old)
    assert changes.proposer_definition(plan) == legacy.proposer_definition()
    assert changes.propose_change(*old) == legacy.expected_change(*old)
    assert changes.propose_change(*old)["outcome"] == "no_patch"
    new = frontier_packet(plan)
    response = changes.propose_change(*new)
    assert response["patch"]["schema_version"] == 4
    assert changes.validate_response(response, *new) == ("patch", None)
    assert canonical(old) == before


@pytest.mark.parametrize("mutation", ["missing_policy", "missing_descriptor", "unknown", "legacy"])
def test_missing_or_hybrid_selector_cannot_fall_back_to_legacy(mutation):
    _, request, evidence = frontier_packet(plan_value())
    evidence = deepcopy(evidence)
    if mutation == "missing_policy":
        del evidence["proposer_policy"]
    elif mutation == "missing_descriptor":
        del evidence["proposer_definition"]
    elif mutation == "unknown":
        evidence["proposer_policy"] = "unregistered"
    else:
        evidence["proposer_definition"] = legacy.proposer_definition()
    args = bind(evidence, request["base_package_ref"])
    with pytest.raises(ValueError):
        changes.propose_change(*args)


def test_removed_both_selector_fields_does_not_reinterpret_frontier_request():
    _, request, evidence = frontier_packet(plan_value())
    evidence = {
        key: value
        for key, value in evidence.items()
        if key not in {"proposer_policy", "proposer_definition"}
    }
    request = request | {"evidence_ref": digest(canonical(evidence))}
    with pytest.raises(ValueError, match="proposer descriptor"):
        changes.propose_change(digest(canonical(request)), request, evidence)


def test_schema4_grant_binds_policy_with_original_eight_field_root(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    value = plan4(tmp_path)
    report = campaign.init_recursive_campaign(value)
    grant, binding = report["grant"], report["binding"]
    assert grant["schema_version"] == 4
    assert grant["proposer_definition"] == frontier.proposer_definition()
    assert len(binding) == 8
    assert not (tmp_path / "source").exists()
    research_plan = ResearchPlan.model_validate_json(canonical(grant["research_plan"]))
    registration = identity_definition(research_plan, grant["source_campaign_id"], binding, grant)
    assert registration["grant"] == grant and registration["binding"] == binding
    assert campaign.init_recursive_campaign(value) == report
    legacy = campaign._normalize(legacy_plan(tmp_path))
    assert campaign._normalize(legacy) == legacy
    for key in ("proposer_policy", "proposer_definition", "exposure_mode", "market_evidence_ref"):
        assert key not in campaign._normalize(legacy_plan(tmp_path))


def test_stage_authorization_uses_grant_selector_and_rejects_program_drift(tmp_path):
    value = campaign._normalize(plan4(tmp_path))
    store = ArtifactStore(tmp_path / "objects")
    plan = ResearchPlan.model_validate_json(canonical(value["research_plan"]))
    receipt = dict(
        manifest_ref=store.put_json({"task_ref": "1" * 64}),
        research_plan_ref=store.put_json(plan.model_dump(mode="json")),
        events_ref="2" * 64,
        feedback_watermark=4,
        files_ref="3" * 64,
        data_ref="4" * 64,
    )
    head = {"research_receipt_ref": store.put_json(receipt), "package_ref": "5" * 64}
    policy = campaign.policy_binding(value)
    program = dict(
        policy,
        root_registration_ref=store.put_json({"identity": {"grant": value}}),
        authority_root=str(tmp_path),
    )
    authority = SimpleNamespace(
        store=store,
        program=program,
        state={"authority_id": "0" * 31 + "1", "program_ref": "6" * 64},
    )
    authorization = authorization_definition(authority, 0, head, None)
    assert {key: authorization[key] for key in policy} == policy
    assert (
        changes.proposer_definition(plan, authorization=authorization)
        == frontier.proposer_definition()
    )
    del program["proposer_policy"]
    with pytest.raises(ValueError, match="stage proposer"):
        authorization_definition(authority, 0, head, None)


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 3},
        {"proposer_policy": "other"},
        {"proposer_definition": {}},
        {"exposure_mode": "historical_market_exploration"},
        {"market_evidence_ref": "f" * 64},
    ],
)
def test_unregistered_policy_or_exposure_hybrid_rejected_before_authority(tmp_path, mutation):
    with pytest.raises(ValueError):
        campaign._normalize(plan4(tmp_path) | mutation)
    assert list(tmp_path.iterdir()) == []
