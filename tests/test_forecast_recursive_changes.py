"""Pure controlled feedback, never worker results or authenticated business receipts."""

from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_task_definition import domain_value, forecast_value

from quant_rsi.core.artifacts import ArtifactStore, canonical, digest
from quant_rsi.forecast.forecast_contracts import ForecastDevelopmentPlan
from quant_rsi.forecast.recursive.forecast_recursive_changes import (
    RegisteredCapPatch,
    ResetOpportunity,
    _FinalFeedback,
    _supports_final_boundary,
    expected_change,
    extend_capsule,
    proposer_definition,
    reset_opportunity,
    validate_registered_evidence,
)
from quant_rsi.research.changes import expected_change as dispatched_change
from quant_rsi.research.changes import request_definition
from quant_rsi.research.researcher import (
    ResearchPlan,
    advance,
    initial_state,
    next_factor,
    package_definition,
)
from quant_rsi.research.task_definition import (
    ForecastRecursiveProfile,
    forecast_profile_ref,
    task_definition,
)


def ref(label):
    return digest(label.encode())


def plan_value(
    *,
    cap=3,
    upper=5,
    cutoff=30,
    instruments=2,
    attempts=8,
    seconds=160,
    threshold=0.0,
    lineage=None,
):
    value = forecast_value(cutoff=cutoff, window=2, instruments=instruments)
    value["policy"].update(window_max=cap, min_improvement=threshold)
    value["development"]["budget"].update(max_attempts=attempts, max_worker_seconds=seconds)
    development = ForecastDevelopmentPlan.model_validate_json(canonical(value["development"]))
    value.update(
        schema_version=3,
        neighbor_domain=domain_value(initial=3, upper=upper),
        forecast_profile=ForecastRecursiveProfile.from_development(development).model_dump(
            mode="json"
        ),
    )
    if cap != 3:
        value["package_lineage"] = lineage or dict(
            base_package_ref=ref("unit parent"),
            patch_ref=ref("unit patch"),
            proposer_ref=ref("unit proposer"),
            request_ref=ref("unit request"),
            evidence_ref=ref("unit evidence"),
            state_policy="reset",
        )
    return ResearchPlan.model_validate_json(canonical(value))


def packet(plan, scores=(-3.0, -2.0), *, proposal_failures=()):
    """Build internally consistent *unit* records through the real pure state functions."""
    package_ref = digest(canonical(package_definition(plan)))
    data_ref = ref("unimported unit data")
    task_ref = digest(
        canonical(
            task_definition(
                plan, data_ref, digest(canonical(plan.development.model_dump(mode="json")))
            )
        )
    )
    state = initial_state("controlled-unit-source", package_ref, ref("memory0"))
    entries, parents, requests, attempts = [], {}, [], []
    for ordinal, score in enumerate(scores):
        factor = next_factor(plan, state)
        assert factor is not None
        parent_ref = digest(canonical(state))
        observation = dict(
            campaign_id=state["campaign_id"],
            ordinal=ordinal,
            task_ref=task_ref,
            visibility="development",
            factor=factor,
            score=score,
            eligible=score is not None,
            status="completed",
            request_ref=ref(f"request{ordinal}"),
            proposal_ref=ref(f"proposal{ordinal}"),
            result_ref=ref(f"result{ordinal}"),
        )
        if ordinal in proposal_failures:
            observation.update(
                factor=None,
                score=None,
                eligible=False,
                status="proposal_failed",
                proposal_ref=None,
                result_ref=None,
            )
        observation_ref = digest(canonical(observation))
        parents[parent_ref] = deepcopy(state)
        entries.append(
            dict(
                observation=observation,
                observation_ref=observation_ref,
                observation_event_seq=ordinal + 1,
                parent_checkpoint_ref=parent_ref,
                parent_incumbent=deepcopy(state["incumbent"]),
            )
        )
        requests.append(
            dict(
                ordinal=ordinal,
                request_ref=observation["request_ref"],
                parent_ref=parent_ref,
                observation_ref=observation_ref,
                proposal_ref=observation["proposal_ref"],
            )
        )
        if ordinal not in proposal_failures:
            attempts.append(
                dict(
                    ordinal=ordinal,
                    proposal_ref=observation["proposal_ref"],
                    status=observation["status"],
                    result_ref=observation["result_ref"],
                    worker_seconds=20,
                )
            )
        state = advance(plan, state, observation, ref(f"memory{ordinal + 1}"), parent_ref)
    evidence = extend_capsule(
        dict(
            kind="DevelopmentEvidence",
            source_campaign_id=state["campaign_id"],
            task_ref=task_ref,
            feedback_watermark=len(entries),
            entries=entries,
            historical_attempts=len(requests),
            historical_worker_slots=sum(a["worker_seconds"] for a in attempts),
        ),
        plan,
        digest(canonical(state)),
        state,
        parents,
        dict(
            profile="registered_forecast_cap_v3",
            source_receipt_ref=ref("source receipt"),
            source_head={"selection_ref": digest(canonical(state["incumbent"]))},
            expected_revision=0,
            source_cas_ref=ref("source cas"),
            development_data_ref=data_ref,
        ),
        requests,
        attempts,
    )
    return rebound(evidence, package_ref)


def rebound(evidence, package_ref):
    """Unit envelope construction, not production registration or receipt replacement."""
    request = request_definition(
        "unit-change",
        package_ref,
        digest(canonical(proposer_definition())),
        digest(canonical(evidence)),
        evidence,
    )
    return digest(canonical(request)), request, evidence


def test_two_controlled_transitions_use_the_current_package_and_final_trace():
    first_plan = plan_value()
    first_packet = packet(first_plan)
    first = dispatched_change(*first_packet)
    patch = RegisteredCapPatch.model_validate_json(canonical(first["patch"]))
    assert first["outcome"] == "patch" and patch.value == 4 and patch.schema_version == 3
    assert patch.forecast_recursive_profile_ref == forecast_profile_ref(first_plan)
    assert patch.reset_opportunity.new_factor == {"op": "momentum", "window": 4}
    lineage = dict(
        base_package_ref=patch.base_package_ref,
        patch_ref=digest(canonical(first["patch"])),
        proposer_ref=patch.proposer_ref,
        request_ref=patch.request_ref,
        evidence_ref=patch.evidence_ref,
        state_policy="reset",
    )
    second_plan = plan_value(cap=4, lineage=lineage)
    second_packet = packet(second_plan, (-3.0, -2.0, -1.0))
    second = expected_change(*second_packet)
    assert second["patch"]["value"] == 5
    assert second["patch"]["base_package_ref"] != first["patch"]["base_package_ref"]
    assert (
        second["patch"]["selected_observation_ref"]
        == (second_packet[2]["entries"][-1]["observation_ref"])
    )
    assert second["patch"]["reset_opportunity"]["required_attempts"] == 4
    assert second["patch"]["reset_opportunity"]["required_registered_seconds"] == 80


@pytest.mark.parametrize("cap", [3, 4])
def test_private_counterfactual_boundary_uses_valid_typed_allowed_feedback(cap):
    """No alternative journal/receipt is invented for the fixed checkpoint background."""
    plan = plan_value(cap=cap)
    parent = dict(
        factor={"op": "momentum", "window": cap - 1},
        score=-3.0,
        result_ref=ref("parent result"),
        proposal_ref=ref("parent proposal"),
        ordinal=0,
    )
    background = canonical({"package": package_definition(plan), "checkpoint": ref("fixed state")})
    decisions = []
    for factor in [{"op": "momentum", "window": cap}, {"op": "mean_return", "window": cap - 1}]:
        winner = dict(
            factor=factor,
            score=-2.0,
            result_ref=ref("winner result"),
            proposal_ref=ref("winner proposal"),
            ordinal=1,
        )
        feedback = _FinalFeedback.model_validate_json(
            canonical(
                dict(
                    selected=winner,
                    candidate=winner,
                    parent=parent,
                    completed=True,
                    eligible=True,
                )
            )
        )
        decisions.append(_supports_final_boundary(plan, feedback))
        assert background == canonical(
            {"package": package_definition(plan), "checkpoint": ref("fixed state")}
        )
    assert decisions == [True, False]


def test_protected_only_changes_cannot_affect_real_response_or_trigger_reads(monkeypatch, tmp_path):
    args = packet(plan_value())
    baseline = expected_change(*args)
    source_dir = Path(
        __import__("quant_rsi.research.researcher", fromlist=["__file__"]).__file__
    ).parent
    original = Path.read_bytes
    reads = []

    def source_only(path):
        reads.append(path)
        if path.parent != source_dir or path.suffix != ".py":
            raise AssertionError("unexpected data/protected file read")
        return original(path)

    def no_store_reads(*args, **kwargs):
        raise AssertionError("proposer has no protected/artifact-store read capability")

    monkeypatch.setattr(Path, "read_bytes", source_only)
    monkeypatch.setattr(ArtifactStore, "json", no_store_reads)
    protected = tmp_path / "protected-only.txt"
    for secret in ["unavailable-score=-100", "unavailable-score=100"]:
        protected.write_text(secret)
        assert expected_change(*args) == baseline
    assert reads and protected not in reads


@pytest.mark.parametrize("scores", [(-3.0,), (-4.0, -3.0, -2.0)])
def test_baseline_or_stale_boundary_improvement_is_not_final_support(scores):
    result = expected_change(*packet(plan_value(), scores))
    assert result["outcome"] == "no_patch"
    assert result["reason"] == "unsupported_final_boundary_feedback"


def test_unresolved_later_observation_does_not_replace_actual_selected_evidence():
    args = packet(plan_value(), (-3.0, -2.0, None))
    result = expected_change(*args)
    assert result["outcome"] == "patch"
    assert result["patch"]["selected_observation_ref"] == args[2]["entries"][1]["observation_ref"]
    assert result["patch"]["reset_opportunity"]["required_attempts"] == 3


def test_gain_must_strictly_exceed_registered_threshold():
    args = packet(plan_value(threshold=0.5), (-3.0, -2.5))
    assert expected_change(*args)["reason"] == "unsupported_final_boundary_feedback"


@pytest.mark.parametrize("attempts,seconds", [(2, 160), (8, 59)])
def test_reset_requires_full_common_prefix_plus_new_trial_grant(attempts, seconds):
    plan = plan_value(attempts=attempts, seconds=seconds)
    args = packet(plan)
    assert reset_opportunity(plan, args[2]) is None
    assert expected_change(*args)["reason"] == "insufficient_reset_prefix_opportunity"


def test_zero_attempt_proposal_failure_still_requires_fresh_reset_capacity():
    args = packet(plan_value(), (-3.0, None, -2.0), proposal_failures=(1,))
    result = expected_change(*args)
    opportunity = ResetOpportunity.model_validate_json(
        canonical(result["patch"]["reset_opportunity"])
    )
    assert args[2]["historical_attempts"] == 3  # Source requests, not charged fits.
    assert args[2]["historical_worker_slots"] == 40
    assert opportunity.prefix_attempts[1] is None
    assert opportunity.required_attempts == 4
    assert opportunity.required_registered_seconds == 80


@pytest.mark.parametrize(
    "plan,scores,reason",
    [
        (dict(upper=3), (-3.0, -2.0), "global_window_bound_reached"),
        (dict(cap=5, upper=7), (-4.0, -3.0, -2.0, -1.0), "registered_policy_cap_exhausted"),
        (dict(cutoff=13, instruments=2), (-3.0, -2.0), "no_admissible_new_factor"),
    ],
)
def test_honest_domain_and_horizon_capacity_exhaustion(plan, scores, reason):
    assert expected_change(*packet(plan_value(**plan), scores))["reason"] == reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "registered_cap_v2"),
        ("forecast_recursive_profile_ref", ref("foreign profile")),
        ("registered_neighbor_domain_ref", ref("foreign domain")),
        ("final_checkpoint_ref", ref("foreign state")),
    ],
)
def test_rebound_envelope_cannot_replace_registered_profile_or_trace(field, value):
    args = packet(plan_value())
    evidence = deepcopy(args[2])
    evidence[field] = value
    with pytest.raises(ValueError):
        expected_change(*rebound(evidence, args[1]["base_package_ref"]))


def test_direct_helper_requires_request_and_evidence_hashes():
    args = packet(plan_value())
    with pytest.raises(ValueError, match="request reference mismatch"):
        expected_change(ref("wrong request"), args[1], args[2])
    changed = deepcopy(args[2])
    changed["entries"][0]["observation"]["score"] = -50.0
    with pytest.raises(ValueError, match="evidence/visibility/watermark"):
        expected_change(args[0], args[1], changed)


@pytest.mark.parametrize("damage", ["parent", "attempt", "selection", "visibility"])
def test_envelope_rehash_does_not_make_invalid_trace_valid(damage):
    args = packet(plan_value())
    evidence = deepcopy(args[2])
    if damage == "parent":
        evidence["entries"][-1]["parent_incumbent"]["score"] = -100.0
    elif damage == "attempt":
        evidence["source_attempts"].pop()
        evidence["historical_worker_slots"] = 20
    elif damage == "selection":
        evidence["final_selection"] = evidence["entries"][-1]["parent_incumbent"]
    else:
        evidence["entries"][-1]["observation"]["visibility"] = "protected"
        evidence["entries"][-1]["observation_ref"] = digest(
            canonical(evidence["entries"][-1]["observation"])
        )
    with pytest.raises(ValueError):
        expected_change(*rebound(evidence, args[1]["base_package_ref"]))


def test_extend_requires_v3_authorization_and_registered_evidence_returns_exact_plan():
    plan = plan_value()
    args = packet(plan)
    assert validate_registered_evidence(args[1], args[2]) == plan
    with pytest.raises(ValueError, match="authenticated trajectory authorization"):
        extend_capsule({}, plan, "", {}, {}, None, [], [])


def test_nonfinite_or_untyped_semantic_feedback_rejects():
    selection = dict(
        factor={"op": "momentum", "window": 3},
        score=float("nan"),
        result_ref=ref("r"),
        proposal_ref=ref("p"),
        ordinal=1,
    )
    with pytest.raises(ValidationError):
        _FinalFeedback.model_validate(
            dict(
                selected=selection,
                candidate=selection,
                parent=None,
                completed=True,
                eligible=True,
            )
        )


def test_descriptor_is_explicitly_v3_forecast_and_has_no_external_tools():
    value = proposer_definition()
    assert value["version"] == 3
    assert value["engine_generation"] == "five_session_recursive_forecast_v3"
    assert value["profile"] == "registered_forecast_cap_v3"
    assert value["external_tools"] == []
