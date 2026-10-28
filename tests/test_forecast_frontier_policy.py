"""Pure controlled feedback tests, not workers, receipts, or generation acceptance.

Packets are internally consistent synthetic development traces produced by the
existing pure state machine. They do not alter any executed history or claim a
positive result from the registered engineering data.
"""

from copy import deepcopy
from pathlib import Path

import pytest
from test_forecast_recursive_changes import packet, plan_value, ref

from quant_rsi import forecast_frontier_policy as frontier
from quant_rsi.artifacts import canonical, digest
from quant_rsi.changes import request_definition
from quant_rsi.forecast_recursive_changes import expected_change as legacy_change
from quant_rsi.researcher import package_definition


def bind(evidence, package_ref):
    request = request_definition(
        "unit-frontier-change",
        package_ref,
        digest(canonical(frontier.proposer_definition())),
        digest(canonical(evidence)),
        evidence,
    )
    return digest(canonical(request)), request, evidence


def frontier_packet(plan, scores=(-4.0, -3.0, -2.0, -3.5), *, proposal_failures=()):
    _, request, evidence = packet(plan, scores, proposal_failures=proposal_failures)
    evidence = evidence | {
        "proposer_policy": frontier.POLICY,
        "proposer_definition": frontier.proposer_definition(),
    }
    return bind(evidence, request["base_package_ref"])


def test_new_descriptor_is_explicit_and_math_profile_stays_v3():
    descriptor = frontier.proposer_definition()
    assert descriptor["version"] == 4
    assert descriptor["proposer_policy"] == frontier.POLICY
    assert descriptor["profile"] == "registered_forecast_cap_v3"
    assert descriptor["engine_generation"] == "five_session_recursive_forecast_v3"
    assert descriptor["external_tools"] == []
    assert descriptor["source_digest"] == digest(Path(frontier.__file__).read_bytes())


@pytest.mark.parametrize(
    "cap,scores,prefix,new_factor",
    [
        (3, (-4.0, -3.0, -2.0, -3.5), 2, {"op": "momentum", "window": 4}),
        (4, (-5.0, -4.0, -3.0, -2.0, -3.5), 3, {"op": "momentum", "window": 5}),
    ],
)
def test_both_terminal_frontiers_stop_at_actual_first_divergence(cap, scores, prefix, new_factor):
    args = frontier_packet(plan_value(cap=cap), scores)
    before = canonical(args)
    response = frontier.expected_change(*args)
    patch = frontier.FrontierCapPatch.model_validate_json(canonical(response["patch"]))
    evidence = args[2]
    selected = evidence["entries"][-2]
    comparison = evidence["entries"][-1]
    assert response["outcome"] == "patch"
    assert patch.schema_version == 4
    assert patch.value == cap + 1
    assert patch.forecast_recursive_profile_ref == evidence["forecast_recursive_profile_ref"]
    assert patch.selected_observation_ref == selected["observation_ref"]
    assert patch.comparison_observation_ref == comparison["observation_ref"]
    assert response["supporting_observations"] == [
        selected["observation_ref"],
        comparison["observation_ref"],
    ]
    # The selected mean-return observation really followed a momentum incumbent.
    assert selected["parent_incumbent"]["factor"]["op"] == "momentum"
    assert selected["observation"]["factor"]["op"] == "mean_return"
    opportunity = patch.reset_opportunity
    assert opportunity.new_factor == new_factor
    assert opportunity.prefix_observation_refs == [
        entry["observation_ref"] for entry in evidence["entries"][:prefix]
    ]
    assert opportunity.prefix_request_refs == [
        row["request_ref"] for row in evidence["source_requests"][:prefix]
    ]
    assert opportunity.checkpoint_ref == evidence["entries"][prefix]["parent_checkpoint_ref"]
    assert opportunity.required_attempts == prefix + 1
    assert opportunity.required_registered_seconds == (prefix + 1) * 20
    assert canonical(args) == before


@pytest.mark.parametrize("scores", [(-3.0, -2.0), (-4.0, -3.0, -2.0, -3.5)])
def test_comparison_before_or_after_selection_uses_original_order(scores):
    args = frontier_packet(plan_value(), scores)
    response = frontier.expected_change(*args)
    assert response["outcome"] == "patch"
    selected_ref, comparison_ref = response["supporting_observations"]
    by_ref = {entry["observation_ref"]: entry["observation"] for entry in args[2]["entries"]}
    selected, comparison = by_ref[selected_ref], by_ref[comparison_ref]
    assert selected["factor"]["op"] == comparison["factor"]["op"]
    assert selected["factor"]["window"] == comparison["factor"]["window"] + 1
    assert (comparison["ordinal"] < selected["ordinal"]) == (len(scores) == 2)


def test_legacy_no_patch_is_preserved_without_rewriting_legacy_envelope():
    legacy = packet(plan_value(), (-4.0, -3.0, -2.0, -3.5))
    original = canonical(legacy)
    response = legacy_change(*legacy)
    assert response["outcome"] == "no_patch"
    assert response["reason"] == "unsupported_final_boundary_feedback"
    with pytest.raises(ValueError, match="descriptor"):
        frontier.expected_change(*legacy)
    new = frontier_packet(plan_value(), (-4.0, -3.0, -2.0, -3.5))
    stripped = {
        key: value
        for key, value in new[2].items()
        if key not in ("proposer_policy", "proposer_definition")
    }
    assert canonical(stripped) == canonical(legacy[2])
    assert frontier.expected_change(*new)["outcome"] == "patch"
    assert canonical(legacy) == original


@pytest.mark.parametrize("comparison_score", [-5.0, -4.5])
def test_comparison_gain_must_strictly_exceed_original_threshold(comparison_score):
    args = frontier_packet(plan_value(threshold=1.0), (-8.0, -6.0, -4.0, comparison_score))
    response = frontier.expected_change(*args)
    assert response["outcome"] == "no_patch"
    assert response["reason"] == "unsupported_terminal_frontier_feedback"


def test_valid_semantic_counterfactual_changes_real_decision_without_forged_history():
    plan = plan_value(threshold=1.0)
    unsupported = frontier_packet(plan, (-8.0, -6.0, -4.0, -5.0))
    supported = frontier_packet(plan, (-8.0, -6.0, -4.0, -5.1))
    assert frontier.validate_registered_evidence(unsupported[1], unsupported[2]) == plan
    assert frontier.validate_registered_evidence(supported[1], supported[2]) == plan
    assert frontier.expected_change(*unsupported)["outcome"] == "no_patch"
    assert frontier.expected_change(*supported)["outcome"] == "patch"
    assert unsupported[0] != supported[0]
    assert unsupported[1]["evidence_ref"] != supported[1]["evidence_ref"]


@pytest.mark.parametrize(
    "scores",
    [
        (-4.0,),  # Only the non-boundary baseline.
        (-4.0, -3.0, -2.0),  # Final mean-return boundary lacks its own U-1 comparison.
        (-4.0, -3.0, -2.0, None),  # Unresolved comparison is not eligible support.
        (-4.0, -3.0, -2.0, -1.0),  # Latest winner is U-1, not boundary U.
    ],
)
def test_absent_ineligible_or_non_boundary_support_is_no_patch(scores):
    response = frontier.expected_change(*frontier_packet(plan_value(), scores))
    assert response["outcome"] == "no_patch"
    assert response["reason"] == "unsupported_terminal_frontier_feedback"


@pytest.mark.parametrize(
    "plan,scores,reason",
    [
        (dict(upper=3), (-3.0, -2.0), "global_window_bound_reached"),
        (dict(cap=5, upper=7), (-4.0, -3.0, -2.0, -1.0), "registered_policy_cap_exhausted"),
        (dict(cutoff=13), (-3.0, -2.0), "no_admissible_new_factor"),
    ],
)
def test_domain_and_maturity_exhaustion_are_honest(plan, scores, reason):
    response = frontier.expected_change(*frontier_packet(plan_value(**plan), scores))
    assert response["outcome"] == "no_patch"
    assert response["reason"] == reason


@pytest.mark.parametrize("attempts,seconds", [(2, 160), (8, 59)])
def test_first_divergence_requires_full_prefix_and_one_full_new_timeout(attempts, seconds):
    args = frontier_packet(plan_value(attempts=attempts, seconds=seconds), (-3.0, -2.0))
    assert frontier.expected_change(*args)["reason"] == "insufficient_first_divergence_opportunity"


def test_zero_worker_proposal_failure_still_requires_request_and_timeout_capacity():
    plan = plan_value()
    args = frontier_packet(plan, (-3.0, None, -2.0), proposal_failures=(1,))
    opportunity = frontier.expected_change(*args)["patch"]["reset_opportunity"]
    assert args[2]["historical_worker_slots"] == 40
    assert opportunity["prefix_attempts"][1] is None
    assert len(opportunity["prefix_request_refs"]) == 3
    assert opportunity["required_attempts"] == 4
    assert opportunity["required_registered_seconds"] == 80
    scarce = frontier_packet(plan_value(attempts=3), (-3.0, None, -2.0), proposal_failures=(1,))
    assert frontier.expected_change(*scarce)["outcome"] == "no_patch"


def test_structural_helper_rejects_patience_stop_before_any_reset_divergence():
    plan = plan_value()
    evidence = deepcopy(frontier_packet(plan, (-3.0, -2.0))[2])
    evidence["final_checkpoint"]["no_progress"] = plan.policy.max_no_progress
    # Structural-only input, not passed off as an authenticated source.
    assert frontier.reset_opportunity(plan, evidence) is None


def test_structural_helper_rejects_divergence_into_an_old_window(monkeypatch):
    plan = plan_value()
    evidence = frontier_packet(plan, (-3.0, -2.0))[2]
    original = frontier.next_factor

    def invalid_decision(candidate, checkpoint):
        if candidate.policy.window_max == 4 and checkpoint["next_position"] == 2:
            return {"op": "mean_return", "window": 2}
        return original(candidate, checkpoint)

    monkeypatch.setattr(frontier, "next_factor", invalid_decision)
    # This unit counterfactual verifies the certificate's own first-divergence gate.
    assert frontier.reset_opportunity(plan, evidence) is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence.pop("proposer_policy"),
        lambda evidence: evidence.update(proposer_policy="unknown"),
        lambda evidence: evidence.pop("proposer_definition"),
        lambda evidence: evidence["proposer_definition"].update(version=3),
        lambda evidence: evidence["proposer_definition"].update(source_digest=ref("wrong source")),
    ],
)
def test_missing_unknown_or_hybrid_descriptor_is_rejected(mutation):
    _, request, evidence = frontier_packet(plan_value())
    mutation(evidence)
    with pytest.raises(ValueError, match="descriptor"):
        frontier.expected_change(*bind(evidence, request["base_package_ref"]))


def test_request_descriptor_cannot_be_silently_replaced():
    _, request, evidence = frontier_packet(plan_value())
    request["proposer_ref"] = ref("legacy or unknown proposer")
    with pytest.raises(ValueError, match="request/proposer"):
        frontier.expected_change(digest(canonical(request)), request, evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence["entries"][-2]["parent_incumbent"].update(score=-100.0),
        lambda evidence: evidence["final_selection"].update(ordinal=0),
        lambda evidence: evidence.update(feedback_watermark=999),
        lambda evidence: evidence["source_requests"][0].update(request_ref=ref("wrong request")),
    ],
)
def test_descriptor_does_not_bypass_original_authenticated_trace(mutation):
    _, request, evidence = frontier_packet(plan_value())
    mutation(evidence)
    with pytest.raises(ValueError):
        frontier.expected_change(*bind(evidence, request["base_package_ref"]))


def test_protected_only_changes_do_not_affect_response_or_trigger_artifact_reads(
    monkeypatch, tmp_path
):
    args = frontier_packet(plan_value())
    protected = tmp_path / "protected-only.json"
    protected.write_text('{"score": 100}')
    read_bytes = Path.read_bytes
    implementations = {
        Path(frontier.__file__).with_name(name)
        for name in (
            "forecast_frontier_policy.py",
            "researcher.py",
            "research.py",
            "research_journal.py",
            "task_definition.py",
        )
    }

    def only_implementation(path):
        if path not in implementations:
            raise AssertionError("policy attempted a non-implementation file read")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", only_implementation)
    from quant_rsi.artifacts import ArtifactStore

    def forbidden(*args, **kwargs):
        raise AssertionError("policy attempted an artifact read")

    monkeypatch.setattr(ArtifactStore, "get", forbidden)
    first = frontier.expected_change(*args)
    protected.write_text('{"score": -100, "outcome": "different"}')
    assert frontier.expected_change(*args) == first


def test_capsule_extension_requires_exact_authorized_descriptor():
    plan = plan_value()
    _, _, evidence = packet(plan)
    base_keys = (
        "kind",
        "source_campaign_id",
        "task_ref",
        "feedback_watermark",
        "entries",
        "historical_attempts",
        "historical_worker_slots",
    )
    base = {key: deepcopy(evidence[key]) for key in base_keys}
    for entry in base["entries"]:
        entry.pop("parent_checkpoint")
    parents = {
        entry["parent_checkpoint_ref"]: entry["parent_checkpoint"] for entry in evidence["entries"]
    }
    authorization = dict(
        profile="registered_forecast_cap_v3",
        source_receipt_ref=evidence["source_receipt_ref"],
        source_head={"selection_ref": evidence["source_selection_ref"]},
        expected_revision=evidence["source_head_revision"],
        source_cas_ref=evidence["source_cas_ref"],
        development_data_ref=evidence["data_ref"],
    )
    arguments = (
        base,
        plan,
        evidence["final_checkpoint_ref"],
        evidence["final_checkpoint"],
        parents,
        authorization,
        evidence["source_requests"],
        evidence["source_attempts"],
    )
    with pytest.raises(ValueError, match="descriptor"):
        frontier.extend_capsule(*arguments)
    authorization.update(
        proposer_policy=frontier.POLICY,
        proposer_definition=frontier.proposer_definition(),
    )
    result = frontier.extend_capsule(*arguments)
    stripped = {
        key: value
        for key, value in result.items()
        if key not in ("proposer_policy", "proposer_definition")
    }
    assert canonical(stripped) == canonical(evidence)
    args = bind(result, digest(canonical(package_definition(plan))))
    assert frontier.validate_registered_evidence(args[1], args[2]) == plan
