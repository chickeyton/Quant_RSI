"""Versioned terminal-frontier admission for a shared forecast search cap.

The original development envelope remains authoritative. This module compares
observed feedback and certifies only the known prefix before the first new
decision; it never predicts the new trial's outcome or grants worker authority.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.forecast.recursive.forecast_recursive_changes import (
    RegisteredCapPatch,
    ResetOpportunity,
)
from quant_rsi.forecast.recursive.forecast_recursive_changes import (
    extend_capsule as _extend_legacy_capsule,
)
from quant_rsi.forecast.recursive.forecast_recursive_changes import (
    validate_registered_evidence as _validate_legacy_evidence,
)
from quant_rsi.market.market_contracts import Hash
from quant_rsi.research.researcher import ResearchPlan, next_factor
from quant_rsi.research.task_definition import (
    domain_ref,
    forecast_profile_ref,
    training_row_capacity,
)

POLICY = "terminal_frontier_first_divergence_v1"


class FrontierCapPatch(RegisteredCapPatch):
    schema_version: Literal[4]
    comparison_observation_ref: Hash


class _Selection(StrictModel):
    """Allowed feedback, not a receipt or an authority claim."""

    factor: FactorSpec
    score: float
    result_ref: Hash
    proposal_ref: Hash
    ordinal: int = Field(ge=0)


class _FrontierFeedback(StrictModel):
    selected: _Selection | None
    boundary: _Selection | None
    comparison: _Selection | None


def _supports_terminal_frontier(plan: ResearchPlan, feedback: _FrontierFeedback) -> bool:
    selected, comparison = feedback.selected, feedback.comparison
    return (
        selected is not None
        and selected == feedback.boundary
        and comparison is not None
        and selected.factor.window == plan.policy.window_max
        and comparison.factor.op == selected.factor.op
        and comparison.factor.window == plan.policy.window_max - 1
        and selected.score - comparison.score > plan.policy.min_improvement
    )


def proposer_definition() -> dict:
    """Canonical policy identity; source hashing reads only this implementation."""
    return dict(
        kind="ProposerPackage",
        version=4,
        algorithm=POLICY,
        proposer_policy=POLICY,
        engine_generation="five_session_recursive_forecast_v3",
        profile="registered_forecast_cap_v3",
        source_digest=digest(Path(__file__).read_bytes()),
        visibility="development",
        allowed_path="policy.window_max",
        step=1,
        max_advances=2,
        neighbor_order="window_first",
        state_policy="reset",
        rng=None,
        rule="final selected eligible U strictly improves on the unique observed "
        "same-operator U-1 result by the registered threshold; the first reset "
        "divergence is a funded, feasible newly admitted U+1 trial",
        intervention="effective_search_set_expansion_not_same_space_algorithm_improvement",
        external_tools=[],
    )


def _require_descriptor(binding: dict) -> dict:
    expected = proposer_definition()
    if binding.get("proposer_policy") != POLICY or canonical(
        binding.get("proposer_definition")
    ) != canonical(expected):
        raise ValueError("missing or unsupported frontier proposer descriptor")
    return expected


def extend_capsule(
    capsule, plan, checkpoint_ref, checkpoint, checkpoints, authorization, requests, attempts
):
    """Extend the original envelope only after an explicit policy authorization."""
    descriptor = _require_descriptor(authorization)
    return _extend_legacy_capsule(
        capsule, plan, checkpoint_ref, checkpoint, checkpoints, authorization, requests, attempts
    ) | {"proposer_policy": POLICY, "proposer_definition": descriptor}


def validate_registered_evidence(request, evidence):
    descriptor = _require_descriptor(evidence)
    if request.get("proposer_ref") != digest(canonical(descriptor)):
        raise ValueError("frontier request/proposer descriptor mismatch")
    # Validate the exact received envelope. Do not rewrite parents, source fields,
    # request/evidence hashes, or the legacy mathematical profile to make it pass.
    return _validate_legacy_evidence(request, evidence)


def _selection(observation):
    if observation["status"] != "completed" or not observation["eligible"]:
        return None
    return {
        key: observation[key]
        for key in ("factor", "score", "result_ref", "proposal_ref", "ordinal")
    }


def _terminal_support(plan, evidence):
    selected = evidence["final_selection"]
    if selected is None:
        return None
    boundaries = [
        entry for entry in evidence["entries"] if _selection(entry["observation"]) == selected
    ]
    comparisons = [
        entry
        for entry in evidence["entries"]
        if (candidate := _selection(entry["observation"])) is not None
        and candidate["factor"]
        == {"op": selected["factor"]["op"], "window": plan.policy.window_max - 1}
    ]
    if len(boundaries) != 1 or len(comparisons) != 1:
        return None
    boundary, comparison = boundaries[0], comparisons[0]
    feedback = _FrontierFeedback.model_validate_json(
        canonical(
            dict(
                selected=selected,
                boundary=_selection(boundary["observation"]),
                comparison=_selection(comparison["observation"]),
            )
        )
    )
    return (boundary, comparison) if _supports_terminal_frontier(plan, feedback) else None


def reset_opportunity(plan: ResearchPlan, evidence):
    """Bound known reset requests through the first divergence, never later scores.

    The caller authenticates the envelope before using this structural helper.
    Proposal failures still consume a prospective request and full timeout slot.
    """
    cap = plan.policy.window_max + 1
    if (
        plan.schema_version != 3
        or cap not in plan.neighbor_domain.caps
        or training_row_capacity(plan.development, cap) < 12
    ):
        return None
    candidate = plan.model_copy(
        update={"policy": plan.policy.model_copy(update={"window_max": cap})}
    )
    prefix = []
    checkpoints = [entry["parent_checkpoint"] for entry in evidence["entries"]]
    checkpoints.append(evidence["final_checkpoint"])
    for position, checkpoint in enumerate(checkpoints):
        if checkpoint["no_progress"] >= plan.policy.max_no_progress:
            return None
        parent_factor = next_factor(plan, checkpoint)
        candidate_factor = next_factor(candidate, checkpoint)
        if candidate_factor != parent_factor:
            count = len(prefix) + 1
            seconds = count * plan.development.budget.worker_timeout_seconds
            if (
                checkpoint["baseline"] is None
                or candidate_factor is None
                or candidate_factor["window"] != cap
                or count > plan.development.budget.max_attempts
                or seconds > plan.development.budget.max_worker_seconds
            ):
                return None
            actual = {attempt["ordinal"]: attempt for attempt in evidence["source_attempts"]}
            return ResetOpportunity(
                prefix_observation_refs=[entry["observation_ref"] for entry in prefix],
                prefix_request_refs=[
                    row["request_ref"] for row in evidence["source_requests"][: len(prefix)]
                ],
                prefix_attempts=[
                    actual.get(row["ordinal"]) for row in evidence["source_requests"][: len(prefix)]
                ],
                checkpoint_ref=digest(canonical(checkpoint)),
                new_factor=candidate_factor,
                required_attempts=count,
                required_registered_seconds=seconds,
            ).model_dump(mode="json")
        if candidate_factor is None:
            return None
        if position < len(evidence["entries"]):
            prefix.append(evidence["entries"][position])
    return None


def expected_change(request_ref, request, evidence):
    """Reproduce admission without authority, data, protected access, or writes."""
    if digest(canonical(request)) != request_ref:
        raise ValueError("change request reference mismatch")
    plan = validate_registered_evidence(request, evidence)
    supporting = _terminal_support(plan, evidence)
    opportunity = None
    reason = "unsupported_terminal_frontier_feedback"
    cap = plan.policy.window_max + 1
    if supporting is not None:
        if cap > plan.neighbor_domain.window_max:
            reason = "global_window_bound_reached"
        elif cap not in plan.neighbor_domain.caps:
            reason = "registered_policy_cap_exhausted"
        elif training_row_capacity(plan.development, cap) < 12:
            reason = "no_admissible_new_factor"
        else:
            opportunity = reset_opportunity(plan, evidence)
            reason = (
                "insufficient_first_divergence_opportunity"
                if opportunity is None
                else "terminal_frontier_with_first_divergence_opportunity"
            )
    patch = None
    if opportunity is not None:
        boundary, comparison = supporting
        patch = FrontierCapPatch(
            kind="PolicyPatch",
            schema_version=4,
            request_ref=request_ref,
            base_package_ref=request["base_package_ref"],
            proposer_ref=request["proposer_ref"],
            evidence_ref=request["evidence_ref"],
            evidence_refs=request["evidence_refs"],
            feedback_watermark=request["feedback_watermark"],
            task_ref=request["task_ref"],
            interface_version=1,
            state_schema_version=1,
            state_policy="reset",
            operation="replace",
            path="policy.window_max",
            value=cap,
            registered_neighbor_domain_ref=domain_ref(plan),
            forecast_recursive_profile_ref=forecast_profile_ref(plan),
            selected_observation_ref=boundary["observation_ref"],
            comparison_observation_ref=comparison["observation_ref"],
            reset_opportunity=opportunity,
            expected_effect="Explore one additional shared-cap window under a fresh paired "
            "reset; not direct same-operator extrapolation, promotion, or proof of improvement",
        ).model_dump(mode="json")
    return dict(
        kind="ChangeResponse",
        request_ref=request_ref,
        outcome="patch" if patch else "no_patch",
        patch=patch,
        reason=reason,
        supporting_observations=(
            [entry["observation_ref"] for entry in supporting] if supporting else []
        ),
        error=None,
    )
