"""Pure v3 forecast feedback rule for a registered, at-most-two-step cap schedule.

Authenticated development trace validation precedes the private semantic decision
seam. A reset opportunity is a structural bound, not a promise of worker outcomes.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field

from quant_rsi.artifacts import canonical, digest
from quant_rsi.contracts_base import StrictModel
from quant_rsi.market_contracts import Hash
from quant_rsi.researcher import ResearchPlan, advance, next_factor, package_definition
from quant_rsi.schemas import FactorSpec
from quant_rsi.task_definition import (
    domain_ref,
    forecast_profile_ref,
    task_definition,
    training_row_capacity,
)


class ResetOpportunity(StrictModel):
    prefix_observation_refs: list[str]
    prefix_request_refs: list[str]
    prefix_attempts: list[dict | None]
    checkpoint_ref: str
    new_factor: dict
    required_attempts: int
    required_registered_seconds: int


class RegisteredCapPatch(StrictModel):
    kind: Literal["PolicyPatch"]
    schema_version: Literal[3]
    request_ref: str
    base_package_ref: str
    proposer_ref: str
    evidence_ref: str
    evidence_refs: list[str]
    feedback_watermark: int
    task_ref: str
    interface_version: Literal[1]
    state_schema_version: Literal[1]
    state_policy: Literal["reset"]
    operation: Literal["replace"]
    path: Literal["policy.window_max"]
    value: int
    registered_neighbor_domain_ref: str
    forecast_recursive_profile_ref: str
    selected_observation_ref: str
    reset_opportunity: ResetOpportunity
    expected_effect: str


class _FeedbackSelection(StrictModel):
    """Allowed semantic feedback only; this is not an authenticated receipt."""

    factor: FactorSpec
    score: float
    result_ref: Hash
    proposal_ref: Hash
    ordinal: int = Field(ge=0)


class _FinalFeedback(StrictModel):
    """Private decision seam, populated from validated development evidence."""

    selected: _FeedbackSelection | None
    candidate: _FeedbackSelection | None
    parent: _FeedbackSelection | None
    completed: bool
    eligible: bool


def _supports_final_boundary(plan: ResearchPlan, feedback: _FinalFeedback) -> bool:
    """Test causal support, without reading data, artifacts, or protected feedback."""
    selected, candidate, parent = feedback.selected, feedback.candidate, feedback.parent
    return (
        feedback.completed
        and feedback.eligible
        and selected is not None
        and candidate == selected
        and parent is not None
        and selected.factor.window == plan.policy.window_max
        and parent.factor.op == selected.factor.op
        and parent.factor.window == plan.policy.window_max - 1
        and selected.score - parent.score > plan.policy.min_improvement
    )


def _final_feedback(selected, entry) -> _FinalFeedback:
    observation = entry["observation"]
    candidate = (
        {
            key: observation[key]
            for key in ("factor", "score", "result_ref", "proposal_ref", "ordinal")
        }
        if observation["status"] == "completed" and observation["eligible"]
        else None
    )
    return _FinalFeedback.model_validate_json(
        canonical(
            dict(
                selected=selected,
                candidate=candidate,
                parent=entry["parent_incumbent"],
                completed=observation["status"] == "completed",
                eligible=observation["eligible"],
            )
        )
    )


def proposer_definition():
    return dict(
        kind="ProposerPackage",
        version=3,
        algorithm="final_forecast_boundary_cap_schedule_v3",
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
        rule="final selected eligible same-op U-1 to U improvement strictly exceeds the "
        "original threshold; registered U+1 has an actual common-prefix reset opportunity",
        intervention="effective_search_set_expansion_not_same_space_algorithm_improvement",
        external_tools=[],
    )


def extend_capsule(
    capsule, plan, checkpoint_ref, checkpoint, parents, authorization, requests, attempts
):
    """Both live source capture and frozen replay use identical canonical inputs."""
    if authorization is None or authorization.get("profile") != "registered_forecast_cap_v3":
        raise ValueError("registered-domain C4 requires authenticated trajectory authorization")
    if plan.schema_version != 3:
        raise ValueError("registered trajectory source requires version 3")
    value = capsule | dict(
        profile="registered_forecast_cap_v3",
        research_plan=plan.model_dump(mode="json"),
        registered_neighbor_domain_ref=domain_ref(plan),
        forecast_recursive_profile_ref=forecast_profile_ref(plan),
        final_checkpoint_ref=checkpoint_ref,
        final_checkpoint=checkpoint,
        final_selection=checkpoint["incumbent"],
        source_receipt_ref=authorization["source_receipt_ref"],
        source_selection_ref=authorization["source_head"]["selection_ref"],
        source_head_revision=authorization["expected_revision"],
        source_cas_ref=authorization["source_cas_ref"],
        data_ref=authorization["development_data_ref"],
        source_requests=requests,
        source_attempts=attempts,
    )
    value["entries"] = [
        entry | {"parent_checkpoint": parents[entry["parent_checkpoint_ref"]]}
        for entry in capsule["entries"]
    ]
    return value


def validate_registered_evidence(request, evidence):
    # Deferred import keeps the shared dispatch module free of an import cycle.
    from quant_rsi.changes import validate_evidence

    validate_evidence(request, evidence)
    plan = ResearchPlan.model_validate_json(canonical(evidence["research_plan"]))
    if (
        evidence.get("profile") != "registered_forecast_cap_v3"
        or plan.schema_version != 3
        or digest(canonical(package_definition(plan))) != request["base_package_ref"]
        or evidence["registered_neighbor_domain_ref"] != domain_ref(plan)
        or evidence["forecast_recursive_profile_ref"] != forecast_profile_ref(plan)
        or digest(
            canonical(
                task_definition(
                    plan,
                    evidence["data_ref"],
                    digest(canonical(plan.development.model_dump(mode="json"))),
                )
            )
        )
        != request["task_ref"]
        or digest(canonical(evidence["final_checkpoint"])) != evidence["final_checkpoint_ref"]
        or evidence["final_checkpoint"]["incumbent"] != evidence["final_selection"]
        or evidence["final_checkpoint"]["next_position"] != len(evidence["entries"])
        or len(evidence["source_requests"]) != len(evidence["entries"])
        or evidence["historical_attempts"] != len(evidence["source_requests"])
        or evidence["historical_worker_slots"]
        != sum(attempt["worker_seconds"] for attempt in evidence["source_attempts"])
    ):
        raise ValueError("registered proposer source/task/final checkpoint mismatch")
    attempts = {attempt["ordinal"]: attempt for attempt in evidence["source_attempts"]}
    if len(attempts) != len(evidence["source_attempts"]):
        raise ValueError("registered source has duplicate actual attempt identities")
    for position, entry in enumerate(evidence["entries"]):
        checkpoint = entry["parent_checkpoint"]
        row, observation = evidence["source_requests"][position], entry["observation"]
        attempt = attempts.pop(row["ordinal"], None)
        if (
            row["ordinal"] != observation["ordinal"]
            or row["request_ref"] != observation["request_ref"]
            or row["parent_ref"] != entry["parent_checkpoint_ref"]
            or row["observation_ref"] != entry["observation_ref"]
            or row["proposal_ref"] != observation["proposal_ref"]
            or (
                attempt is None
                and (
                    observation["status"] != "proposal_failed"
                    or observation["result_ref"] is not None
                )
            )
            or (
                attempt is not None
                and (
                    attempt["proposal_ref"] != row["proposal_ref"]
                    or attempt["status"] != observation["status"]
                    or attempt["result_ref"] != observation["result_ref"]
                )
            )
        ):
            raise ValueError("registered source request/attempt/result binding mismatch")
        if (
            digest(canonical(checkpoint)) != entry["parent_checkpoint_ref"]
            or checkpoint["incumbent"] != entry["parent_incumbent"]
            or checkpoint["next_position"] != entry["observation"]["ordinal"]
            or checkpoint["campaign_id"] != evidence["source_campaign_id"]
            or checkpoint["package_ref"] != request["base_package_ref"]
            or (
                observation["status"] != "proposal_failed"
                and next_factor(plan, checkpoint) != observation["factor"]
            )
        ):
            raise ValueError("registered proposer common source prefix mismatch")
        after = (
            evidence["entries"][position + 1]["parent_checkpoint"]
            if position + 1 < len(evidence["entries"])
            else evidence["final_checkpoint"]
        )
        if (
            advance(
                plan,
                checkpoint,
                entry["observation"],
                after["memory_ref"],
                entry["parent_checkpoint_ref"],
            )
            != after
        ):
            raise ValueError(
                "registered proposer selected checkpoint is not the actual trace result"
            )
    if attempts:
        raise ValueError("registered source has an unbound actual attempt")
    return plan


def reset_opportunity(plan, evidence):
    """Prove only a structural resource opportunity, never future worker outcomes."""
    candidate = plan.model_copy(
        update={"policy": plan.policy.model_copy(update={"window_max": plan.policy.window_max + 1})}
    )
    selected = evidence["final_selection"]
    if selected is None:
        return None
    prefix = []
    for entry in evidence["entries"]:
        checkpoint = entry["parent_checkpoint"]
        if (
            next_factor(candidate, checkpoint) != next_factor(plan, checkpoint)
            or checkpoint["no_progress"] >= plan.policy.max_no_progress
        ):
            return None
        prefix.append(entry)
        if entry["observation"]["ordinal"] == selected["ordinal"]:
            break
    after = (
        evidence["entries"][len(prefix)]["parent_checkpoint"]
        if len(prefix) < len(evidence["entries"])
        else evidence["final_checkpoint"]
    )
    budget = plan.development.budget
    proposed = next_factor(candidate, after)
    count = len(prefix) + 1
    if (
        after["baseline"] is None
        or after["no_progress"] >= plan.policy.max_no_progress
        or proposed is None
        or proposed["window"] != candidate.policy.window_max
        or count > budget.max_attempts
        or count * budget.worker_timeout_seconds > budget.max_worker_seconds
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
        checkpoint_ref=digest(canonical(after)),
        new_factor=proposed,
        required_attempts=count,
        required_registered_seconds=count * budget.worker_timeout_seconds,
    ).model_dump(mode="json")


def expected_change(request_ref, request, evidence):
    """Verify and reproduce the response; no authority, receipt, or state is written."""
    if digest(canonical(request)) != request_ref:
        raise ValueError("change request reference mismatch")
    plan = validate_registered_evidence(request, evidence)
    selected, supporting, opportunity = evidence["final_selection"], None, None
    reason = "unsupported_final_boundary_feedback"
    for entry in evidence["entries"]:
        if _supports_final_boundary(plan, _final_feedback(selected, entry)):
            supporting = entry
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
                "insufficient_reset_prefix_opportunity"
                if opportunity is None
                else ("final_boundary_improvement_with_reset_opportunity")
            )
    patch = None
    if opportunity is not None:
        patch = RegisteredCapPatch(
            kind="PolicyPatch",
            schema_version=3,
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
            selected_observation_ref=supporting["observation_ref"],
            reset_opportunity=opportunity,
            expected_effect="Expand the reachable factor set by one registered window; "
            "development experiment only, not promotion or proof of improvement",
        ).model_dump(mode="json")
    return dict(
        kind="ChangeResponse",
        request_ref=request_ref,
        outcome="patch" if patch else "no_patch",
        patch=patch,
        reason=reason,
        supporting_observations=[supporting["observation_ref"]] if supporting else [],
        error=None,
    )
