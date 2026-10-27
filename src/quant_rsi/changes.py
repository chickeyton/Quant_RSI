"""Frozen evidence-driven proposer and the only permitted package intervention."""

import math
from pathlib import Path
from typing import Literal

from quant_rsi.artifacts import canonical, digest
from quant_rsi.researcher import ResearchPlan, package_definition
from quant_rsi.schemas import StrictModel


class PolicyPatch(StrictModel):
    kind: Literal["PolicyPatch"]
    schema_version: Literal[1]
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
    path: Literal["policy.neighbor_order"]
    value: Literal["operator_first"]
    expected_effect: str


def registered_proposer(binding=None):
    """Dispatch only an exact explicit selector; an absent selector retains the legacy policy."""
    if binding is not None and any(
        key in binding for key in ("proposer_policy", "proposer_definition")
    ):
        from quant_rsi import forecast_frontier_policy

        forecast_frontier_policy._require_descriptor(binding)
        return forecast_frontier_policy
    from quant_rsi import forecast_recursive_changes

    return forecast_recursive_changes


def proposer_definition(plan=None, *, authorization=None):
    if plan is not None and plan.neighbor_domain is not None:
        return registered_proposer(authorization).proposer_definition()
    if authorization is not None and any(
        key in authorization for key in ("proposer_policy", "proposer_definition")
    ):
        raise ValueError("frontier policy requires the registered forecast profile")
    return {
        "kind": "ProposerPackage",
        "version": 1,
        "algorithm": "last_matched_neighbor_pair_v1",
        "source_digest": digest(Path(__file__).read_bytes()),
        "rule": "latest eligible swap versus last prior eligible window with the same incumbent; "
        "propose operator_first iff swap score is strictly greater",
        "visibility": "development",
        "allowed_path": "policy.neighbor_order",
        "allowed_transition": ["window_first", "operator_first"],
        "state_policy": "reset",
        "rng": None,
        "external_tools": [],
    }


def request_definition(request_id, base_ref, proposer_ref, evidence_ref, evidence):
    return {
        "kind": "ChangeRequest",
        "request_id": request_id,
        "base_package_ref": base_ref,
        "proposer_ref": proposer_ref,
        "evidence_ref": evidence_ref,
        "task_ref": evidence["task_ref"],
        "evidence_refs": [e["observation_ref"] for e in evidence["entries"]],
        "feedback_watermark": evidence["feedback_watermark"],
        "visibility": "development",
        "interface_version": 1,
        "state_schema_version": 1,
        "state_policy": "reset",
        "allowed_path": (
            "policy.window_max"
            if evidence.get("profile") == "registered_forecast_cap_v3"
            else "policy.neighbor_order"
        ),
    }


def validate_evidence(request, evidence):
    expected = request_definition(
        request["request_id"],
        request["base_package_ref"],
        request["proposer_ref"],
        request["evidence_ref"],
        evidence,
    )
    if (
        canonical(request) != canonical(expected)
        or digest(canonical(evidence)) != request["evidence_ref"]
    ):
        raise ValueError("change request evidence/visibility/watermark mismatch")
    previous = 0
    for position, entry in enumerate(evidence["entries"]):
        observation = entry["observation"]
        sequence = entry["observation_event_seq"]
        if (
            observation["visibility"] != "development"
            or observation["campaign_id"] != evidence["source_campaign_id"]
            or observation["task_ref"] != evidence["task_ref"]
            or observation["ordinal"] != position
            or digest(canonical(observation)) != entry["observation_ref"]
            or type(sequence) is not int
            or not previous < sequence <= request["feedback_watermark"]
        ):
            raise ValueError("unavailable or foreign development evidence")
        previous = sequence
        if observation["status"] == "completed":
            score = observation["score"]
            unresolved_local = (
                evidence.get("profile") == "registered_forecast_cap_v3"
                and evidence["research_plan"]["development"]["data"]["source"]
                == "local_event_snapshot"
                and score is None
                and observation["eligible"] is False
            )
            if not unresolved_local and (
                type(score) not in (int, float) or not math.isfinite(score)
            ):
                raise ValueError("invalid trusted observation score")
        elif observation["score"] is not None or observation["eligible"]:
            raise ValueError("failed observation has performance")
    if previous != request["feedback_watermark"]:
        raise ValueError("evidence does not reach declared feedback watermark")


def expected_change(request_ref, request, evidence):
    """Pure verifier, not a new proposer request or a sampling opportunity."""
    if digest(canonical(request)) != request_ref:
        raise ValueError("change request reference mismatch")
    validate_evidence(request, evidence)
    if evidence.get("profile") == "registered_forecast_cap_v3":
        proposer = registered_proposer(evidence)
        if request["proposer_ref"] != digest(canonical(proposer.proposer_definition())):
            raise ValueError("registered request differs from its explicit proposer descriptor")
        return proposer.expected_change(request_ref, request, evidence)
    if any(key in evidence for key in ("proposer_policy", "proposer_definition")):
        raise ValueError("frontier policy requires the registered forecast evidence profile")
    windows = {}
    pair = None
    for entry in evidence["entries"]:
        observation, incumbent = entry["observation"], entry["parent_incumbent"]
        if incumbent is None or observation["status"] != "completed" or not observation["eligible"]:
            continue
        anchor = digest(canonical(incumbent))
        candidate, origin = observation["factor"], incumbent["factor"]
        if candidate["op"] == origin["op"] and abs(candidate["window"] - origin["window"]) == 1:
            windows[anchor] = entry
        elif (
            candidate["op"] != origin["op"]
            and candidate["window"] == origin["window"]
            and anchor in windows
        ):
            pair = (windows[anchor], entry)
    better = pair is not None and pair[1]["observation"]["score"] > pair[0]["observation"]["score"]
    patch = None
    if better:
        patch = PolicyPatch(
            kind="PolicyPatch",
            schema_version=1,
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
            path="policy.neighbor_order",
            value="operator_first",
            expected_effect="Try the operator neighbor earlier under identical development rules; "
            "this is a hypothesis, not an improvement or promotion verdict.",
        ).model_dump(mode="json")
    return {
        "kind": "ChangeResponse",
        "request_ref": request_ref,
        "outcome": "patch" if better else "no_patch",
        "patch": patch,
        "reason": "operator_better_in_latest_pair"
        if better
        else ("operator_not_better" if pair else "insufficient_matched_evidence"),
        "supporting_observations": [e["observation_ref"] for e in pair] if pair else [],
        "error": None,
    }


def propose_change(request_ref, request, evidence):
    """Called only after the single durable change request has been reserved."""
    return expected_change(request_ref, request, evidence)


def validate_response(response, request_ref, request, evidence):
    expected = expected_change(request_ref, request, evidence)
    if canonical(response) == canonical(expected):
        if response["patch"] is not None:
            from quant_rsi.forecast_frontier_policy import FrontierCapPatch
            from quant_rsi.forecast_recursive_changes import RegisteredCapPatch

            patch_type = PolicyPatch
            if request["allowed_path"] == "policy.window_max":
                patch_type = (
                    FrontierCapPatch if "proposer_policy" in evidence else RegisteredCapPatch
                )
            patch_type.model_validate_json(canonical(response["patch"]))
        return response["outcome"], None
    if (
        isinstance(response, dict)
        and set(response) == set(expected)
        and response["kind"] == "ChangeResponse"
        and response["request_ref"] == request_ref
        and response["outcome"] == "failed"
        and response["patch"] is None
        and response["reason"] is None
        and response["supporting_observations"] == []
        and isinstance(response["error"], str)
        and response["error"].startswith("change_failed: ")
    ):
        return "failed", response["error"]
    return "rejected", "patch_policy_mismatch"


def experimental_plan(base_plan, patch_ref, request_ref, request):
    document = base_plan.model_dump(mode="json")
    if base_plan.neighbor_domain is None:
        document["policy"]["neighbor_order"] = "operator_first"
    else:
        document["policy"]["window_max"] += 1
    document["package_lineage"] = {
        "base_package_ref": request["base_package_ref"],
        "patch_ref": patch_ref,
        "proposer_ref": request["proposer_ref"],
        "request_ref": request_ref,
        "evidence_ref": request["evidence_ref"],
        "state_policy": "reset",
    }
    return ResearchPlan.model_validate_json(canonical(document))


def validate_lineage(plan, store):
    if plan.neighbor_domain is not None:
        return validate_registered_lineage(plan, store)
    lineage = plan.package_lineage
    if lineage is None:
        return
    document = plan.model_dump(mode="json")
    document["package_lineage"] = None
    document["policy"]["neighbor_order"] = "window_first"
    base_plan = ResearchPlan.model_validate_json(canonical(document))
    base = package_definition(base_plan)
    if (
        store.json(lineage.base_package_ref) != base
        or digest(canonical(base)) != lineage.base_package_ref
    ):
        raise ValueError("candidate exact base package mismatch")
    request = store.json(lineage.request_ref)
    evidence = store.json(lineage.evidence_ref)
    if store.json(lineage.proposer_ref) != proposer_definition():
        raise ValueError("candidate proposer package mismatch")
    expected = expected_change(lineage.request_ref, request, evidence)
    if expected["outcome"] != "patch" or store.json(lineage.patch_ref) != expected["patch"]:
        raise ValueError("candidate patch differs from frozen proposer")
    if experimental_plan(base_plan, lineage.patch_ref, lineage.request_ref, request) != plan:
        raise ValueError("candidate package policy or lineage mismatch")


def validate_registered_lineage(plan, store):
    """Authenticate exact immediate parents; never replace their prior lineage with gen0."""
    from quant_rsi.task_definition import verify_domain

    verify_domain(plan, store)
    seen = set()
    current = plan
    while current.package_lineage is not None:
        lineage = current.package_lineage
        if lineage.base_package_ref in seen or len(seen) >= 2:
            raise ValueError("registered lineage is cyclic or exceeds its frozen bound")
        seen.add(lineage.base_package_ref)
        base = store.json(lineage.base_package_ref)
        document = current.model_dump(mode="json")
        document.update(policy=base["policy"], package_lineage=base["lineage"])
        parent = ResearchPlan.model_validate_json(canonical(document))
        if base != package_definition(parent):
            raise ValueError("registered lineage exact immediate parent mismatch")
        request, evidence = store.json(lineage.request_ref), store.json(lineage.evidence_ref)
        if (
            request["base_package_ref"] != lineage.base_package_ref
            or request["proposer_ref"] != lineage.proposer_ref
            or request["evidence_ref"] != lineage.evidence_ref
            or evidence.get("research_plan") != parent.model_dump(mode="json")
            or store.json(lineage.proposer_ref)
            != proposer_definition(parent, authorization=evidence)
        ):
            raise ValueError("registered lineage parent/source/request mismatch")
        expected = expected_change(lineage.request_ref, request, evidence)
        if (
            expected["outcome"] != "patch"
            or store.json(lineage.patch_ref) != expected["patch"]
            or experimental_plan(parent, lineage.patch_ref, lineage.request_ref, request) != current
        ):
            raise ValueError("registered lineage differs from the frozen boundary proposer")
        current = parent
    if (
        current.policy.window_max != current.neighbor_domain.initial_window_max
        or plan.policy.window_max != current.policy.window_max + len(seen)
    ):
        raise ValueError("registered lineage does not end at its preregistered root")


def lineage_artifact_refs(plan, store):
    """Return the validated bounded ancestor closure for reset arm initialization."""
    validate_lineage(plan, store)
    refs = set()
    if plan.neighbor_domain is not None:
        from quant_rsi.task_definition import domain_ref, forecast_profile_ref

        refs.add(domain_ref(plan))
        refs.add(forecast_profile_ref(plan))
    lineage = plan.package_lineage
    while lineage is not None:
        refs.update(
            getattr(lineage, key)
            for key in (
                "base_package_ref",
                "patch_ref",
                "proposer_ref",
                "request_ref",
                "evidence_ref",
            )
        )
        parent = store.json(lineage.base_package_ref)
        from quant_rsi.researcher import PackageLineage

        lineage = (
            PackageLineage.model_validate(parent["lineage"])
            if parent["lineage"] is not None
            else None
        )
    return refs
