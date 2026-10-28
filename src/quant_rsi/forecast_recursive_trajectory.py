"""Two real recursive cap transitions under one immutable cumulative authority."""

import uuid

from quant_rsi import changes
from quant_rsi.artifacts import canonical, digest
from quant_rsi.researcher import ResearchPlan
from quant_rsi.task_definition import domain_ref, forecast_profile_ref, verify_domain
from quant_rsi.trajectory_evidence import source_evidence_capsule, verify_research_receipt
from quant_rsi.trajectory_state import occupied_capacity, require_owner

UNITS = ("worker_attempts", "registered_worker_seconds")


def terminal_proposer():
    return dict(
        kind="TrajectoryTerminalProposer",
        version=3,
        algorithm="registered_cap_and_capacity_limits_v1",
        qualification="limits_are_not_exhaustive_factor_search",
    )


def authorization_definition(authority, index, source_head, source_cas_ref):
    state, program, store = authority.state, authority.program, authority.store
    receipt = store.json(source_head["research_receipt_ref"])
    manifest = store.json(receipt["manifest_ref"])
    plan = ResearchPlan.model_validate_json(store.get(receipt["research_plan_ref"]))
    from quant_rsi.forecast_recursive_campaign import policy_binding

    registration = store.json(program["root_registration_ref"])
    policy = policy_binding(registration["identity"]["grant"])
    if any(program.get(key) != value for key, value in policy.items()):
        raise ValueError("stage proposer differs from its original grant")
    return dict(
        **policy,
        kind="TrajectoryC4Authorization",
        profile="registered_forecast_cap_v3",
        authority_id=state["authority_id"],
        program_ref=state["program_ref"],
        stage=index,
        request_id=uuid.uuid5(uuid.UUID(state["authority_id"]), f"stage:{index}").hex,
        child_campaign_id=uuid.uuid5(uuid.UUID(state["authority_id"]), f"c4:{index}").hex,
        child_directory=f"children/stage_{index}",
        base_package_ref=source_head["package_ref"],
        source_head=source_head,
        expected_revision=index,
        source_cas_ref=source_cas_ref,
        source_receipt_ref=source_head["research_receipt_ref"],
        source_events_ref=receipt["events_ref"],
        feedback_watermark=receipt["feedback_watermark"],
        task_ref=manifest["task_ref"],
        registered_neighbor_domain_ref=domain_ref(plan),
        forecast_recursive_profile_ref=forecast_profile_ref(plan),
        research_plan_ref=receipt["research_plan_ref"],
        source_files_ref=receipt["files_ref"],
        development_data_ref=receipt["data_ref"],
        source_directory=program["authority_root"] + f"/sources/stage_{index}",
        capacity=dict(
            worker_attempts=2 * plan.development.budget.max_attempts,
            registered_worker_seconds=2 * plan.development.budget.max_worker_seconds,
        ),
        round_steps=1,
    )


def next_stop(authority):
    """Only frozen structural bounds; never claims all factors were evaluated."""
    state, program, store = authority.state, authority.program, authority.store
    receipt = store.json(state["head"]["research_receipt_ref"])
    plan = ResearchPlan.model_validate_json(store.get(receipt["research_plan_ref"]))
    if plan.policy.window_max >= plan.neighbor_domain.window_max:
        return "global_window_bound_reached"
    if plan.policy.window_max + 1 not in plan.neighbor_domain.caps:
        return "registered_policy_cap_exhausted"
    if state["generation"] >= program["input"]["max_generations"]:
        return "generation_limit"
    if len(state["stages"]) >= program["input"]["max_proposal_stages"]:
        return "proposal_stage_limit"
    capacity = dict(
        worker_attempts=2 * plan.development.budget.max_attempts,
        registered_worker_seconds=2 * plan.development.budget.max_worker_seconds,
    )
    occupied = occupied_capacity(program, state)
    if any(occupied[u] + capacity[u] > program["input"]["budget"][u] for u in UNITS):
        return "capacity_exhausted"
    return None


def terminal_definition(authority, reason):
    state, program = authority.state, authority.program
    source = authority.store.json(state["head"]["research_receipt_ref"])
    return dict(
        kind="RegisteredTrajectoryStop",
        schema_version=3,
        reason=reason,
        program_ref=state["program_ref"],
        head=state["head"],
        revision=state["revision"],
        source_events_ref=source["events_ref"],
        feedback_watermark=source["feedback_watermark"],
        proposer_ref=program["saturation_proposer_ref"],
        registered_neighbor_domain_ref=program["registered_neighbor_domain_ref"],
        occupied=occupied_capacity(program, state),
        zero_new_workers=True,
        exhaustive_factor_search_claim=False,
    )


def reserve_stage(authority):
    from quant_rsi.trajectory import package_operation, seal

    require_owner(authority.root, authority.owner_fd)
    state = authority.state
    if state["pending"] is not None or len(state["stages"]) != state["generation"]:
        raise ValueError("next stage requires the actual committed previous candidate head")
    reason = next_stop(authority)
    if reason is not None:
        ref = authority.store.put_json(terminal_definition(authority, reason))
        seal(authority, reason, terminal_evidence_ref=ref)
        return
    index = len(state["stages"])
    previous = state["stages"][-1] if index else None
    cas_ref = state["decisions"][previous["request_id"] + ":head"] if previous else None
    authorization = authorization_definition(authority, index, state["head"], cas_ref)
    ref = authority.store.put_json(authorization)
    stage = dict(
        ordinal=index,
        request_id=authorization["request_id"],
        authorization_ref=ref,
        capacity=authorization["capacity"],
        consumed=dict.fromkeys(UNITS, 0),
        status="authorized",
        receipts=[],
        candidate_head=None,
    )

    def change(next_state):
        next_state["stages"].append(stage)
        next_state["pending"] = package_operation(
            authorization, ref, 0, {"parent": 1, "candidate": 1}
        )

    authority.update("trajectory_stage_reserved", change)


def resume_step(authority, owner_fd):
    from quant_rsi.trajectory import advance_head, execute_package_round, seal

    state, program = authority.state, authority.program
    if str(authority.root) != program["authority_root"]:
        raise ValueError("a copied or relocated trajectory cannot dispatch")
    if state["pending"] is not None:
        try:
            execute_package_round(authority, owner_fd)
        except ValueError as error:
            seal(
                authority,
                "required_child_operation_failed",
                failure=dict(type=type(error).__name__, message=str(error)),
            )
    elif len(state["stages"]) == state["generation"]:
        reserve_stage(authority)
    else:
        stage = state["stages"][-1]
        if stage["status"] != "evaluated" or not stage["receipts"]:
            raise ValueError("registered trajectory has unexplained unfinished stage state")
        pair = authority.store.json(stage["receipts"][-1]["package_receipt_ref"])
        if pair["decision"]["outcome"] != "patch":
            seal(
                authority,
                "proposal_no_patch"
                if pair["decision"]["outcome"] == "no_patch"
                else "change_proposal_rejected",
            )
        else:
            advance_head(
                authority,
                request_id=stage["request_id"] + ":head",
                expected_head=state["head"]["package_ref"],
                expected_revision=state["revision"],
            )


def verify_authority(authority):
    from quant_rsi.trajectory import head_from

    state, program, store = authority.state, authority.program, authority.store
    initial_ref = program["initial_head"]["research_receipt_ref"]
    initial = store.json(initial_ref)
    root_plan = ResearchPlan.model_validate_json(store.get(initial["research_plan_ref"]))
    from quant_rsi.forecast_recursive_registration import verify_frozen_handoff

    verify_frozen_handoff(program, initial, store)
    verify_domain(root_plan, store)
    initial_manifest = store.json(initial["manifest_ref"])
    if (
        program.get("profile") != "registered_forecast_cap_v3"
        or program["input"]["schema_version"] != 3
        or root_plan.schema_version != 3
        or root_plan.package_lineage is not None
        or any(k in initial_manifest for k in ("launch_context_ref", "bound_view_ref"))
        or program["registered_neighbor_domain_ref"] != domain_ref(root_plan)
        or program["forecast_recursive_profile_ref"] != forecast_profile_ref(root_plan)
        or program["initial_head"] != head_from(initial_ref, initial)
        or program["initial_costs"] != initial["costs"]
        or program["source_files_ref"] != initial["files_ref"]
        or program["source_plan_ref"] != initial["research_plan_ref"]
        or program["source_data_ref"] != initial["data_ref"]
        or program["input"]["source_checkpoint_ref"] != initial["checkpoint_ref"]
        or store.json(program["proposer_ref"])
        != changes.proposer_definition(root_plan, authorization=program)
        or store.json(program["saturation_proposer_ref"]) != terminal_proposer()
    ):
        raise ValueError("registered trajectory root/domain/descriptor binding mismatch")
    head, cas_ref, history = program["initial_head"], None, [initial["semantic_ref"]]
    decisions = {}
    for index, stage in enumerate(state["stages"]):
        if index >= program["input"]["max_generations"]:
            raise ValueError("registered stage exceeds the frozen generation limit")
        authorization = authorization_definition(authority, index, head, cas_ref)
        if (
            store.json(stage["authorization_ref"]) != authorization
            or stage["request_id"] != authorization["request_id"]
            or stage["capacity"] != authorization["capacity"]
        ):
            raise ValueError("registered stage authorization differs from its actual source head")
        source = store.json(head["research_receipt_ref"])
        verify_research_receipt(store, source)
        capsule = source_evidence_capsule(store, source, authorization)
        source_files = store.json(source["files_ref"])
        actual = dict.fromkeys(UNITS, 0)
        previous_costs = actual
        for round_index, pair in enumerate(stage["receipts"]):
            package = store.json(pair["package_receipt_ref"])
            manifest, files = store.json(package["manifest_ref"]), store.json(package["files_ref"])
            if (
                pair["round"] != round_index
                or manifest.get("trajectory_authorization_ref") != stage["authorization_ref"]
                or package["campaign_id"] != authorization["child_campaign_id"]
                or manifest["proposer_ref"] != program["proposer_ref"]
                or manifest["source_files_ref"] != source["files_ref"]
                or manifest["plan_ref"] != source["research_plan_ref"]
                or manifest["evidence_ref"] != digest(canonical(capsule))
                or store.json(manifest["evidence_ref"]) != capsule
                or any(files.get("evidence/" + name) != ref for name, ref in source_files.items())
            ):
                raise ValueError("registered C4 input differs from its actual stage source")
            actual = {
                unit: sum(store.json(ref)["costs"][unit] for ref in pair["arm_receipts"].values())
                for unit in UNITS
            }
            if any(actual[u] < previous_costs[u] for u in UNITS):
                raise ValueError("registered stage receipt refunded actual worker costs")
            previous_costs = actual
        if actual != stage["consumed"]:
            raise ValueError("registered stage costs differ from latest cumulative receipts")
        if stage["receipts"]:
            last_package = store.json(stage["receipts"][-1]["package_receipt_ref"])
            complete = (
                last_package["decision"]["outcome"] != "patch"
                or last_package["completion_ref"] is not None
            )
            expected_status = "evaluated" if complete else "running"
        else:
            expected_status = "authorized"
        if stage["status"] != expected_status:
            raise ValueError("registered stage status differs from its latest actual receipt")
        if stage["candidate_head"] is not None:
            if not stage["receipts"] or stage["status"] != "evaluated":
                raise ValueError("registered head lacks a completed stage")
            pair = stage["receipts"][-1]
            package = store.json(pair["package_receipt_ref"])
            candidate_ref = pair["arm_receipts"]["candidate"]
            candidate = store.json(candidate_ref)
            candidate_head = head_from(candidate_ref, candidate)
            if (
                package["completion_ref"] is None
                or package["decision"]["outcome"] != "patch"
                or candidate["package_ref"] != package["decision"]["candidate_package_ref"]
                or candidate_head != stage["candidate_head"]
                or candidate_head["semantic_ref"] in history
                or any(
                    store.json(store.json(ref)["selection_ref"])["role"] != "active"
                    for ref in pair["arm_receipts"].values()
                )
            ):
                raise ValueError("registered head lacks its actual valid paired candidate")
            request_id = stage["request_id"] + ":head"
            expected = dict(
                kind="TrajectoryExperimentalHeadReceipt",
                request=dict(
                    kind="TrajectoryHeadCAS",
                    request_id=request_id,
                    expected_head=head["package_ref"],
                    expected_revision=index,
                ),
                previous_head=head,
                head=candidate_head,
                revision=index + 1,
                package_receipt_ref=pair["package_receipt_ref"],
                qualification="experimental_development_head_not_accepted_or_release",
            )
            cas_ref = state["decisions"].get(request_id)
            if cas_ref is None or store.json(cas_ref) != expected:
                raise ValueError("registered CAS edge differs from its own frozen revision")
            decisions[request_id] = cas_ref
            head = candidate_head
            history.append(head["semantic_ref"])
        elif index != len(state["stages"]) - 1:
            raise ValueError("registered trajectory advanced without its immediate candidate")
    if (
        state["head"] != head
        or state["decisions"] != decisions
        or state["generation"] != len(decisions)
        or state["semantic_history"] != history
    ):
        raise ValueError("registered trajectory head/CAS history mismatch")
    verify_pending(authority)
    if state["terminal_ref"] is not None:
        terminal = store.json(state["terminal_ref"])
        ref = terminal.get("terminal_evidence_ref")
        bounds = {
            "global_window_bound_reached",
            "registered_policy_cap_exhausted",
            "generation_limit",
            "proposal_stage_limit",
            "capacity_exhausted",
        }
        if (terminal["reason"] in bounds) != (ref is not None):
            raise ValueError("registered terminal is missing its committed bound evidence")
        if ref is not None:
            reason = next_stop(authority)
            if (
                reason is None
                or terminal["reason"] != reason
                or store.json(ref) != terminal_definition(authority, reason)
            ):
                raise ValueError("registered zero-worker terminal evidence mismatch")


def verify_pending(authority):
    from quant_rsi.trajectory import package_operation

    state, store = authority.state, authority.store
    expected = None
    if state["stages"]:
        stage = state["stages"][-1]
        if stage["status"] in {"authorized", "running"}:
            authorization = store.json(stage["authorization_ref"])
            targets = {"parent": 1, "candidate": 1}
            if stage["receipts"]:
                package = store.json(stage["receipts"][-1]["package_receipt_ref"])
                targets = {
                    arm: package["reports"][arm]["committed_observations"] + 1
                    for arm in ("parent", "candidate")
                }
            expected = package_operation(
                authorization, stage["authorization_ref"], len(stage["receipts"]), targets
            )
    if state["status"] == "ACTIVE":
        actual = state["pending"]
    else:
        terminal = store.json(state["terminal_ref"])
        actual = terminal["unresolved_operation"]
        if expected is not None and terminal["reason"] != "required_child_operation_failed":
            raise ValueError("registered unresolved operation lacks its required failure stop")
    if actual != expected:
        raise ValueError("registered pending operation differs from its fixed stage/round targets")


def execution_counts(authority):
    """Count original source and each stage's latest cumulative arms exactly once."""
    state, program, store = authority.state, authority.program, authority.store
    refs = [program["initial_head"]["research_receipt_ref"]]
    for stage in state["stages"]:
        if stage["receipts"]:
            refs.extend(stage["receipts"][-1]["arm_receipts"].values())
    if len(refs) != len(set(refs)):
        raise ValueError("recursive costs contain a duplicated original child receipt")
    receipts = [store.json(ref) for ref in refs]
    confirmed, unknown = 0, []
    for receipt in receipts:
        for attempt in receipt["costs"]["attempts"]:
            if attempt["worker_seconds"] == 0:
                continue
            if attempt["status"] == "completed":
                result = store.json(attempt["result_ref"])
                metadata = store.json(result["worker_metadata_ref"])
                # Existing M0 worker has one fit; frozen research replay validates this result.
                if not metadata.get("actual_fit_completed_at"):
                    raise ValueError("completed forecast attempt lacks actual fit receipt")
                confirmed += 1
            else:
                unknown.append(
                    dict(
                        campaign_id=receipt["campaign_id"],
                        ordinal=attempt["ordinal"],
                        status=attempt["status"],
                    )
                )
    return dict(
        proposal_reservations=sum(r["costs"]["proposal_reservations"] for r in receipts),
        kernel_attempt_rows=sum(r["costs"]["kernel_attempt_rows"] for r in receipts),
        charged_worker_attempts=sum(r["costs"]["worker_attempts"] for r in receipts),
        registered_worker_seconds=sum(r["costs"]["registered_worker_seconds"] for r in receipts),
        confirmed_actual_fits=confirmed,
        charged_attempts_with_unconfirmed_fit_count=unknown,
        proposal_stages=len(state["stages"]),
        paired_stages=sum(
            bool(stage["receipts"])
            and store.json(stage["receipts"][-1]["package_receipt_ref"])["decision"]["outcome"]
            == "patch"
            and store.json(stage["receipts"][-1]["package_receipt_ref"])["completion_ref"]
            is not None
            for stage in state["stages"]
        ),
        semantic_advances=state["generation"],
        historical_source_counted_once=True,
        intermediate_source_rereads_charge=0,
    )
