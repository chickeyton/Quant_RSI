"""Finite, development-only package trajectory with a single durable authority."""

import sqlite3
import uuid
from pathlib import Path

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.forecast.recursive.forecast_recursive_registration import (
    registered_trajectory_creation,
    registered_trajectory_resume,
    trajectory_units,
)
from quant_rsi.research import changes, package_experiment, research, researcher
from quant_rsi.research.researcher import ResearchPlan
from quant_rsi.research.trajectory_evidence import (
    capture_package,
    capture_research,
    source_evidence_capsule,
    verify_package_receipt,
    verify_research_receipt,
)
from quant_rsi.research.trajectory_state import (
    TrajectoryLedger,
    TrajectoryPlan,
    anchor_path,
    initialize_ledger,
    locked_trajectory,
    occupied_capacity,
    require_owner,
    trajectory_root,
)

UNITS = ("worker_attempts", "registered_worker_seconds")


def substantive(plan):
    return {key: value for key, value in plan.items() if key != "name"}


def head_from(ref, receipt):
    return {
        key: receipt[key]
        for key in ("package_ref", "semantic_ref", "checkpoint_ref", "selection_ref")
    } | {"research_receipt_ref": ref}


def materialize(store, files_ref, destination):
    for relative, ref in store.json(files_ref).items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid frozen source path")
        target = destination / path
        raw = store.get(ref)
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("frozen trajectory source changed")
        else:
            atomic_write(target, raw)


@registered_trajectory_creation
def init_trajectory(plan: TrajectoryPlan):
    """Freeze once; an identical repeated initialization never rereads its source."""
    with locked_trajectory(bootstrap=True) as (root, owner_fd):
        if anchor_path(root).exists():
            with TrajectoryLedger(root) as authority:
                if substantive(authority.program["input"]) != substantive(
                    plan.model_dump(mode="json")
                ):
                    raise ValueError("trajectory authority already has different frozen inputs")
                return summarize(authority)
        registration = None
        if plan.schema_version == 3:
            from quant_rsi.forecast.recursive.forecast_recursive_registration import begin_handoff

            registration = begin_handoff(plan, owner_fd)
        marker = root.parent / "trajectory_initialization.json"
        if marker.exists():
            saved = decode(marker.read_bytes())
            if (
                substantive(saved["input"]) != substantive(plan.model_dump(mode="json"))
                or saved["runtime"] != controller.runtime_identity()
                or (
                    registration is not None
                    and saved["authority_id"] != registration["handoff"]["authority_id"]
                )
            ):
                raise ValueError("trajectory initialization identity conflict")
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError("trajectory data exists without its initialization authority")
            saved = dict(
                kind="TrajectoryInitialization",
                authority_id=(
                    registration["handoff"]["authority_id"] if registration else uuid.uuid4().hex
                ),
                input=plan.model_dump(mode="json"),
                runtime=controller.runtime_identity(),
                program_ref=None,
            )
            atomic_write(marker, canonical(saved))
        root.mkdir(exist_ok=True)
        store = ArtifactStore(root / "artifacts")
        if saved["program_ref"] is None:
            source = plan.source_directory.resolve()
            if source == root or root in source.parents or source in root.parents:
                raise ValueError("trajectory source and authority must be disjoint")
            if registration is not None:
                dry_ref, _ = capture_research(
                    source,
                    None,
                    publish=None,
                    initial=True,
                    expected_checkpoint=plan.source_checkpoint_ref,
                )
                if dry_ref != registration["handoff"]["source_receipt_ref"]:
                    raise ValueError("root source changed after its irreversible handoff")
            receipt_ref, receipt = capture_research(
                source,
                store,
                publish=True,
                initial=True,
                expected_checkpoint=plan.source_checkpoint_ref,
            )
            costs = receipt["costs"]
            source_plan = ResearchPlan.model_validate_json(store.get(receipt["research_plan_ref"]))
            if (plan.schema_version == 3) != (source_plan.neighbor_domain is not None):
                raise ValueError("trajectory version must match the explicitly registered root")
            if any(costs[unit] > getattr(plan.budget, unit) for unit in UNITS):
                raise ValueError(
                    "trajectory budget does not cover its actual historical source costs"
                )
            program = dict(
                kind="DevelopmentTrajectoryProgram",
                authority_id=saved["authority_id"],
                input=saved["input"],
                runtime_ref=store.put_json(saved["runtime"]),
                source_files_ref=receipt["files_ref"],
                source_plan_ref=receipt["research_plan_ref"],
                source_data_ref=receipt["data_ref"],
                initial_costs=costs,
                initial_head=head_from(receipt_ref, receipt),
                proposer_ref=store.put_json(changes.proposer_definition(source_plan)),
                saturation_proposer_ref=store.put_json(saturation_proposer()),
                boundary="development_only_no_protected_evidence_or_promotion",
            )
            if plan.schema_version == 3:
                from quant_rsi.forecast.recursive.forecast_recursive_campaign import policy_binding
                from quant_rsi.forecast.recursive.forecast_recursive_trajectory import (
                    terminal_proposer,
                )
                from quant_rsi.research.task_definition import domain_ref, forecast_profile_ref

                policy = policy_binding(registration["identity"]["grant"])
                program.update(
                    **policy,
                    proposer_ref=store.put_json(
                        changes.proposer_definition(source_plan, authorization=policy)
                    ),
                    schema_version=3,
                    profile="registered_forecast_cap_v3",
                    authority_root=str(root),
                    registered_neighbor_domain_ref=domain_ref(source_plan),
                    forecast_recursive_profile_ref=forecast_profile_ref(source_plan),
                    saturation_proposer_ref=store.put_json(terminal_proposer()),
                    root_registration_ref=store.put_json(registration),
                    recursive_campaign_binding=registration["identity"]["binding"],
                    initial_capacity=trajectory_units(
                        registration["identity"]["grant"]["source_allowance"]
                    ),
                )
            saved["program_ref"] = store.put_json(program)
            atomic_write(marker, canonical(saved))
        program = store.json(saved["program_ref"])
        if registration is not None:
            from quant_rsi.forecast.recursive.forecast_recursive_registration import (
                verify_frozen_handoff,
            )

            if store.json(program["root_registration_ref"]) != registration:
                raise ValueError("trajectory initialization registration changed")
            initial = store.json(program["initial_head"]["research_receipt_ref"])
            verify_frozen_handoff(program, initial, store)
            verify_research_receipt(store, initial)
            if (root / "control.sqlite").exists():
                with sqlite3.connect(
                    f"{(root / 'control.sqlite').as_uri()}?mode=ro", uri=True
                ) as db:
                    tables = db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                    if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                        raise ValueError("trajectory partial initialization journal is corrupt")
                if tables:
                    recovery_anchor = dict(
                        kind="TrajectoryAuthority",
                        authority_id=saved["authority_id"],
                        program_ref=saved["program_ref"],
                    )
                    with TrajectoryLedger(
                        root, bootstrap_anchor=recovery_anchor, readonly=True
                    ) as prior:
                        summarize(prior)
        materialize(store, program["source_files_ref"], root / "source")
        anchor = dict(
            kind="TrajectoryAuthority",
            authority_id=saved["authority_id"],
            program_ref=saved["program_ref"],
        )
        state = dict(
            authority_id=anchor["authority_id"],
            program_ref=anchor["program_ref"],
            status="ACTIVE",
            stages=[],
            head=program["initial_head"],
            generation=0,
            revision=0,
            semantic_history=[program["initial_head"]["semantic_ref"]],
            pending=None,
            terminal_ref=None,
            decisions={},
        )
        initialize_ledger(root, anchor, state, owner_fd=owner_fd)
        atomic_write(anchor_path(root), canonical(anchor))
        marker.unlink()
        require_owner(root, owner_fd)
        with TrajectoryLedger(root) as authority:
            return summarize(authority)


def saturation_proposer():
    return dict(
        kind="TrajectorySaturationProposer",
        version=1,
        source_digest=digest(Path(__file__).read_bytes()),
        grammar=["window_first", "operator_first"],
        rule="authenticated operator_first candidate has no remaining policy transition",
        external_tools=[],
        rng=None,
    )


def reserve_stage(authority):
    require_owner(authority.root, authority.owner_fd)
    state, program = authority.state, authority.program
    if state["stages"] or state["pending"] is not None:
        raise ValueError("initial trajectory stage already reserved")
    plan = ResearchPlan.model_validate_json(authority.store.get(program["source_plan_ref"]))
    capacity = dict(
        worker_attempts=2 * plan.development.budget.max_attempts,
        registered_worker_seconds=2 * plan.development.budget.max_worker_seconds,
    )
    occupied = occupied_capacity(program, state)
    if any(occupied[unit] + capacity[unit] > program["input"]["budget"][unit] for unit in UNITS):
        seal(authority, "insufficient_child_capacity")
        return
    request_id = uuid.uuid5(uuid.UUID(state["authority_id"]), "stage:0").hex
    authorization = dict(
        kind="TrajectoryC4Authorization",
        authority_id=state["authority_id"],
        program_ref=state["program_ref"],
        stage=0,
        request_id=request_id,
        child_campaign_id=uuid.uuid5(uuid.UUID(state["authority_id"]), "c4:0").hex,
        child_directory="children/stage_0",
        base_package_ref=state["head"]["package_ref"],
        research_plan_ref=program["source_plan_ref"],
        source_files_ref=program["source_files_ref"],
        development_data_ref=program["source_data_ref"],
        source_directory=str(authority.root / "source"),
        capacity=capacity,
        round_steps=1,
    )
    ref = authority.store.put_json(authorization)
    stage = dict(
        ordinal=0,
        request_id=request_id,
        authorization_ref=ref,
        capacity=capacity,
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


def package_operation(authorization, ref, index, targets):
    return dict(
        kind="package_round",
        stage=authorization["stage"],
        authorization_ref=ref,
        child_campaign_id=authorization["child_campaign_id"],
        steps=1,
        round_index=index,
        targets=targets,
    )


def child_binding(authority):
    stage = authority.state["stages"][authority.state["pending"]["stage"]]
    auth = authority.store.json(stage["authorization_ref"])
    context = dict(
        kind="TrajectoryC4Context",
        authority_root=str(authority.root),
        authority_id=authority.anchor["authority_id"],
        authorization_ref=stage["authorization_ref"],
    )
    return auth, context, authority.root / auth["child_directory"]


def package_snapshot(authority, child):
    if not (child / "manifest.json").exists() or (child / "package_initialization.json").exists():
        return None
    ref, receipt = capture_package(child, authority.store, publish=True)
    pending = authority.state["pending"]
    manifest = authority.store.json(receipt["manifest_ref"])
    if (
        receipt["campaign_id"] != pending["child_campaign_id"]
        or manifest.get("trajectory_authorization_ref") != pending["authorization_ref"]
    ):
        raise ValueError("trajectory child receipt belongs to another authorization")
    return ref, receipt


def round_committed(receipt, pending):
    decision = receipt["decision"]
    if decision is not None and decision["outcome"] != "patch":
        return True
    rounds = receipt["rounds"]
    if len(rounds) == pending["round_index"]:
        return False
    if len(rounds) != pending["round_index"] + 1:
        raise ValueError("child advanced beyond the parent round")
    row = rounds[-1]
    from quant_rsi.research.package_journal import targets_for

    if row["steps"] != 1 or targets_for(row) != pending["targets"]:
        raise ValueError("child round target differs from the parent reservation")
    return row["receipt_ref"] is not None


def finish_committed_pair(authority, child, owner_fd):
    """Recover only a missing C4 completion receipt, without reopening a round."""
    require_owner(authority.root, owner_fd)
    pending = authority.state["pending"]
    with controller.controller_lock(child):
        journal, store, manifest, plan = package_experiment.open_package(child, replay=True)
        try:
            decision, reports = package_experiment.verify_package(
                child, journal, store, manifest, plan, replay=True
            )
            rounds = journal.rounds()
            if (
                manifest.get("trajectory_authorization_ref") != pending["authorization_ref"]
                or manifest["campaign_id"] != pending["child_campaign_id"]
                or not rounds
                or rounds[-1]["ordinal"] != pending["round_index"]
                or rounds[-1]["receipt_ref"] is None
            ):
                raise ValueError("completion recovery lacks its authorized committed round")
            if all(r is not None and r["execution_state"] == "completed" for r in reports.values()):
                journal.complete(
                    store.put_json(
                        package_experiment.comparison_receipt(manifest, decision, reports)
                    )
                )
        finally:
            journal.close()


def execute_package_round(authority, owner_fd):
    require_owner(authority.root, owner_fd)
    auth, context, child = child_binding(authority)
    if auth.get("profile") == "registered_forecast_cap_v3":
        materialize(authority.store, auth["source_files_ref"], Path(auth["source_directory"]))
    pending = authority.state["pending"]
    snapshot = package_snapshot(authority, child)
    if snapshot is None or not round_committed(snapshot[1], pending):
        if (child / "manifest.json").exists() or (child / "package_initialization.json").exists():
            package_experiment.resume_package(child, steps=1, trajectory_owner_fd=owner_fd)
        else:
            package_experiment.create_package(
                Path(auth["source_directory"]),
                child,
                state_policy="reset",
                steps=1,
                trajectory_context=context,
                trajectory_owner_fd=owner_fd,
            )
        snapshot = package_snapshot(authority, child)
    if snapshot is None or not round_committed(snapshot[1], pending):
        raise ValueError("required child operation did not produce its round receipt")
    ref, receipt = snapshot
    decision = receipt["decision"]
    if decision["outcome"] == "patch" and receipt["completion_ref"] is None:
        finish_committed_pair(authority, child, owner_fd)
        ref, receipt = package_snapshot(authority, child)
    arms, costs = {}, dict.fromkeys(UNITS, 0)
    if decision["outcome"] == "patch":
        for arm in ("parent", "candidate"):
            child_ref, child_receipt = capture_research(
                child / "arms" / arm, authority.store, publish=True
            )
            arms[arm] = child_ref
            for unit in UNITS:
                costs[unit] += child_receipt["costs"][unit]
    complete = decision["outcome"] != "patch" or receipt["completion_ref"] is not None
    evidence = dict(package_receipt_ref=ref, arm_receipts=arms, round=pending["round_index"])
    next_operation = None
    if not complete:
        targets = {
            arm: receipt["reports"][arm]["committed_observations"] + 1
            for arm in ("parent", "candidate")
        }
        next_operation = package_operation(
            auth, context["authorization_ref"], len(receipt["rounds"]), targets
        )

    def change(state):
        stage = state["stages"][pending["stage"]]
        stage["consumed"] = costs
        stage["receipts"].append(evidence)
        stage["status"] = "evaluated" if complete else "running"
        state["pending"] = next_operation

    authority.update("trajectory_child_operation_committed", change)
    if decision["outcome"] == "no_patch":
        seal(
            authority,
            "proposal_no_patch"
            if authority.program.get("schema_version") == 3
            else "first_proposal_no_patch",
        )
    elif decision["outcome"] != "patch":
        seal(authority, "change_proposal_rejected")


def candidate_evidence(authority):
    stage = authority.state["stages"][-1]
    if stage["status"] != "evaluated" or not stage["receipts"]:
        raise ValueError("candidate requires a completed paired development stage")
    pair = stage["receipts"][-1]
    package = authority.store.json(pair["package_receipt_ref"])
    verify_package_receipt(authority.store, package, pair["arm_receipts"])
    if package["completion_ref"] is None or package["decision"]["outcome"] != "patch":
        raise ValueError("candidate requires its actual full C4 comparison")
    for arm in ("parent", "candidate"):
        receipt = authority.store.json(pair["arm_receipts"][arm])
        selection = authority.store.json(receipt["selection_ref"])
        if selection["role"] != "active":
            return None
        report = authority.store.json(receipt["development_report_ref"])
        if report["execution_state"] != "completed":
            raise ValueError("candidate evidence contains an unfinished required arm")
    ref = pair["arm_receipts"]["candidate"]
    candidate = authority.store.json(ref)
    plan = ResearchPlan.model_validate_json(authority.store.get(candidate["research_plan_ref"]))
    changes.validate_lineage(plan, authority.store)
    if (
        plan.package_lineage is None
        or plan.policy.neighbor_order
        != ("window_first" if plan.neighbor_domain is not None else "operator_first")
        or candidate["package_ref"] != package["decision"]["candidate_package_ref"]
        or authority.store.json(candidate["package_ref"]) != researcher.package_definition(plan)
    ):
        raise ValueError("candidate package/lineage does not match the actual C4 decision")
    return head_from(ref, candidate)


def advance_head(authority, *, request_id, expected_head, expected_revision):
    state = authority.state
    request = dict(
        kind="TrajectoryHeadCAS",
        request_id=request_id,
        expected_head=expected_head,
        expected_revision=expected_revision,
    )
    if request_id in state["decisions"]:
        receipt = authority.store.json(state["decisions"][request_id])
        if receipt["request"] != request:
            raise ValueError("trajectory head request identity conflict")
        return receipt
    require_owner(authority.root, authority.owner_fd)
    if state["status"] != "ACTIVE" or state["pending"] is not None:
        raise ValueError("trajectory head change requires a quiescent active authority")
    if (
        type(expected_revision) is not int
        or expected_revision != state["revision"]
        or expected_head != state["head"]["package_ref"]
    ):
        raise ValueError("trajectory head compare-and-swap conflict")
    head = candidate_evidence(authority)
    if head is None:
        seal(authority, "no_authoritative_valid_incumbent")
        return None
    if head["semantic_ref"] in state["semantic_history"]:
        seal(authority, "duplicate_semantic_behavior")
        return None
    receipt = dict(
        kind="TrajectoryExperimentalHeadReceipt",
        request=request,
        previous_head=state["head"],
        head=head,
        revision=state["revision"] + 1,
        package_receipt_ref=state["stages"][-1]["receipts"][-1]["package_receipt_ref"],
        qualification="experimental_development_head_not_accepted_or_release",
    )
    ref = authority.store.put_json(receipt)

    def change(next_state):
        next_state.update(
            head=head, revision=state["revision"] + 1, generation=state["generation"] + 1
        )
        next_state["semantic_history"].append(head["semantic_ref"])
        next_state["decisions"][request_id] = ref
        next_state["stages"][-1]["candidate_head"] = head

    authority.update("trajectory_experimental_head_advanced", change)
    return receipt


def saturate(authority):
    require_owner(authority.root, authority.owner_fd)
    state = authority.state
    head = candidate_evidence(authority)
    if state["generation"] != 1 or head != state["head"] or len(state["stages"]) != 1:
        raise ValueError("saturation requires its actual advanced candidate head")
    if authority.store.json(authority.program["saturation_proposer_ref"]) != saturation_proposer():
        raise ValueError("trajectory saturation proposer changed")
    request_id = uuid.uuid5(uuid.UUID(state["authority_id"]), "stage:1").hex
    receipt = authority.store.json(head["research_receipt_ref"])
    request = dict(
        kind="TrajectorySaturationRequest",
        request_id=request_id,
        stage=1,
        package_ref=head["package_ref"],
        semantic_ref=head["semantic_ref"],
        checkpoint_ref=head["checkpoint_ref"],
        research_receipt_ref=head["research_receipt_ref"],
        feedback_watermark=receipt["feedback_watermark"],
        events_ref=receipt["events_ref"],
        proposer_ref=authority.program["saturation_proposer_ref"],
    )
    request_ref = authority.store.put_json(request)
    response_ref = authority.store.put_json(
        dict(
            kind="TrajectorySaturationResponse",
            request_ref=request_ref,
            outcome="no_patch",
            reason="grammar_exhausted",
        )
    )
    stage = dict(
        ordinal=1,
        request_id=request_id,
        authorization_ref=request_ref,
        capacity=dict.fromkeys(UNITS, 0),
        consumed=dict.fromkeys(UNITS, 0),
        status="no_patch",
        request_ref=request_ref,
        response_ref=response_ref,
        receipts=[],
    )
    authority.update(
        "trajectory_saturation_committed", lambda next_state: next_state["stages"].append(stage)
    )
    seal(authority, "grammar_exhausted")


def seal(authority, reason, *, failure=None, terminal_evidence_ref=None):
    state = authority.state
    if state["status"] == "SEALED":
        return
    require_owner(authority.root, authority.owner_fd)
    receipt = dict(
        kind="TrajectoryTerminalReceipt",
        reason=reason,
        program_ref=state["program_ref"],
        prior_state_ref=authority.state_ref,
        head=state["head"],
        stages=state["stages"],
        failure=failure,
        unresolved_operation=state["pending"],
        occupied=occupied_capacity(authority.program, state),
        qualification="finite_development_trajectory_no_independent_profitability_claim",
        protected_evidence="not_accessed",
        promotion="not_requested",
    )
    if terminal_evidence_ref is not None:
        receipt["terminal_evidence_ref"] = terminal_evidence_ref
    ref = authority.store.put_json(receipt)
    authority.update(
        "trajectory_sealed",
        lambda next_state: next_state.update(status="SEALED", terminal_ref=ref, pending=None),
    )


@registered_trajectory_resume
def resume_trajectory(*, steps=1):
    research.validate_steps(steps)
    with locked_trajectory() as (root, owner_fd):
        with TrajectoryLedger(root, owner_fd=owner_fd) as authority:
            return _resume_locked(authority, owner_fd, steps)


def _resume_locked(authority, owner_fd, steps):
    summarize(authority)
    for _ in range(steps):
        state = authority.state
        if state["status"] == "SEALED":
            break
        if authority.program.get("schema_version") == 3:
            from quant_rsi.forecast.recursive.forecast_recursive_trajectory import resume_step

            resume_step(authority, owner_fd)
            continue
        if not state["stages"]:
            reserve_stage(authority)
        elif state["pending"] is not None:
            try:
                execute_package_round(authority, owner_fd)
            except ValueError as error:
                seal(
                    authority,
                    "required_child_operation_failed",
                    failure=dict(type=type(error).__name__, message=str(error)),
                )
        elif state["generation"] == 0:
            stage = state["stages"][0]
            pair = authority.store.json(stage["receipts"][-1]["package_receipt_ref"])
            outcome = pair["decision"]["outcome"]
            if outcome != "patch":
                seal(
                    authority,
                    "first_proposal_no_patch"
                    if outcome == "no_patch"
                    else "change_proposal_rejected",
                )
            else:
                advance_head(
                    authority,
                    request_id=stage["request_id"] + ":head",
                    expected_head=state["head"]["package_ref"],
                    expected_revision=0,
                )
        elif len(state["stages"]) == 2:
            seal(authority, "grammar_exhausted")
        elif len(state["stages"]) >= authority.program["input"]["max_proposal_stages"]:
            seal(authority, "proposal_stage_limit")
        else:
            saturate(authority)
    return summarize(authority)


def verify_authority_receipts(authority):
    """Bind every committed parent decision and cost to its sealed child evidence."""
    state, program, store = authority.state, authority.program, authority.store
    if program.get("schema_version") == 3:
        from quant_rsi.forecast.recursive.forecast_recursive_trajectory import verify_authority

        verify_authority(authority)
        verify_terminal(authority)
        return
    initial_ref = program["initial_head"]["research_receipt_ref"]
    initial = store.json(initial_ref)
    if (
        store.json(program["proposer_ref"]) != changes.proposer_definition()
        or store.json(program["saturation_proposer_ref"]) != saturation_proposer()
    ):
        raise ValueError("trajectory frozen proposer descriptor changed")
    if (
        program["initial_head"] != head_from(initial_ref, initial)
        or program["initial_costs"] != initial["costs"]
        or program["source_files_ref"] != initial["files_ref"]
        or program["source_plan_ref"] != initial["research_plan_ref"]
        or program["source_data_ref"] != initial["data_ref"]
        or program["input"]["source_checkpoint_ref"] != initial["checkpoint_ref"]
    ):
        raise ValueError("trajectory frozen source/cost/checkpoint binding mismatch")
    source_files = store.json(program["source_files_ref"])
    capsule = source_evidence_capsule(store, initial)
    for stage in state["stages"]:
        authorization = store.json(stage["authorization_ref"])
        if stage["ordinal"] == 0:
            if (
                authorization["kind"] != "TrajectoryC4Authorization"
                or authorization["authority_id"] != state["authority_id"]
                or authorization["program_ref"] != state["program_ref"]
                or authorization["request_id"] != stage["request_id"]
                or authorization["capacity"] != stage["capacity"]
                or authorization["base_package_ref"] != program["initial_head"]["package_ref"]
                or authorization["source_files_ref"] != program["source_files_ref"]
                or authorization["development_data_ref"] != program["source_data_ref"]
            ):
                raise ValueError("trajectory stage authorization differs from its program")
            for pair in stage["receipts"]:
                package = store.json(pair["package_receipt_ref"])
                manifest = store.json(package["manifest_ref"])
                files = store.json(package["files_ref"])
                if (
                    manifest.get("trajectory_authorization_ref") != stage["authorization_ref"]
                    or package["campaign_id"] != authorization["child_campaign_id"]
                ):
                    raise ValueError("trajectory receipt belongs to another authorized child")
                if (
                    manifest["proposer_ref"] != program["proposer_ref"]
                    or manifest["source_files_ref"] != program["source_files_ref"]
                    or manifest["plan_ref"] != program["source_plan_ref"]
                    or manifest["evidence_ref"] != digest(canonical(capsule))
                    or store.json(manifest["evidence_ref"]) != capsule
                    or any(
                        files.get("evidence/" + name) != ref for name, ref in source_files.items()
                    )
                ):
                    raise ValueError(
                        "trajectory C4 proposer input differs from its frozen initial source"
                    )
            actual = dict.fromkeys(UNITS, 0)
            if stage["receipts"]:
                for ref in stage["receipts"][-1]["arm_receipts"].values():
                    costs = store.json(ref)["costs"]
                    for unit in UNITS:
                        actual[unit] += costs[unit]
            if actual != stage["consumed"]:
                raise ValueError("trajectory confirmed consumption differs from child receipts")
        else:
            source = store.json(state["head"]["research_receipt_ref"])
            expected = dict(
                kind="TrajectorySaturationRequest",
                request_id=stage["request_id"],
                stage=1,
                package_ref=state["head"]["package_ref"],
                semantic_ref=state["head"]["semantic_ref"],
                checkpoint_ref=state["head"]["checkpoint_ref"],
                research_receipt_ref=state["head"]["research_receipt_ref"],
                feedback_watermark=source["feedback_watermark"],
                events_ref=source["events_ref"],
                proposer_ref=program["saturation_proposer_ref"],
            )
            if (
                state["generation"] != 1
                or authorization != expected
                or stage["request_ref"] != stage["authorization_ref"]
                or stage["capacity"] != dict.fromkeys(UNITS, 0)
                or store.json(stage["response_ref"])
                != dict(
                    kind="TrajectorySaturationResponse",
                    request_ref=stage["request_ref"],
                    outcome="no_patch",
                    reason="grammar_exhausted",
                )
            ):
                raise ValueError(
                    "trajectory saturation receipt differs from its authenticated head"
                )
    if len(state["decisions"]) != state["generation"]:
        raise ValueError("trajectory semantic head lacks its CAS receipt")
    for request_id, ref in state["decisions"].items():
        receipt = store.json(ref)
        stage = state["stages"][0]
        pair = stage["receipts"][-1]
        candidate_ref = pair["arm_receipts"]["candidate"]
        candidate_head = head_from(candidate_ref, store.json(candidate_ref))
        expected_request = dict(
            kind="TrajectoryHeadCAS",
            request_id=request_id,
            expected_head=program["initial_head"]["package_ref"],
            expected_revision=0,
        )
        expected = dict(
            kind="TrajectoryExperimentalHeadReceipt",
            request=expected_request,
            previous_head=program["initial_head"],
            head=candidate_head,
            revision=1,
            package_receipt_ref=pair["package_receipt_ref"],
            qualification="experimental_development_head_not_accepted_or_release",
        )
        if (
            receipt != expected
            or state["head"] != candidate_head
            or stage["candidate_head"] != candidate_head
            or store.json(pair["package_receipt_ref"])["completion_ref"] is None
        ):
            raise ValueError(
                "trajectory head CAS receipt does not bind its complete actual candidate"
            )
    verify_terminal(authority)


def verify_terminal(authority):
    state, program, store = authority.state, authority.program, authority.store
    if state["terminal_ref"] is not None:
        terminal = store.json(state["terminal_ref"])
        last_event_ref = authority.connection.execute(
            "SELECT ref FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()[0]
        event = store.json(last_event_ref)
        prior = store.json(terminal["prior_state_ref"])
        if (
            terminal["kind"] != "TrajectoryTerminalReceipt"
            or event["kind"] != "trajectory_sealed"
            or event["before"] != terminal["prior_state_ref"]
            or terminal["program_ref"] != state["program_ref"]
            or terminal["head"] != state["head"]
            or terminal["stages"] != state["stages"]
            or terminal["unresolved_operation"] != prior["pending"]
            or terminal["occupied"] != occupied_capacity(program, prior)
            or (prior | dict(status="SEALED", terminal_ref=state["terminal_ref"], pending=None))
            != state
        ):
            raise ValueError("trajectory terminal receipt differs from its committed final state")


def summarize(authority):
    state, program = authority.state, authority.program
    verify_authority_receipts(authority)
    verify_research_receipt(
        authority.store, authority.store.json(program["initial_head"]["research_receipt_ref"])
    )
    for stage in state["stages"]:
        authority.store.get(stage["authorization_ref"])
        for pair in stage["receipts"]:
            verify_package_receipt(
                authority.store,
                authority.store.json(pair["package_receipt_ref"]),
                pair["arm_receipts"],
            )
        if "response_ref" in stage:
            authority.store.get(stage["response_ref"])
    used = {
        unit: program["initial_costs"][unit]
        + sum(stage["consumed"][unit] for stage in state["stages"])
        for unit in UNITS
    }
    occupied = occupied_capacity(program, state)
    terminal = authority.store.json(state["terminal_ref"]) if state["terminal_ref"] else None
    report = dict(
        authority_id=state["authority_id"],
        program_ref=state["program_ref"],
        state_ref=authority.state_ref,
        status=state["status"],
        generation=state["generation"],
        revision=state["revision"],
        head=state["head"],
        stages=state["stages"],
        pending=state["pending"],
        terminal=terminal,
        budget=dict(
            limit=program["input"]["budget"],
            confirmed_consumed=used,
            occupied=occupied,
            unused_irrevocable_commitment={unit: occupied[unit] - used[unit] for unit in UNITS},
            unreconciled_child_costs=state["pending"] is not None
            or bool(terminal and terminal["unresolved_operation"]),
        ),
        boundary=program["boundary"],
    )
    if program.get("schema_version") == 3:
        from quant_rsi.forecast.recursive.forecast_recursive_trajectory import execution_counts

        report["execution_counts"] = execution_counts(authority)
        report["recursive_campaign_binding"] = program["recursive_campaign_binding"]
        # The shared grant reserves both pairs upfront, even if no later stage activates.
        reserved = program["input"]["budget"]
        report["shared_campaign_budget"] = dict(
            reserved=reserved,
            confirmed_consumed=used,
            activated_trajectory_commitment=occupied,
            unactivated_reserved_commitment={
                unit: reserved[unit] - occupied[unit] for unit in UNITS
            },
            unused_irrevocable_commitment={unit: reserved[unit] - used[unit] for unit in UNITS},
            unreconciled_child_costs=report["budget"]["unreconciled_child_costs"],
        )
    return report


def inspect_trajectory():
    root = trajectory_root()
    with TrajectoryLedger(root, readonly=True) as authority:
        return summarize(authority)
