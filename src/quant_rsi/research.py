"""Finite-step development research over the shared, trusted Experiment Kernel."""

import os
import sqlite3
import uuid
from pathlib import Path

from quant_rsi import controller, researcher
from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.data import MarketData, synthetic_data
from quant_rsi.forecast_contracts import parse_development_plan
from quant_rsi.forecast_recursive_registration import (
    effective_budget,
    registered_creation,
    registered_resume,
)
from quant_rsi.journal import TERMINAL
from quant_rsi.research_journal import ResearchJournal
from quant_rsi.researcher import ResearchPlan, advance, factor_identity, initial_state
from quant_rsi.schemas import FactorSpec, Proposal


def authorize_managed(
    root,
    plan,
    campaign_id,
    context,
    view,
    owner_fd,
    trajectory_owner_fd,
    *,
    steps=None,
    committed=0,
):
    """Check actual parent authority before any managed research publication."""
    if plan.development.plan_type == "forecast_development" and (
        context is not None or view is not None or plan.package_lineage is not None
    ):
        if (
            plan.schema_version != 3
            or view is not None
            or context is None
            or "trajectory_authorization_ref" not in context
        ):
            raise ValueError("forecast development does not support C4/C5 managed research")
    if context is None:
        if view is not None:
            raise ValueError("bound development data requires a controller grant")
        if plan.schema_version == 3:
            from quant_rsi.forecast_recursive_registration import require_root_identity

            require_root_identity(plan, root, campaign_id, trajectory_owner_fd)
        return
    if owner_fd is None and plan.development.data.source != "local_event_snapshot":
        raise ValueError("managed research arm requires package resume")
    if plan.schema_version == 3:
        from quant_rsi.forecast_recursive_campaign import require_recursive_owner

        require_recursive_owner()
    arguments = dict(steps=steps, committed=committed)
    if view is not None:
        from quant_rsi.local_evidence import authorized_view

        if view != authorized_view(root, plan, campaign_id, context, owner_fd, **arguments):
            raise ValueError("bound development view differs from the authority grant")
        validate_bound_view(view, plan)
    else:
        from quant_rsi.package_experiment import authorized_local_input

        authorized_local_input(
            root,
            plan,
            campaign_id,
            context,
            owner_fd,
            trajectory_owner_fd=trajectory_owner_fd,
            **arguments,
        )


def preflight_managed(root, owner_fd, trajectory_owner_fd, *, steps=None):
    """Read persisted identity without creating locks, journals, or artifacts."""
    marker = root / "research_initialization.json"
    manifest_path = root / "manifest.json"
    if marker.exists():
        saved = decode(marker.read_bytes())
        authorize_managed(
            root,
            ResearchPlan.model_validate_json(canonical(saved["plan"])),
            saved["campaign_id"],
            saved.get("launch_context"),
            saved.get("bound_view"),
            owner_fd,
            trajectory_owner_fd,
        )
    elif manifest_path.exists():
        manifest = decode(manifest_path.read_bytes())
        if "launch_context_ref" not in manifest:
            store = ArtifactStore(root / "artifacts")
            plan = ResearchPlan.model_validate_json(store.get(manifest["research_plan_ref"]))
            if plan.schema_version == 3:
                authorize_managed(
                    root, plan, manifest["campaign_id"], None, None, owner_fd, trajectory_owner_fd
                )
            return
        if owner_fd is None:
            raise ValueError("managed research arm requires package resume")
        store = ArtifactStore(root / "artifacts")
        plan = ResearchPlan.model_validate_json(store.get(manifest["research_plan_ref"]))
        with sqlite3.connect(f"{(root / 'journal.sqlite').as_uri()}?mode=ro", uri=True) as db:
            committed = db.execute(
                "SELECT COUNT(*) FROM research_requests WHERE observation_ref IS NOT NULL"
            ).fetchone()[0]
        authorize_managed(
            root,
            plan,
            manifest["campaign_id"],
            store.json(manifest["launch_context_ref"]),
            store.json(manifest["bound_view_ref"]) if "bound_view_ref" in manifest else None,
            owner_fd,
            trajectory_owner_fd,
            steps=steps,
            committed=committed,
        )


@registered_creation
def create_research(
    plan: ResearchPlan,
    root: Path,
    *,
    steps: int = 1,
    campaign_id: str | None = None,
    package_artifacts: dict | None = None,
    launch_context: dict | None = None,
    owner_fd: int | None = None,
    initialize_only: bool = False,
    bound_view: dict | None = None,
    trajectory_owner_fd: int | None = None,
) -> dict:
    validate_steps(steps)
    root = root.resolve()
    preflight_managed(root, owner_fd, trajectory_owner_fd, steps=steps)
    authorize_managed(
        root, plan, campaign_id, launch_context, bound_view, owner_fd, trajectory_owner_fd
    )
    if plan.development.data.source == "local_event_snapshot":
        if bound_view is not None:
            from quant_rsi.local_evidence import authorized_view

            if bound_view != authorized_view(root, plan, campaign_id, launch_context, owner_fd):
                raise ValueError("local C5 view differs from the authority grant")
            validate_bound_view(bound_view, plan)
        elif launch_context is not None:
            from quant_rsi.package_experiment import authorized_local_input

            authorized_local_input(
                root,
                plan,
                campaign_id,
                launch_context,
                owner_fd,
                trajectory_owner_fd=trajectory_owner_fd,
            )
    root.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(root):
        preflight_managed(root, owner_fd, trajectory_owner_fd, steps=steps)
        authorize_managed(
            root, plan, campaign_id, launch_context, bound_view, owner_fd, trajectory_owner_fd
        )
        marker = root / "research_initialization.json"
        if marker.exists():
            previous = decode(marker.read_bytes())
            if (
                previous["plan"] != plan.model_dump(mode="json")
                or (campaign_id is not None and previous["campaign_id"] != campaign_id)
                or previous.get("package_artifacts") != package_artifacts
                or previous.get("launch_context") != launch_context
                or previous.get("bound_view") != bound_view
            ):
                raise ValueError("unfinished research initialization belongs to a different plan")
        else:
            if any(
                p.name != ".controller.lock" and not p.name.startswith(".pending-")
                for p in root.iterdir()
            ):
                raise FileExistsError("research campaign exists; use research resume")
            atomic_write(
                marker,
                canonical(
                    {
                        "kind": "research_initialization_v1",
                        "campaign_id": campaign_id or uuid.uuid4().hex,
                        "plan": plan.model_dump(mode="json"),
                        "runtime": controller.runtime_identity(),
                        "package_artifacts": package_artifacts,
                        "launch_context": launch_context,
                        "bound_view": bound_view,
                    }
                ),
            )
        finish_initialization(root, owner_fd=owner_fd, trajectory_owner_fd=trajectory_owner_fd)
    if initialize_only:
        return {"initialized": True}
    return resume_research(
        root, steps=steps, owner_fd=owner_fd, trajectory_owner_fd=trajectory_owner_fd
    )


def finish_initialization(root: Path, *, owner_fd=None, trajectory_owner_fd=None):
    marker = root / "research_initialization.json"
    if not marker.exists():
        return
    saved = decode(marker.read_bytes())
    if saved["kind"] != "research_initialization_v1":
        raise ValueError("unknown research initialization")
    if saved["runtime"] != controller.runtime_identity():
        raise ValueError("source/runtime changed during research initialization")
    plan = ResearchPlan.model_validate_json(canonical(saved["plan"]))
    authorize_managed(
        root,
        plan,
        saved["campaign_id"],
        saved.get("launch_context"),
        saved.get("bound_view"),
        owner_fd,
        trajectory_owner_fd,
    )
    store = ArtifactStore(root / "artifacts")
    if plan.schema_version == 3:
        store.put_json(plan.neighbor_domain.model_dump(mode="json"))
        store.put_json(plan.forecast_profile.model_dump(mode="json"))
    for ref, value in (saved.get("package_artifacts") or {}).items():
        if digest(canonical(value)) != ref:
            raise ValueError("package artifact reference mismatch")
        store.put_json(value)
    if plan.package_lineage is not None:
        from quant_rsi.changes import validate_lineage

        validate_lineage(plan, store)
    bound_view = saved.get("bound_view")
    if plan.development.data.source == "local_event_snapshot":
        from quant_rsi.local_research import export_local_data

        if bound_view is not None:
            from quant_rsi.local_evidence import authorized_view

            if bound_view != authorized_view(
                root, plan, saved["campaign_id"], saved.get("launch_context"), owner_fd
            ):
                raise ValueError("local C5 bootstrap view differs from the authority grant")
            validate_bound_view(bound_view, plan)
            local_data = bound_view["data"]
        elif saved.get("launch_context") is None:
            local_data = export_local_data(plan.development.data)
        else:
            from quant_rsi.package_experiment import authorized_local_input

            local_data = authorized_local_input(
                root,
                plan,
                saved["campaign_id"],
                saved["launch_context"],
                owner_fd,
                trajectory_owner_fd=trajectory_owner_fd,
            )
        data_ref = store.put_json(local_data)
    elif bound_view is None:
        data_ref = store.put_json(synthetic_data(plan.development.data).to_json())
    else:
        validate_bound_view(bound_view, plan)
        if saved.get("launch_context") is None:
            raise ValueError("bound development data requires a controller grant")
        data_ref = store.put_json(bound_view["data"])
    template_ref = store.put_json(plan.development.model_dump(mode="json"))
    from quant_rsi.task_definition import task_definition

    task_ref = store.put_json(task_definition(plan, data_ref, template_ref))
    package_ref = store.put_json(researcher.package_definition(plan))
    memory_ref = store.put_json(
        {
            "kind": "MemorySnapshot",
            "schema_version": 1,
            "campaign_id": saved["campaign_id"],
            "task_ref": task_ref,
            "parent_ref": None,
            "observation_refs": [],
        }
    )
    manifest = {
        "kind": "ResearchRun",
        "schema_version": 1,
        "campaign_id": saved["campaign_id"],
        "research_plan_ref": store.put_json(saved["plan"]),
        "data_ref": data_ref,
        "development_template_ref": template_ref,
        "task_ref": task_ref,
        "runtime_ref": store.put_json(saved["runtime"]),
        "package_ref": package_ref,
        "initial_checkpoint_ref": store.put_json(
            initial_state(saved["campaign_id"], package_ref, memory_ref)
        ),
    }
    if saved.get("launch_context") is not None:
        manifest["launch_context_ref"] = store.put_json(saved["launch_context"])
    if plan.schema_version == 3 and saved.get("launch_context") is None:
        from quant_rsi.forecast_recursive_registration import require_root_identity

        manifest["root_registration_identity_ref"] = store.put_json(
            require_root_identity(plan, root, saved["campaign_id"], trajectory_owner_fd)
        )
    if bound_view is not None:
        manifest["bound_view_ref"] = store.put_json(bound_view)
    path = root / "manifest.json"
    if path.exists() and path.read_bytes() != canonical(manifest):
        raise ValueError("research initialization identity conflict")
    if not path.exists():
        atomic_write(path, canonical(manifest))
    journal = ResearchJournal(root, digest(canonical(manifest)))
    journal.close()
    marker.unlink()
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def open_research(root: Path, *, verify_runtime=True):
    raw = (root / "manifest.json").read_bytes()
    manifest = decode(raw)
    if manifest.get("kind") != "ResearchRun":
        raise ValueError("not a research campaign")
    journal = ResearchJournal(root)
    try:
        row = journal.connection.execute(
            "SELECT value FROM metadata WHERE key='manifest_digest'"
        ).fetchone()
        if row is None or row[0] != digest(raw):
            raise ValueError("research manifest integrity failure")
        store = ArtifactStore(root / "artifacts")
        plan = ResearchPlan.model_validate_json(store.get(manifest["research_plan_ref"]))
        if store.json(manifest["development_template_ref"]) != plan.development.model_dump(
            mode="json"
        ):
            raise ValueError("research template binding mismatch")
        if verify_runtime and store.json(manifest["runtime_ref"]) != controller.runtime_identity():
            raise ValueError("source/runtime changed; research requires recorded version")
        if verify_runtime and store.json(manifest["package_ref"]) != researcher.package_definition(
            plan
        ):
            raise ValueError("researcher package changed")
        store.get(manifest["runtime_ref"])
        store.get(manifest["package_ref"])
        if plan.package_lineage is not None:
            from quant_rsi.changes import validate_lineage

            validate_lineage(plan, store)
        from quant_rsi.forecast_recursive_registration import verify_root_manifest
        from quant_rsi.task_definition import task_definition, verify_domain

        verify_domain(plan, store)
        verify_root_manifest(plan, store, manifest)
        if store.json(manifest["task_ref"]) != task_definition(
            plan, manifest["data_ref"], manifest["development_template_ref"]
        ):
            raise ValueError("research task binding mismatch")
        data = MarketData.from_json(store.json(manifest["data_ref"]))
        from quant_rsi.local_research import LocalResearchData

        if isinstance(data, LocalResearchData):
            if data.spec != plan.development.data:
                raise ValueError("local research data differs from the registered specification")
        elif plan.development.data.source == "local_event_snapshot":
            raise ValueError("local research requires its frozen local input")
        if "bound_view_ref" in manifest:
            view = store.json(manifest["bound_view_ref"])
            validate_bound_view(view, plan)
            if digest(canonical(view["data"])) != manifest["data_ref"]:
                raise ValueError("bound view differs from actual development input")
        return journal, store, manifest, plan, data
    except Exception:
        journal.close()
        raise


def validate_bound_view(view, plan):
    if set(view) != {"kind", "origin_ref", "data"} or view["kind"] != "BoundDevelopmentView":
        raise ValueError("invalid bound development view")
    if not isinstance(view["origin_ref"], str) or len(view["origin_ref"]) != 64:
        raise ValueError("invalid bound development origin")
    data = MarketData.from_json(view["data"])
    if plan.development.data.source == "local_event_snapshot":
        if view["data"]["kind"] != "DerivedLocalResearchData" or data.spec != plan.development.data:
            raise ValueError("local developer-visible data cannot be used as protected bound data")
        if digest(canonical(view["data"]["projection_lineage"])) != view["origin_ref"]:
            raise ValueError("local projection origin binding mismatch")
    elif view["data"]["kind"] != "synthetic_weekday_quotes_not_market_data":
        raise ValueError("local developer-visible data cannot be used as protected bound data")
    if (len(data.dates), len(data.instruments)) != (
        plan.development.data.sessions,
        plan.development.data.instruments,
    ):
        raise ValueError("development view shape differs from registered plan")


def request_body(manifest, ordinal, parent_ref, memory, watermark):
    return {
        "kind": "ProposalRequest",
        "campaign_id": manifest["campaign_id"],
        "ordinal": ordinal,
        "package_ref": manifest["package_ref"],
        "checkpoint_ref": parent_ref,
        "task_ref": manifest["task_ref"],
        "visibility": "development",
        "evidence_refs": memory["observation_refs"],
        "feedback_watermark": watermark,
        "budget_ledger_id": manifest["campaign_id"],
        "rng": None,
    }


def sealed_experiment(store, manifest, plan, row, proposal, *, publish=True):
    def reference(value):
        return store.put_json(value) if publish else digest(canonical(value))

    document = plan.development.model_dump(mode="json")
    document.update(baseline=proposal["name"], proposals=[proposal])
    child = parse_development_plan(canonical(document))
    context = {
        "campaign_id": manifest["campaign_id"],
        "research_plan_ref": manifest["research_plan_ref"],
        "package_ref": manifest["package_ref"],
        "request_ref": row["request_ref"],
        "checkpoint_ref": row["parent_ref"],
        "reservation_ordinal": row["ordinal"],
    }
    experiment = {
        "schema_version": 1,
        "plan_ref": reference(child.model_dump(mode="json")),
        "data_ref": manifest["data_ref"],
        "runtime_ref": manifest["runtime_ref"],
        "research_context_ref": reference(context),
    }
    return child, experiment


def commit_proposal(root, journal, store, manifest, plan, row, checkpoint):
    path = root / "proposals" / f"{row['ordinal']:04d}.json"
    if path.exists():
        response = decode(path.read_bytes())
    else:
        try:
            proposal = researcher.propose(plan, checkpoint)
            # Invalid factors remain representable for the Kernel's rejection ledger.
            proposal = Proposal.model_validate_json(canonical(proposal)).model_dump(mode="json")
            if proposal["name"] != f"research_{row['ordinal']:04d}":
                raise ValueError("proposal name does not identify its reserved step")
            error = None
        except Exception as exc:
            proposal, error = None, f"proposal_failed: {type(exc).__name__}: {exc}"
        response = {
            "kind": "ProposalResponse",
            "request_ref": row["request_ref"],
            "proposal": proposal,
            "error": error,
        }
        atomic_write(path, canonical(response))
    response_ref = store.put_json(response)
    decision = validate_response(response, response_ref, row["request_ref"], plan, checkpoint)
    proposal = decision["proposal"]
    if proposal is not None:
        _, experiment = sealed_experiment(store, manifest, plan, row, proposal)
        proposal_ref, experiment_ref = store.put_json(proposal), store.put_json(experiment)
    else:
        proposal_ref = experiment_ref = None
    journal.commit_decision(
        row["ordinal"],
        store.put_json(decision),
        proposal_ref,
        experiment_ref,
        None if proposal is None else proposal["name"],
        plan.development.budget.max_attempts,
    )


def validate_response(response, response_ref, request_ref, plan, checkpoint):
    """Check deterministic policy content; retain a mismatched response as a rejected decision."""
    if (
        not isinstance(response, dict)
        or set(response) != {"kind", "request_ref", "proposal", "error"}
        or response["kind"] != "ProposalResponse"
        or response["request_ref"] != request_ref
    ):
        raise ValueError("proposal response binding mismatch")
    proposal, error = response["proposal"], response["error"]
    if proposal is not None:
        proposal = Proposal.model_validate_json(canonical(proposal)).model_dump(mode="json")
        if proposal["name"] != f"research_{checkpoint['next_position']:04d}" or error is not None:
            raise ValueError("proposal response step identity mismatch")
        if proposal != researcher.expected_proposal(plan, checkpoint):
            error = "proposal_policy_mismatch"
    elif not isinstance(error, str) or not error.startswith("proposal_failed:"):
        raise ValueError("missing proposal has no recorded failure")
    return {
        "kind": "ResearchDecision",
        "request_ref": request_ref,
        "response_ref": response_ref,
        "proposal": proposal,
        "error": error,
    }


def make_observation(store, manifest, plan, row, attempt):
    decision = store.json(row["decision_ref"])
    proposal = decision["proposal"]
    report = None
    if attempt is not None and attempt["status"] == "completed":
        report = store.json(store.json(attempt["result_ref"])["report_ref"])
    return {
        "kind": "ResearchObservation",
        "campaign_id": manifest["campaign_id"],
        "task_ref": manifest["task_ref"],
        "visibility": "development",
        "ordinal": row["ordinal"],
        "request_ref": row["request_ref"],
        "decision_ref": row["decision_ref"],
        "proposal_ref": row["proposal_ref"],
        "factor": None if proposal is None else proposal["factor"],
        "status": "proposal_failed" if attempt is None else attempt["status"],
        "error": decision["error"] if attempt is None else attempt["error"],
        "result_ref": None if attempt is None else attempt["result_ref"],
        "eligible": False if report is None else report["eligible"],
        "score": None if report is None else report["metrics"][plan.development.primary_metric],
        "availability": "only_after_research_step_commit",
    }


def derived_state(store, plan, previous_ref, previous, observation_ref):
    memory = store.json(previous["memory_ref"])
    observation = store.json(observation_ref)
    if observation["task_ref"] != memory["task_ref"]:
        raise ValueError("observation belongs to another research task")
    next_memory = memory | {
        "parent_ref": previous["memory_ref"],
        "observation_refs": memory["observation_refs"] + [observation_ref],
    }
    memory_ref = digest(canonical(next_memory))
    checkpoint = advance(plan, previous, observation, memory_ref, previous_ref)
    return next_memory, checkpoint


def verify_trace(journal, store, manifest, plan, data, *, replay=False, semantic=True):
    """Validate availability, bindings, budget mapping and the committed checkpoint chain."""
    initial_ref = manifest["initial_checkpoint_ref"]
    head_ref, checkpoint = initial_ref, store.json(initial_ref)
    memory = store.json(checkpoint["memory_ref"])
    if memory != {
        "kind": "MemorySnapshot",
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "task_ref": manifest["task_ref"],
        "parent_ref": None,
        "observation_refs": [],
    } or checkpoint != initial_state(
        manifest["campaign_id"], manifest["package_ref"], checkpoint["memory_ref"]
    ):
        raise ValueError("initial research state binding mismatch")
    requests, attempts = journal.requests(), {a["ordinal"]: a for a in journal.attempts()}
    watermark = 0
    verified = 0
    for ordinal, row in enumerate(requests):
        expected_request = request_body(manifest, ordinal, head_ref, memory, watermark)
        if (
            row["ordinal"] != ordinal
            or row["parent_ref"] != head_ref
            or store.json(row["request_ref"]) != expected_request
            or row["request_event_seq"] <= watermark
        ):
            raise ValueError("research request/evidence availability mismatch")
        require_event(
            journal,
            row["request_event_seq"],
            "research_request_reserved",
            ordinal,
            {
                "request_ref": row["request_ref"],
                "parent_ref": row["parent_ref"],
                "attempt_units": 1,
            },
        )
        attempt = attempts.get(ordinal)
        if row["decision_ref"] is not None:
            decision = store.json(row["decision_ref"])
            if decision["request_ref"] != row["request_ref"]:
                raise ValueError("research decision binding mismatch")
            response = store.json(decision["response_ref"])
            if semantic and decision != validate_response(
                response, decision["response_ref"], row["request_ref"], plan, checkpoint
            ):
                raise ValueError("recorded decision differs from frozen deterministic policy")
            if decision["proposal"] is not None:
                proposal = store.json(row["proposal_ref"])
                if (
                    attempt is None
                    or attempt["proposal_ref"] != row["proposal_ref"]
                    or proposal != decision["proposal"]
                ):
                    raise ValueError("research attempt mapping mismatch")
                if decision["error"] is not None and (
                    attempt["status"] not in {"reserved", "rejected"}
                    or attempt["worker_seconds"] != 0
                ):
                    raise ValueError("rejected research response was executed")
                child, expected_experiment = sealed_experiment(
                    store, manifest, plan, row, proposal, publish=False
                )
                experiment = store.json(row["experiment_ref"])
                if experiment != expected_experiment:
                    raise ValueError("research experiment binding mismatch")
                store.get(experiment["plan_ref"])
                store.get(experiment["research_context_ref"])
                if attempt["status"] == "completed":
                    bundle = store.json(attempt["result_ref"])
                    for ref in bundle.values():
                        store.get(ref)
                    expected_job = controller.job_for(ordinal, proposal, experiment, child)
                    metadata = store.json(bundle["worker_metadata_ref"])
                    if (
                        store.json(bundle["job_ref"]) != expected_job
                        or metadata["job_digest"] != expected_job["job_digest"]
                        or metadata["model_digest"] != bundle["model_ref"]
                    ):
                        raise ValueError("research result computation binding mismatch")
                if replay and attempt["status"] == "completed":
                    bundle = store.json(attempt["result_ref"])
                    controller.replay_result(
                        store,
                        bundle,
                        controller.job_for(ordinal, proposal, experiment, child),
                        child,
                        data,
                    )
                    verified += 1
            elif (
                attempt is not None
                or row["proposal_ref"] is not None
                or row["experiment_ref"] is not None
            ):
                raise ValueError("unbacked research attempt")
        elif attempt is not None:
            raise ValueError("attempt has no sealed research decision")
        if row["observation_ref"] is None:
            if ordinal != len(requests) - 1:
                raise ValueError("research advanced past an unobserved request")
            continue
        if attempt is not None and attempt["status"] not in TERMINAL:
            raise ValueError("uncommitted worker result used as feedback")
        observation = store.json(row["observation_ref"])
        if observation != make_observation(store, manifest, plan, row, attempt):
            raise ValueError("research observation provenance mismatch")
        if row["observation_event_seq"] <= row["request_event_seq"]:
            raise ValueError("invalid observation availability cursor")
        require_event(
            journal,
            row["observation_event_seq"],
            "research_step_committed",
            ordinal,
            {
                "observation_ref": row["observation_ref"],
                "checkpoint_ref": row["checkpoint_ref"],
                "parent_ref": row["parent_ref"],
            },
        )
        if semantic:
            next_memory, next_checkpoint = derived_state(
                store, plan, head_ref, checkpoint, row["observation_ref"]
            )
            if canonical(next_checkpoint) != store.get(row["checkpoint_ref"]) or canonical(
                next_memory
            ) != store.get(next_checkpoint["memory_ref"]):
                raise ValueError("research checkpoint transition mismatch")
        head_ref = row["checkpoint_ref"]
        checkpoint = store.json(head_ref)
        memory = store.json(checkpoint["memory_ref"])
        watermark = row["observation_event_seq"]
    if (journal.state()["head_ref"] or initial_ref) != head_ref:
        raise ValueError("research checkpoint head mismatch")
    if not set(attempts).issubset({r["ordinal"] for r in requests}):
        raise ValueError("kernel attempt bypassed campaign reservation")
    budget = effective_budget(plan, store, manifest)
    if (
        len(requests) > budget["max_attempts"]
        or sum(a["worker_seconds"] for a in attempts.values()) > budget["max_worker_seconds"]
    ):
        raise ValueError("campaign budget conservation failure")
    return head_ref, checkpoint, memory, watermark, verified


def require_event(journal, sequence, kind, ordinal, payload):
    event = journal.connection.execute(
        "SELECT * FROM events WHERE sequence=?", (sequence,)
    ).fetchone()
    if (
        event is None
        or event["kind"] != kind
        or event["ordinal"] != ordinal
        or event["payload"] != canonical(payload).decode()
    ):
        raise ValueError("research provenance event mismatch")


def stop_reason(plan, checkpoint, journal, *, budget=None):
    if checkpoint["next_position"] and checkpoint["baseline"] is None:
        return "baseline_unavailable"
    budget = budget or plan.development.budget.model_dump(mode="json")
    if len(journal.requests()) >= budget["max_attempts"]:
        return "attempt_budget_exhausted"
    used = sum(a["worker_seconds"] for a in journal.attempts())
    if used + budget["worker_timeout_seconds"] > budget["max_worker_seconds"]:
        return "worker_budget_exhausted"
    if checkpoint["no_progress"] >= plan.policy.max_no_progress:
        return "no_progress_limit"
    if researcher.next_factor(plan, checkpoint) is None:
        return "registered_neighborhood_exhausted"
    return None


def validate_steps(steps):
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer between 1 and 100")


@registered_resume
def resume_research(
    root: Path,
    *,
    steps: int = 1,
    owner_fd: int | None = None,
    trajectory_owner_fd: int | None = None,
) -> dict:
    validate_steps(steps)
    root = root.resolve()
    preflight_managed(root, owner_fd, trajectory_owner_fd, steps=steps)
    with controller.controller_lock(root) as lock_fd:
        preflight_managed(root, owner_fd, trajectory_owner_fd, steps=steps)
        finish_initialization(root, owner_fd=owner_fd, trajectory_owner_fd=trajectory_owner_fd)
        journal, store, manifest, plan, data = open_research(root)
        try:
            if "launch_context_ref" in manifest and owner_fd is None:
                raise ValueError("managed research arm requires package resume")
            if (
                "launch_context_ref" in manifest
                and plan.development.data.source == "local_event_snapshot"
            ):
                context = store.json(manifest["launch_context_ref"])
                arguments = dict(
                    steps=steps,
                    committed=sum(r["observation_ref"] is not None for r in journal.requests()),
                )
                if "bound_view_ref" in manifest:
                    from quant_rsi.local_evidence import authorized_view

                    view = authorized_view(
                        root, plan, manifest["campaign_id"], context, owner_fd, **arguments
                    )
                    if digest(canonical(view)) != manifest["bound_view_ref"]:
                        raise ValueError("local C5 view differs from its frozen grant")
                    frozen = view["data"]
                else:
                    from quant_rsi.package_experiment import authorized_local_input

                    frozen = authorized_local_input(
                        root,
                        plan,
                        manifest["campaign_id"],
                        context,
                        owner_fd,
                        trajectory_owner_fd=trajectory_owner_fd,
                        **arguments,
                    )
                if digest(canonical(frozen)) != manifest["data_ref"]:
                    raise ValueError("managed local input differs from its frozen evidence")
            if owner_fd is not None:
                lock_fd = (lock_fd, owner_fd)
            if trajectory_owner_fd is not None:
                lock_fd = (
                    *(lock_fd if isinstance(lock_fd, tuple) else (lock_fd,)),
                    trajectory_owner_fd,
                )
            if plan.schema_version == 3:
                from quant_rsi.forecast_recursive_campaign import require_recursive_owner

                shared_fd = require_recursive_owner().owner_fd
                lock_fd = tuple(
                    dict.fromkeys(
                        (*(lock_fd if isinstance(lock_fd, tuple) else (lock_fd,)), shared_fd)
                    )
                )
            trace = verify_trace(journal, store, manifest, plan, data)
            if journal.state()["execution_state"] == "completed":
                return summarize(journal, store, manifest, plan, trace)
            completed_here = 0
            while True:
                head_ref, checkpoint, memory, watermark, _ = trace
                requests = journal.requests()
                pending = (
                    requests[-1] if requests and requests[-1]["observation_ref"] is None else None
                )
                if pending is None:
                    reason = stop_reason(
                        plan, checkpoint, journal, budget=effective_budget(plan, store, manifest)
                    )
                    if reason is not None:
                        journal.stop("completed", reason)
                        break
                    if completed_here >= steps:
                        journal.stop("paused", "step_limit")
                        break
                    request = request_body(manifest, len(requests), head_ref, memory, watermark)
                    ordinal = journal.reserve_request(
                        store.put_json(request),
                        head_ref,
                        manifest["initial_checkpoint_ref"],
                        effective_budget(plan, store, manifest)["max_attempts"],
                    )
                    pending = journal.requests()[ordinal]
                ordinal = pending["ordinal"]
                if pending["decision_ref"] is None:
                    commit_proposal(root, journal, store, manifest, plan, pending, checkpoint)
                    pending = journal.requests()[ordinal]
                if pending["proposal_ref"] is not None:
                    proposal = store.json(pending["proposal_ref"])
                    experiment = store.json(pending["experiment_ref"])
                    child = parse_development_plan(store.get(experiment["plan_ref"]))
                    attempt = next(a for a in journal.attempts() if a["ordinal"] == ordinal)
                    if attempt["status"] == "running":
                        controller.recover_attempt(
                            root, attempt, experiment, child, data, store, journal
                        )
                    elif attempt["status"] == "reserved":
                        decision = store.json(pending["decision_ref"])
                        if decision["error"] is not None:
                            journal.finish(ordinal, "rejected", error=decision["error"])
                        elif factor_identity(proposal["factor"]) in checkpoint["visited"]:
                            journal.finish(ordinal, "rejected", error="duplicate_factor")
                        else:
                            try:
                                factor = FactorSpec.model_validate_json(
                                    canonical(proposal["factor"])
                                )
                            except ValueError:
                                # The Kernel records its ordinary invalid-factor rejection.
                                factor = None
                            if (
                                factor is not None
                                and not plan.policy.window_min
                                <= factor.window
                                <= plan.policy.window_max
                            ):
                                journal.finish(
                                    ordinal, "rejected", error="outside_registered_search_range"
                                )
                            else:
                                controller.execute_attempt(
                                    root,
                                    ordinal,
                                    proposal,
                                    experiment,
                                    child,
                                    data,
                                    store,
                                    journal,
                                    lock_fd,
                                )
                attempt = next((a for a in journal.attempts() if a["ordinal"] == ordinal), None)
                observation = make_observation(store, manifest, plan, pending, attempt)
                observation_ref = store.put_json(observation)
                next_memory, next_checkpoint = derived_state(
                    store, plan, head_ref, checkpoint, observation_ref
                )
                store.put_json(next_memory)
                checkpoint_ref = store.put_json(next_checkpoint)
                journal.commit_observation(
                    ordinal,
                    head_ref,
                    manifest["initial_checkpoint_ref"],
                    observation_ref,
                    checkpoint_ref,
                )
                completed_here += 1
                trace = verify_trace(journal, store, manifest, plan, data)
            report = summarize(journal, store, manifest, plan, trace)
            atomic_write(root / "summary.json", canonical(report))
            return report
        finally:
            journal.close()


def summarize(journal, store, manifest, plan, trace):
    head_ref, checkpoint, _, _, verified = trace
    requests, attempts = journal.requests(), journal.attempts()
    used_workers = sum(a["worker_seconds"] for a in attempts)
    budget = effective_budget(plan, store, manifest)
    baseline = checkpoint["baseline"]
    results = []
    for row in requests:
        item = {
            key: row[key]
            for key in (
                "ordinal",
                "request_ref",
                "parent_ref",
                "decision_ref",
                "observation_ref",
                "checkpoint_ref",
                "observation_event_seq",
            )
        }
        if row["decision_ref"]:
            item["decision"] = store.json(row["decision_ref"])
            item["input_evidence_refs"] = store.json(row["request_ref"])["evidence_refs"]
        attempt = next((a for a in attempts if a["ordinal"] == row["ordinal"]), None)
        item["worker_seconds_charged"] = 0 if attempt is None else attempt["worker_seconds"]
        item["elapsed_seconds"] = None if attempt is None else attempt["elapsed_seconds"]
        if row["observation_ref"]:
            observation = store.json(row["observation_ref"])
            item["observation"] = observation
            item["delta_vs_baseline"] = (
                None
                if baseline is None or observation["score"] is None
                else (observation["score"] - baseline["score"])
            )
            event = journal.connection.execute(
                "SELECT timestamp FROM events WHERE sequence=?", (row["observation_event_seq"],)
            ).fetchone()
            item["observation_available_at"] = event[0]
            prior = store.json(row["parent_ref"])["incumbent"]
            item["delta_vs_prior_incumbent"] = (
                None
                if prior is None or observation["score"] is None
                else observation["score"] - prior["score"]
            )
            next_selection = store.json(row["checkpoint_ref"])["incumbent"]
            item["selected_as_development_incumbent"] = (
                next_selection is not None and next_selection["ordinal"] == row["ordinal"]
            )
        results.append(item)
    state = journal.state()
    return {
        "campaign_id": manifest["campaign_id"],
        "execution_state": state["execution_state"],
        "stop_reason": state["stop_reason"],
        "research_outcome": "no_valid_incumbent"
        if checkpoint["incumbent"] is None
        else "development_candidate",
        "evidence_status": "development_only",
        "promotion_decision": "not_requested",
        "learning_scope": "frozen_package_memory_search_state",
        "data_source": plan.development.data.source,
        "data_ref": manifest["data_ref"],
        "research_plan_ref": manifest["research_plan_ref"],
        "training_mode": "M0",
        "capacity_status": "unverified",
        "package_ref": manifest["package_ref"],
        "checkpoint_ref": head_ref,
        "incumbent": checkpoint["incumbent"],
        "baseline": baseline,
        "no_progress": checkpoint["no_progress"],
        "attempts_reserved": len(requests),
        "attempt_limit": budget["max_attempts"],
        "attempts_remaining": budget["max_attempts"] - len(requests),
        "pending_requests": sum(r["observation_ref"] is None for r in requests),
        "kernel_attempts": len(attempts),
        "unmapped_proposal_reservations": len(requests) - len(attempts),
        "worker_seconds_charged": used_workers,
        "worker_seconds_remaining": budget["max_worker_seconds"] - used_workers,
        "worker_seconds_limit": budget["max_worker_seconds"],
        "worker_charging_rule": "full_timeout_slot_per_started_attempt_no_refund",
        "results": results,
        "verified_experiments": verified,
    }


def inspect_research(root: Path, *, replay=False) -> dict:
    root = root.resolve()
    with controller.controller_lock(root):
        journal, store, manifest, plan, data = open_research(root, verify_runtime=replay)
        try:
            trace = verify_trace(
                journal, store, manifest, plan, data, replay=replay, semantic=replay
            )
            report = summarize(journal, store, manifest, plan, trace)
            if replay:
                report["replay_method"] = "recorded_lineage_and_kernel_accounting_no_new_research"
            return report
        finally:
            journal.close()
