"""Bounded development experiments on immutable researcher package variants."""

import fcntl
import os
import uuid
from pathlib import Path

from quant_rsi import changes, controller, research, researcher
from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.package_journal import PackageJournal, targets_for
from quant_rsi.researcher import ResearchPlan
from quant_rsi.trajectory_delegation import authorize_package, verify_delegation


def evidence_capsule(root, *, replay=True):
    """Read the committed source chain; never accept a user-written observation list."""
    with controller.controller_lock(root):
        journal, store, manifest, plan, data = research.open_research(root)
        try:
            if plan.development.plan_type == "forecast_development":
                raise ValueError("C4/C5 package evidence does not support forecast development")
            if plan.package_lineage is not None or plan.policy.neighbor_order != "window_first":
                raise ValueError("C4 evidence requires this engine's window_first base package")
            trace = research.verify_trace(journal, store, manifest, plan, data, replay=replay)
            requests = journal.requests()
            if not requests or any(r["observation_ref"] is None for r in requests):
                raise ValueError("evidence must have committed observations and no pending request")
            capsule = {
                "kind": "DevelopmentEvidence",
                "source_campaign_id": manifest["campaign_id"],
                "source_manifest_digest": digest(canonical(manifest)),
                "source_events_digest": digest(canonical(journal.events())),
                "task_ref": manifest["task_ref"],
                "feedback_watermark": trace[3],
                "entries": [
                    {
                        "observation_ref": r["observation_ref"],
                        "observation": store.json(r["observation_ref"]),
                        "observation_event_seq": r["observation_event_seq"],
                        "parent_checkpoint_ref": r["parent_ref"],
                        "parent_incumbent": store.json(r["parent_ref"])["incumbent"],
                    }
                    for r in requests
                ],
                "historical_attempts": len(requests),
                "historical_worker_slots": sum(a["worker_seconds"] for a in journal.attempts()),
                "reuse": "historical_development_not_independent_validation",
            }
            files = {
                str(p.relative_to(root)): digest(p.read_bytes())
                for p in [
                    root / "manifest.json",
                    root / "journal.sqlite",
                    *sorted((root / "artifacts").iterdir()),
                ]
                if p.is_file()
            }
            return plan, capsule, files
        finally:
            journal.close()


def create_package(
    evidence: Path,
    root: Path,
    *,
    state_policy: str,
    steps=1,
    trajectory_context=None,
    trajectory_owner_fd=None,
):
    research.validate_steps(steps)
    if state_policy != "reset":
        raise ValueError("only explicit reset is supported; checkpoint migration is forbidden")
    evidence, root = evidence.resolve(), root.resolve()
    if evidence == root or root in evidence.parents or evidence in root.parents:
        raise ValueError("evidence and package directories must be disjoint")
    binding = authorize_package(
        root,
        trajectory_owner_fd,
        context=trajectory_context,
        source=evidence,
        steps=steps,
    )
    root.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(root):
        marker = root / "package_initialization.json"
        if marker.exists():
            saved = decode(marker.read_bytes())
            if saved["source_directory"] != str(evidence):
                raise ValueError("package initialization belongs to different evidence")
        else:
            if any(
                p.name != ".controller.lock" and not p.name.startswith(".pending-")
                for p in root.iterdir()
            ):
                raise FileExistsError("package campaign exists; use package resume")
            plan, capsule, files = evidence_capsule(evidence)
            saved = {
                "kind": "package_initialization_v1",
                "campaign_id": uuid.uuid4().hex
                if binding is None
                else binding["authorization"]["child_campaign_id"],
                "runtime": controller.runtime_identity(),
                "source_directory": str(evidence),
                "source_files": files,
                "evidence": capsule,
                "plan": plan.model_dump(mode="json"),
            }
            if binding is not None:
                saved["trajectory_context"] = binding["context"]
                saved["trajectory_authorization"] = binding["authorization"]
            atomic_write(marker, canonical(saved))
        finish_initialization(root, trajectory_owner_fd=trajectory_owner_fd)
    return resume_package(root, steps=steps, trajectory_owner_fd=trajectory_owner_fd)


def finish_initialization(root, *, trajectory_owner_fd=None):
    marker = root / "package_initialization.json"
    if not marker.exists():
        return
    binding = authorize_package(root, trajectory_owner_fd)
    saved = decode(marker.read_bytes())
    if (
        saved["kind"] != "package_initialization_v1"
        or saved["runtime"] != controller.runtime_identity()
    ):
        raise ValueError("package initialization source/runtime mismatch")
    source = Path(saved["source_directory"])
    missing = [
        name
        for name, checksum in saved["source_files"].items()
        if not (root / "evidence" / name).exists()
    ]
    if missing:
        with controller.controller_lock(source):
            for name in missing:
                data = (source / name).read_bytes()
                if digest(data) != saved["source_files"][name]:
                    raise ValueError("evidence changed before snapshot was fully committed")
                atomic_write(root / "evidence" / name, data)
    verify_snapshot(root / "evidence", saved["source_files"])
    plan, evidence, _ = evidence_capsule(root / "evidence")
    if plan.model_dump(mode="json") != saved["plan"] or evidence != saved["evidence"]:
        raise ValueError("evidence snapshot differs from authorized source")
    store = ArtifactStore(root / "artifacts")
    base_ref = store.put_json(researcher.package_definition(plan))
    evidence_ref = store.put_json(evidence)
    proposer_ref = store.put_json(changes.proposer_definition())
    request = changes.request_definition(
        saved["campaign_id"] + ":change:0", base_ref, proposer_ref, evidence_ref, evidence
    )
    budget = plan.development.budget
    manifest = {
        "kind": "PackageExperiment",
        "schema_version": 1,
        "campaign_id": saved["campaign_id"],
        "runtime_ref": store.put_json(saved["runtime"]),
        "plan_ref": store.put_json(saved["plan"]),
        "base_package_ref": base_ref,
        "evidence_ref": evidence_ref,
        "source_files_ref": store.put_json(saved["source_files"]),
        "proposer_ref": proposer_ref,
        "request_ref": store.put_json(request),
        "accepted_pointer_ref": store.put_json(
            {
                "kind": "ReadOnlyLaunchBaseline",
                "package_ref": base_ref,
                "scope": "engineering_baseline_not_a_promotion_registry",
            }
        ),
        "limits": {
            "change_requests": 1,
            "research_attempts": 2 * budget.max_attempts,
            "worker_seconds": 2 * budget.max_worker_seconds,
        },
    }
    if binding is not None:
        manifest["trajectory_context_ref"] = store.put_json(binding["context"])
        manifest["trajectory_authorization_ref"] = store.put_json(binding["authorization"])
        verify_delegation(manifest, store)
    path = root / "manifest.json"
    if path.exists() and path.read_bytes() != canonical(manifest):
        raise ValueError("package initialization identity conflict")
    if not path.exists():
        atomic_write(path, canonical(manifest))
    journal = PackageJournal(root, digest(canonical(manifest)))
    journal.close()
    marker.unlink()
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def verify_snapshot(root, files):
    for name, checksum in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid evidence snapshot path")
        if digest((root / name).read_bytes()) != checksum:
            raise ValueError("evidence snapshot content changed")


def verify_package_header(journal, store, manifest):
    """Authenticate the same pure C4 inputs for live and frozen journals."""
    if manifest.get("kind") != "PackageExperiment":
        raise ValueError("not a package experiment")
    if store.json(manifest["runtime_ref"]) != controller.runtime_identity():
        raise ValueError("package experiment requires its recorded source/runtime")
    verify_delegation(manifest, store)
    plan = ResearchPlan.model_validate_json(store.get(manifest["plan_ref"]))
    if plan.package_lineage is not None or plan.policy.neighbor_order != "window_first":
        raise ValueError("package experiment requires a window_first base")
    if store.json(manifest["base_package_ref"]) != researcher.package_definition(plan):
        raise ValueError("package exact base mismatch")
    if store.json(manifest["proposer_ref"]) != changes.proposer_definition():
        raise ValueError("frozen change proposer mismatch")
    if store.json(manifest["accepted_pointer_ref"]) != {
        "kind": "ReadOnlyLaunchBaseline",
        "package_ref": manifest["base_package_ref"],
        "scope": "engineering_baseline_not_a_promotion_registry",
    }:
        raise ValueError("read-only accepted launch reference changed")
    request = changes.request_definition(
        manifest["campaign_id"] + ":change:0",
        manifest["base_package_ref"],
        manifest["proposer_ref"],
        manifest["evidence_ref"],
        store.json(manifest["evidence_ref"]),
    )
    if store.json(manifest["request_ref"]) != request:
        raise ValueError("change request exact input binding mismatch")
    budget = plan.development.budget
    if manifest["limits"] != {
        "change_requests": 1,
        "research_attempts": 2 * budget.max_attempts,
        "worker_seconds": 2 * budget.max_worker_seconds,
    }:
        raise ValueError("outer budget mismatch")
    if journal.attempts():
        raise ValueError("unexpected unallocated outer Kernel work")
    return plan


def open_package(root, *, replay=False):
    raw = (root / "manifest.json").read_bytes()
    manifest = decode(raw)
    if manifest.get("kind") != "PackageExperiment":
        raise ValueError("not a package experiment")
    journal = PackageJournal(root)
    try:
        if dict(journal.connection.execute("SELECT key,value FROM metadata")) != {
            "manifest_digest": digest(raw)
        }:
            raise ValueError("package manifest integrity failure")
        store = ArtifactStore(root / "artifacts")
        plan = verify_package_header(journal, store, manifest)
        verify_snapshot(root / "evidence", store.json(manifest["source_files_ref"]))
        source_plan, evidence, _ = evidence_capsule(root / "evidence", replay=replay)
        if source_plan != plan or evidence != store.json(manifest["evidence_ref"]):
            raise ValueError("source evidence is not the committed allowed snapshot")
        return journal, store, manifest, plan
    except Exception:
        journal.close()
        raise


def decision_content(store, manifest, plan, response):
    """Derive every candidate/grant byte without publishing artifacts."""
    objects = {}

    def ref(value):
        key = digest(canonical(value))
        objects[key] = value
        return key

    request = store.json(manifest["request_ref"])
    outcome, error = changes.validate_response(
        response, manifest["request_ref"], request, store.json(manifest["evidence_ref"])
    )
    decision = {
        "kind": "ChangeDecision",
        "request_ref": manifest["request_ref"],
        "response_ref": ref(response),
        "outcome": outcome,
        "error": error,
        "patch_ref": None,
        "candidate_package_ref": None,
        "grants": {},
    }
    if outcome == "patch":
        decision["patch_ref"] = ref(response["patch"])
        candidate = changes.experimental_plan(
            plan, decision["patch_ref"], manifest["request_ref"], request
        )
        decision["candidate_package_ref"] = ref(researcher.package_definition(candidate))
        for arm, selected in (("parent", plan), ("candidate", candidate)):
            grant = {
                "kind": "IrrevocableResearchGrant",
                "outer_campaign_id": manifest["campaign_id"],
                "arm": arm,
                "request_ref": manifest["request_ref"],
                "campaign_id": uuid.uuid5(uuid.UUID(manifest["campaign_id"]), arm).hex,
                "directory": "arms/" + arm,
                "state_policy": "reset",
                "research_plan_ref": ref(selected.model_dump(mode="json")),
                "package_ref": ref(researcher.package_definition(selected)),
                "budget": selected.development.budget.model_dump(mode="json"),
            }
            if "trajectory_authorization_ref" in manifest:
                grant["trajectory_authorization_ref"] = manifest["trajectory_authorization_ref"]
            decision["grants"][arm] = ref(grant)
    decision_ref = ref(decision)
    return decision_ref, decision, objects


def commit_change(root, journal, store, manifest, plan, *, trajectory_owner_fd=None):
    authorize_package(root, trajectory_owner_fd)
    path = root / "change_response.json"
    if path.exists():
        raw = path.read_bytes()
        try:
            response = decode(raw)
            canonical(response)
        except (ValueError, UnicodeError):
            response = {"kind": "UnparseableChangeResponse", "raw_ref": store.put(raw)}
    else:
        try:
            response = changes.propose_change(
                manifest["request_ref"],
                store.json(manifest["request_ref"]),
                store.json(manifest["evidence_ref"]),
            )
            canonical(response)
        except Exception as exc:
            response = {
                "kind": "ChangeResponse",
                "request_ref": manifest["request_ref"],
                "outcome": "failed",
                "patch": None,
                "reason": None,
                "supporting_observations": [],
                "error": f"change_failed: {type(exc).__name__}: {exc}",
            }
        atomic_write(path, canonical(response))
    decision_ref, decision, objects = decision_content(store, manifest, plan, response)
    for key, value in objects.items():
        if store.put_json(value) != key:
            raise ValueError("decision artifact identity changed")
    journal.decide(decision_ref, decision["grants"])


def launch_context(manifest, arm, grant_ref):
    context = {
        "package_campaign_id": manifest["campaign_id"],
        "arm": arm,
        "grant_ref": grant_ref,
        "state_policy": "reset",
    }
    if "trajectory_authorization_ref" in manifest:
        context["trajectory_authorization_ref"] = manifest["trajectory_authorization_ref"]
    return context


def authorized_local_input(
    root,
    plan,
    campaign_id,
    context,
    owner_fd,
    *,
    steps=None,
    committed=0,
    trajectory_owner_fd=None,
):
    """Resolve a local arm only through its actual locked C4 allocation and frozen source."""
    context_keys = {
        "package_campaign_id",
        "arm",
        "grant_ref",
        "state_policy",
    }
    if not isinstance(context, dict) or set(context) not in (
        context_keys,
        context_keys | {"trajectory_authorization_ref"},
    ):
        raise ValueError("local development requires an actual C4 grant")
    root = root.resolve()
    outer = root.parent.parent
    lock_path = outer / ".controller.lock"
    try:
        if owner_fd is None or not os.path.samestat(os.fstat(owner_fd), lock_path.stat()):
            raise ValueError("local C4 launch requires its owning controller lock")
        # A separate open must be blocked, while the inherited owning description
        # must already hold the lock. A matching filename or an arbitrary FD is not authority.
        with lock_path.open("r+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("local C4 controller lock is not held")
        fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, TypeError) as exc:
        raise ValueError("local C4 launch requires its owning controller lock") from exc
    authorize_package(outer, trajectory_owner_fd)
    journal, store, manifest, base = open_package(outer)
    try:
        decision = verify_decision(journal, store, manifest, base)
        if decision is None or decision["outcome"] != "patch":
            raise ValueError("local C4 launch has no committed allocation")
        arm = context["arm"]
        rows = {row["arm"]: row for row in journal.grants()}
        if arm not in rows or context != launch_context(manifest, arm, rows[arm]["grant_ref"]):
            raise ValueError("local C4 launch context differs from its actual grant")
        row = rows[arm]
        grant = store.json(row["grant_ref"])
        if (
            root != (outer / grant["directory"]).resolve()
            or campaign_id != grant["campaign_id"]
            or plan.model_dump(mode="json") != store.json(grant["research_plan_ref"])
            or researcher.package_definition(plan) != store.json(grant["package_ref"])
            or plan.development != base.development
        ):
            raise ValueError("local C4 arm/package/plan/input binding mismatch")
        rounds = journal.rounds()
        if (
            journal.state()["completion_ref"] is not None
            or not rounds
            or rounds[-1]["receipt_ref"] is not None
        ):
            raise ValueError("local C4 work requires an unfinished durable round")
        current = rounds[-1]
        targets = targets_for(current)
        require_event(
            journal,
            "package_round_started",
            {"steps": current["steps"], "targets": targets},
            current["ordinal"],
        )
        if steps is None:
            if row["initialized"]:
                raise ValueError("initialized local C4 arm cannot be recreated")
        elif not row["initialized"] or steps > targets[arm] - committed:
            raise ValueError("local C4 work exceeds its committed round target")
        source = ArtifactStore(outer / "evidence" / "artifacts")
        source_manifest = decode((outer / "evidence" / "manifest.json").read_bytes())
        evidence = store.json(manifest["evidence_ref"])
        value = source.json(source_manifest["data_ref"])
        from quant_rsi.data import MarketData

        data = MarketData.from_json(value)
        if (
            (
                plan.development.data.source == "local_event_snapshot"
                and data.spec != plan.development.data
            )
            or digest(canonical(source_manifest)) != evidence["source_manifest_digest"]
            or source_manifest["task_ref"] != evidence["task_ref"]
            or source.json(evidence["task_ref"])
            != {
                "data_ref": source_manifest["data_ref"],
                "development_template_ref": digest(
                    canonical(plan.development.model_dump(mode="json"))
                ),
            }
        ):
            raise ValueError("local C4 input differs from the frozen source task")
        return value
    finally:
        journal.close()


def arm_report(root, store, manifest, row, *, replay=False):
    grant = store.json(row["grant_ref"])
    path = root / grant["directory"]
    if not (path / "manifest.json").exists() or (path / "research_initialization.json").exists():
        if row["initialized"]:
            raise ValueError("initialized child is missing or reset; grant cannot be reused")
        if path.exists():
            if (path / "research_initialization.json").exists():
                saved = decode((path / "research_initialization.json").read_bytes())
                if (
                    saved["campaign_id"] != grant["campaign_id"]
                    or saved["plan"] != store.json(grant["research_plan_ref"])
                    or saved["launch_context"]
                    != launch_context(manifest, row["arm"], row["grant_ref"])
                ):
                    raise ValueError("pending child initialization conflicts with its grant")
            elif any(
                p.name != ".controller.lock" and not p.name.startswith(".pending-")
                for p in path.iterdir()
            ):
                raise ValueError("ambiguous child contents; refusing to recreate its ledger")
        return None
    with controller.controller_lock(path):
        journal, child_store, child_manifest, plan, data = research.open_research(path)
        try:
            if (
                child_manifest["campaign_id"] != grant["campaign_id"]
                or child_manifest["package_ref"] != grant["package_ref"]
                or child_manifest["research_plan_ref"] != grant["research_plan_ref"]
                or child_store.json(child_manifest["launch_context_ref"])
                != launch_context(manifest, row["arm"], row["grant_ref"])
                or child_manifest["task_ref"] != store.json(manifest["evidence_ref"])["task_ref"]
                or plan.development.budget.model_dump(mode="json") != grant["budget"]
            ):
                raise ValueError("child package/task/identity/grant binding mismatch")
            trace = research.verify_trace(
                journal, child_store, child_manifest, plan, data, replay=replay
            )
            report = research.summarize(journal, child_store, child_manifest, plan, trace)
            if not row["initialized"] and report["attempts_reserved"]:
                raise ValueError("child executed before durable outer initialization receipt")
            report["committed_observations"] = sum(
                r["observation_ref"] is not None for r in journal.requests()
            )
            return report
        finally:
            journal.close()


def require_event(journal, kind, payload, ordinal=None):
    matching = [e for e in journal.events() if e["kind"] == kind and e["ordinal"] == ordinal]
    if len(matching) != 1 or matching[0]["payload"] != payload:
        raise ValueError("package ledger event binding mismatch: " + kind)


def verify_decision(journal, store, manifest, plan):
    """Verify the actual reserved decision and grants without recursively opening arms."""
    state = journal.state()
    grants = journal.grants()
    if state["request_ref"] is None:
        if state["decision_ref"] is not None or grants or journal.rounds():
            raise ValueError("unreserved package work")
        return None
    if state["request_ref"] != manifest["request_ref"]:
        raise ValueError("package request reservation mismatch")
    require_event(journal, "change_reserved", {"request_ref": state["request_ref"]}, 0)
    if state["decision_ref"] is None:
        if grants or journal.rounds():
            raise ValueError("grants exist before candidate decision")
        return None
    recorded = store.json(state["decision_ref"])
    response = store.json(recorded["response_ref"])
    if isinstance(response, dict) and response.get("kind") == "UnparseableChangeResponse":
        store.get(response["raw_ref"])
    ref, decision, objects = decision_content(store, manifest, plan, response)
    if ref != state["decision_ref"] or decision != recorded:
        raise ValueError("recorded patch violates frozen proposer semantics")
    for key, value in objects.items():
        if store.json(key) != value:
            raise ValueError("missing or conflicting package artifact")
    if {r["arm"]: r["grant_ref"] for r in grants} != decision["grants"]:
        raise ValueError("candidate grants differ from committed allocation")
    if decision["outcome"] != "patch" and (journal.rounds() or state["completion_ref"] is not None):
        raise ValueError("non-candidate decision has experimental work")
    require_event(journal, "change_decided", {"decision_ref": ref, "grants": decision["grants"]}, 0)
    return decision


def verify_package(root, journal, store, manifest, plan, *, replay=False):
    decision = verify_decision(journal, store, manifest, plan)
    if decision is None:
        return None, {}
    grants = journal.grants()
    arms_path = root / "arms"
    if arms_path.exists() and {p.name for p in arms_path.iterdir()} - set(decision["grants"]):
        raise ValueError("unallocated child research directory")
    reports = {}
    for row in grants:
        reports[row["arm"]] = arm_report(root, store, manifest, row, replay=replay)
    return verify_package_records(journal, store, manifest, decision, reports)


def verify_package_records(journal, store, manifest, decision, reports):
    """Verify journal boundaries against already authenticated child reports."""
    if set(reports) != set(decision["grants"]):
        raise ValueError("child reports differ from committed grants")
    for row in journal.grants():
        report = reports[row["arm"]]
        if row["initialized"]:
            require_event(
                journal,
                "arm_initialized",
                {"arm": row["arm"], "grant_ref": row["grant_ref"]},
                0 if row["arm"] == "parent" else 1,
            )
        elif report is not None and report["attempts_reserved"]:
            raise ValueError("child executed before durable outer initialization receipt")
    state = journal.state()
    previous_counts = dict.fromkeys(decision["grants"], 0)
    for index, row in enumerate(journal.rounds()):
        targets = targets_for(row)
        if row["ordinal"] != index or set(targets) != set(decision["grants"]):
            raise ValueError("invalid package round")
        if targets != {arm: count + row["steps"] for arm, count in previous_counts.items()}:
            raise ValueError("package round targets changed")
        require_event(
            journal, "package_round_started", {"steps": row["steps"], "targets": targets}, index
        )
        if row["receipt_ref"] is None:
            if index != len(journal.rounds()) - 1:
                raise ValueError("unfinished package round precedes new round")
            break
        receipt = store.json(row["receipt_ref"])
        require_event(
            journal, "package_round_committed", {"receipt_ref": row["receipt_ref"]}, index
        )
        if receipt["round"] != index or set(receipt["arms"]) != set(targets):
            raise ValueError("package round receipt mismatch")
        for arm, saved in receipt["arms"].items():
            current = reports[arm]
            if (
                current is None
                or current["committed_observations"] < saved["committed_observations"]
            ):
                raise ValueError("previously observed child state disappeared")
            count = saved["committed_observations"]
            head = (
                current["results"][count - 1]["checkpoint_ref"]
                if count
                else (
                    current["results"][0]["parent_ref"]
                    if current["results"]
                    else current["checkpoint_ref"]
                )
            )
            if saved["checkpoint_ref"] != head or count > targets[arm]:
                raise ValueError("package round receipt does not identify the committed boundary")
            if saved["execution_state"] == "completed" and (
                current["execution_state"] != "completed"
                or count != current["committed_observations"]
            ):
                raise ValueError("package round falsely claims a terminal child")
            if (
                saved["committed_observations"] < targets[arm]
                and saved["execution_state"] != "completed"
            ):
                raise ValueError("package round stopped before its declared boundary")
            previous_counts[arm] = saved["committed_observations"]
    if state["completion_ref"] is not None:
        if not reports or any(
            r is None or r["execution_state"] != "completed" for r in reports.values()
        ):
            raise ValueError("package completion precedes terminal children")
        require_event(journal, "package_completed", {"completion_ref": state["completion_ref"]})
        completion = store.json(state["completion_ref"])
        if completion != comparison_receipt(manifest, decision, reports):
            raise ValueError("package completion differs from committed child outcomes")
    return decision, reports


def comparison_receipt(manifest, decision, reports):
    available = {arm: (report["incumbent"] if report else None) for arm, report in reports.items()}
    parent, candidate = available.get("parent"), available.get("candidate")
    return {
        "kind": "DevelopmentPackageComparison",
        "campaign_id": manifest["campaign_id"],
        "decision": decision,
        "accepted_pointer_ref": manifest["accepted_pointer_ref"],
        "children": {
            arm: None
            if report is None
            else {
                key: report[key]
                for key in (
                    "campaign_id",
                    "package_ref",
                    "checkpoint_ref",
                    "execution_state",
                    "stop_reason",
                    "incumbent",
                    "attempts_reserved",
                    "worker_seconds_charged",
                    "committed_observations",
                )
            }
            for arm, report in reports.items()
        },
        "candidate_minus_parent_development_score": None
        if parent is None or candidate is None
        else candidate["score"] - parent["score"],
        "evidence_status": "reused_development_only",
        "promotion_decision": "not_requested",
    }


def summarize(journal, store, manifest, decision, reports):
    granted_attempts = sum(
        store.json(r["grant_ref"])["budget"]["max_attempts"] for r in journal.grants()
    )
    granted_workers = sum(
        store.json(r["grant_ref"])["budget"]["max_worker_seconds"] for r in journal.grants()
    )
    attempts = sum(r["attempts_reserved"] for r in reports.values() if r is not None)
    workers = sum(r["worker_seconds_charged"] for r in reports.values() if r is not None)
    limits = manifest["limits"]
    if not (
        0 <= attempts <= granted_attempts <= limits["research_attempts"]
        and 0 <= workers <= granted_workers <= limits["worker_seconds"]
    ):
        raise ValueError("outer/child resource conservation failure")
    terminal = decision is not None and (
        decision["outcome"] != "patch" or journal.state()["completion_ref"] is not None
    )
    return {
        "campaign_id": manifest["campaign_id"],
        "execution_state": "completed" if terminal else "paused",
        "stop_reason": decision["outcome"]
        if terminal and decision["outcome"] != "patch"
        else ("development_comparison_complete" if terminal else "finite_round_or_pending"),
        "base_package_ref": manifest["base_package_ref"],
        "accepted_pointer": store.json(manifest["accepted_pointer_ref"]),
        "request_ref": manifest["request_ref"],
        "decision": decision,
        "committed_completion_ref": journal.state()["completion_ref"],
        "change_requests_reserved": int(journal.state()["request_ref"] is not None),
        "change_requests_limit": 1,
        "historical_evidence_cost": {
            k: store.json(manifest["evidence_ref"])[k]
            for k in ("historical_attempts", "historical_worker_slots", "reuse")
        },
        "research_budget": {
            "attempt_limit": limits["research_attempts"],
            "attempts_granted": granted_attempts,
            "attempts_consumed": attempts,
            "attempts_outstanding": granted_attempts - attempts,
            "attempts_unallocated": limits["research_attempts"] - granted_attempts,
            "worker_slot_limit": limits["worker_seconds"],
            "worker_slots_granted": granted_workers,
            "worker_slots_consumed": workers,
            "worker_slots_outstanding": granted_workers - workers,
            "worker_slots_unallocated": limits["worker_seconds"] - granted_workers,
            "grant_rule": "full_irrevocable_grants_no_refund",
        },
        "arms": reports,
        "comparison": comparison_receipt(manifest, decision, reports),
        "evidence_status": "development_only",
        "promotion_decision": "not_requested",
    }


def resume_package(root: Path, *, steps=1, trajectory_owner_fd=None):
    research.validate_steps(steps)
    root = root.resolve()
    authorize_package(root, trajectory_owner_fd, steps=steps)
    with controller.controller_lock(root) as owner_fd:
        finish_initialization(root, trajectory_owner_fd=trajectory_owner_fd)
        journal, store, manifest, plan = open_package(root)
        try:
            decision, reports = verify_package(root, journal, store, manifest, plan)
            if decision is not None and (
                decision["outcome"] != "patch" or journal.state()["completion_ref"] is not None
            ):
                return summarize(journal, store, manifest, decision, reports)
            journal.reserve(manifest["request_ref"])
            if decision is None:
                commit_change(
                    root, journal, store, manifest, plan, trajectory_owner_fd=trajectory_owner_fd
                )
                decision, reports = verify_package(root, journal, store, manifest, plan)
            if decision["outcome"] == "patch":
                rounds = journal.rounds()
                if (
                    rounds
                    and rounds[-1]["receipt_ref"] is not None
                    and all(
                        r is not None and r["execution_state"] == "completed"
                        for r in reports.values()
                    )
                ):
                    journal.complete(
                        store.put_json(comparison_receipt(manifest, decision, reports))
                    )
                    return summarize(journal, store, manifest, decision, reports)
                counts = {
                    arm: r["committed_observations"] if r else 0 for arm, r in reports.items()
                }
                row = journal.begin_round(steps, {arm: n + steps for arm, n in counts.items()})
                targets = targets_for(row)
                for arm in ("parent", "candidate"):
                    grant_row = next(r for r in journal.grants() if r["arm"] == arm)
                    grant = store.json(grant_row["grant_ref"])
                    child_root = root / grant["directory"]
                    if not grant_row["initialized"]:
                        if reports[arm] is None:
                            selected = ResearchPlan.model_validate_json(
                                store.get(grant["research_plan_ref"])
                            )
                            artifact_refs = [
                                manifest["base_package_ref"],
                                manifest["proposer_ref"],
                                manifest["request_ref"],
                                manifest["evidence_ref"],
                            ]
                            if decision["patch_ref"]:
                                artifact_refs.append(decision["patch_ref"])
                            if "trajectory_authorization_ref" in manifest:
                                artifact_refs.extend(
                                    [
                                        manifest["trajectory_context_ref"],
                                        manifest["trajectory_authorization_ref"],
                                    ]
                                )
                            research.create_research(
                                selected,
                                child_root,
                                campaign_id=grant["campaign_id"],
                                package_artifacts={ref: store.json(ref) for ref in artifact_refs},
                                launch_context=launch_context(
                                    manifest, arm, grant_row["grant_ref"]
                                ),
                                owner_fd=owner_fd,
                                initialize_only=True,
                                trajectory_owner_fd=trajectory_owner_fd,
                            )
                        journal.mark_initialized(arm)
                        grant_row = next(r for r in journal.grants() if r["arm"] == arm)
                        reports[arm] = arm_report(root, store, manifest, grant_row)
                    current = reports[arm]
                    remaining = targets[arm] - current["committed_observations"]
                    if remaining > 0 and current["execution_state"] != "completed":
                        research.resume_research(
                            child_root,
                            steps=remaining,
                            owner_fd=owner_fd,
                            trajectory_owner_fd=trajectory_owner_fd,
                        )
                    reports[arm] = arm_report(root, store, manifest, grant_row)
                receipt = {
                    "kind": "PackageRoundReceipt",
                    "round": row["ordinal"],
                    "arms": {
                        arm: {
                            key: report[key]
                            for key in (
                                "committed_observations",
                                "execution_state",
                                "checkpoint_ref",
                            )
                        }
                        for arm, report in reports.items()
                    },
                }
                journal.finish_round(row["ordinal"], store.put_json(receipt))
                if all(r["execution_state"] == "completed" for r in reports.values()):
                    journal.complete(
                        store.put_json(comparison_receipt(manifest, decision, reports))
                    )
            decision, reports = verify_package(root, journal, store, manifest, plan)
            report = summarize(journal, store, manifest, decision, reports)
            atomic_write(root / "summary.json", canonical(report))
            return report
        finally:
            journal.close()


def inspect_package(root: Path, *, replay=False):
    root = root.resolve()
    with controller.controller_lock(root):
        journal, store, manifest, plan = open_package(root, replay=replay)
        try:
            decision, reports = verify_package(root, journal, store, manifest, plan, replay=replay)
            return summarize(journal, store, manifest, decision, reports)
        finally:
            journal.close()
