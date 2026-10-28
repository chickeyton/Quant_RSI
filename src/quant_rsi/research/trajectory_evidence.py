"""Closed development evidence and semantic identities for finite trajectories."""

from pathlib import Path

from quant_rsi.core import controller
from quant_rsi.core.artifacts import canonical, decode, digest
from quant_rsi.core.journal import TERMINAL
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.evaluation import local_research
from quant_rsi.research import package_experiment, research


def semantic_definition(package):
    if package.get("kind") != "ResearcherPackage":
        raise ValueError("trajectory semantic identity requires a researcher package")
    ignored = {"lineage", "name", "display_name", "request_id", "generation"}
    return {
        "kind": "ResearcherSemanticIdentity",
        "definition": {key: value for key, value in package.items() if key not in ignored},
    }


def emitter(store, publish):
    def emit(raw):
        ref = digest(raw)
        if publish:
            if store.put(raw) != ref:
                raise ValueError("trajectory evidence identity changed")
        elif publish is False and store.get(ref) != raw:
            raise ValueError("sealed trajectory evidence differs from its committed bytes")
        return ref

    return emit


def charged_costs(journal):
    attempts = journal.attempts()
    elapsed = [row["elapsed_seconds"] for row in attempts if row["elapsed_seconds"] is not None]
    return {
        "worker_attempts": sum(row["worker_seconds"] > 0 for row in attempts),
        "registered_worker_seconds": sum(row["worker_seconds"] for row in attempts),
        "kernel_attempt_rows": len(attempts),
        "proposal_reservations": len(journal.requests()),
        "unmapped_proposal_reservations": len(journal.requests()) - len(attempts),
        "known_elapsed_seconds": sum(elapsed),
        "elapsed_missing_for_charged_attempts": [
            row["ordinal"]
            for row in attempts
            if row["worker_seconds"] > 0 and row["elapsed_seconds"] is None
        ],
        "attempts": attempts,
        "units": "registered charged worker attempts and timeout slots; elapsed is separate",
    }


def snapshot_files(root, emit, *, recursive):
    paths = (
        sorted(root.rglob("*"))
        if recursive
        else [
            root / "manifest.json",
            root / "journal.sqlite",
            *sorted((root / "artifacts").iterdir()),
        ]
    )
    return {
        str(path.relative_to(root)): emit(path.read_bytes())
        for path in paths
        if path.is_file() and path.name != ".controller.lock" and "__pycache__" not in path.parts
    }


def require_settled_research(journal, report):
    if (
        not journal.requests()
        or report["execution_state"] not in {"paused", "completed"}
        or report["pending_requests"]
        or any(row["status"] not in TERMINAL for row in journal.attempts())
    ):
        raise ValueError("trajectory source has unresolved research state or costs")


def capture_research(root, output_store, *, publish, initial=False, expected_checkpoint=None):
    """publish=None performs complete live verification and hashing without CAS publication."""
    root = Path(root).resolve()
    raw_manifest = decode((root / "manifest.json").read_bytes())
    if initial and any(key in raw_manifest for key in ("launch_context_ref", "bound_view_ref")):
        raise ValueError("trajectory initial source must be standalone C2, not a managed arm")
    emit = emitter(output_store, publish)

    def save(value):
        return emit(canonical(value))

    with controller.controller_lock(root):
        journal, store, manifest, plan, data = research.open_research(root)
        try:
            if plan.development.data.source != "local_event_snapshot":
                raise ValueError("trajectory admission requires actual local C2 research")
            if initial and (
                plan.package_lineage is not None or plan.policy.neighbor_order != "window_first"
            ):
                raise ValueError(
                    "trajectory initial source requires an unlineaged window_first package"
                )
            trace = research.verify_trace(journal, store, manifest, plan, data, replay=True)
            report = research.summarize(journal, store, manifest, plan, trace)
            requests = journal.requests()
            require_settled_research(journal, report)
            if expected_checkpoint is not None and trace[0] != expected_checkpoint:
                raise ValueError("trajectory source checkpoint differs from its registered input")
            files = snapshot_files(root, emit, recursive=False)
            incumbent = trace[1]["incumbent"]
            selection = {
                "kind": "TrajectoryDevelopmentSelection",
                "campaign_id": manifest["campaign_id"],
                "package_ref": manifest["package_ref"],
                "checkpoint_ref": trace[0],
                "data_ref": manifest["data_ref"],
                "research_plan_ref": manifest["research_plan_ref"],
                "role": "missing" if incumbent is None else "active",
                "incumbent": incumbent,
                "model_ref": None,
                "training_ref": None,
                "reason": "no_authoritative_valid_incumbent" if incumbent is None else None,
                "qualification": "development_selection_not_independent_scientific_evidence",
            }
            if incumbent is not None:
                bundle = store.json(incumbent["result_ref"])
                if store.json(bundle["report_ref"])["eligible"] is not True:
                    raise ValueError("trajectory incumbent has no eligible authoritative result")
                factor = FactorSpec.model_validate_json(canonical(incumbent["factor"]))
                metadata = store.json(bundle["worker_metadata_ref"])
                predictions = store.json(bundle["predictions_ref"])
                local_research.verify_local_result(
                    data,
                    plan.development,
                    factor,
                    metadata | {"predictions": predictions},
                    store.get(bundle["model_ref"]).decode(),
                )
                x, y, expected, mask = local_research.training_view(data, plan.development, factor)
                training_ref = save(
                    {
                        "kind": "TrajectoryLocalTrainingEvidence",
                        "data_ref": manifest["data_ref"],
                        "development_plan_ref": save(plan.development.model_dump(mode="json")),
                        "actual_mask_ref": save(mask),
                        "feature_values_ref": save(x.tolist()),
                        "mature_labels_ref": save(y.tolist()),
                        "result_ref": incumbent["result_ref"],
                        **expected,
                    }
                )
                selection.update(
                    model_ref=bundle["model_ref"],
                    training_ref=training_ref,
                    result_ref=incumbent["result_ref"],
                    predictions_ref=bundle["predictions_ref"],
                    worker_metadata_ref=bundle["worker_metadata_ref"],
                )
            package = store.json(manifest["package_ref"])
            receipt = {
                "kind": "TrajectoryResearchReceipt",
                "campaign_id": manifest["campaign_id"],
                "manifest_ref": save(manifest),
                "files_ref": save(files),
                "research_plan_ref": manifest["research_plan_ref"],
                "package_ref": manifest["package_ref"],
                "semantic_ref": save(semantic_definition(package)),
                "data_ref": manifest["data_ref"],
                "checkpoint_ref": trace[0],
                "feedback_watermark": trace[3],
                "events_ref": save(journal.events()),
                "requests_ref": save(requests),
                "selection_ref": save(selection),
                "development_report_ref": save(report),
                "costs": charged_costs(journal),
                "historical_exposure": "already_exposed_development_history_not_new_validation",
            }
            if plan.schema_version == 3:
                receipt["attempts_ref"] = save(journal.attempts())
            return save(receipt), receipt
        finally:
            journal.close()


def capture_package(root, output_store, *, publish):
    root = Path(root).resolve()
    emit = emitter(output_store, publish)
    with controller.controller_lock(root):
        journal, store, manifest, plan = package_experiment.open_package(root, replay=True)
        try:
            decision, reports = package_experiment.verify_package(
                root, journal, store, manifest, plan, replay=True
            )
            receipt = {
                "kind": "TrajectoryPackageReceipt",
                "campaign_id": manifest["campaign_id"],
                "manifest_ref": emit(canonical(manifest)),
                "files_ref": emit(canonical(snapshot_files(root, emit, recursive=True))),
                "decision_ref": journal.state()["decision_ref"],
                "decision": decision,
                "completion_ref": journal.state()["completion_ref"],
                "rounds": journal.rounds(),
                "grants": journal.grants(),
                "reports": reports,
            }
            return emit(canonical(receipt)), receipt
        finally:
            journal.close()


def verify_snapshot_receipt(store, receipt):
    files = store.json(receipt["files_ref"])
    if not files or "manifest.json" not in files or "journal.sqlite" not in files:
        raise ValueError("trajectory receipt lacks its actual manifest/journal snapshot")
    for relative, ref in files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid relative trajectory snapshot path")
        store.get(ref)
    if files["manifest.json"] != receipt["manifest_ref"]:
        raise ValueError("trajectory manifest differs from its snapshot receipt")


def source_evidence_capsule(store, receipt, authorization=None):
    """Reconstruct the C4 proposer input from its frozen initial research journal."""
    manifest = store.json(receipt["manifest_ref"])
    requests = store.json(receipt["requests_ref"])
    capsule = {
        "kind": "DevelopmentEvidence",
        "source_campaign_id": manifest["campaign_id"],
        "source_manifest_digest": receipt["manifest_ref"],
        "source_events_digest": receipt["events_ref"],
        "task_ref": manifest["task_ref"],
        "feedback_watermark": receipt["feedback_watermark"],
        "entries": [
            {
                "observation_ref": row["observation_ref"],
                "observation": store.json(row["observation_ref"]),
                "observation_event_seq": row["observation_event_seq"],
                "parent_checkpoint_ref": row["parent_ref"],
                "parent_incumbent": store.json(row["parent_ref"])["incumbent"],
            }
            for row in requests
        ],
        "historical_attempts": len(requests),
        "historical_worker_slots": receipt["costs"]["registered_worker_seconds"],
        "reuse": "historical_development_not_independent_validation",
    }
    from quant_rsi.research.researcher import ResearchPlan

    plan = ResearchPlan.model_validate_json(store.get(receipt["research_plan_ref"]))
    if plan.neighbor_domain is not None:
        from quant_rsi.research.changes import registered_proposer

        capsule = registered_proposer(authorization).extend_capsule(
            capsule,
            plan,
            receipt["checkpoint_ref"],
            store.json(receipt["checkpoint_ref"]),
            {row["parent_ref"]: store.json(row["parent_ref"]) for row in requests},
            authorization,
            requests,
            store.json(receipt["attempts_ref"]),
        )
    return capsule


def frozen_journal(store, receipt, journal_type):
    """Deserialize a sealed SQLite image in memory; never open a mutable child."""
    import sqlite3

    verify_snapshot_receipt(store, receipt)
    files = store.json(receipt["files_ref"])
    journal = journal_type.__new__(journal_type)
    journal.connection = sqlite3.connect(":memory:")
    journal.connection.deserialize(store.get(files["journal.sqlite"]))
    journal.connection.row_factory = sqlite3.Row
    journal.connection.execute("PRAGMA query_only=ON")
    if dict(journal.connection.execute("SELECT key,value FROM metadata")) != {
        "manifest_digest": receipt["manifest_ref"]
    }:
        journal.close()
        raise ValueError("frozen trajectory journal/manifest identity mismatch")
    return journal


def verify_research_receipt(store, receipt):
    from quant_rsi.core.data import MarketData
    from quant_rsi.forecast.recursive.forecast_recursive_registration import verify_root_manifest
    from quant_rsi.research import changes, researcher
    from quant_rsi.research.research_journal import ResearchJournal
    from quant_rsi.research.researcher import ResearchPlan
    from quant_rsi.research.task_definition import task_definition, verify_domain

    journal = frozen_journal(store, receipt, ResearchJournal)
    try:
        manifest = store.json(receipt["manifest_ref"])
        plan = ResearchPlan.model_validate_json(store.get(manifest["research_plan_ref"]))
        verify_domain(plan, store)
        verify_root_manifest(plan, store, manifest)
        if (
            manifest["kind"] != "ResearchRun"
            or plan.development.data.source != "local_event_snapshot"
            or store.json(manifest["runtime_ref"]) != controller.runtime_identity()
            or store.json(manifest["package_ref"]) != researcher.package_definition(plan)
            or store.json(manifest["development_template_ref"])
            != plan.development.model_dump(mode="json")
            or store.json(manifest["task_ref"])
            != task_definition(
                plan,
                manifest["data_ref"],
                manifest["development_template_ref"],
            )
        ):
            raise ValueError("frozen research manifest/package/task mismatch")
        changes.validate_lineage(plan, store)
        for key in ("campaign_id", "package_ref", "research_plan_ref", "data_ref"):
            if receipt[key] != manifest[key]:
                raise ValueError("trajectory research receipt identity mismatch")
        data = MarketData.from_json(store.json(manifest["data_ref"]))
        trace = research.verify_trace(journal, store, manifest, plan, data, replay=True)
        report = research.summarize(journal, store, manifest, plan, trace)
        require_settled_research(journal, report)
        if plan.schema_version == 3 and store.json(receipt["attempts_ref"]) != journal.attempts():
            raise ValueError("registered research attempt map differs from actual journal")
        if (
            receipt["checkpoint_ref"] != trace[0]
            or receipt["feedback_watermark"] != trace[3]
            or store.json(receipt["events_ref"]) != journal.events()
            or store.json(receipt["requests_ref"]) != journal.requests()
            or store.json(receipt["development_report_ref"]) != report
            or receipt["costs"] != charged_costs(journal)
            or store.json(receipt["semantic_ref"])
            != semantic_definition(store.json(manifest["package_ref"]))
        ):
            raise ValueError("trajectory research receipt differs from its frozen actual journal")
        selection = store.json(receipt["selection_ref"])
        incumbent = trace[1]["incumbent"]
        expected = dict(
            kind="TrajectoryDevelopmentSelection",
            campaign_id=manifest["campaign_id"],
            package_ref=manifest["package_ref"],
            checkpoint_ref=trace[0],
            data_ref=manifest["data_ref"],
            research_plan_ref=manifest["research_plan_ref"],
            role="missing" if incumbent is None else "active",
            incumbent=incumbent,
            model_ref=None,
            training_ref=None,
            reason="no_authoritative_valid_incumbent" if incumbent is None else None,
            qualification="development_selection_not_independent_scientific_evidence",
        )
        if incumbent is not None:
            bundle = store.json(incumbent["result_ref"])
            if store.json(bundle["report_ref"])["eligible"] is not True:
                raise ValueError("frozen trajectory selection lacks an eligible result")
            factor = FactorSpec.model_validate_json(canonical(incumbent["factor"]))
            metadata, predictions = (
                store.json(bundle["worker_metadata_ref"]),
                store.json(bundle["predictions_ref"]),
            )
            local_research.verify_local_result(
                data,
                plan.development,
                factor,
                metadata | {"predictions": predictions},
                store.get(bundle["model_ref"]).decode(),
            )
            x, y, training_metadata, mask = local_research.training_view(
                data, plan.development, factor
            )

            def expected_ref(value):
                ref = digest(canonical(value))
                if store.json(ref) != value:
                    raise ValueError("frozen trajectory training evidence mismatch")
                return ref

            training = dict(
                kind="TrajectoryLocalTrainingEvidence",
                data_ref=manifest["data_ref"],
                development_plan_ref=expected_ref(plan.development.model_dump(mode="json")),
                actual_mask_ref=expected_ref(mask),
                feature_values_ref=expected_ref(x.tolist()),
                mature_labels_ref=expected_ref(y.tolist()),
                result_ref=incumbent["result_ref"],
                **training_metadata,
            )
            expected.update(
                model_ref=bundle["model_ref"],
                training_ref=expected_ref(training),
                result_ref=incumbent["result_ref"],
                predictions_ref=bundle["predictions_ref"],
                worker_metadata_ref=bundle["worker_metadata_ref"],
            )
        if selection != expected:
            raise ValueError("trajectory selected model/mask differs from its actual incumbent")
        return report
    finally:
        journal.close()


def verify_package_receipt(store, receipt, arm_receipts):
    from quant_rsi.research.package_journal import PackageJournal

    journal = frozen_journal(store, receipt, PackageJournal)
    try:
        manifest = store.json(receipt["manifest_ref"])
        plan = package_experiment.verify_package_header(journal, store, manifest)
        decision = package_experiment.verify_decision(journal, store, manifest, plan)
        if (
            receipt["campaign_id"] != manifest["campaign_id"]
            or receipt["decision"] != decision
            or receipt["decision_ref"] != journal.state()["decision_ref"]
            or receipt["completion_ref"] != journal.state()["completion_ref"]
            or receipt["rounds"] != journal.rounds()
            or receipt["grants"] != journal.grants()
        ):
            raise ValueError("frozen C4 receipt differs from its actual journal")
        package_files = store.json(receipt["files_ref"])
        captured_arms = {
            Path(name).parts[1]
            for name in package_files
            if len(Path(name).parts) > 1 and Path(name).parts[0] == "arms"
        }
        if captured_arms - set(decision["grants"]):
            raise ValueError("unallocated child research directory")
        if set(arm_receipts) != set(decision["grants"]):
            raise ValueError("frozen C4 child receipts differ from committed grants")
        reports = {}
        for row in journal.grants():
            arm = row["arm"]
            child = store.json(arm_receipts[arm])
            report = verify_research_receipt(store, child)
            child_manifest = store.json(child["manifest_ref"])
            grant = store.json(row["grant_ref"])
            child_prefix = grant["directory"] + "/"
            if child_prefix + "research_initialization.json" in package_files or any(
                package_files.get(child_prefix + name) != ref
                for name, ref in store.json(child["files_ref"]).items()
            ):
                raise ValueError("frozen C4 child snapshot differs from its granted receipt")
            if (
                child["campaign_id"] != grant["campaign_id"]
                or child["package_ref"] != grant["package_ref"]
                or child["research_plan_ref"] != grant["research_plan_ref"]
                or store.json(child_manifest["launch_context_ref"])
                != package_experiment.launch_context(manifest, arm, row["grant_ref"])
                or child_manifest["task_ref"] != store.json(manifest["evidence_ref"])["task_ref"]
                or store.json(child["research_plan_ref"])["development"]["budget"]
                != grant["budget"]
            ):
                raise ValueError("frozen C4 child identity/grant binding mismatch")
            report["committed_observations"] = sum(
                row["observation_ref"] is not None for row in store.json(child["requests_ref"])
            )
            reports[arm] = report
        if reports != receipt["reports"]:
            raise ValueError("frozen C4 reports differ from actual child observations")
        package_experiment.verify_package_records(journal, store, manifest, decision, reports)
    finally:
        journal.close()
