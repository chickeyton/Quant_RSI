import fcntl
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from quant_rsi.accounting import evaluate
from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.data import MarketData, synthetic_data
from quant_rsi.journal import TERMINAL, BudgetExhausted, Journal
from quant_rsi.schemas import DevelopmentPlan, FactorSpec


def runtime_identity() -> dict:
    package = Path(__file__).parent
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "dependencies": {
            distribution.metadata["Name"].lower(): distribution.version
            for distribution in importlib.metadata.distributions()
        },
        "sources": {p.name: digest(p.read_bytes()) for p in sorted(package.glob("*.py"))},
    }


@contextmanager
def controller_lock(root: Path):
    # The worker inherits this descriptor, so a dead parent cannot permit concurrent recovery.
    with (root / ".controller.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another controller or its worker still owns this run") from exc
        yield handle.fileno()


def create_run(plan: DevelopmentPlan, root: Path) -> dict:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with controller_lock(root):
        marker = root / "initialization.json"
        if marker.exists():
            saved = decode(marker.read_bytes())
            if saved["plan"] != plan.model_dump(mode="json"):
                raise ValueError("unfinished initialization belongs to a different plan")
        else:
            # A crash before publishing the first record leaves only the lock/temp files.
            if any(
                p.name != ".controller.lock" and not p.name.startswith(".pending-")
                for p in root.iterdir()
            ):
                raise FileExistsError("run already exists; use resume without replacing its plan")
            atomic_write(
                marker,
                canonical(
                    {
                        "kind": "quant_rsi_initialization_v1",
                        "plan": plan.model_dump(mode="json"),
                        "runtime": runtime_identity(),
                    }
                ),
            )
        finish_initialization(root)
    return resume_run(root)


def finish_initialization(root: Path) -> None:
    """Complete only a marked creation; callers hold the run lock, before any dispatch."""
    marker = root / "initialization.json"
    if not marker.exists():
        return
    saved = decode(marker.read_bytes())
    if saved["kind"] != "quant_rsi_initialization_v1":
        raise ValueError("unknown initialization record")
    plan = DevelopmentPlan.model_validate_json(canonical(saved["plan"]))
    if saved["runtime"] != runtime_identity():
        raise ValueError("source/runtime changed during initialization")
    store = ArtifactStore(root / "artifacts")
    manifest = {
        "schema_version": 1,
        "plan_ref": store.put_json(saved["plan"]),
        "data_ref": store.put_json(synthetic_data(plan.data).to_json()),
        "runtime_ref": store.put_json(saved["runtime"]),
    }
    path = root / "manifest.json"
    if path.exists():
        if path.read_bytes() != canonical(manifest):
            raise ValueError("initialization manifest conflicts with saved identity")
    else:
        atomic_write(path, canonical(manifest))
    journal = Journal(root, digest(canonical(manifest)))
    journal.close()
    # Journal schema + identity + run_created are committed before this marker is removed.
    # No attempt may start until removal. Missing/damaged unmarked DBs are never rebuilt.
    marker.unlink()
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def open_run(root: Path, *, verify_runtime: bool = True):
    manifest_bytes = (root / "manifest.json").read_bytes()
    journal = Journal(root)
    try:
        row = journal.connection.execute(
            "SELECT value FROM metadata WHERE key='manifest_digest'"
        ).fetchone()
        if row is None:
            raise ValueError("run has no committed initialization identity")
        expected = row[0]
        if digest(manifest_bytes) != expected:
            raise ValueError("run manifest integrity failure")
        manifest = decode(manifest_bytes)
        store = ArtifactStore(root / "artifacts")
        plan = DevelopmentPlan.model_validate_json(store.get(manifest["plan_ref"]))
        data = MarketData.from_json(store.json(manifest["data_ref"]))
        recorded_runtime = store.json(manifest["runtime_ref"])
        if verify_runtime and recorded_runtime != runtime_identity():
            raise ValueError("source/runtime changed; replay/resume requires the recorded version")
        return journal, store, manifest, plan, data
    except Exception:
        journal.close()
        raise


def job_for(ordinal: int, proposal: dict, manifest: dict, plan: DevelopmentPlan):
    binding = {
        "ordinal": ordinal,
        "proposal_index": 0 if "research_context_ref" in manifest else ordinal,
        "proposal": proposal,
        "manifest": manifest,
        "timeout_seconds": plan.budget.worker_timeout_seconds,
    }
    return {"job_digest": digest(canonical(binding)), "binding": binding}


def accept_worker_result(work: Path, job: dict, store: ArtifactStore, plan, data) -> str:
    if decode((work / "job.json").read_bytes()) != job:
        raise ValueError("executed job differs from the controller-bound input")
    result = decode((work / "result.json").read_bytes())
    if not isinstance(result, dict) or not isinstance(result.get("model_text"), str):
        raise ValueError("invalid worker result schema")
    if result["job_digest"] != job["job_digest"]:
        raise ValueError("worker output belongs to another bound computation")
    if digest(result["model_text"].encode()) != result["model_digest"]:
        raise ValueError("model artifact digest mismatch")
    if plan.data.source == "local_event_snapshot":
        from quant_rsi.local_research import evaluate_local, verify_local_result

        factor = FactorSpec.model_validate_json(canonical(job["binding"]["proposal"]["factor"]))
        verify_local_result(data, plan, factor, result, result["model_text"])
        report = evaluate_local(data, plan, result["predictions"])
    else:
        cutoff = plan.data.train_cutoff
        if (
            result["training_information_cutoff"] != data.timestamp(cutoff, "close")
            or result["last_training_label_available"] != data.timestamp(cutoff, "open")
            or result["training_rows"]
            != (cutoff - 1 - job["binding"]["proposal"]["factor"]["window"]) * len(data.instruments)
        ):
            raise ValueError("worker training metadata differs from bound mature-label split")
        report = evaluate(
            data, plan.data.train_cutoff, result["predictions"], plan.portfolio, plan.annualization
        )
        report["eligible"] = report["metrics"]["max_drawdown"] >= -plan.portfolio.max_drawdown
        report["eligibility_reasons"] = [] if report["eligible"] else ["drawdown_limit"]
    model_ref = store.put(result.pop("model_text").encode())
    predictions_ref = store.put_json(result.pop("predictions"))
    return store.put_json(
        {
            "model_ref": model_ref,
            "predictions_ref": predictions_ref,
            "worker_metadata_ref": store.put_json(result),
            "job_ref": store.put_json(job),
            "report_ref": store.put_json(report),
            "log_ref": store.put((work / "worker.log").read_bytes()),
        }
    )


def execute_attempt(root: Path, ordinal, proposal, manifest, plan, data, store, journal, lock_fd):
    try:
        FactorSpec.model_validate_json(canonical(proposal["factor"]))
    except (ValidationError, ValueError) as exc:
        journal.finish(ordinal, "rejected", error="invalid_factor: " + str(exc).splitlines()[0])
        return
    try:
        journal.start(ordinal, plan.budget.worker_timeout_seconds, plan.budget.max_worker_seconds)
    except BudgetExhausted as exc:
        journal.finish(ordinal, "budget_exhausted", error=str(exc))
        return
    work = root / "work" / f"{ordinal:04d}"
    job = job_for(ordinal, proposal, manifest, plan)
    atomic_write(work / "job.json", canonical(job))
    start = time.monotonic()
    error = None
    with (work / "worker.log").open("wb") as log:
        try:
            process = subprocess.run(
                [sys.executable, "-m", "quant_rsi.worker", str(work / "job.json")],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=plan.budget.worker_timeout_seconds + 2,
                pass_fds=lock_fd if isinstance(lock_fd, tuple) else (lock_fd,),
                env={
                    "PATH": os.defpath,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONHASHSEED": str(plan.data.seed),
                },
            )
            if process.returncode:
                error = f"worker_exit_{process.returncode}"
        except subprocess.TimeoutExpired:
            error = "worker_timeout"
        except OSError as exc:
            error = f"worker_launch_error: {exc}"
    elapsed = time.monotonic() - start
    if error:
        journal.finish(ordinal, "failed", error=error, elapsed=elapsed)
        return
    try:
        result_ref = accept_worker_result(work, job, store, plan, data)
    except (ValueError, KeyError, OSError) as exc:
        journal.finish(ordinal, "failed", error="invalid_output: " + str(exc), elapsed=elapsed)
        return
    # A crash here leaves running + complete atomic output, recoverable without training again.
    journal.finish(ordinal, "completed", result_ref=result_ref, elapsed=elapsed)


def recover_attempt(root, attempt, manifest, plan, data, store, journal):
    ordinal = attempt["ordinal"]
    work = root / "work" / f"{ordinal:04d}"
    proposal = store.json(attempt["proposal_ref"])
    job = job_for(ordinal, proposal, manifest, plan)
    try:
        if decode((work / "job.json").read_bytes()) != job:
            raise ValueError("interrupted job binding mismatch")
        result_ref = accept_worker_result(work, job, store, plan, data)
    except (ValueError, KeyError, OSError):
        journal.finish(ordinal, "interrupted", error="no_valid_committed_worker_output")
    else:
        journal.finish(ordinal, "completed", result_ref=result_ref)


def summarize(journal, store, plan) -> dict:
    rows = journal.attempts()
    results = []
    eligible = []
    for row in rows:
        item = {key: row[key] for key in ("ordinal", "name", "status", "error", "elapsed_seconds")}
        if row["status"] == "completed":
            bundle = store.json(row["result_ref"])
            report = store.json(bundle["report_ref"])
            item.update(
                metrics=report["metrics"], eligible=report["eligible"], result_ref=row["result_ref"]
            )
            if report["eligible"]:
                eligible.append(item)
        results.append(item)
    incomplete = len(rows) != len(plan.proposals) or any(
        row["status"] == "budget_exhausted" for row in rows
    )
    running = any(row["status"] not in TERMINAL for row in rows)
    status = "running" if running else "budget_exhausted" if incomplete else "completed"
    best = max(eligible, key=lambda x: x["metrics"][plan.primary_metric], default=None)
    return {
        "execution_status": status,
        "evidence_status": "development_only",
        "promotion_decision": "not_requested",
        "data_source": "synthetic",
        "capacity_status": "unverified",
        "training_mode": "M0",
        "attempts_used": len(rows),
        "attempt_limit": plan.budget.max_attempts,
        "worker_seconds_charged": sum(row["worker_seconds"] for row in rows),
        "worker_seconds_limit": plan.budget.max_worker_seconds,
        "worker_charging_rule": "full_timeout_slot_per_started_attempt_no_refund",
        "development_incumbent": None if best is None else best["name"],
        "results": results,
    }


def resume_run(root: Path) -> dict:
    root = root.resolve()
    with controller_lock(root) as lock_fd:
        finish_initialization(root)
        journal, store, manifest, plan, data = open_run(root)
        try:
            for ordinal, candidate in enumerate(plan.proposals):
                prior = {row["ordinal"]: row for row in journal.attempts()}.get(ordinal)
                if prior and prior["status"] in TERMINAL:
                    continue
                if prior and prior["status"] == "running":
                    recover_attempt(root, prior, manifest, plan, data, store, journal)
                    continue
                proposal = candidate.model_dump(mode="json")
                if prior is None:
                    try:
                        journal.reserve(
                            ordinal,
                            candidate.name,
                            store.put_json(proposal),
                            plan.budget.max_attempts,
                        )
                    except BudgetExhausted:
                        break
                execute_attempt(
                    root, ordinal, proposal, manifest, plan, data, store, journal, lock_fd
                )
            report = summarize(journal, store, plan)
            atomic_write(root / "summary.json", canonical(report))
            return report
        finally:
            journal.close()


def inspect_run(root: Path, *, replay: bool = False) -> dict:
    root = root.resolve()
    with controller_lock(root):
        journal, store, manifest, plan, data = open_run(root, verify_runtime=replay)
        try:
            verified = 0
            for row in journal.attempts():
                store.get(row["proposal_ref"])
                if not replay or row["status"] != "completed":
                    continue
                bundle = store.json(row["result_ref"])
                expected_job = job_for(
                    row["ordinal"], store.json(row["proposal_ref"]), manifest, plan
                )
                replay_result(store, bundle, expected_job, plan, data)
                verified += 1
            report = summarize(journal, store, plan)
            if replay:
                report["replay"] = {
                    "verified_experiments": verified,
                    "method": "stored_predictions_accounting_no_new_training",
                }
            return report
        finally:
            journal.close()


def replay_result(store, bundle, expected_job, plan, data):
    for ref in bundle.values():
        store.get(ref)
    if store.json(bundle["job_ref"]) != expected_job:
        raise ValueError("stored job binding mismatch")
    metadata = store.json(bundle["worker_metadata_ref"])
    if (
        metadata["job_digest"] != expected_job["job_digest"]
        or metadata["model_digest"] != bundle["model_ref"]
    ):
        raise ValueError("stored worker identity mismatch")
    if plan.data.source == "local_event_snapshot":
        from quant_rsi.local_research import evaluate_local, verify_local_result

        predictions = store.json(bundle["predictions_ref"])
        factor = FactorSpec.model_validate_json(
            canonical(expected_job["binding"]["proposal"]["factor"])
        )
        verify_local_result(
            data,
            plan,
            factor,
            metadata | {"predictions": predictions},
            store.get(bundle["model_ref"]).decode(),
        )
        calculated = evaluate_local(data, plan, predictions)
    else:
        calculated = evaluate(
            data,
            plan.data.train_cutoff,
            store.json(bundle["predictions_ref"]),
            plan.portfolio,
            plan.annualization,
        )
        calculated["eligible"] = (
            calculated["metrics"]["max_drawdown"] >= -plan.portfolio.max_drawdown
        )
        calculated["eligibility_reasons"] = [] if calculated["eligible"] else ["drawdown_limit"]
    if canonical(calculated) != store.get(bundle["report_ref"]):
        raise ValueError("authoritative replay disagrees with stored accounting")
    return calculated
