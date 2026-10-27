"""First-use presentation around the existing managed forecast workflow."""

import os
import platform
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError

from pydantic import ValidationError

from quant_rsi import controller
from quant_rsi import enhanced_forecast_workflow as workflow
from quant_rsi import forecast_campaign as campaign
from quant_rsi.artifacts import atomic_write, canonical
from quant_rsi.evidence import authority_root
from quant_rsi.forecast_workflow_config import ForecastWorkflowConfig

HANDLED_ERRORS = (ValueError, OSError, sqlite3.DatabaseError, subprocess.TimeoutExpired)
ACTION_ERRORS = (*HANDLED_ERRORS, KeyboardInterrupt)
STAGES = [
    "Collect the explicitly configured target, industry proxy and market proxy.",
    "Validate and freeze inputs and the registered B0/B1/B2 comparison.",
    "Run at most nine development fits on the three fixed folds.",
    "Use the remaining publication slot for the selected recipe.",
    "Inspect the forecast, controls, timing and qualifications; save JSON and HTML.",
]


def load_profile(path):
    return ForecastWorkflowConfig.model_validate_json(path.read_bytes()).model_dump(mode="json")


def retry_command(arguments):
    return shlex.join(
        ["env", "QUANT_RSI_WORKSPACE=" + str(authority_root().parent), "quant-rsi", *arguments]
    )


def command(action, workflow_id, *arguments):
    return retry_command(["forecast-workflow", action, workflow_id, *arguments])


def _original_authority():
    base = authority_root()
    for path in (
        base / ".forecast_campaign.lock",
        base / "forecast_campaign_identity.json",
        base / "forecast_campaigns" / "identity.json",
        base / "forecast_campaigns" / "journal.sqlite",
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "missing or corrupt campaign authority; restore the original complete workspace "
                "if previously initialized; otherwise explicitly register a finite campaign first"
            )
    return base


def _context(
    workflow_id, *, campaign_id=None, profile=None, runtime=False, fresh=False, replay=False
):
    """Read existing authority under its normal locks; never bootstrap or repair it."""
    base = _original_authority()
    root = workflow.workflow_root(workflow_id)
    with campaign.run_scope(workflow_id, required=True) as managed:
        if campaign_id is not None and campaign_id != managed.binding["campaign_id"]:
            raise ValueError("run is allocated to a different campaign")
        if profile is not None:
            normalized = ForecastWorkflowConfig.model_validate(profile).model_dump(mode="json")
            if normalized != managed.config:
                raise ValueError("profile differs from the frozen registered configuration")
        parent = campaign.inspect_campaign(managed.binding["campaign_id"])
        child = next(row for row in parent["runs"] if row["run_id"] == workflow_id)
        report = None
        if (root / "manifest.json").exists():
            if (root / ".controller.lock").is_symlink() or not (
                root / ".controller.lock"
            ).is_file():
                raise ValueError("missing or corrupt workflow controller authority")
            with controller.controller_lock(root):
                journal, manifest, config = workflow._open(
                    root, verify_runtime=runtime or replay, managed=managed
                )
                try:
                    result = workflow._bound_report(root, journal, config, replay=replay)
                    report = workflow._display(root, journal, manifest, result)
                    if fresh and result is not None and not workflow._terminal(result):
                        inputs = workflow._check_inputs(
                            root, workflow._payload(journal, "inputs_committed"), config
                        )
                        workflow._freshness(root, inputs, config)
                finally:
                    journal.close()
        return base.parent, parent, child, report


def budget_view(parent, child):
    actual, allocation = child["actual"], child["allocation"]
    return {
        "run_maximum": allocation,
        "run_committed_proposal_attempts": actual["fit_reservations"],
        "run_charged_worker_slots": actual["charged_fits"],
        "run_reserved_worker_seconds": actual["reserved_worker_seconds"],
        "run_remaining_proposal_attempts": allocation["max_attempts"] - actual["fit_reservations"],
        "run_remaining_worker_seconds": (
            allocation["max_worker_seconds"] - actual["reserved_worker_seconds"]
        ),
        "campaign_maximum": parent["budget"],
        "campaign_allocated": parent["allocated"],
        "campaign_actual": parent["actual"],
        "qualification": (
            "Attempts include rejected zero-charge proposals. Worker seconds are reserved timeout "
            "slots, not elapsed time. Remaining capacity is not transferable and does not "
            "guarantee another admissible fit."
        ),
    }


def _check_environment():
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux":
        raise ValueError("unsupported runtime: this workflow requires Python 3.12 on Linux")
    workspace = authority_root().parent
    if not os.access(workspace, os.W_OK | os.X_OK):
        raise OSError("workspace is not writable")


def preflight(workflow_id, *, campaign_id=None, profile=None):
    _check_environment()
    workspace, parent, child, report = _context(
        workflow_id, campaign_id=campaign_id, profile=profile, runtime=True, fresh=True
    )
    prepared = report is not None and report["experiment"] is not None
    terminal = prepared and workflow._terminal(report["experiment"])
    return {
        "kind": "ForecastWorkflowPreflight",
        "schema_version": 1,
        "workflow_id": workflow_id,
        "campaign_id": parent["campaign_id"],
        "workspace": str(workspace),
        "mode": "enhanced_B0_B1_B2_comparison",
        "status": report["status"]
        if terminal
        else "ready_to_run"
        if prepared
        else "ready_to_prepare",
        "profile": child["config"],
        "runtime": {"python": platform.python_version(), "platform": platform.system()},
        "planned_stages": STAGES,
        "budget": budget_view(parent, child),
        "data_access": {
            "status": "verified_saved_inputs" if prepared else "not_checked",
            "reason": (
                "Original saved inputs were verified; terminal results do not require fresh data."
                if terminal
                else "Original saved inputs and source freshness were verified."
                if prepared
                else "No network request was made. Explicit prepare checks source access, "
                "market metadata and data coverage before any training."
            ),
        },
        "commands": {
            "prepare": command("prepare", workflow_id, "--campaign-id", parent["campaign_id"]),
            "run": command("run", workflow_id, "--max-fits", "1"),
            "status": command("status", workflow_id),
        },
        "model_training_performed": False,
        "fit_reservations_created": False,
    }


def execute(workflow_id, *, campaign_id, profile, max_fits):
    if type(max_fits) is not int or not 0 <= max_fits <= 10:
        raise ValueError("max_fits must be an integer from 0 to 10")
    _check_environment()
    _context(workflow_id, campaign_id=campaign_id, profile=profile, runtime=True)
    # The existing runner recovers a saved dispatch/publication before checking freshness.
    # A read-only freshness preflight here would incorrectly block that zero-fit recovery.
    workflow.prepare_workflow(workflow_id, campaign_id=campaign_id, config=profile)
    report = workflow.run_workflow(workflow_id, max_fits=max_fits)
    parent = campaign.inspect_campaign(campaign_id)
    return report, parent


def report_context(workflow_id, *, replay=False):
    _, parent, _, report = _context(workflow_id, replay=replay)
    if report is None:
        raise ValueError("workflow is not prepared; repeat prepare for the original registered run")
    return report, parent


def _failure_kind(error, stage):
    message = str(error).lower()
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    if "stale" in message or "latest complete session" in message or "future-dated" in message:
        return "stale_inputs"
    if "runtime" in message:
        return "runtime_mismatch"
    if "attempt budget exhausted" in message:
        return "budget_exhausted"
    if "another" in message and ("owns" in message or "controller" in message):
        return "busy"
    if (
        isinstance(error, (URLError, ConnectionError))
        or "public source" in message
        or "public response" in message
    ):
        return "network_failure"
    if isinstance(error, sqlite3.DatabaseError) or any(
        word in message for word in ("authority", "ledger", "corrupt", "hash mismatch", "identity")
    ):
        return "corrupt_authority"
    if "configuration" in message or "different campaign" in message:
        return "frozen_configuration"
    if "not registered" in message or "unknown forecast campaign" in message:
        return "unregistered_run"
    if isinstance(error, FileNotFoundError):
        return "missing_input" if stage == "profile" else "missing_authority"
    if isinstance(error, ValidationError) or stage in {"profile", "prepare", "preflight"}:
        return "unsupported_input"
    return "stage_failed"


def failure_report(workflow_id, error, *, stage, retry=None):
    parent, report = None, None
    try:
        _, parent, _, report = _context(workflow_id)
    except HANDLED_ERRORS:
        pass
    category = _failure_kind(error, stage)
    recovery = None
    if parent is not None and category in {"network_failure", "timeout", "busy", "interrupted"}:
        if retry is not None:
            recovery = retry
        else:
            action = "run" if report is not None and report.get("experiment") else "prepare"
            extra = ("--max-fits", "1") if action == "run" else ()
            recovery = command(action, workflow_id, *extra)
    if report is not None and category == "stale_inputs":
        result = report.get("experiment")
        attempts = result["attempts"] if result else []
        completed_publication = (
            result is not None
            and len(attempts) == result["budget"]["max_attempts"]
            and attempts[-1]["status"] == "completed"
            and result.get("forecast") is None
        )
        if completed_publication or any(
            row["status"] in {"reserved", "running"} for row in attempts
        ):
            recovery = shlex.join(
                [
                    "env",
                    "QUANT_RSI_WORKSPACE=" + report["workspace"],
                    "quant-rsi",
                    "forecast-compare",
                    "run",
                    workflow_id,
                    "--protocol-ref",
                    result["protocol_ref"],
                    "--max-fits",
                    "0",
                ]
            )
    if report is None:
        child = (
            next((row for row in parent["runs"] if row["run_id"] == workflow_id), None)
            if parent is not None
            else None
        )
        report = {
            "workflow_id": workflow_id,
            "status": "failed",
            "workspace": os.environ.get("QUANT_RSI_WORKSPACE"),
            "config": child["config"] if child else {},
            "managed_run": child["binding"] if child else None,
            "experiment": None,
            "development_comparison": None,
            "coverage": {},
        }
    else:
        report = dict(report)
        stage = report["status"] if stage in {"execute", "prepare"} else stage
    report["failure"] = {
        "category": category,
        "stage": stage,
        "message": str(error)
        or "Execution interrupted; retained results and charges remain in the original run.",
        "recovery_command": recovery,
        "recoverable": recovery is not None,
        "budget_unavailable_reason": (
            "Original authority could not be verified; consumed or reserved budget is unknown."
            if parent is None
            else None
        ),
    }
    return report, parent


def write_reports(report, parent, output):
    from quant_rsi.forecast_report import render_forecast_report

    output = Path(output).resolve()
    base = authority_root()
    if output == base or base in output.parents:
        raise ValueError("report output must be outside the product authority")
    html = render_forecast_report(report, campaign_report=parent)
    atomic_write(output / "forecast.json", canonical(report))
    atomic_write(output / "forecast.html", html.encode("utf-8"))
    return {"json": str(output / "forecast.json"), "html": str(output / "forecast.html")}
