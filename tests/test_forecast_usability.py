import json
import shlex
from urllib.error import URLError

import pytest
from test_enhanced_forecast_workflow import setup_workflow as _workflow_fixture
from test_forecast_comparison import dummy_worker  # noqa: F401

from quant_rsi import controller, market_contracts
from quant_rsi import forecast_context_collection as collection
from quant_rsi import forecast_context_data as context
from quant_rsi import forecast_usability as usability
from quant_rsi.artifacts import canonical, digest
from quant_rsi.cli import main

setup_workflow = _workflow_fixture


def fingerprint(root):
    return {
        str(path.relative_to(root)): digest(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def profile_file(tmp_path, config):
    path = tmp_path / "instrument.json"
    path.write_bytes(canonical(config))
    return path


def offline_workers(request, monkeypatch):
    ingest = context.subprocess.run
    fake = request.getfixturevalue("dummy_worker")
    run_model = controller.subprocess.run

    def dispatch(command, **kwargs):
        if command[-2].startswith("import sys;"):
            return ingest(command, **kwargs)
        return run_model(command, **kwargs)

    monkeypatch.setattr(controller.subprocess, "run", dispatch)
    return fake


def result(capsys):
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(("target", "industry"), [("SOXL", "XSD"), ("KO", "XLP")])
def test_guided_profile_preflight_execution_reports_and_terminal_duplicates(
    target, industry, setup_workflow, tmp_path, request, monkeypatch, capsys
):
    prepare, calls = setup_workflow
    run_id = target.lower() + "-guided"
    config, _ = prepare(target, industry, workflow_id=run_id, run=False)
    profile = profile_file(tmp_path, config)
    workspace = tmp_path / "workspace"
    before = fingerprint(workspace)
    assert main(["forecast-workflow", "preflight", run_id, "--profile", str(profile)]) == 0
    check = result(capsys)
    assert check["status"] == "ready_to_prepare"
    assert check["data_access"]["status"] == "not_checked"
    assert check["profile"]["target"] == target and check["profile"]["industry"] == industry
    assert check["budget"]["run_remaining_proposal_attempts"] == 10
    assert check["model_training_performed"] is check["fit_reservations_created"] is False
    assert calls["collect"] == 0 and fingerprint(workspace) == before

    fake = offline_workers(request, monkeypatch)
    output = tmp_path / "reports"
    arguments = [
        "forecast-workflow",
        "execute",
        run_id,
        "--campaign-id",
        "campaign-" + run_id,
        "--profile",
        str(profile),
        "--max-fits",
        "10",
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    finished = result(capsys)
    assert finished["status"] == "complete"
    assert fake["fits"] == 10 and calls["collect"] == 1
    assert len(finished["experiment"]["attempts"]) == 10
    stored = json.loads((output / "forecast.json").read_bytes())
    assert stored["workflow_id"] == run_id
    html = (output / "forecast.html").read_text()
    assert target in html and industry in html and "<!doctype html>" in html.lower()
    from test_forecast_report import Page

    page = Page(html)
    assert [
        "Registered attempt ceiling",
        str(check["budget"]["campaign_maximum"]["max_fit_reservations"]),
        str(check["budget"]["run_maximum"]["max_attempts"]),
    ] in page.rows
    before = fingerprint(workspace)
    saved_reports = fingerprint(output)

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal repeat must not collect or dispatch")

    monkeypatch.setattr(collection, "collect_context", forbidden)
    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    assert main(arguments) == 0
    repeated = result(capsys)
    assert repeated == finished
    assert fingerprint(workspace) == before and fingerprint(output) == saved_reports
    assert fake["fits"] == 10
    assert main(["forecast-workflow", "report", run_id, "--output", str(output)]) == 0
    result(capsys)
    assert fingerprint(workspace) == before and fingerprint(output) == saved_reports


def test_network_failure_returns_same_run_recovery_and_preserves_identity(
    setup_workflow, tmp_path, request, monkeypatch, capsys
):
    prepare, calls = setup_workflow
    config, _ = prepare(run=False)
    profile = profile_file(tmp_path, config)
    collect = collection.collect_context

    def unavailable(*args, **kwargs):
        raise URLError("offline connection failure")

    monkeypatch.setattr(collection, "collect_context", unavailable)
    arguments = [
        "forecast-workflow",
        "execute",
        "sample",
        "--campaign-id",
        "campaign-sample",
        "--profile",
        str(profile),
        "--max-fits",
        "10",
        "--output",
        str(tmp_path / "report"),
    ]
    assert main(arguments) == 1
    failed = result(capsys)
    assert failed["failure"]["category"] == "network_failure"
    assert failed["failure"]["recoverable"] is True
    assert failed["failure"]["stage"] == "collection_started"
    command = shlex.split(failed["failure"]["recovery_command"])
    assert command[2:] == ["quant-rsi", *arguments]
    manifest = tmp_path / "workspace/.quant_rsi/enhanced_forecast_workflows/sample/manifest.json"
    original = manifest.read_bytes()
    monkeypatch.setattr(collection, "collect_context", collect)
    fake = offline_workers(request, monkeypatch)
    assert main(command[3:]) == 0
    recovered = result(capsys)
    assert recovered["status"] == "complete"
    assert manifest.read_bytes() == original and fake["fits"] == 10 and calls["collect"] == 1


def test_interrupted_dispatch_reuses_saved_worker_output_without_a_second_charge(
    setup_workflow, tmp_path, request, monkeypatch, capsys
):
    prepare, _ = setup_workflow
    prepare()
    fake = request.getfixturevalue("dummy_worker")
    dispatch = controller.subprocess.run
    interrupted = False

    def lose_response(command, **kwargs):
        nonlocal interrupted
        completed = dispatch(command, **kwargs)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return completed

    monkeypatch.setattr(controller.subprocess, "run", lose_response)
    arguments = ["forecast-workflow", "run", "sample", "--max-fits", "10"]
    assert main(arguments) == 130
    failed = result(capsys)
    assert failed["failure"]["category"] == "interrupted"
    assert failed["failure"]["recoverable"] is True
    assert len(failed["experiment"]["attempts"]) == 1
    assert failed["experiment"]["reserved_worker_seconds"] == 180
    retry = shlex.split(failed["failure"]["recovery_command"])
    monkeypatch.setattr(controller.subprocess, "run", dispatch)
    assert main(retry[3:]) == 0
    recovered = result(capsys)
    assert recovered["status"] == "complete"
    assert fake["launches"] == list(range(10)) and fake["fits"] == 10
    assert recovered["experiment"]["reserved_worker_seconds"] == 1800


def test_stale_preflight_never_collects_reserves_or_refreshes(
    setup_workflow, tmp_path, monkeypatch, capsys
):
    prepare, calls = setup_workflow
    config, _ = prepare()
    profile = profile_file(tmp_path, config)
    workspace = tmp_path / "workspace"
    before = fingerprint(workspace)
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-28T21:00:00+00:00")
    assert main(["forecast-workflow", "preflight", "sample"]) == 1
    failed = result(capsys)
    assert failed["failure"]["category"] == "stale_inputs"
    assert failed["failure"]["recovery_command"] is None
    assert failed["experiment"]["attempts"] == []
    assert calls["collect"] == 1 and fingerprint(workspace) == before
    assert (
        main(
            [
                "forecast-workflow",
                "execute",
                "sample",
                "--campaign-id",
                "campaign-sample",
                "--profile",
                str(profile),
                "--max-fits",
                "10",
            ]
        )
        == 1
    )
    failed = result(capsys)
    assert failed["failure"]["category"] == "stale_inputs"
    assert failed["experiment"]["attempts"] == []
    assert calls["collect"] == 1 and fingerprint(workspace) == before


def test_unsupported_profile_is_rejected_before_collection(setup_workflow, tmp_path, capsys):
    prepare, calls = setup_workflow
    config, _ = prepare(run=False)
    config["horizon_sessions"] = 10
    profile = profile_file(tmp_path, config)
    before = fingerprint(tmp_path / "workspace")
    assert main(["forecast-workflow", "preflight", "sample", "--profile", str(profile)]) == 1
    failed = result(capsys)
    assert failed["failure"]["category"] == "unsupported_input"
    assert failed["failure"]["stage"] == "profile"
    assert calls["collect"] == 0 and fingerprint(tmp_path / "workspace") == before


def test_missing_authority_stays_missing_with_unknown_budget(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "empty"
    workspace.mkdir()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    assert main(["forecast-workflow", "preflight", "sample"]) == 1
    failed = result(capsys)
    assert failed["failure"]["category"] == "corrupt_authority"
    assert failed["failure"]["budget_unavailable_reason"]
    assert failed["experiment"] is None and failed["failure"]["recovery_command"] is None
    assert list(workspace.iterdir()) == []


def test_damaged_authority_is_not_recreated(setup_workflow, tmp_path, capsys):
    prepare, calls = setup_workflow
    prepare(run=False)
    workspace = tmp_path / "workspace"
    (workspace / ".quant_rsi/forecast_campaign_identity.json").unlink()
    before = fingerprint(workspace)
    assert main(["forecast-workflow", "preflight", "sample"]) == 1
    failed = result(capsys)
    assert failed["failure"]["category"] == "corrupt_authority"
    assert failed["failure"]["budget_unavailable_reason"]
    assert calls["collect"] == 0 and fingerprint(workspace) == before


def test_remaining_attempts_include_rejected_zero_charge_proposals():
    child = {
        "allocation": {"max_attempts": 10, "max_worker_seconds": 1800},
        "actual": {"fit_reservations": 3, "charged_fits": 2, "reserved_worker_seconds": 360},
    }
    parent = {"budget": {}, "allocated": {}, "actual": child["actual"]}
    budget = usability.budget_view(parent, child)
    assert budget["run_remaining_proposal_attempts"] == 7
    assert budget["run_charged_worker_slots"] == 2
    assert budget["run_reserved_worker_seconds"] == 360


@pytest.mark.parametrize("recovery_path", ["execute", "zero_fit_command"])
def test_stale_completed_publication_recovers_without_collection_or_new_charge(
    setup_workflow, tmp_path, request, monkeypatch, capsys, recovery_path
):
    from quant_rsi import forecast_experiment as experiment

    prepare, calls = setup_workflow
    config, _ = prepare()
    profile = profile_file(tmp_path, config)
    fake = request.getfixturevalue("dummy_worker")
    publish = experiment.publish

    def interrupt_before_publication(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(experiment, "publish", interrupt_before_publication)
    arguments = [
        "forecast-workflow",
        "execute",
        "sample",
        "--campaign-id",
        "campaign-sample",
        "--profile",
        str(profile),
        "--max-fits",
        "10",
        "--output",
        str(tmp_path / "report"),
    ]
    assert main(arguments) == 130
    interrupted = result(capsys)
    assert len(interrupted["experiment"]["attempts"]) == 10
    assert interrupted["experiment"]["attempts"][-1]["status"] == "completed"
    assert interrupted["experiment"]["forecast"] is None
    assert fake["fits"] == 10

    monkeypatch.setattr(experiment, "publish", publish)
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-28T21:00:00+00:00")
    workspace = tmp_path / "workspace"
    original_manifest = workspace / ".quant_rsi/enhanced_forecast_workflows/sample/manifest.json"
    original_bytes = original_manifest.read_bytes()
    before = fingerprint(workspace)
    assert main(["forecast-workflow", "preflight", "sample"]) == 1
    failed = result(capsys)
    assert failed["failure"]["category"] == "stale_inputs"
    assert failed["failure"]["recoverable"] is True
    retry = shlex.split(failed["failure"]["recovery_command"])
    assert retry[3:6] == ["forecast-compare", "run", "sample"]
    assert retry[-2:] == ["--max-fits", "0"]
    assert fingerprint(workspace) == before

    assert main(arguments if recovery_path == "execute" else retry[3:]) == 0
    result(capsys)
    recovered, parent = usability.report_context("sample")
    assert recovered["status"] == "complete"
    assert recovered["experiment"]["forecast"] is not None
    assert fake["fits"] == 10 and fake["launches"] == list(range(10))
    assert calls["collect"] == 1
    assert original_manifest.read_bytes() == original_bytes
    assert parent["actual"]["fit_reservations"] == 10
    assert parent["actual"]["reserved_worker_seconds"] == 1800
