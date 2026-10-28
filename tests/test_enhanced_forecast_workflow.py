import os
import shutil
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_forecast_comparison import dummy_worker  # noqa: F401
from test_forecast_context_data import audit_package

from quant_rsi.core import controller
from quant_rsi.core.artifacts import canonical, decode, digest
from quant_rsi.forecast import enhanced_forecast_workflow as workflow
from quant_rsi.forecast import forecast_context_data as context
from quant_rsi.forecast import forecast_experiment as experiment
from quant_rsi.forecast.forecast_campaign import init_campaign
from quant_rsi.market import market_contracts

NOW = "2026-10-26T21:00:00+00:00"


@pytest.fixture
def setup_workflow(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: NOW)
    calls = {"collect": 0}

    def ingest_child(command, **kwargs):
        assert command[-2].startswith("import sys;"), "unexpected model dispatch"
        assert len(kwargs["pass_fds"]) == 2
        inherited_paths = {os.readlink(f"/proc/self/fd/{fd}") for fd in kwargs["pass_fds"]}
        assert any(path.endswith("/.forecast_campaign.lock") for path in inherited_paths)
        assert any("/enhanced_forecast_workflows/" in path for path in inherited_paths)
        with monkeypatch.context() as child:
            child.setenv("QUANT_RSI_WORKSPACE", kwargs["env"]["QUANT_RSI_WORKSPACE"])
            context._ingest_prepared(command[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(context.subprocess, "run", ingest_child)

    def prepare(
        target="SOXL",
        industry="XSD",
        market="SPY",
        workflow_id="sample",
        *,
        run=True,
        campaign_plan=None,
    ):
        from quant_rsi.forecast import forecast_context_collection as collection

        config = dict(
            schema_version=1,
            kind="ForecastWorkflowConfig",
            target=target,
            industry=industry,
            market=market,
        )
        init_campaign(
            campaign_plan
            or dict(
                kind="ForecastCampaign",
                schema_version=1,
                campaign_id="campaign-" + workflow_id,
                budget=dict(max_fit_reservations=10, max_worker_seconds=1800),
                runs=[dict(instrument_id="US/" + target, run_id=workflow_id, config=config)],
            )
        )
        roles = dict(target=target, industry=industry, market=market)
        audit = audit_package(tmp_path / workflow_id, roles=roles, history=756, cash=False)

        def collect(output, *, roles):
            calls["collect"] += 1
            assert roles == dict(target=target, industry=industry, market=market)
            shutil.copytree(audit, output, dirs_exist_ok=True)
            return Path(output)

        monkeypatch.setattr(collection, "collect_context", collect)
        return config, workflow.prepare_workflow(workflow_id, config=config) if run else None

    return prepare, calls


def test_two_profiles_share_one_grant_and_terminal_repeats_are_read_only(
    setup_workflow, monkeypatch, request, capsys
):
    from quant_rsi.cli import main
    from quant_rsi.forecast.forecast_campaign import inspect_campaign

    prepare, calls = setup_workflow
    cases = [("KO", "XLP", "ko-run"), ("SOXL", "XSD", "soxl-run")]
    plan = dict(
        kind="ForecastCampaign",
        schema_version=1,
        campaign_id="two-instruments",
        budget=dict(max_fit_reservations=20, max_worker_seconds=3600),
        runs=[
            dict(
                instrument_id="US/" + target,
                run_id=run_id,
                config=dict(
                    kind="ForecastWorkflowConfig",
                    schema_version=1,
                    target=target,
                    industry=industry,
                    market="SPY",
                ),
            )
            for target, industry, run_id in cases
        ],
    )
    prepared = {}
    for target, industry, run_id in cases:
        config, report = prepare(target, industry, workflow_id=run_id, campaign_plan=plan)
        prepared[run_id] = (config, report)
        assert report["status"] == "frozen"
        assert report["experiment"]["fit_reservations"] == 0
        assert report["experiment"]["budget"]["max_attempts"] == 10
        assert [len(f["evaluation_origins"]) for f in report["experiment"]["folds"]] == [25] * 3

    fake = request.getfixturevalue("dummy_worker")
    dispatch = controller.subprocess.run
    inherited = []

    def record_locks(command, **kwargs):
        descriptors = kwargs["pass_fds"]
        assert len(set(descriptors)) == len(descriptors)
        for descriptor in descriptors:
            os.fstat(descriptor)
        inherited.append(len(descriptors))
        return dispatch(command, **kwargs)

    monkeypatch.setattr(controller.subprocess, "run", record_locks)
    # Direct old CLI must resolve the same registered parent and consume the same child budget.
    assert (
        main(
            [
                "forecast-compare",
                "run",
                "ko-run",
                "--protocol-ref",
                prepared["ko-run"][1]["protocol_ref"],
                "--max-fits",
                "1",
            ]
        )
        == 0
    )
    direct = decode(capsys.readouterr().out.encode())
    assert direct["fit_reservations"] == 1
    reports = {"ko-run": workflow.run_workflow("ko-run", max_fits=10)}
    assert fake["fits"] == 10

    commit = experiment.commit_event

    def crash_after_publication(journal, kind, payload):
        if kind == "forecast_publication_confirmed":
            raise RuntimeError("publication confirmation interrupted")
        return commit(journal, kind, payload)

    monkeypatch.setattr(experiment, "commit_event", crash_after_publication)
    with pytest.raises(RuntimeError, match="confirmation interrupted"):
        workflow.run_workflow("soxl-run", max_fits=10)
    assert fake["fits"] == 20
    monkeypatch.setattr(experiment, "commit_event", commit)
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-28T21:00:00+00:00")
    reports["soxl-run"] = workflow.run_workflow("soxl-run")
    assert fake["fits"] == 20
    assert inherited == [2] + [3] * 19
    for target, _, run_id in cases:
        report = reports[run_id]
        assert report["status"] == "complete"
        assert report["experiment"]["fit_reservations"] == 10
        assert report["experiment"]["forecast"]["instrument"] == target
        assert report["experiment"]["forecast"]["publication_model_validation_metrics"] is None
        assert (
            report["development_comparison"]["naive_controls"]["unchanged_price"]["observations"]
            == 75
        )
        assert report["development_comparison"]["predictive_validity_established"] is False
    assert "SOXL" not in canonical(reports["ko-run"]["experiment"]).decode()
    assert "XSD" not in canonical(reports["ko-run"]["experiment"]).decode()
    campaign = inspect_campaign("two-instruments")
    assert campaign["actual"]["fit_reservations"] == 20
    assert campaign["actual"]["reserved_worker_seconds"] == 3600
    assert campaign["allocated"] == {"max_attempts": 20, "max_worker_seconds": 3600}

    def fingerprint():
        return {
            str(p): digest(p.read_bytes())
            for p in workflow.authority_root().rglob("*")
            if p.is_file()
        }

    before = fingerprint()

    def forbidden(*args, **kwargs):
        pytest.fail("completed workflows must not collect, ingest, or dispatch")

    from quant_rsi.forecast import forecast_context_collection as collection

    monkeypatch.setattr(collection, "collect_context", forbidden)
    monkeypatch.setattr(context, "prepare_context_data", forbidden)
    monkeypatch.setattr(experiment, "resume_experiment", forbidden)
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2027-06-01T21:00:00+00:00")
    for _, _, run_id in cases:
        config, frozen = prepared[run_id]
        report = reports[run_id]
        assert workflow.prepare_workflow(run_id, config=config) == report
        assert workflow.run_workflow(run_id) == report
        assert workflow.inspect_workflow(run_id) == report
        assert workflow.inspect_workflow(run_id, replay=True) == report
        assert frozen["protocol_ref"] == report["protocol_ref"]
    replayed = inspect_campaign("two-instruments", replay=True)
    for row in replayed["runs"]:
        assert row.pop("experiment_report") == reports[row["run_id"]]["experiment"]
    assert replayed == campaign
    assert fingerprint() == before
    assert calls["collect"] == 2


def test_config_changes_and_missing_authority_never_reset(setup_workflow, monkeypatch):
    prepare, calls = setup_workflow
    config, prepared = prepare()
    changed = config | {"industry": "XLP"}
    with pytest.raises(ValueError, match="another configuration"):
        workflow.prepare_workflow("sample", config=changed)
    with pytest.raises(ValueError, match="configuration reference"):
        workflow.run_workflow("sample", config_ref="0" * 64, protocol_ref=prepared["protocol_ref"])
    with pytest.raises(ValueError, match="protocol reference"):
        workflow.run_workflow("sample", config_ref=prepared["config_ref"], protocol_ref="0" * 64)
    root = experiment.experiment_root("sample")
    (root / "journal.sqlite").unlink()
    with pytest.raises(ValueError, match="initialized journal"):
        workflow.prepare_workflow("sample", config=config)
    assert not (root / "journal.sqlite").exists()
    assert calls["collect"] == 1


def test_missing_first_seen_registry_never_reimports(setup_workflow):
    prepare, calls = setup_workflow
    config, _ = prepare()
    root = workflow.workflow_root("sample") / "preparation"
    (root / "workspace" / ".quant_rsi" / "market" / "data.sqlite").unlink()
    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        workflow.prepare_workflow("sample", config=config)
    assert not (root / "workspace" / ".quant_rsi" / "market" / "data.sqlite").exists()
    assert calls["collect"] == 1


def test_resume_after_partial_ingest_preserves_first_receipt(setup_workflow, monkeypatch):
    prepare, calls = setup_workflow
    config, _ = prepare(run=False)
    original = context.ingest
    first = []

    def interrupted(*args, **kwargs):
        if kwargs["request_id"] == "context.market.v1":
            raise RuntimeError("interrupted after first import")
        result = original(*args, **kwargs)
        first.append(result)
        return result

    monkeypatch.setattr(context, "ingest", interrupted)
    with pytest.raises(RuntimeError, match="first import"):
        workflow.prepare_workflow("sample", config=config)
    assert len(first) == 1
    monkeypatch.setattr(context, "ingest", original)
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-26T21:01:00+00:00")
    report = workflow.prepare_workflow("sample", config=config)
    imports = decode(
        (workflow.workflow_root("sample") / "preparation" / "imports.json").read_bytes()
    )
    assert imports["industry"]["receipt"] == first[0]
    assert report["experiment"]["fit_reservations"] == 0
    assert calls["collect"] == 1


def test_resume_between_input_freeze_and_experiment_commit(setup_workflow, monkeypatch):
    prepare, calls = setup_workflow
    config, _ = prepare(run=False)
    commit = workflow._commit

    def interrupt(journal, stage, value):
        if stage == "experiment_committed":
            raise RuntimeError("after protocol freeze")
        return commit(journal, stage, value)

    monkeypatch.setattr(workflow, "_commit", interrupt)
    with pytest.raises(RuntimeError, match="protocol freeze"):
        workflow.prepare_workflow("sample", config=config)
    before = (experiment.experiment_root("sample") / "manifest.json").read_bytes()
    monkeypatch.setattr(workflow, "_commit", commit)
    recovered = workflow.prepare_workflow("sample", config=config)
    assert recovered["experiment"]["fit_reservations"] == 0
    assert (experiment.experiment_root("sample") / "manifest.json").read_bytes() == before
    assert calls["collect"] == 1


def test_stale_frozen_inputs_refuse_new_dispatch(setup_workflow, monkeypatch, capsys):
    prepare, calls = setup_workflow
    _, prepared = prepare()
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-28T21:00:00+00:00")
    with pytest.raises(ValueError, match="stale"):
        workflow.run_workflow(
            "sample",
            config_ref=prepared["config_ref"],
            protocol_ref=prepared["protocol_ref"],
        )
    with pytest.raises(ValueError, match="stale"):
        experiment.resume_experiment("sample", protocol_ref=prepared["protocol_ref"])
    from quant_rsi.cli import main

    assert (
        main(["forecast-compare", "run", "sample", "--protocol-ref", prepared["protocol_ref"]]) != 0
    )
    assert "stale" in capsys.readouterr().err
    assert (
        experiment.resume_experiment(
            "sample",
            protocol_ref=prepared["protocol_ref"],
            max_fits=0,
        )["fit_reservations"]
        == 0
    )
    report = workflow.inspect_workflow("sample")
    assert report["experiment"]["fit_reservations"] == 0
    assert calls["collect"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"target": "aapl"},
        {"market": "SOXL"},
        {"currency": "EUR"},
        {"horizon_sessions": 10},
        {"industry": None},
        {"source": "other"},
    ],
)
def test_unsupported_configuration_rejected_before_collection(change, tmp_path, monkeypatch):
    config = (
        dict(
            schema_version=1,
            kind="ForecastWorkflowConfig",
            target="SOXL",
            industry="XSD",
            market="SPY",
        )
        | change
    )
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    with pytest.raises(ValueError):
        init_campaign(
            dict(
                kind="ForecastCampaign",
                schema_version=1,
                campaign_id="invalid",
                budget=dict(max_fit_reservations=10, max_worker_seconds=1800),
                runs=[
                    dict(
                        instrument_id="US/" + str(config["target"]), run_id="invalid", config=config
                    )
                ],
            )
        )
    assert not (tmp_path / ".quant_rsi").exists()


def test_shared_entry_guards_reject_unknown_runs_and_copied_or_lost_authority(
    setup_workflow, monkeypatch, tmp_path
):
    from quant_rsi.forecast.forecast_campaign import init_campaign

    prepare, _ = setup_workflow
    config, prepared = prepare()
    original_workspace = Path(os.environ["QUANT_RSI_WORKSPACE"])
    root = experiment.experiment_root("sample")
    inputs = decode((workflow.workflow_root("sample") / "preparation" / "inputs.json").read_bytes())
    with pytest.raises(ValueError):
        experiment.prepare_experiment(inputs, "unallocated")
    assert not experiment.experiment_root("unallocated").exists()
    with pytest.raises(ValueError, match="fixed registered fold"):
        experiment.prepare_experiment(inputs, "sample", fold_origins=((80, 85, 90),))
    assert experiment.prepare_experiment(inputs, "sample")["fit_reservations"] == 0

    isolated = tmp_path / "copied-child-workspace"
    copied = isolated / ".quant_rsi" / "forecast_experiments" / "sample"
    shutil.copytree(root, copied)
    child_before = {str(p): digest(p.read_bytes()) for p in copied.rglob("*") if p.is_file()}
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(isolated))
    for operation in (
        lambda: experiment.prepare_experiment(inputs, "sample"),
        lambda: experiment.open_experiment(copied),
        lambda: experiment.resume_experiment("sample", protocol_ref=prepared["protocol_ref"]),
    ):
        with pytest.raises(ValueError):
            operation()
    assert {
        str(p): digest(p.read_bytes()) for p in copied.rglob("*") if p.is_file()
    } == child_before

    copied_workspace = tmp_path / "copied-entire-workspace"
    shutil.copytree(original_workspace, copied_workspace)
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(copied_workspace))
    with pytest.raises(ValueError):
        workflow.prepare_workflow("sample", config=config)
    with pytest.raises(ValueError):
        experiment.resume_experiment("sample", protocol_ref=prepared["protocol_ref"])

    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(original_workspace))
    shutil.rmtree(workflow.authority_root() / "forecast_campaigns")
    assert (workflow.authority_root() / "forecast_campaign_identity.json").exists()
    with pytest.raises(ValueError):
        experiment.prepare_experiment(inputs, "new-standalone-id")
    with pytest.raises(ValueError):
        experiment.resume_experiment("sample", protocol_ref=prepared["protocol_ref"])
    with pytest.raises(ValueError):
        init_campaign(
            dict(
                kind="ForecastCampaign",
                schema_version=1,
                campaign_id="campaign-sample",
                budget=dict(max_fit_reservations=10, max_worker_seconds=1800),
                runs=[dict(instrument_id="US/SOXL", run_id="sample", config=config)],
            )
        )
    assert not (workflow.authority_root() / "forecast_campaigns" / "journal.sqlite").exists()


def test_direct_batch_rechecks_freshness_before_each_new_dispatch(
    setup_workflow, monkeypatch, request
):
    prepare, _ = setup_workflow
    _, prepared = prepare()
    fake = request.getfixturevalue("dummy_worker")
    dispatch = controller.subprocess.run

    def expire_after_one_job(*args, **kwargs):
        result = dispatch(*args, **kwargs)
        monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-28T21:00:00+00:00")
        return result

    monkeypatch.setattr(controller.subprocess, "run", expire_after_one_job)
    with pytest.raises(ValueError, match="stale"):
        experiment.resume_experiment(
            "sample",
            protocol_ref=prepared["protocol_ref"],
            max_fits=10,
        )
    report = workflow.inspect_workflow("sample")
    assert fake["fits"] == report["experiment"]["fit_reservations"] == 1
    assert len(report["experiment"]["attempts"]) == 1
    assert report["experiment"]["reserved_worker_seconds"] == 180


@pytest.mark.parametrize("child", ["workflow", "experiment"])
@pytest.mark.parametrize("empty", [True, False])
def test_matching_pre_ack_empty_schema_can_resume_but_damage_rejects(
    setup_workflow, monkeypatch, child, empty
):
    from quant_rsi.core.journal import Journal

    prepare, calls = setup_workflow
    config, _ = prepare(run=False)
    root = (
        workflow.workflow_root("sample")
        if child == "workflow"
        else experiment.experiment_root("sample")
    )
    original = Journal.initialize

    def interrupt(self, manifest_digest):
        database = self.connection.execute("PRAGMA database_list").fetchone()[2]
        if Path(database).parent == root:
            raise RuntimeError("before schema transaction")
        return original(self, manifest_digest)

    with monkeypatch.context() as patch:
        patch.setattr(Journal, "initialize", interrupt)
        with pytest.raises(RuntimeError, match="before schema transaction"):
            workflow.prepare_workflow("sample", config=config)
    marker = (root / "bootstrap.json").read_bytes()
    with sqlite3.connect(root / "journal.sqlite") as database:
        assert database.execute("SELECT name FROM sqlite_master").fetchall() == []
        if not empty:
            database.execute("CREATE TABLE damaged (value TEXT)")
    if empty:
        report = workflow.prepare_workflow("sample", config=config)
        assert report["experiment"]["fit_reservations"] == 0
        assert not (root / "bootstrap.json").exists()
        assert calls["collect"] == 1
    else:
        with pytest.raises((ValueError, sqlite3.DatabaseError)):
            workflow.prepare_workflow("sample", config=config)
        assert (root / "bootstrap.json").read_bytes() == marker
        assert not (root / "manifest.json").exists()
