"""Recursive grant components only: no ingestion, fitting or research dispatch."""

import fcntl
import os
import shutil

import pytest
from test_forecast_campaign import plan as comparison_plan
from test_task_definition import recursive_value

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.journal import Journal
from quant_rsi.forecast import forecast_campaign as comparison
from quant_rsi.forecast.forecast_contracts import ForecastDevelopmentPlan
from quant_rsi.forecast.recursive import forecast_recursive_campaign as recursive
from quant_rsi.research.task_definition import ForecastRecursiveProfile


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    path = tmp_path / "workspace"
    path.mkdir()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(path))
    return path


def plan(workspace):
    research = recursive_value()
    research["development"]["budget"] = {
        "max_attempts": 8,
        "worker_timeout_seconds": 180,
        "max_worker_seconds": 1440,
    }
    development = ForecastDevelopmentPlan.model_validate_json(canonical(research["development"]))
    research["forecast_profile"] = ForecastRecursiveProfile.from_development(
        development
    ).model_dump(mode="json")
    return {
        "kind": "ForecastRecursiveCampaign",
        "schema_version": 3,
        "campaign_id": "recursive-one",
        "source_directory": str(workspace / "source"),
        "source_campaign_id": "source-one",
        "research_plan": research,
        "source_allowance": {"max_attempts": 4, "max_worker_seconds": 720},
        "budget": {"max_attempts": 36, "max_worker_seconds": 6480},
        "max_proposal_stages": 2,
        "max_generations": 2,
    }


def journal_state(workspace):
    journal = Journal(workspace / ".quant_rsi/forecast_campaigns")
    try:
        return {
            "events": journal.events(),
            "attempts": list(journal.connection.execute("SELECT * FROM attempts")),
        }
    finally:
        journal.close()


def test_grant_reserves_full_template_once_without_source_or_fit(workspace):
    value = plan(workspace)
    report = recursive.init_recursive_campaign(value)
    assert report["reserved"] == {"max_attempts": 36, "max_worker_seconds": 6480}
    assert report["grant"]["research_plan"]["development"]["budget"]["max_attempts"] == 8
    assert report["grant"]["source_allowance"]["max_attempts"] == 4
    assert report["exposure"] == recursive.EXPOSURE
    assert report["exposure"]["prior_exposure"] == "engineering_fixture"
    assert report["grant"]["research_plan_ref"] == digest(
        canonical(report["grant"]["research_plan"])
    )
    assert set(report["binding"]) == {
        "authority_ref",
        "grant_ref",
        "workspace",
        "campaign_id",
        "source_directory",
        "source_campaign_id",
        "research_plan_ref",
        "exposure_ref",
    }
    before = journal_state(workspace)
    assert not before["attempts"]
    assert [event["kind"] for event in before["events"]] == [
        "run_created",
        "recursive_campaign_registered",
    ]
    assert not (workspace / "source").exists()
    (workspace / "source").mkdir()
    assert recursive.init_recursive_campaign(value) == report
    assert journal_state(workspace) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget", {"max_attempts": 35, "max_worker_seconds": 6480}),
        ("budget", {"max_attempts": 36, "max_worker_seconds": 6479}),
        ("source_allowance", {"max_attempts": 9, "max_worker_seconds": 720}),
        ("source_allowance", {"max_attempts": 4, "max_worker_seconds": 179}),
        ("source_allowance", {"max_attempts": True, "max_worker_seconds": 720}),
        ("max_proposal_stages", 3),
        ("max_generations", 1),
        ("source_directory", "relative/source"),
    ],
)
def test_invalid_plan_rejected_before_authority_creation(workspace, field, value):
    supplied = plan(workspace)
    supplied[field] = value
    with pytest.raises(ValueError):
        recursive.init_recursive_campaign(supplied)
    assert not (workspace / ".quant_rsi").exists()


def test_existing_source_and_authority_overlap_rejected_without_grant(workspace):
    value = plan(workspace)
    (workspace / "source").mkdir()
    with pytest.raises(ValueError, match="before it exists"):
        recursive.init_recursive_campaign(value)
    assert len(journal_state(workspace)["events"]) == 1
    value["source_directory"] = str(workspace / ".quant_rsi" / "research")
    with pytest.raises(ValueError, match="overlaps"):
        recursive.init_recursive_campaign(value)
    assert len(journal_state(workspace)["events"]) == 1


def test_nested_scope_keeps_original_ofd_and_selects_only_source_id(workspace):
    value = plan(workspace)
    report = recursive.init_recursive_campaign(value)
    with recursive.recursive_scope(campaign_id="source-one") as outer:
        assert recursive.require_recursive_owner() is outer
        with recursive.recursive_scope(
            plan=value["research_plan"],
            source_directory=value["source_directory"],
            grant_ref=report["binding"]["grant_ref"],
        ) as nested:
            assert nested is outer
            with (workspace / ".quant_rsi/.forecast_campaign.lock").open("r+") as probe:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="selectors disagree"):
            with recursive.recursive_scope(campaign_id="recursive-one"):
                pass
    with pytest.raises(ValueError, match="active original"):
        recursive.require_recursive_owner()
    with pytest.raises(ValueError, match="unknown or conflicting"):
        with recursive.recursive_scope(campaign_id="recursive-one"):
            pass
    with recursive.recursive_scope(source_directory=value["source_directory"]):
        assert recursive.require_recursive_owner().binding == report["binding"]


def test_wrong_template_or_changed_grant_cannot_reuse_allowance(workspace):
    value = plan(workspace)
    recursive.init_recursive_campaign(value)
    before = journal_state(workspace)
    changed = plan(workspace)
    changed["research_plan"]["policy"]["max_no_progress"] += 1
    with pytest.raises(ValueError, match="another immutable"):
        recursive.init_recursive_campaign(changed)
    with pytest.raises(ValueError, match="frozen full template"):
        with recursive.recursive_scope(campaign_id="source-one", plan=changed["research_plan"]):
            pass
    assert journal_state(workspace) == before


@pytest.mark.parametrize("identity", ["source_campaign_id", "source_directory"])
def test_new_name_cannot_reclaim_same_source(workspace, identity):
    original = plan(workspace)
    recursive.init_recursive_campaign(original)
    before = journal_state(workspace)
    changed = plan(workspace)
    changed["campaign_id"] = "renamed"
    changed[identity] = "source-two" if identity == "source_campaign_id" else str(workspace / "two")
    with pytest.raises(ValueError, match="already reserved"):
        recursive.init_recursive_campaign(changed)
    assert journal_state(workspace) == before


def test_v1_and_recursive_grants_coexist_without_allowance_inheritance(workspace):
    old = comparison.init_campaign(comparison_plan())
    value = plan(workspace)
    recursive.init_recursive_campaign(value)
    assert comparison.inspect_campaign("research-one") == old
    assert comparison.init_campaign(comparison_plan()) == old
    with pytest.raises(ValueError, match="not a forecast-comparison"):
        comparison.inspect_campaign("recursive-one")
    with recursive.recursive_scope(campaign_id="source-one") as scope:
        assert scope.grant["budget"] == value["budget"]
    assert len(journal_state(workspace)["events"]) == 3


@pytest.mark.parametrize("recursive_first", [False, True])
def test_cross_kind_source_id_collision_rejected_before_event(workspace, recursive_first):
    old = comparison_plan()
    new = plan(workspace)
    new["source_campaign_id"] = old["runs"][0]["run_id"]
    if recursive_first:
        recursive.init_recursive_campaign(new)
        before = journal_state(workspace)
        with pytest.raises(ValueError, match="already allocated"):
            comparison.init_campaign(old)
    else:
        comparison.init_campaign(old)
        before = journal_state(workspace)
        with pytest.raises(ValueError, match="already reserved"):
            recursive.init_recursive_campaign(new)
    assert journal_state(workspace) == before


@pytest.mark.parametrize(
    "relative", ["forecast_campaign_identity.json", "forecast_campaigns/journal.sqlite"]
)
def test_missing_ready_authority_does_not_reinitialize(workspace, relative):
    value = plan(workspace)
    recursive.init_recursive_campaign(value)
    (workspace / ".quant_rsi" / relative).unlink()
    with pytest.raises(ValueError):
        with recursive.recursive_scope(campaign_id="source-one"):
            pass
    with pytest.raises(ValueError):
        recursive.init_recursive_campaign(value)


def test_copying_authority_to_new_workspace_fails_closed(workspace, monkeypatch, tmp_path):
    recursive.init_recursive_campaign(plan(workspace))
    copied = tmp_path / "copied"
    shutil.copytree(workspace, copied)
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(copied))
    with pytest.raises(ValueError, match="identity"):
        with recursive.recursive_scope(campaign_id="source-one"):
            pass


@pytest.mark.parametrize("which", ["grant_ref", "research_plan_ref", "exposure_ref"])
def test_missing_cas_member_cannot_be_repaired_by_reentry(workspace, which):
    report = recursive.init_recursive_campaign(plan(workspace))
    ref = report["binding"][which]
    (workspace / ".quant_rsi/forecast_campaigns/artifacts" / ref).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        with recursive.recursive_scope(campaign_id="source-one"):
            pass


@pytest.mark.parametrize("replacement", ["closed", "independent", "unrelated"])
def test_reject_closed_reopened_or_recycled_owner_descriptor(workspace, replacement):
    recursive.init_recursive_campaign(plan(workspace))
    with recursive.recursive_scope(campaign_id="source-one") as scope:
        original = scope.owner_fd
        os.close(original)
        fd = None
        if replacement != "closed":
            path = (
                workspace / ".quant_rsi/.forecast_campaign.lock"
                if replacement == "independent"
                else workspace / "other"
            )
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            assert fd == original
        try:
            with pytest.raises(ValueError, match="actual owner descriptor|identity changed"):
                recursive.require_recursive_owner()
        finally:
            if fd is not None:
                os.close(fd)
    with recursive.recursive_scope(campaign_id="source-one"):
        recursive.require_recursive_owner()


def test_scope_exception_releases_without_changing_grant(workspace):
    value = plan(workspace)
    report = recursive.init_recursive_campaign(value)
    before = journal_state(workspace)
    with pytest.raises(RuntimeError, match="injected"):
        with recursive.recursive_scope(campaign_id="source-one"):
            raise RuntimeError("injected caller failure")
    with recursive.recursive_scope(campaign_id="source-one") as scope:
        assert scope.binding == report["binding"]
    assert journal_state(workspace) == before


def test_registration_transaction_failure_does_not_duplicate_reservation(workspace, monkeypatch):
    original = Journal.event

    def interrupt(self, ordinal, kind, payload):
        original(self, ordinal, kind, payload)
        if kind == "recursive_campaign_registered":
            raise RuntimeError("injected before commit")

    with monkeypatch.context() as patch:
        patch.setattr(Journal, "event", interrupt)
        with pytest.raises(RuntimeError, match="before commit"):
            recursive.init_recursive_campaign(plan(workspace))
    assert len(journal_state(workspace)["events"]) == 1
    result = recursive.init_recursive_campaign(plan(workspace))
    assert result["reserved"]["max_attempts"] == 36
    assert len(journal_state(workspace)["events"]) == 2


def test_committed_registration_response_loss_reuses_original_grant(workspace, monkeypatch):
    with monkeypatch.context() as patch:

        def lost(scope):
            raise RuntimeError("injected response loss")

        patch.setattr(recursive, "_report", lost)
        with pytest.raises(RuntimeError, match="response loss"):
            recursive.init_recursive_campaign(plan(workspace))
    before = journal_state(workspace)
    assert len(before["events"]) == 2
    result = recursive.init_recursive_campaign(plan(workspace))
    assert result["binding"]["grant_ref"] == before["events"][1]["payload"]["grant_ref"]
    assert journal_state(workspace) == before


def test_actual_exec_child_retains_authority_fd_after_controller_scope(workspace):
    import json
    import subprocess
    import sys

    recursive.init_recursive_campaign(plan(workspace))
    child = None
    try:
        with recursive.recursive_scope(campaign_id="source-one") as scope:
            stat = os.fstat(scope.owner_fd)
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import json,os,sys; s=os.fstat(int(sys.argv[1])); "
                    "print(json.dumps([s.st_dev,s.st_ino]),flush=True); sys.stdin.read(1)",
                    str(scope.owner_fd),
                ],
                pass_fds=(scope.owner_fd,),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert json.loads(child.stdout.readline()) == [stat.st_dev, stat.st_ino]
        with pytest.raises(ValueError, match="another campaign controller or worker"):
            with recursive.recursive_scope(campaign_id="source-one"):
                pass
        child.communicate("x", timeout=5)
        assert child.returncode == 0
        with recursive.recursive_scope(campaign_id="source-one"):
            recursive.require_recursive_owner()
    finally:
        if child is not None and child.poll() is None:
            child.communicate("x", timeout=5)
