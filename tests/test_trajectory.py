"""Actual local development trajectory; invented market fixture, no performance claim."""

import shutil
import sqlite3

import lightgbm as lgb
import pytest
from test_local_packages import local_c4_fixture as _local_c4_fixture
from test_local_packages import local_workspace

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, canonical, digest
from quant_rsi.evaluation import evidence, local_research
from quant_rsi.research import changes, package_experiment, research, trajectory
from quant_rsi.research.trajectory_evidence import verify_package_receipt
from quant_rsi.research.trajectory_state import (
    TrajectoryLedger,
    TrajectoryPlan,
    locked_trajectory,
    trajectory_root,
)

local_c4_fixture = _local_c4_fixture


def files(root):
    return {
        str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()
    }


def missing_artifact_fails_closed(root, ref):
    path = root / "artifacts" / ref
    raw = path.read_bytes()
    path.unlink()
    before = files(root)
    try:
        for action in (trajectory.inspect_trajectory, trajectory.resume_trajectory):
            with pytest.raises(FileNotFoundError):
                action()
            assert files(root) == before
    finally:
        path.write_bytes(raw)


def rejected_package_closures(root, report, audit_root):
    """Derive corruption cases from one actual pair, without new workers."""
    shutil.copytree(root / "artifacts", audit_root / "artifacts")
    store = ArtifactStore(audit_root / "artifacts")
    pair = report["stages"][0]["receipts"][-1]
    original = store.json(pair["package_receipt_ref"])
    original_files = store.json(original["files_ref"])
    child = store.json(pair["arm_receipts"]["parent"])
    child_files = store.json(child["files_ref"])
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.deserialize(store.get(child_files["journal.sqlite"]))
    attempt = dict(connection.execute("SELECT * FROM attempts LIMIT 1").fetchone())
    connection.close()
    verify_package_receipt(store, original, pair["arm_receipts"])
    cases = {
        "outer_attempt": "unallocated outer Kernel",
        "missing_initialization": "event binding mismatch: arm_initialized",
        "uninitialized_work": "before durable outer initialization",
        "extra_arm": "unallocated child research directory",
        "child_snapshot": "child snapshot differs",
        "child_pending": "child snapshot differs",
    }
    for case, error in cases.items():
        receipt, captured = dict(original), dict(original_files)
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.deserialize(store.get(captured["journal.sqlite"]))
        with connection:
            if case == "outer_attempt":
                columns = ",".join(attempt)
                parameters = ",".join("?" for _ in attempt)
                connection.execute(
                    f"INSERT INTO attempts ({columns}) VALUES ({parameters})",
                    list(attempt.values()),
                )
            elif case == "missing_initialization":
                connection.execute("DELETE FROM events WHERE kind='arm_initialized'")
            elif case == "uninitialized_work":
                connection.execute("UPDATE package_grants SET initialized=0")
            elif case == "extra_arm":
                captured["arms/unallocated/audit.json"] = store.put_json({"audit": True})
            elif case == "child_snapshot":
                captured["arms/parent/journal.sqlite"] = captured["journal.sqlite"]
            elif case == "child_pending":
                captured["arms/parent/research_initialization.json"] = store.put_json({})
        receipt["grants"] = [
            dict(row) for row in connection.execute("SELECT * FROM package_grants ORDER BY arm")
        ]
        captured["journal.sqlite"] = store.put(connection.serialize())
        connection.close()
        receipt["files_ref"] = store.put_json(captured)
        before = files(audit_root)
        with pytest.raises(ValueError, match=error):
            verify_package_receipt(store, receipt, pair["arm_receipts"])
        assert files(audit_root) == before


def setup_plan(fixture, tmp_path, monkeypatch, *, budget=None):
    local_workspace(fixture, tmp_path, monkeypatch)
    source = tmp_path / "source"
    shutil.copytree(fixture["root"] / "source", source)
    return TrajectoryPlan.model_validate_json(
        canonical(
            dict(
                name="finite-development",
                source_directory=str(source),
                source_checkpoint_ref=fixture["source"]["checkpoint_ref"],
                state_policy="reset",
                max_proposal_stages=2,
                budget=budget or dict(worker_attempts=20, registered_worker_seconds=400),
            )
        )
    )


def test_actual_finite_chain_seals_saturation_at_fully_occupied_capacity(
    local_c4_fixture, tmp_path, monkeypatch
):
    plan = setup_plan(local_c4_fixture, tmp_path, monkeypatch)

    def forbidden_protected_authority(*args, **kwargs):
        pytest.fail("development trajectory accessed protected C5 authority")

    monkeypatch.setattr(evidence.Authority, "__init__", forbidden_protected_authority)
    original = files(plan.source_directory)
    initial = trajectory.init_trajectory(plan)
    assert initial["generation"] == 0
    assert initial["budget"]["confirmed_consumed"] == dict(
        worker_attempts=4, registered_worker_seconds=80
    )
    plan.source_directory.rename(tmp_path / "source-offline")
    market = tmp_path / ".quant_rsi/market"
    if market.exists():
        market.rename(tmp_path / "market-offline")
    same = trajectory.init_trajectory(plan.model_copy(update={"name": "renamed-handle"}))
    assert same == initial
    advanced = trajectory.resume_trajectory(steps=7)
    assert advanced["status"] == "ACTIVE" and advanced["generation"] == 1
    with (
        locked_trajectory() as (root, owner_fd),
        TrajectoryLedger(root, owner_fd=owner_fd) as authority,
    ):
        receipt_ref = next(iter(authority.state["decisions"].values()))
        receipt = authority.store.json(receipt_ref)
        args = {
            key: receipt["request"][key]
            for key in ("request_id", "expected_head", "expected_revision")
        }
        assert trajectory.advance_head(authority, **args) == receipt
        with pytest.raises(ValueError, match="compare-and-swap"):
            trajectory.advance_head(authority, **(args | {"request_id": "stale-new-request"}))
    report = trajectory.resume_trajectory(steps=30)
    assert report["status"] == "SEALED", report
    assert report["terminal"]["reason"] == "grammar_exhausted", report["terminal"]
    assert report["generation"] == report["revision"] == 1
    assert len(report["stages"]) == 2
    assert report["stages"][1]["capacity"] == dict(worker_attempts=0, registered_worker_seconds=0)
    assert report["budget"]["occupied"] == plan.budget.model_dump()
    assert report["budget"]["confirmed_consumed"] == dict(
        worker_attempts=13, registered_worker_seconds=260
    )
    assert files(tmp_path / "source-offline") == original
    root = trajectory_root()
    frozen_store = ArtifactStore(tmp_path / "frozen-trajectory")
    with TrajectoryLedger(root, readonly=True) as authority:
        for artifact in (root / "artifacts").iterdir():
            frozen_store.put(authority.store.get(artifact.name))
        closure = dict(
            authority_root=str(root),
            anchor=authority.anchor,
            control_ref=frozen_store.put(authority.connection.serialize()),
            state_ref=authority.state_ref,
        )
    before = files(root)

    def forbidden(*args, **kwargs):
        pytest.fail("readonly trajectory attempted live inputs, proposal, fit or publication")

    with monkeypatch.context() as patch:
        for obj, name in [
            (ArtifactStore, "put"),
            (local_research, "export_local_data"),
            (changes, "propose_change"),
            (lgb, "train"),
            (controller, "execute_attempt"),
            (research, "resume_research"),
            (package_experiment, "resume_package"),
        ]:
            patch.setattr(obj, name, forbidden)
        assert trajectory.inspect_trajectory() == report
        assert trajectory.resume_trajectory(steps=30) == report
        with TrajectoryLedger.frozen(frozen_store, closure) as frozen:
            assert trajectory.summarize(frozen) == report
    assert files(root) == before
    with locked_trajectory() as (root, _), TrajectoryLedger(root) as authority:
        request_id, receipt_ref = next(iter(authority.state["decisions"].items()))
        receipt = authority.store.json(receipt_ref)
        args = {
            key: receipt["request"][key]
            for key in ("request_id", "expected_head", "expected_revision")
        }
        assert trajectory.advance_head(authority, **args) == receipt
        with pytest.raises(ValueError):
            trajectory.advance_head(authority, **(args | {"request_id": "stale-new-request"}))
    missing_artifact_fails_closed(root, receipt_ref)
    store = ArtifactStore(root / "artifacts")
    program = store.json(report["program_ref"])
    missing_artifact_fails_closed(root, program["saturation_proposer_ref"])
    rejected_package_closures(root, report, tmp_path / "derived-package-audit")


def test_insufficient_budget_stops_before_child_grants(local_c4_fixture, tmp_path, monkeypatch):
    plan = setup_plan(
        local_c4_fixture,
        tmp_path,
        monkeypatch,
        budget=dict(worker_attempts=4, registered_worker_seconds=80),
    )
    trajectory.init_trajectory(plan)
    report = trajectory.resume_trajectory(steps=30)
    assert report["terminal"]["reason"] == "insufficient_child_capacity"
    assert not report["stages"]
    assert not (trajectory_root() / "children").exists()


def test_lost_child_response_reconciles_original_round_without_dispatch(
    local_c4_fixture, tmp_path, monkeypatch
):
    trajectory.init_trajectory(setup_plan(local_c4_fixture, tmp_path, monkeypatch))
    trajectory.resume_trajectory(steps=1)
    original_create = package_experiment.create_package

    def lost_response(*args, **kwargs):
        original_create(*args, **kwargs)
        raise SystemExit("committed child result, lost outer response")

    with monkeypatch.context() as patch:
        patch.setattr(package_experiment, "create_package", lost_response)
        with pytest.raises(SystemExit):
            trajectory.resume_trajectory(steps=1)
    before = trajectory.inspect_trajectory()
    assert before["pending"]["round_index"] == 0
    assert before["budget"]["unreconciled_child_costs"]
    child = trajectory_root() / "children/stage_0"
    unchanged = files(child)
    with pytest.raises(ValueError, match="owner lock"):
        package_experiment.resume_package(child)
    with pytest.raises(ValueError, match="managed research"):
        research.resume_research(child / "arms/parent")
    assert files(child) == unchanged

    def forbidden(*args, **kwargs):
        pytest.fail("committed child response was dispatched again")

    with monkeypatch.context() as patch:
        patch.setattr(package_experiment, "resume_package", forbidden)
        patch.setattr(controller, "execute_attempt", forbidden)
        report = trajectory.resume_trajectory(steps=1)
    assert report["pending"]["round_index"] == 1
    assert report["budget"]["confirmed_consumed"] == dict(
        worker_attempts=6, registered_worker_seconds=120
    )
    original_update = TrajectoryLedger.update

    def interrupted_commit(authority, kind, change):
        original_update(authority, kind, change)
        if kind == "trajectory_child_operation_committed":
            raise SystemExit("after child receipt transaction commits")

    with monkeypatch.context() as patch:
        patch.setattr(TrajectoryLedger, "update", interrupted_commit)
        with pytest.raises(SystemExit):
            trajectory.resume_trajectory(steps=1)
    committed = trajectory.inspect_trajectory()
    assert committed["pending"]["round_index"] == 2
    resumed = trajectory.resume_trajectory(steps=1)
    assert resumed["pending"]["round_index"] == 3


def test_first_no_patch_is_not_a_fake_generation(local_c4_fixture, tmp_path, monkeypatch):
    plan = setup_plan(local_c4_fixture, tmp_path, monkeypatch)
    source = tmp_path / "short-source"
    one = research.create_research(local_c4_fixture["plan"], source, steps=1)
    value = plan.model_dump(mode="json") | {
        "source_directory": str(source),
        "source_checkpoint_ref": one["checkpoint_ref"],
    }
    trajectory.init_trajectory(TrajectoryPlan.model_validate_json(canonical(value)))
    original_update = TrajectoryLedger.update

    def interrupted_no_patch(authority, kind, change):
        original_update(authority, kind, change)
        if kind == "trajectory_child_operation_committed":
            raise SystemExit("after no_patch receipt, before terminal receipt")

    with monkeypatch.context() as patch:
        patch.setattr(TrajectoryLedger, "update", interrupted_no_patch)
        with pytest.raises(SystemExit):
            trajectory.resume_trajectory(steps=20)
    report = trajectory.resume_trajectory(steps=20)
    assert report["status"] == "SEALED"
    assert report["terminal"]["reason"] == "first_proposal_no_patch"
    assert report["generation"] == 0 and len(report["stages"]) == 1
    assert report["stages"][0]["consumed"] == dict(worker_attempts=0, registered_worker_seconds=0)
    assert report["budget"]["confirmed_consumed"] == dict(
        worker_attempts=1, registered_worker_seconds=20
    )


def test_checkpoint_admission_and_readonly_corruption_fail_closed(
    local_c4_fixture, tmp_path, monkeypatch
):
    plan = setup_plan(local_c4_fixture, tmp_path, monkeypatch)
    trajectory.init_trajectory(plan)
    root = trajectory_root()
    with TrajectoryLedger(root, readonly=True) as authority:
        with pytest.raises(ValueError, match="owner lock"):
            authority.update("unauthorized", lambda state: state.update(pending=None))
        receipt = authority.store.json(authority.program["initial_head"]["research_receipt_ref"])
        model_ref = authority.store.json(receipt["selection_ref"])["model_ref"]
        proposer_refs = [
            authority.program[key] for key in ("proposer_ref", "saturation_proposer_ref")
        ]
    for ref in proposer_refs:
        missing_artifact_fails_closed(root, ref)
    missing_artifact_fails_closed(root, model_ref)
    with pytest.raises(ValueError, match="different frozen inputs"):
        trajectory.init_trajectory(plan.model_copy(update={"source_checkpoint_ref": "0" * 64}))
    model = root / "artifacts" / model_ref
    model.write_bytes(model.read_bytes() + b"corrupt")
    before = files(root)
    for action in (trajectory.inspect_trajectory, trajectory.resume_trajectory):
        with pytest.raises(ValueError):
            action()
        assert files(root) == before


def test_pending_source_is_rejected_without_proposal_or_child_grant(
    local_c4_fixture, tmp_path, monkeypatch
):
    plan = setup_plan(local_c4_fixture, tmp_path, monkeypatch)
    source = tmp_path / "pending-source"
    first = research.create_research(local_c4_fixture["plan"], source, steps=1)

    def interrupted_before_dispatch(*args, **kwargs):
        raise SystemExit("original research request is reserved, worker not dispatched")

    with monkeypatch.context() as patch:
        patch.setattr(controller, "execute_attempt", interrupted_before_dispatch)
        with pytest.raises(SystemExit):
            research.resume_research(source, steps=1)
    before = files(source)
    value = plan.model_dump(mode="json") | {
        "source_directory": str(source),
        "source_checkpoint_ref": first["checkpoint_ref"],
    }
    with pytest.raises(ValueError):
        trajectory.init_trajectory(TrajectoryPlan.model_validate_json(canonical(value)))
    assert files(source) == before
    assert not (trajectory_root() / "children").exists()
    assert not (trajectory_root().parent / "trajectory_anchor.json").exists()
