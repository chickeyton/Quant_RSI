"""Recovery gates on actual committed pairs; synthetic fixtures, no live evidence."""

import fcntl
import shutil
from collections import Counter
from contextlib import ExitStack, contextmanager

import lightgbm as lgb
import pytest
from test_local_packages import local_c4_fixture as _local_c4_fixture
from test_trajectory import files, setup_plan

from quant_rsi.core import controller
from quant_rsi.core.artifacts import canonical
from quant_rsi.research import package_experiment, research, trajectory
from quant_rsi.research.package_journal import PackageJournal
from quant_rsi.research.trajectory_state import TrajectoryLedger, locked_trajectory, trajectory_root

local_c4_fixture = _local_c4_fixture


def forbidden(*args, **kwargs):
    pytest.fail("committed evidence must not dispatch or fit again")


@pytest.fixture(scope="module")
def committed_pair_copy(local_c4_fixture, tmp_path_factory):
    workspace = tmp_path_factory.mktemp("unfinished_pair")
    with pytest.MonkeyPatch.context() as patch:
        trajectory.init_trajectory(setup_plan(local_c4_fixture, workspace, patch))
        trajectory.resume_trajectory(steps=1)
        original = package_experiment.create_package

        def lost_response(*args, **kwargs):
            original(*args, **kwargs)
            raise SystemExit("after committed round")

        patch.setattr(package_experiment, "create_package", lost_response)
        with pytest.raises(SystemExit, match="after committed round"):
            trajectory.resume_trajectory(steps=1)
    backup = tmp_path_factory.mktemp("unfinished_backup") / "workspace"
    shutil.copytree(workspace, backup)
    return workspace, backup


@pytest.fixture
def unfinished_pair(committed_pair_copy, monkeypatch):
    workspace, backup = committed_pair_copy
    # Restore only this module's private temporary workspace, at its bound path.
    shutil.rmtree(workspace)
    shutil.copytree(backup, workspace)
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(workspace))
    return trajectory_root() / "children/stage_0"


def no_dispatch(monkeypatch):
    monkeypatch.setattr(package_experiment, "create_package", forbidden)
    monkeypatch.setattr(package_experiment, "resume_package", forbidden)
    monkeypatch.setattr(controller, "execute_attempt", forbidden)
    monkeypatch.setattr(lgb, "train", forbidden)


def test_unfinished_pair_keeps_evidence_costs_and_targets_with_two_fewer_arm_replays(
    unfinished_pair, monkeypatch
):
    child = unfinished_pair
    no_dispatch(monkeypatch)
    original_verify = research.verify_trace
    calls = Counter()

    def counted(journal, *args, **kwargs):
        if kwargs.get("replay"):
            path = journal.connection.execute("PRAGMA database_list").fetchone()[2]
            calls[path] += 1
        return original_verify(journal, *args, **kwargs)

    monkeypatch.setattr(research, "verify_trace", counted)
    with locked_trajectory() as (root, owner_fd):
        with TrajectoryLedger(root, owner_fd=owner_fd) as authority:
            before_head = canonical(authority.state["head"])
            unchanged = files(child)
            # Execute the old no-write detour against the same real committed pair.
            ref, receipt = trajectory.package_snapshot(authority, child)
            assert receipt["completion_ref"] is None
            assert all(r["execution_state"] == "paused" for r in receipt["reports"].values())
            trajectory.finish_committed_pair(authority, child, owner_fd)
            assert trajectory.package_snapshot(authority, child) == (ref, receipt)
            expected_arms = {}
            expected_costs = dict(worker_attempts=0, registered_worker_seconds=0)
            for arm in ("parent", "candidate"):
                arm_ref, arm_receipt = trajectory.capture_research(
                    child / "arms" / arm, authority.store, publish=True
                )
                expected_arms[arm] = arm_ref
                for unit in expected_costs:
                    expected_costs[unit] += arm_receipt["costs"][unit]
            old_calls = calls.copy()
            assert files(child) == unchanged

            calls.clear()
            trajectory.execute_package_round(authority, owner_fd)
            stage = authority.state["stages"][0]
            assert stage["receipts"][-1] == dict(
                package_receipt_ref=ref, arm_receipts=expected_arms, round=0
            )
            assert (
                stage["consumed"]
                == expected_costs
                == dict(worker_attempts=2, registered_worker_seconds=40)
            )
            assert stage["status"] == "running"
            assert authority.state["pending"]["round_index"] == 1
            assert authority.state["pending"]["targets"] == dict(parent=2, candidate=2)
            assert canonical(authority.state["head"]) == before_head
            assert authority.state["generation"] == authority.state["revision"] == 0
            assert files(child) == unchanged
            for arm in ("parent", "candidate"):
                path = str(child / "arms" / arm / "journal.sqlite")
                assert old_calls[path] == 4
                assert calls[path] == 2


@pytest.mark.parametrize("gate", ["first_child_lock", "second_child_lock", "owner_lock"])
def test_post_snapshot_recovery_gate_rejects_contention_or_lost_owner(
    unfinished_pair, monkeypatch, gate
):
    child = unfinished_pair
    no_dispatch(monkeypatch)
    with locked_trajectory() as (root, owner_fd), ExitStack() as held:
        with TrajectoryLedger(root, owner_fd=owner_fd) as authority:
            original_snapshot = trajectory.package_snapshot
            original_snapshot(authority, child)
            before = files(root)
            original_state = authority.state_ref

            ready = False
            child_locks = 0
            original_lock = controller.controller_lock

            @contextmanager
            def contend_after_snapshot(path):
                nonlocal child_locks
                if ready and path == child:
                    child_locks += 1
                    target = 1 if gate == "first_child_lock" else 2
                    if gate != "owner_lock" and child_locks == target:
                        held.enter_context(original_lock(child))
                with original_lock(path) as fd:
                    yield fd

            def snapshot_then_contend(*args, **kwargs):
                nonlocal ready
                result = original_snapshot(*args, **kwargs)
                ready = True
                if gate == "owner_lock":
                    fcntl.flock(owner_fd, fcntl.LOCK_UN)
                return result

            monkeypatch.setattr(controller, "controller_lock", contend_after_snapshot)
            monkeypatch.setattr(trajectory, "package_snapshot", snapshot_then_contend)
            monkeypatch.setattr(trajectory, "capture_research", forbidden)
            error = "lock is not held" if gate == "owner_lock" else "another controller"
            with pytest.raises(ValueError, match=error):
                trajectory.execute_package_round(authority, owner_fd)
            assert authority.state_ref == original_state
            assert files(root) == before


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_unfinished_pair_still_rejects_damaged_package_evidence(
    unfinished_pair, monkeypatch, damage
):
    child = unfinished_pair
    no_dispatch(monkeypatch)
    with locked_trajectory() as (root, owner_fd):
        with TrajectoryLedger(root, owner_fd=owner_fd) as authority:
            _, receipt = trajectory.package_snapshot(authority, child)
            artifact = child / "artifacts" / receipt["decision_ref"]
            if damage == "missing":
                artifact.unlink()
            else:
                artifact.write_bytes(b"corrupt committed decision")
            before = files(root)
            previous = authority.state_ref
            with pytest.raises(FileNotFoundError if damage == "missing" else ValueError):
                trajectory.execute_package_round(authority, owner_fd)
            assert authority.state_ref == previous
            assert files(root) == before


def test_both_complete_recovery_and_lost_completion_response_are_once_only(
    unfinished_pair, monkeypatch
):
    child = unfinished_pair
    original_complete = PackageJournal.complete

    def before_completion(*args, **kwargs):
        raise SystemExit("before completion write")

    with monkeypatch.context() as patch:
        patch.setattr(PackageJournal, "complete", before_completion)
        with pytest.raises(SystemExit, match="before completion write"):
            trajectory.resume_trajectory(steps=10)

    no_dispatch(monkeypatch)
    with locked_trajectory() as (root, owner_fd):
        with TrajectoryLedger(root, owner_fd=owner_fd) as authority:
            _, receipt = trajectory.package_snapshot(authority, child)
            assert receipt["completion_ref"] is None
            assert all(r["execution_state"] == "completed" for r in receipt["reports"].values())
            previous_state = authority.state_ref
            previous_head = canonical(authority.state["head"])
            completion_calls = []

            def after_completion(journal, ref):
                original_complete(journal, ref)
                completion_calls.append(ref)
                raise SystemExit("after completion write")

            with monkeypatch.context() as patch:
                patch.setattr(PackageJournal, "complete", after_completion)
                with pytest.raises(SystemExit, match="after completion write"):
                    trajectory.execute_package_round(authority, owner_fd)
            assert authority.state_ref == previous_state
            assert len(completion_calls) == 1
            completed_ref, completed = trajectory.package_snapshot(authority, child)
            assert completed["completion_ref"] == completion_calls[0]
            unchanged = files(child)

            monkeypatch.setattr(trajectory, "finish_committed_pair", forbidden)
            trajectory.execute_package_round(authority, owner_fd)
            stage = authority.state["stages"][0]
            assert stage["receipts"][-1]["package_receipt_ref"] == completed_ref
            assert stage["status"] == "evaluated"
            assert authority.state["pending"] is None
            assert canonical(authority.state["head"]) == previous_head
            assert files(child) == unchanged
            journal = PackageJournal(child)
            try:
                assert journal.state()["completion_ref"] == completion_calls[0]
                events = journal.connection.execute(
                    "SELECT COUNT(*) FROM events WHERE kind='package_completed'"
                ).fetchone()[0]
                assert events == 1
            finally:
                journal.close()
