from pathlib import Path

import pytest

from quant_rsi.core import controller
from quant_rsi.core.artifacts import canonical, decode, digest
from quant_rsi.core.journal import Journal
from quant_rsi.core.schemas import DevelopmentPlan


def one_candidate(plan):
    document = plan.model_dump(mode="json")
    document["proposals"] = document["proposals"][1:2]
    document["baseline"] = document["proposals"][0]["name"]
    return DevelopmentPlan.model_validate_json(canonical(document))


def state(root):
    journal = Journal(root)
    try:
        return journal.attempts(), journal.events()
    finally:
        journal.close()


@pytest.mark.parametrize("mutation", ["stale_digest", "rehashed", "duplicate_outer_factor"])
def test_actual_worker_factor_is_bound_to_original_plan(tmp_path, plan, monkeypatch, mutation):
    original_run = controller.subprocess.run

    def corrupt_job(args, **kwargs):
        path = Path(args[-1])
        job = decode(path.read_bytes())
        if mutation == "duplicate_outer_factor":
            job["factor"] = {"op": "mean_return", "window": 5}
        else:
            job["binding"]["proposal"]["factor"]["op"] = "mean_return"
            if mutation == "rehashed":
                job["job_digest"] = digest(canonical(job["binding"]))
        path.write_bytes(canonical(job))
        return original_run(args, **kwargs)

    monkeypatch.setattr(controller.subprocess, "run", corrupt_job)
    root = tmp_path / "run"
    report = controller.create_run(one_candidate(plan), root)
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["error"] == "worker_exit_1"
    assert report["attempts_used"] == 1 and report["worker_seconds_charged"] == 20
    assert not (root / "work/0000/result.json").exists()
    assert controller.inspect_run(root, replay=True)["replay"]["verified_experiments"] == 0
    before = state(root)
    assert controller.resume_run(root) == report
    assert state(root) == before


@pytest.mark.parametrize("target", ["job", "result"])
def test_parent_rejects_input_or_returned_identity_mismatch(tmp_path, plan, monkeypatch, target):
    original_accept = controller.accept_worker_result

    def corrupt_after_execution(work, *args):
        path = work / (target + ".json")
        document = decode(path.read_bytes())
        if target == "job":
            document["binding"]["proposal"]["factor"]["op"] = "mean_return"
        else:
            document["job_digest"] = "0" * 64
        path.write_bytes(canonical(document))
        return original_accept(work, *args)

    monkeypatch.setattr(controller, "accept_worker_result", corrupt_after_execution)
    root = tmp_path / "run"
    report = controller.create_run(one_candidate(plan), root)
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["error"].startswith("invalid_output:")
    assert report["worker_seconds_charged"] == 20
    assert controller.inspect_run(root, replay=True)["replay"]["verified_experiments"] == 0


@pytest.mark.parametrize(
    "stage", ["before_manifest", "before_journal", "before_commit", "after_commit"]
)
@pytest.mark.parametrize("retry", ["resume", "create"])
def test_initialization_interruption_recovers_once(tmp_path, plan, monkeypatch, stage, retry):
    root = tmp_path / "run"
    plan = one_candidate(plan)
    original_write = controller.atomic_write
    original_event = Journal.event

    def write(path, data):
        if stage == "before_manifest" and path.name == "manifest.json":
            raise RuntimeError("injected initialization interruption")
        return original_write(path, data)

    def journal(*args, **kwargs):
        if stage == "before_journal":
            raise RuntimeError("injected initialization interruption")
        instance = Journal(*args, **kwargs)
        if stage == "after_commit":
            instance.close()
            raise RuntimeError("injected initialization interruption")
        return instance

    def event(self, ordinal, kind, payload):
        original_event(self, ordinal, kind, payload)
        if stage == "before_commit" and kind == "run_created":
            raise RuntimeError("injected initialization interruption")

    with monkeypatch.context() as injected:
        injected.setattr(controller, "atomic_write", write)
        injected.setattr(controller, "Journal", journal)
        injected.setattr(Journal, "event", event)
        with pytest.raises(RuntimeError, match="injected"):
            controller.create_run(plan, root)
    assert (root / "initialization.json").exists()
    if retry == "resume":
        report = controller.resume_run(root)
    else:
        report = controller.create_run(plan, root)
    assert report["results"][0]["status"] == "completed"
    assert report["attempts_used"] == 1 and report["worker_seconds_charged"] == 20
    assert not (root / "initialization.json").exists()
    rows, events = state(root)
    assert sum(e["kind"] == "run_created" for e in events) == 1
    assert sum(e["kind"] == "worker_slot_reserved" for e in events) == 1
    assert controller.resume_run(root) == report
    assert state(root) == (rows, events)


def test_retry_before_first_record_and_different_plan_rejection(tmp_path, plan, monkeypatch):
    root = tmp_path / "early"
    original_write = controller.atomic_write

    def crash(path, data):
        if path.name == "initialization.json":
            raise RuntimeError("injected before first record")
        return original_write(path, data)

    with monkeypatch.context() as injected:
        injected.setattr(controller, "atomic_write", crash)
        with pytest.raises(RuntimeError):
            controller.create_run(one_candidate(plan), root)
    assert controller.create_run(one_candidate(plan), root)["results"][0]["status"] == "completed"

    pending = tmp_path / "pending"
    with monkeypatch.context() as injected:

        def crash_before_journal(*args, **kwargs):
            raise RuntimeError("injected before journal")

        injected.setattr(controller, "Journal", crash_before_journal)
        with pytest.raises(RuntimeError):
            controller.create_run(plan, pending)
    before = (pending / "initialization.json").read_bytes()
    with pytest.raises(ValueError, match="different plan"):
        controller.create_run(one_candidate(plan), pending)
    assert (pending / "initialization.json").read_bytes() == before
    assert not (pending / "journal.sqlite").exists()


@pytest.mark.parametrize(
    "damage", ["none", "missing_identity", "missing_database", "partial_schema"]
)
def test_committed_attempts_are_preserved_and_never_reinitialized(tmp_path, plan, damage):
    root = tmp_path / "run"
    plan = one_candidate(plan)
    report = controller.create_run(plan, root)
    before = state(root)
    manifest_digest = digest((root / "manifest.json").read_bytes())
    if damage == "none":
        journal = Journal(root, manifest_digest)
        journal.close()
        assert controller.resume_run(root) == report
        assert state(root) == before
    elif damage == "missing_database":
        (root / "journal.sqlite").rename(root / "saved_journal.sqlite")
        with pytest.raises(ValueError, match="no initialized journal"):
            controller.resume_run(root)
        assert not (root / "journal.sqlite").exists()
    else:
        journal = Journal(root)
        with journal.connection:
            if damage == "missing_identity":
                journal.connection.execute("DELETE FROM metadata")
            else:
                journal.connection.execute("DROP TABLE metadata")
        journal.close()
        with pytest.raises(ValueError, match="identity|schema"):
            Journal(root, manifest_digest)
        assert state(root) == before
    with pytest.raises(FileExistsError):
        controller.create_run(plan, root)
