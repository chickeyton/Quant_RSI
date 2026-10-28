import subprocess

import pytest

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, canonical, decode, digest
from quant_rsi.core.journal import Journal
from quant_rsi.core.schemas import DevelopmentPlan


def modified(plan, *, one=False, **budget):
    document = plan.model_dump(mode="json")
    document["budget"].update(budget)
    if one:
        document["proposals"] = document["proposals"][:1]
    return DevelopmentPlan.model_validate_json(canonical(document))


def journal_state(root):
    journal = Journal(root)
    try:
        return journal.attempts(), journal.events()
    finally:
        journal.close()


def forbid_worker(*args, **kwargs):
    pytest.fail("replay/recovery/resume must not dispatch another worker")


def test_runtime_identity_covers_nested_sources_without_name_collisions(tmp_path, monkeypatch):
    package = tmp_path / "quant_rsi"
    sources = {
        "__init__.py": "",
        "worker.py": "worker = 1\n",
        "core/controller.py": "controller = 1\n",
        "core/shared.py": "core = 1\n",
        "forecast/recursive/shared.py": "forecast = 1\n",
    }
    for name, body in sources.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    (package / "ignored.json").write_text("{}")
    monkeypatch.setattr(controller, "__file__", str(package / "core/controller.py"))

    before = controller.runtime_identity()
    assert before["sources"] == {name: digest(body.encode()) for name, body in sources.items()}
    assert list(before["sources"]) == sorted(sources)
    assert controller.runtime_identity() == before

    changed = "forecast/recursive/shared.py"
    (package / changed).write_text("forecast = 2\n")
    after = controller.runtime_identity()
    assert after["sources"][changed] != before["sources"][changed]
    assert {key: value for key, value in after["sources"].items() if key != changed} == {
        key: value for key, value in before["sources"].items() if key != changed
    }


def test_real_worker_demo_replay_and_idempotent_resume(tmp_path, plan, monkeypatch):
    root = tmp_path / "run"
    result = controller.create_run(plan, root)
    assert [r["status"] for r in result["results"]] == ["completed", "completed", "rejected"]
    assert result["attempts_used"] == 3
    assert result["worker_seconds_charged"] == 40
    assert result["evidence_status"] == "development_only"
    assert result["promotion_decision"] == "not_requested"
    before = journal_state(root)
    monkeypatch.setattr(controller.subprocess, "run", forbid_worker)
    replay = controller.inspect_run(root, replay=True)
    assert replay.pop("replay")["verified_experiments"] == 2
    assert replay == result == controller.resume_run(root)
    assert journal_state(root) == before


@pytest.mark.parametrize(
    "budget,expected,charged",
    [
        ({"max_attempts": 1}, ["completed"], 20),
        ({"max_worker_seconds": 20}, ["completed", "budget_exhausted", "rejected"], 20),
    ],
)
def test_budgets_stop_dispatch_and_keep_attempt_history(tmp_path, plan, budget, expected, charged):
    result = controller.create_run(modified(plan, **budget), tmp_path / "run")
    assert [r["status"] for r in result["results"]] == expected
    assert result["worker_seconds_charged"] == charged
    assert result["execution_status"] == "budget_exhausted"


def test_timeout_is_not_converted_to_cash_performance(tmp_path, plan, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(controller.subprocess, "run", timeout)
    root = tmp_path / "run"
    result = controller.create_run(modified(plan, one=True), root)
    assert result["results"][0]["status"] == "failed"
    assert result["results"][0]["error"] == "worker_timeout"
    assert "metrics" not in result["results"][0]
    assert result["development_incumbent"] is None
    assert result["attempts_used"] == 1 and result["worker_seconds_charged"] == 20
    monkeypatch.setattr(controller.subprocess, "run", forbid_worker)
    assert controller.resume_run(root) == result


def test_recover_output_after_crash_without_second_training(tmp_path, plan, monkeypatch):
    root = tmp_path / "run"
    original_finish = Journal.finish

    def crash_at_commit(self, ordinal, status, **kwargs):
        if status == "completed":
            raise RuntimeError("injected controller crash before commit")
        return original_finish(self, ordinal, status, **kwargs)

    monkeypatch.setattr(Journal, "finish", crash_at_commit)
    with pytest.raises(RuntimeError, match="injected"):
        controller.create_run(modified(plan, one=True), root)
    assert journal_state(root)[0][0]["status"] == "running"
    monkeypatch.setattr(Journal, "finish", original_finish)
    monkeypatch.setattr(controller.subprocess, "run", forbid_worker)
    result = controller.resume_run(root)
    assert result["results"][0]["status"] == "completed"
    assert result["attempts_used"] == 1 and result["worker_seconds_charged"] == 20
    assert controller.inspect_run(root, replay=True)["replay"]["verified_experiments"] == 1


def test_crash_without_output_becomes_interrupted_and_is_not_retried(tmp_path, plan, monkeypatch):
    root = tmp_path / "run"

    def crash(*args, **kwargs):
        raise RuntimeError("injected controller crash before worker output")

    monkeypatch.setattr(controller.subprocess, "run", crash)
    with pytest.raises(RuntimeError, match="injected"):
        controller.create_run(modified(plan, one=True), root)
    monkeypatch.setattr(controller.subprocess, "run", forbid_worker)
    result = controller.resume_run(root)
    assert result["results"][0]["status"] == "interrupted"
    assert result["worker_seconds_charged"] == 20
    assert controller.resume_run(root) == result


@pytest.mark.parametrize("target", ["manifest", "predictions"])
def test_tampering_is_detected_on_replay(tmp_path, plan, target):
    root = tmp_path / "run"
    controller.create_run(modified(plan, one=True), root)
    if target == "manifest":
        path = root / "manifest.json"
    else:
        store = ArtifactStore(root / "artifacts")
        bundle = store.json(journal_state(root)[0][0]["result_ref"])
        path = root / "artifacts" / bundle["predictions_ref"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity failure"):
        controller.inspect_run(root, replay=True)


def test_second_controller_cannot_acquire_run_lock(tmp_path):
    with controller.controller_lock(tmp_path):
        with pytest.raises(ValueError, match="still owns"):
            with controller.controller_lock(tmp_path):
                pytest.fail("acquired occupied lock")


def test_worker_claimed_metrics_are_not_authoritative(tmp_path, plan, monkeypatch):
    original_accept = controller.accept_worker_result

    def inject_claim(work, *args):
        path = work / "result.json"
        result = decode(path.read_bytes())
        result["claimed_metrics"] = {"total_return": 999.0}
        path.write_bytes(canonical(result))
        return original_accept(work, *args)

    monkeypatch.setattr(controller, "accept_worker_result", inject_claim)
    result = controller.create_run(modified(plan, one=True), tmp_path / "run")
    assert result["results"][0]["status"] == "completed"
    assert result["results"][0]["metrics"]["total_return"] < 1.0


def test_run_directory_can_move_without_changing_computation(tmp_path, plan):
    original = tmp_path / "original"
    before = controller.create_run(modified(plan, one=True), original)
    moved = tmp_path / "moved"
    original.rename(moved)
    assert controller.inspect_run(moved, replay=True)["replay"]["verified_experiments"] == 1
    assert controller.resume_run(moved) == before


@pytest.mark.parametrize("malformation", ["missing_predictions", "bad_rows", "training_cutoff"])
def test_invalid_output_is_recorded_as_failure(tmp_path, plan, monkeypatch, malformation):
    original_accept = controller.accept_worker_result

    def corrupt_output(work, *args):
        path = work / "result.json"
        result = decode(path.read_bytes())
        if malformation == "missing_predictions":
            result.pop("predictions")
        elif malformation == "bad_rows":
            result["predictions"] = [None]
        else:
            result["last_training_label_available"] = "2099-01-01T09:00:00+00:00"
        path.write_bytes(canonical(result))
        return original_accept(work, *args)

    monkeypatch.setattr(controller, "accept_worker_result", corrupt_output)
    result = controller.create_run(modified(plan, one=True), tmp_path / "run")
    assert result["results"][0]["status"] == "failed"
    assert result["results"][0]["error"].startswith("invalid_output:")
    assert result["worker_seconds_charged"] == 20
    assert result["development_incumbent"] is None


def test_runtime_change_allows_status_but_prevents_replay(tmp_path, plan, monkeypatch):
    root = tmp_path / "run"
    controller.create_run(modified(plan, one=True), root)
    monkeypatch.setattr(controller, "runtime_identity", lambda: {"different": "environment"})
    assert controller.inspect_run(root)["results"][0]["status"] == "completed"
    with pytest.raises(ValueError, match="source/runtime changed"):
        controller.inspect_run(root, replay=True)
    with pytest.raises(ValueError, match="source/runtime changed"):
        controller.resume_run(root)
