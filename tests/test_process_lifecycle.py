import os
import signal
import subprocess
import sys

import pytest

from quant_rsi.core.artifacts import ArtifactStore, canonical
from quant_rsi.core.controller import controller_lock, job_for
from quant_rsi.core.schemas import DevelopmentPlan


def test_child_retains_run_lock_after_parent_descriptor_closes(tmp_path):
    child = None
    try:
        with controller_lock(tmp_path) as fd:
            child = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(fd,),
            )
        with pytest.raises(ValueError, match="still owns"):
            with controller_lock(tmp_path):
                pytest.fail("worker's inherited lock was lost")
        child.communicate(b"x", timeout=5)
        assert child.returncode == 0
        with controller_lock(tmp_path):
            pass
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout=5)


def test_worker_self_deadline_terminates_blocked_io(tmp_path, plan):
    # No controller timeout is involved: block the real worker opening its data artifact.
    work = tmp_path / "work" / "0000"
    work.mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    data_ref = "a" * 64
    os.mkfifo(artifacts / data_ref)
    document = plan.model_dump(mode="json")
    document["budget"]["worker_timeout_seconds"] = 1
    plan = DevelopmentPlan.model_validate_json(canonical(document))
    store = ArtifactStore(artifacts)
    manifest = {
        "schema_version": 1,
        "plan_ref": store.put_json(document),
        "data_ref": data_ref,
        "runtime_ref": store.put_json({}),
    }
    job = job_for(0, plan.proposals[0].model_dump(mode="json"), manifest, plan)
    path = work / "job.json"
    path.write_bytes(canonical(job))
    with (work / "worker.log").open("wb") as log:
        process = subprocess.run(
            [sys.executable, "-m", "quant_rsi.worker", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    assert process.returncode == -signal.SIGALRM
    assert not (work / "result.json").exists()
