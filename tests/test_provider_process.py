"""Actual CLI and local provider processes; offline fake Codex, never a paid call."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_research_study import command

from quant_rsi.research.llm import LLMHypothesizer
from quant_rsi.research.provider_process import provider_lock
from quant_rsi.research.study_resume import run_lock


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("Local fixture did not finish within the declared 5 second deadline")


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / "codex"
    script.write_text(
        f"#!{sys.executable}\n"
        "import json,os,signal,subprocess,sys,time\nfrom pathlib import Path\n"
        "root=Path(os.environ['PROVIDER_FIXTURE'])\n"
        "if (root/'dynamic').exists():\n"
        " from quant_rsi.research.demo import simulated_hypothesis\n"
        " data=json.loads(sys.stdin.read().split('\\nINPUT_JSON:\\n',1)[1])\n"
        " reply=simulated_hypothesis(data['context'],data['catalog'],"
        "previous=data['previous'],feedback=data['feedback'])\n"
        " schema=Path(sys.argv[sys.argv.index('--output-schema')+1])\n"
        " keys=json.loads(schema.read_text())['properties']\n"
        " reply={k:v for k,v in reply.items() if k in keys}\n"
        " Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps(reply))\n"
        " sys.exit(0)\n"
        "if (root/'succeed').exists():\n"
        " Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('{}')\n"
        " sys.exit(0)\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'],"
        "start_new_session=True)\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "(root/'pids.json').write_text(json.dumps([os.getpid(),child.pid,os.getppid()]))\n"
        "time.sleep(60)\n"
    )
    script.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PROVIDER_FIXTURE", str(tmp_path))
    yield tmp_path
    if (tmp_path / "pids.json").exists():
        for pid in json.loads((tmp_path / "pids.json").read_text()):
            for sig in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass


def pids(root):
    path = root / "pids.json"
    return wait_for(lambda: json.loads(path.read_text()) if path.exists() else None)


def gone(ids):
    return all(not Path(f"/proc/{pid}").exists() for pid in ids)


@pytest.mark.parametrize("ending", ["kill", "interrupt"])
def test_actual_study_owner_loss_blocks_resume_until_cleanup(fake_codex, ending):
    root = fake_codex / "study"
    with (fake_codex / "owner.log").open("w") as log:
        owner = subprocess.Popen(
            [sys.executable, "-m", "quant_rsi", *command(root, "--llm", "codex")],
            stdout=log,
            stderr=log,
        )
        try:
            direct, child, guardian = pids(fake_codex)
            # Freeze supervision so a racing continuation must prove ownership, not timing luck.
            os.kill(guardian, signal.SIGSTOP)
            owner.send_signal(signal.SIGKILL if ending == "kill" else signal.SIGINT)
            owner.wait(timeout=8)
            resumed = subprocess.run(
                [sys.executable, "-m", "quant_rsi", "research-resume", str(root)],
                stdout=log,
                stderr=log,
                timeout=5,
            )
            assert resumed.returncode != 0
            with pytest.raises(ValueError, match="previous local provider"):
                with run_lock(root):
                    pass
            assert not list((root / "checkpoints").glob("hypothesis*.json"))
            os.kill(guardian, signal.SIGCONT)
            wait_for(lambda: gone([direct, child]))
            wait_for(
                lambda: (
                    "finished_at"
                    in json.loads(next((root / "provider_attempts").glob("*.json")).read_text())
                )
            )
            with run_lock(root):
                pass
            attempt = json.loads(next((root / "provider_attempts").glob("*.json")).read_text())
            assert attempt["status"] == "owner_lost"
            assert attempt["remote_completion"] == "unknown"
            assert (root / "config.json").exists()
            original_identity = (root / "checkpoints/identity.json").read_bytes()
            (fake_codex / "dynamic").touch()
            resumed = subprocess.run(
                [sys.executable, "-m", "quant_rsi", "research-resume", str(root)],
                stdout=log,
                stderr=log,
                timeout=15,
            )
            assert resumed.returncode == 0
            assert (root / "checkpoints/identity.json").read_bytes() == original_identity
            assert len(list((root / "provider_attempts").glob("*.json"))) == 3
            assert json.loads((root / "result.json").read_bytes())["summary"]["factors_changed"]
        finally:
            if owner.poll() is None:
                owner.kill()
                owner.wait(timeout=5)


def test_timeout_removes_detached_descendant_and_preserves_failure(fake_codex):
    root = fake_codex / "study"
    with (fake_codex / "timeout.log").open("w") as log:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "quant_rsi",
                *command(root, "--llm", "codex", "--llm-timeout", "1"),
            ],
            stdout=log,
            stderr=log,
            timeout=10,
        )
    assert run.returncode != 0
    direct, child, _ = pids(fake_codex)
    wait_for(lambda: gone([direct, child]))
    attempt = json.loads(next((root / "provider_attempts").glob("*.json")).read_text())
    assert attempt["status"] == "timeout"
    assert attempt["local_cleanup"] == "completed"
    assert "timed out" in (root / "failure.json").read_text()
    assert not list((root / "checkpoints").glob("hypothesis*.json"))


def test_success_and_explicit_unlock_keep_distinct_operation_ownership(fake_codex):
    root = fake_codex / "study"
    root.mkdir()
    (fake_codex / "succeed").touch()
    with run_lock(root):
        response = LLMHypothesizer("codex")._codex_response("fixture", {}, search=False)
        assert response["text"] == "{}"
        operation = provider_lock(root)
    # run_lock explicitly unlocks its own fd, but must not release a provider operation.
    try:
        with pytest.raises(ValueError, match="previous local provider"):
            with run_lock(root):
                pass
    finally:
        os.close(operation)
    with run_lock(root):
        pass
    attempt = json.loads(next((root / "provider_attempts").glob("*.json")).read_text())
    assert attempt["status"] == "local_completed"
    assert attempt["remote_completion"] == "unknown"


def test_interruption_after_local_response_before_checkpoint_then_actual_resume(fake_codex):
    root = fake_codex / "study"
    (fake_codex / "dynamic").touch()
    script = fake_codex / "commit_gap.py"
    script.write_text(
        "import os,sys\n"
        "from quant_rsi.research import study_resume\n"
        "from quant_rsi.cli import main\n"
        "original=study_resume.atomic_write\n"
        "def crash(path, data):\n"
        " if path.name=='hypothesis_1.json': os._exit(17)\n"
        " return original(path,data)\n"
        "study_resume.atomic_write=crash\n"
        "raise SystemExit(main(sys.argv[1:]))\n"
    )
    with (fake_codex / "commit.log").open("w") as log:
        run = subprocess.run(
            [sys.executable, str(script), *command(root, "--llm", "codex")],
            stdout=log,
            stderr=log,
            timeout=10,
        )
        assert run.returncode == 17
        assert not (root / "checkpoints/hypothesis_1.json").exists()
        attempt_file = next((root / "provider_attempts").glob("*.json"))
        evidence = attempt_file.read_bytes()
        assert json.loads(evidence)["status"] == "local_completed"
        resumed = subprocess.run(
            [sys.executable, "-m", "quant_rsi", "research-resume", str(root)],
            stdout=log,
            stderr=log,
            timeout=15,
        )
        assert resumed.returncode == 0
        assert attempt_file.read_bytes() == evidence
        assert len(list((root / "provider_attempts").glob("*.json"))) == 3
