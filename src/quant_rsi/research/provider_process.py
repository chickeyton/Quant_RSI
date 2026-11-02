"""Linux local-provider guardian: own the operation until its process tree is gone."""

import ctypes
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical

STUDY_ROOT = ContextVar("provider_study_root", default=None)


def provider_lock(root):
    fd = os.open(Path(root) / ".provider.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise ValueError(
            "A previous local provider operation is still owned; retry after cleanup"
        ) from None
    return fd


def run_provider(command, *, input, text, cwd, timeout, stdout, stderr, check=False):
    root = Path(cwd)
    study = STUDY_ROOT.get()
    owner = provider_lock(study) if study else None
    read_fd = write_fd = None
    process = None
    try:
        read_fd, write_fd = os.pipe()
        attempt = (
            (study / "provider_attempts" / (uuid.uuid4().hex + ".json"))
            if study
            else root / "attempt.json"
        )
        attempt.parent.mkdir(exist_ok=True)
        request = root / "provider_request.json"
        request.write_text(
            json.dumps({"command": command, "timeout": timeout, "attempt": str(attempt)})
        )
        (root / "provider_input.txt").write_text(input)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "quant_rsi.research.provider_process",
                str(request),
                str(read_fd),
            ],
            pass_fds=(read_fd,) + ((owner,) if owner is not None else ()),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        os.close(read_fd)
        read_fd = None
        if owner is not None:
            os.close(owner)  # Close only: the guardian retains ownership across owner death.
            owner = None
        process.wait(timeout=timeout + 10)
    finally:
        if write_fd is not None:
            os.close(write_fd)  # EOF also arrives automatically on owner SIGKILL.
        if read_fd is not None:
            os.close(read_fd)
        if owner is not None:
            os.close(owner)
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # A surviving guardian keeps the operation lock; never redispatch over it.
    state = json.loads(attempt.read_text())
    if state["status"] == "timeout":
        raise subprocess.TimeoutExpired(command, timeout)
    if state["status"] == "not_found":
        raise FileNotFoundError(command[0])
    if state["status"] != "local_completed":
        raise ValueError("Local provider ended without a response; remote completion is unknown")
    return subprocess.CompletedProcess(command, state["returncode"])


def _children():
    return [int(x) for x in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()]


def _signal(pid, sig, *, group=False):
    try:
        (os.killpg if group else os.kill)(pid, sig)
    except ProcessLookupError:
        pass


def _cleanup(process):
    # Subreaper adoption includes descendants that detached from the provider's group.
    _signal(process.pid, signal.SIGTERM, group=True)
    deadline = time.monotonic() + 0.25
    while True:
        process.poll()
        for pid in _children():
            _signal(pid, signal.SIGTERM if time.monotonic() < deadline else signal.SIGKILL)
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
        if not _children():
            return
        if time.monotonic() >= deadline:
            _signal(process.pid, signal.SIGKILL, group=True)
        time.sleep(0.02)


def guard(request, owner_pipe):
    request = Path(request)
    config = json.loads(request.read_text())
    attempt = Path(config["attempt"])
    state = {"status": "started", "remote_completion": "unknown", "started_at": time.time()}
    atomic_write(attempt, canonical(state))
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError("Cannot establish provider descendant ownership")
    process = None
    try:
        with (request.parent / "provider_input.txt").open() as prompt:
            process = subprocess.Popen(
                config["command"], stdin=prompt, cwd=request.parent, start_new_session=True
            )
        deadline = time.monotonic() + config["timeout"]
        while process.poll() is None:
            if select.select([owner_pipe], [], [], 0.02)[0] and os.read(owner_pipe, 1) == b"":
                state["status"] = "owner_lost"
                break
            if time.monotonic() >= deadline:
                state["status"] = "timeout"
                break
        else:
            state.update(status="local_completed", returncode=process.returncode)
    except FileNotFoundError:
        state["status"] = "not_found"
    finally:
        if process is not None:
            _cleanup(process)
        state.update(finished_at=time.time(), local_cleanup="completed")
        atomic_write(attempt, canonical(state))


if __name__ == "__main__":
    guard(sys.argv[1], int(sys.argv[2]))
