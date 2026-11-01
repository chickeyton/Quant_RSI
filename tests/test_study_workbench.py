"""Local HTTP controls launch the same CLI and reuse its recoverable artifacts."""

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study
from quant_rsi.research.study_workbench import Workbench, make_server


def test_launcher_real_offline_process_history_and_durable_serial_queue(tmp_path):
    workbench = Workbench(tmp_path)
    item = workbench.start({"symbol": "AAPL", "llm": "simulated", "data": "synthetic"})
    queued = workbench.start({"symbol": "KO", "llm": "simulated", "data": "synthetic"})
    assert queued["status"] == "queued"
    # Reopening while the child starts must not dispatch another worker.
    reopened = Workbench(tmp_path)
    reopened.tick()
    assert reopened.item(reopened.task(item["id"]))["status"] == "running"
    assert not reopened.children
    process = workbench.children[item["id"]]
    assert process.wait(timeout=30) == 0
    restarted = Workbench(tmp_path)
    saved = restarted.item(restarted.task(item["id"]))
    assert saved["status"] == "completed" and saved["symbol"] == "AAPL"
    with pytest.raises(ValueError, match="Only an interrupted"):
        restarted.retry(item["id"])
    assert (restarted.task(item["id"]) / "study/report.html").is_file()
    restarted.tick()
    assert restarted.children[queued["id"]].wait(timeout=30) == 0
    overview = restarted.overview()
    assert len(overview["tasks"]) == 2
    assert all(t["development"]["sample_count"] == 20 for t in overview["tasks"])
    assert all(t["terminal"]["status"] == "completed" for t in overview["tasks"])
    # A worker dying after publication must not make an immutable completed study resumable.
    (restarted.task(item["id"]) / "worker.json").write_text('{"status":"running"}')
    assert restarted.item(restarted.task(item["id"]))["status"] == "completed"
    assert not restarted.item(restarted.task(item["id"]))["resumable"]


def test_retry_in_new_process_reuses_completed_fit(tmp_path, monkeypatch):
    workbench = Workbench(tmp_path)
    task = workbench.tasks / ("a" * 24)
    task.mkdir()
    (task / "task.json").write_text(json.dumps({"symbol": "KO", "created_at": 0}))
    original = study.simulated_hypothesis

    def fail_revision(*args, **kwargs):
        if kwargs.get("previous"):
            raise ValueError("Provider temporarily unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", fail_revision)
    assert main(command(task / "study", "--llm", "simulated")) == 1
    checkpoint = task / "study/checkpoints/fit_1.json"
    before = checkpoint.stat().st_mtime_ns
    item = Workbench(tmp_path).listing()[0]
    assert item["status"] == "failed" and item["resumable"]
    workbench.retry(item["id"])
    with pytest.raises(ValueError):
        workbench.retry(item["id"])
    assert workbench.children[item["id"]].wait(timeout=30) == 0
    assert checkpoint.stat().st_mtime_ns == before
    assert Workbench(tmp_path).listing()[0]["status"] == "completed"


def test_http_origin_token_inputs_and_history(tmp_path):
    workbench = Workbench(tmp_path)
    with make_server(workbench, "test-token") as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        base = origin + "/test-token/"
        try:
            with urlopen(base) as response:
                assert b"Start a study" in response.read()
            with urlopen(base + "tasks") as response:
                assert json.load(response) == {"tasks": []}
            for path, headers, body in (
                ("tasks", {"Origin": "https://other.example"}, {"symbol": "KO"}),
                ("tasks", {}, {"symbol": "KO"}),
                ("tasks", {"Origin": origin}, {"symbol": "../../escape"}),
                ("tasks", {"Origin": origin}, {"symbol": "KO", "llm": []}),
                ("tasks", {"Origin": origin}, {"symbol": "KO", "output": "/tmp/arbitrary"}),
                ("batches", {"Origin": origin}, {"symbols": ["KO", "ko"], "options": {}}),
                ("plans", {"Origin": origin}, {"enabled": True}),
                ("plans", {"Origin": "https://other.example"}, {}),
                ("plans/../../run", {"Origin": origin}, {}),
                ("tasks/../../cancel", {"Origin": origin}, {}),
                ("tasks/../../retry", {"Origin": origin}, {}),
            ):
                request = Request(
                    base + path,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json", **headers},
                )
                with pytest.raises(HTTPError) as error:
                    urlopen(request)
                assert error.value.code == 400
            with pytest.raises(HTTPError):
                urlopen(origin + "/wrong-token/tasks")
            assert workbench.listing() == []
            # Initialization can fail before the CLI creates its study directory.
            task = workbench.tasks / ("c" * 24)
            task.mkdir()
            (task / "worker.json").write_text(
                json.dumps({"status": "failed", "error": "Interpreter unavailable"})
            )
            (task / "task.json").write_text(
                json.dumps({"symbol": "KO", "created_at": 0, "queue_state": "dispatched"})
            )
            with urlopen(base + f"tasks/{task.name}/state") as response:
                state = json.load(response)
            assert state["runtime"]["status"] == "failed"
            assert state["symbol"] == "KO"
            assert state["failure"]["error"] == "Interpreter unavailable"
        finally:
            server.shutdown()
            thread.join(timeout=5)
