"""The local UI observes real study artifacts; providers and browser launch stay offline."""

import json
import shutil
import subprocess
import threading
from importlib.resources import files
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study, study_ui
from quant_rsi.research.study_progress import StudyProgress


def test_empty_output_renderer_keeps_initial_live_page_polling():
    # Exercise the actual JS on the empty-output path that runs before polling starts.
    # The minimal DOM only supplies the elements needed by this path; browser QA covers layout.
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the optional renderer smoke test")
    cards = files("quant_rsi.research").joinpath("study_ui_cards.js").read_text()
    script = """
const assert = require('node:assert/strict');
global.document = {createElement: tag => ({tag})};
for (const view of [undefined, null]) {
  let child;
  StudyCards.render({replaceChildren: element => {child = element;}}, view, 'Waiting for input');
  assert.equal(child.tag, 'p');
  assert.equal(child.textContent, 'Waiting for input');
}
"""
    result = subprocess.run(
        [node, "-e", cards + script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


def test_state_keeps_partial_trace_and_escapes_provider_markup(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"symbol": "KO", "llm": "simulated"}))
    progress = StudyProgress(tmp_path, "simulated", quiet=True)
    progress.start("load_prices")
    progress.finish("load_prices", {"provider": "synthetic", "sessions": 120})
    progress.start("initial_hypothesis")
    hostile = '</script><img src=x onerror="alert(1)">'
    event = {"stage": "news", "output": {"headline": hostile}}
    (tmp_path / "trace.jsonl").write_text(json.dumps(event) + '\n{"stage":')
    state = study_ui.read_state(tmp_path)
    assert state["runtime"]["current_stage"] == "initial_hypothesis"
    assert [event["stage"] for event in state["events"]] == ["load_prices", "news"]
    assert state["events"][0]["text"]
    html = study_ui.page(state)
    assert hostile not in html
    assert "\\u003c/script>" in html
    assert "__STUDY_BOOTSTRAP__" not in html
    assert '"live": false' in html


def test_server_only_serves_token_scoped_read_only_endpoints(tmp_path):
    (tmp_path / "config.json").write_text('{"symbol":"KO"}')
    (tmp_path / "secret.txt").write_text("must not be served")
    with study_ui.make_server(tmp_path, "test-token") as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/test-token/state", timeout=3) as response:
                assert json.load(response)["symbol"] == "KO"
                assert response.headers["Cache-Control"] == "no-store"
            with urlopen(base + "/test-token/", timeout=3) as response:
                assert b"THE RESEARCH LOOP" in response.read()
            for path in (
                "/",
                "/config.json",
                "/test-token/../secret.txt",
                "/test-token/secret.txt",
            ):
                with pytest.raises(HTTPError) as exc:
                    urlopen(base + path, timeout=3)
                assert exc.value.code == 404
        finally:
            server.shutdown()
            thread.join(timeout=3)


@pytest.mark.parametrize("fail", [False, True])
def test_ui_command_publishes_live_and_saved_terminal_state(tmp_path, monkeypatch, fail):
    urls, processes = [], []
    monkeypatch.setattr(study_ui.webbrowser, "open", lambda url, new: urls.append(url) or True)
    start = study_ui.start_view

    def launch(root):
        process = start(root)
        assert process is not None
        processes.append(process)
        # The browser endpoint exists before the first model/provider is invoked.
        with urlopen(urls[-1] + "state", timeout=3) as response:
            assert json.load(response)["runtime"]["stages"] == []
        return process

    monkeypatch.setattr(study_ui, "start_view", launch)
    if fail:

        def hypothesis(*args, **kwargs):
            raise ValueError("controlled UI provider failure")

        monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    root = tmp_path / "study"
    try:
        assert main(command(root, "--llm", "simulated", "--news-research", "--ui")) == int(fail)
        with urlopen(urls[0] + "state", timeout=3) as response:
            state = json.load(response)
        assert state["runtime"]["status"] == ("failed" if fail else "completed")
        assert bool(state["report"]) != fail
        if fail:
            assert state["failure"]["error"] == "controlled UI provider failure"
            assert state["runtime"]["stages"][-1]["status"] == "failed"
        else:
            assert state["runtime"]["counts"]["model_stages_completed"] == 2
            assert state["selection"]["selected_attempt"] in {"initial", "revised"}
        saved = (root / "live.html").read_text()
        assert '"live": false' in saved
        assert f'"status": "{state["runtime"]["status"]}"' in saved
    finally:
        for process in processes:
            process.terminate()
            process.wait(timeout=5)


def test_headless_and_no_ui_do_not_start_a_viewer(tmp_path, monkeypatch):
    def unexpected(root):
        pytest.fail("noninteractive or --no-ui invocation launched a browser")

    monkeypatch.setattr(study_ui, "start_view", unexpected)
    assert main(command(tmp_path / "headless", "--llm", "simulated", "--json")) == 0
    assert main(command(tmp_path / "no-ui", "--llm", "simulated", "--no-ui")) == 0


def test_ui_start_failure_does_not_abort_research(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("browser server unavailable")

    monkeypatch.setattr(study_ui.subprocess, "Popen", unavailable)
    root = tmp_path / "study"
    assert main(command(root, "--llm", "simulated", "--ui")) == 0
    assert study_ui.read_state(root)["runtime"]["status"] == "completed"
    assert '"status": "completed"' in (root / "live.html").read_text()


@pytest.mark.parametrize(
    "tty,flags,enabled",
    [
        (True, [], True),
        (True, ["--json"], False),
        (False, ["--ui"], True),
        (True, ["--no-ui"], False),
    ],
)
def test_cli_ui_defaults_and_overrides(tmp_path, monkeypatch, tty, flags, enabled):
    observed = []

    def run(args):
        observed.append(args.ui)
        return {"study": {"conclusion": "fixture"}}

    monkeypatch.setattr(study, "run_study", run)
    monkeypatch.setattr("sys.stderr.isatty", lambda: tty)
    assert main(command(tmp_path / "study", *flags)) == 0
    assert observed == [enabled]


def test_interrupted_study_saves_failed_view(tmp_path, monkeypatch):
    monkeypatch.setattr(study_ui, "start_view", lambda root: None)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(study, "simulated_hypothesis", interrupted)
    root = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        main(command(root, "--llm", "simulated", "--ui"))
    state = study_ui.read_state(root)
    assert state["runtime"]["status"] == "failed"
    assert state["failure"]["error_type"] == "KeyboardInterrupt"
    assert '"live": false' in (root / "live.html").read_text()
