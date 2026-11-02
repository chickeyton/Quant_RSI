"""Read-only local study viewer. Existing runtime/trace artifacts own all progress."""

import argparse
import json
import secrets
import select
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write
from quant_rsi.research.study_ui_views import visual_output


def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_state(root):
    config = _read_json(root / "config.json", {})
    runtime = _read_json(root / "runtime.json", {"status": "starting", "stages": []})
    events = []
    previous = {}

    def present(stage, output):
        text = json.dumps(output, indent=2, ensure_ascii=False)
        return {
            "stage": stage,
            "text": text[:18000],
            "truncated": len(text) > 18000,
            "view": visual_output(stage, output, previous, config.get("symbol", "…")),
        }

    try:
        lines = (root / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # a writer may still be appending the last line
        events.append(present(event["stage"], event["output"]))
        previous[event["stage"]] = event["output"]
    for stage in runtime.get("stages", []):
        if "output" in stage:
            events.insert(0, present(stage["stage"], stage["output"]))
    result = _read_json(root / "result.json", {}) if runtime["status"] == "completed" else {}
    report = ""
    if result:
        try:
            report = (root / "report.html").read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
    try:
        updated = (root / "runtime.json").stat().st_mtime
    except FileNotFoundError:
        updated = time.time()
    history = None
    if config.get("history_dir"):
        from quant_rsi.research.study_history import history_view

        try:
            history = history_view(Path(config["history_dir"]), config.get("symbol"))
        except (OSError, ValueError) as exc:
            history = {"entries": [], "notice": f"Study history unavailable: {exc}"}
    return {
        "symbol": config.get("symbol", "…"),
        "config": {
            key: config.get(key)
            for key in (
                "llm",
                "model",
                "prices",
                "news",
                "news_research",
                "research_rounds",
                "fit_budget",
                "comparison_fit_budget",
                "publication_fit_budget",
                "walk_forward_fit_budget",
                "question",
            )
        },
        "iterations": {
            key: result.get("iterations", {}).get(key)
            for key in ("completed_fits", "selected_round", "stop_reason")
        },
        "runtime": runtime,
        "updated_at": updated,
        "events": events,
        "failure": _read_json(root / "failure.json", None),
        "conclusion": result.get("study", {}).get("conclusion"),
        "selection": result.get("study", {}).get("selection"),
        "report": report,
        "history": history,
    }


def page(state, *, live=False):
    # No provider text is executable markup, including a literal closing script tag.
    bootstrap = json.dumps({"state": state, "live": live}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    template = files("quant_rsi.research").joinpath("study_ui.html").read_text(encoding="utf-8")
    cards = files("quant_rsi.research").joinpath("study_ui_cards.js").read_text(encoding="utf-8")
    template = template.replace("__STUDY_CARDS__", cards)
    return template.replace("__STUDY_BOOTSTRAP__", bootstrap)


def save_view(root):
    atomic_write(root / "live.html", page(read_state(root)).encode("utf-8"))


def make_server(root, token):
    prefix = f"/{token}/"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {prefix, prefix + "state"}:
                self.send_error(404)
                return
            self.server.last_request = time.monotonic()
            state = read_state(root)
            is_state = self.path.endswith("/state")
            body = json.dumps(state, ensure_ascii=False) if is_state else page(state, live=True)
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8" if is_state else "text/html; charset=utf-8",
            )
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            try:
                self.wfile.write(body.encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 1
    server.last_request = time.monotonic()
    return server


def start_view(root):
    """Start an isolated read-only viewer; never block or fail the research on UI errors."""
    process = None
    try:
        save_view(root)
        with (root / "ui-server.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", __name__, "--serve", str(root)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                start_new_session=True,
            )
        if not select.select([process.stdout], [], [], 5)[0]:
            raise OSError("viewer startup timed out")
        url = process.stdout.readline().strip()
        process.stdout.close()
        if not url.startswith("http://127.0.0.1:"):
            raise OSError("viewer failed to start; see ui-server.log")
        print(f"[quant-rsi] Live UI: {url}", file=sys.stderr, flush=True)
        try:
            if not webbrowser.open(url, new=2):
                print(
                    "[quant-rsi] Open the Live UI URL in a browser on this machine.",
                    file=sys.stderr,
                )
        except webbrowser.Error:
            print(
                "[quant-rsi] Browser unavailable; open the Live UI URL manually.", file=sys.stderr
            )
        return process
    except OSError as exc:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        print(
            f"[quant-rsi] UI unavailable: {exc}; research continues in the terminal.",
            file=sys.stderr,
        )
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", type=Path, required=True)
    args = parser.parse_args()
    token = secrets.token_urlsafe(24)
    with make_server(args.serve.resolve(), token) as server:
        print(f"http://127.0.0.1:{server.server_port}/{token}/", flush=True)
        # The child outlives the research command so its final page can still be read.
        # Closing the tab (or a completed viewer stopping its polling) releases it.
        while time.monotonic() - server.last_request < 120:
            server.handle_request()


if __name__ == "__main__":
    main()
