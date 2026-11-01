"""Single-user loopback launcher. Studies remain CLI processes and ordinary directories."""

import fcntl
import json
import os
import re
import secrets
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical
from quant_rsi.research.study_history import history_view
from quant_rsi.research.study_queue import stock_symbol, study_arguments, study_options, watchlist
from quant_rsi.research.study_resume import run_lock
from quant_rsi.research.study_ui import _read_json, page, read_state


class Workbench:
    def __init__(self, workspace):
        self.root = Path(workspace).resolve()
        if self.root.is_relative_to(Path(__file__).resolve().parents[3]):
            raise ValueError("Keep workbench tasks outside the source checkout")
        self.tasks = self.root / "tasks"
        self.tasks.mkdir(parents=True, exist_ok=True)
        self.children = {}
        self.plans_file = self.root / "plans.json"
        self.summary_cache = {}

    def task(self, task_id):
        if not re.fullmatch(r"[0-9a-f]{24}", task_id):
            raise ValueError("Invalid task ID")
        task = self.tasks / task_id
        if not (task / "task.json").is_file():
            raise ValueError("Task not found in this workspace")
        return task

    def item(self, task):
        item = _read_json(task / "task.json", {})
        study = task / "study"
        runtime = _read_json(study / "runtime.json", {})
        process = self.children.get(task.name)
        running = process is not None and process.poll() is None
        for directory in (task, study):
            if directory.exists():
                try:
                    with run_lock(directory):
                        pass
                except ValueError:
                    running = True
        worker = _read_json(task / "worker.json", {})
        status = (
            "running" if running else worker.get("status", runtime.get("status", "interrupted"))
        )
        if not running and item.get("queue_state") in {"queued", "cancelled"}:
            status = item["queue_state"]
        if status in {"running", "starting"} and not running:
            status = "interrupted"
        if not running and runtime.get("status") == "completed":
            status = "completed"
        failure = _read_json(study / "failure.json", {})
        error = worker.get("error") or failure.get("error")
        if status in {"failed", "interrupted"} and (task / "worker.log").exists():
            with (task / "worker.log").open("rb") as stream:
                stream.seek(max(0, (task / "worker.log").stat().st_size - 2048))
                lines = stream.read().decode("utf-8", errors="replace").splitlines()
            # CLI's final diagnostic explains initialization/resume refusal too.
            if lines:
                error = lines[-1][:1000]
        return {k: v for k, v in item.items() if k != "arguments"} | {
            "id": task.name,
            "status": status,
            "stage": runtime.get("current_stage"),
            "error": error if status in {"failed", "interrupted"} else None,
            "resumable": not running
            and status in {"failed", "interrupted"}
            and (study / "checkpoints/identity.json").exists(),
            "reused_operations": runtime.get("reused_operations", 0),
            "observation_reason": _read_json(task / "observation.json", {}).get("reason"),
        }

    def listing(self):
        return sorted(
            (self.item(p.parent) for p in self.tasks.glob("*/task.json")),
            key=lambda item: item.get("order", item["created_at"] * 1_000_000_000),
            reverse=True,
        )

    def _launch(self, task, arguments=None):
        manifest = _read_json(task / "task.json", {})
        if arguments is not None:
            manifest["arguments"] = arguments
        manifest["queue_state"] = "dispatched"
        # Hold the task lock before spawning, and inherit it across exec. A reopened
        # workbench sees the worker even before it creates its study/runtime files.
        fd = os.open(task / ".study.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            atomic_write(task / "task.json", canonical(manifest))
            with (task / "worker.log").open("ab") as log:
                self.children[task.name] = subprocess.Popen(
                    [sys.executable, "-m", "quant_rsi.research.study_queue", str(task)],
                    stdout=log,
                    stderr=log,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    pass_fds=(fd,),
                )
        except OSError as exc:
            atomic_write(task / "worker.json", canonical({"status": "failed", "error": str(exc)}))
        finally:
            # Do not LOCK_UN: the child owns the same open file description.
            os.close(fd)

    def _enqueue(self, symbol, options, *, action="research", plan_id=None, batch_id=None):
        task = self.tasks / secrets.token_hex(12)
        task.mkdir()
        manifest = options | {
            "symbol": symbol,
            "created_at": time.time(),
            "order": time.time_ns(),
            "action": action,
            "queue_state": "queued",
            "plan_id": plan_id,
            "batch_id": batch_id,
            "arguments": study_arguments(task, self.root, symbol, options),
        }
        atomic_write(task / "task.json", canonical(manifest))
        return task

    def start(self, options):
        if not isinstance(options, dict):
            raise ValueError("Unexpected study options")
        symbol = stock_symbol(options.get("symbol"))
        config = study_options({k: v for k, v in options.items() if k != "symbol"})
        task = self._enqueue(symbol, config)
        self.tick()
        return self.item(task)

    def batch(self, symbols, options, *, plan_id=None):
        symbols, options = watchlist(symbols), study_options(options)
        batch_id = secrets.token_hex(12)
        tasks = []
        for symbol in symbols:
            for action in ("observe", "research"):
                task = self._enqueue(
                    symbol, options, action=action, plan_id=plan_id, batch_id=batch_id
                )
                tasks.append(self.item(task))
        return {"batch_id": batch_id, "tasks": tasks}

    def plans(self):
        return _read_json(self.plans_file, [])

    def save_plan(self, options, plan_id=None):
        if not isinstance(options, dict) or set(options) != {
            "name",
            "symbols",
            "options",
            "interval_hours",
            "enabled",
        }:
            raise ValueError("Unexpected template fields")
        name = options["name"]
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("A template needs a name (1 to 80 characters)")
        interval = options["interval_hours"]
        if type(interval) is not int or not 1 <= interval <= 168:
            raise ValueError("Schedule interval must be 1 to 168 hours")
        if type(options["enabled"]) is not bool:
            raise ValueError("enabled must be boolean")
        plan = options | {
            "name": name.strip(),
            "symbols": watchlist(options["symbols"]),
            "options": study_options(options["options"]),
        }
        plans = self.plans()
        previous = next((p for p in plans if p["id"] == plan_id), None)
        if plan_id is not None and previous is None:
            raise ValueError("Template not found")
        plan.update(id=plan_id or secrets.token_hex(12), next_run=time.time() + interval * 3600)
        if previous:
            plans[plans.index(previous)] = plan
        else:
            plans.append(plan)
        atomic_write(self.plans_file, canonical(plans))
        return plan

    def run_plan(self, plan_id):
        plans = self.plans()
        plan = next((p for p in plans if p["id"] == plan_id), None)
        if plan is None:
            raise ValueError("Template not found")
        if any(
            t.get("plan_id") == plan_id and t["status"] in {"queued", "running"}
            for t in self.listing()
        ):
            raise ValueError("This template already has unfinished queued or running tasks")
        batch = self.batch(plan["symbols"], plan["options"], plan_id=plan_id)
        plan.update(
            next_run=time.time() + plan["interval_hours"] * 3600, last_batch=batch["batch_id"]
        )
        atomic_write(self.plans_file, canonical(plans))
        return batch

    def cancel(self, task_id):
        task = self.task(task_id)
        if self.item(task)["status"] != "queued":
            raise ValueError("Only queued tasks can be cancelled")
        manifest = _read_json(task / "task.json", {})
        atomic_write(task / "task.json", canonical(manifest | {"queue_state": "cancelled"}))
        return self.item(task)

    def tick(self):
        # Missed intervals coalesce into one batch, never a catch-up storm.
        items = self.listing()
        for plan in self.plans():
            active = any(
                t.get("plan_id") == plan["id"] and t["status"] in {"queued", "running"}
                for t in items
            )
            if plan["enabled"] and plan["next_run"] <= time.time() and not active:
                self.run_plan(plan["id"])
        items = self.listing()
        if not any(t["status"] == "running" for t in items):
            queued = [t for t in reversed(items) if t["status"] == "queued"]
            if queued:
                self._launch(self.task(queued[0]["id"]))

    def retry(self, task_id):
        task = self.task(task_id)
        if not self.item(task)["resumable"]:
            raise ValueError("Only an interrupted or failed resumable study can continue")
        manifest = _read_json(task / "task.json", {})
        atomic_write(
            task / "task.json",
            canonical(
                manifest
                | {
                    "queue_state": "queued",
                    "arguments": ["research-resume", str(task / "study")],
                }
            ),
        )
        self.tick()
        return self.item(task)

    def overview(self):
        rows = []
        for item in self.listing():
            task = self.task(item["id"])
            row = {
                k: item.get(k)
                for k in (
                    "id",
                    "symbol",
                    "created_at",
                    "status",
                    "action",
                    "llm",
                    "data",
                    "model",
                    "selection_metric",
                    "rounds",
                    "max_fits",
                    "error",
                    "plan_id",
                    "batch_id",
                )
            }
            if item["status"] == "completed" and item.get("action", "research") == "research":
                path = task / "study/result.json"
                if path.exists():
                    # Completed reports are immutable. Avoid rereading large traces every poll.
                    if item["id"] not in self.summary_cache:
                        result = _read_json(path, {})
                        iterations = result.get("iterations", {})
                        selected = iterations.get("selected_round")
                        if selected:
                            prediction = iterations["attempts"][selected - 1]["prediction"]
                            strategy = prediction["strategy"]
                            self.summary_cache[item["id"]] = {
                                "selected_round": selected,
                                "development": prediction["development"]["metrics"],
                                "baseline": prediction["development"]["baselines"][
                                    "unchanged_price"
                                ]["metrics"],
                                "strategy": strategy["strategy"]["metrics"],
                                "buy_and_hold": strategy["buy_and_hold"]["metrics"],
                                "terminal": result["summary"].get("terminal_evaluation"),
                            }
                    row.update(self.summary_cache.get(item["id"], {}))
            if item.get("action") == "observe":
                observation = _read_json(task / "observation.json", {})
                row["observation_reason"] = observation.get("reason")
            rows.append(row)
        return {
            "tasks": rows,
            "histories": {
                data: history_view(self.root / "history" / data) for data in ("daily", "synthetic")
            },
            "notice": "Development metrics reflect adaptive selection, not independent accuracy. "
            "Different stocks, periods and settings are not portfolio returns. "
            "Matured outcomes appear separately; synthetic runs never establish maturity.",
        }


def make_server(workbench, token, port=0):
    prefix = f"/{token}/"

    class Server(HTTPServer):
        scheduler_error = None

        def service_actions(self):
            try:
                workbench.tick()
                self.scheduler_error = None
            except (ValueError, OSError) as exc:
                self.scheduler_error = str(exc)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json"):
            raw = (json.dumps(body) if content_type == "application/json" else body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def route(self):
            if self.headers.get(
                "Host"
            ) != f"127.0.0.1:{self.server.server_port}" or not self.path.startswith(prefix):
                raise ValueError("Use the local workbench URL printed by the command")
            return self.path[len(prefix) :]

        def do_GET(self):
            try:
                route = self.route()
                if route == "":
                    self.respond(
                        200,
                        files("quant_rsi.research").joinpath("study_workbench.html").read_text(),
                        "text/html",
                    )
                elif route == "tasks":
                    self.respond(200, {"tasks": workbench.listing()})
                elif route == "plans":
                    self.respond(
                        200,
                        {
                            "plans": workbench.plans(),
                            "scheduler_error": self.server.scheduler_error,
                        },
                    )
                elif route == "overview":
                    self.respond(200, workbench.overview())
                elif re.fullmatch(r"tasks/[0-9a-f]{24}/(state)?", route):
                    task = workbench.task(route.split("/")[1])
                    state = read_state(task / "study")
                    item = workbench.item(task)
                    state["symbol"] = item["symbol"]
                    if item["status"] in {"failed", "interrupted"}:
                        state["runtime"]["status"] = "failed"
                        state["failure"] = {
                            "error": item["error"] or "Worker stopped. Continue from the workbench."
                        }
                    if route.endswith("/state"):
                        self.respond(200, state)
                    else:
                        self.respond(200, page(state, live=True), "text/html")
                else:
                    self.respond(404, {"error": "Not found"})
            except (ValueError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

        def do_POST(self):
            try:
                route = self.route()
                origin = f"http://127.0.0.1:{self.server.server_port}"
                if (
                    self.headers.get("Origin") != origin
                    or self.headers.get("Content-Type") != "application/json"
                ):
                    raise ValueError("Start studies from this local workbench")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                options = json.loads(self.rfile.read(length))
                if route == "tasks":
                    result = workbench.start(options)
                elif (
                    route == "batches"
                    and isinstance(options, dict)
                    and set(options) == {"symbols", "options"}
                ):
                    result = workbench.batch(options["symbols"], options["options"])
                    workbench.tick()
                elif route == "plans":
                    result = workbench.save_plan(options)
                elif re.fullmatch(r"plans/[0-9a-f]{24}", route):
                    result = workbench.save_plan(options, route.split("/")[1])
                elif re.fullmatch(r"plans/[0-9a-f]{24}/run", route) and options == {}:
                    result = workbench.run_plan(route.split("/")[1])
                    workbench.tick()
                elif re.fullmatch(r"tasks/[0-9a-f]{24}/retry", route) and options == {}:
                    result = workbench.retry(route.split("/")[1])
                elif re.fullmatch(r"tasks/[0-9a-f]{24}/cancel", route) and options == {}:
                    result = workbench.cancel(route.split("/")[1])
                else:
                    raise ValueError("Unknown action")
                self.respond(202, result)
            except (ValueError, OSError) as exc:
                self.respond(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    return Server(("127.0.0.1", port), Handler)


def serve(workspace, *, port=0, open_browser=True):
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    token = secrets.token_urlsafe(24)
    workbench = Workbench(workspace)
    with run_lock(workbench.root), make_server(workbench, token, port) as server:
        url = f"http://127.0.0.1:{server.server_port}/{token}/"
        print(f"Research workbench: {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Workbench stopped. Research workers continue; reopen the same workspace.")
