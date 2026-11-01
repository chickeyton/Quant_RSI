"""Workbench queue inputs and a small worker around the existing study CLI."""

import json
import re
import sys
import time
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical
from quant_rsi.research.study_history import collection_snapshot, history_view, observe_prices
from quant_rsi.research.study_iterations import validate_limits


def stock_symbol(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Z][A-Z0-9.-]{0,14}", value.strip().upper()
    ):
        raise ValueError("Enter a stock symbol, for example KO or AAPL")
    return value.strip().upper()


def study_options(options):
    defaults = {
        "llm": "codex",
        "model": "lightgbm",
        "data": "daily",
        "llm_model": "",
        "news_research": True,
        "financials": False,
        "financials_source": "auto",
        "only_new_evidence": False,
        "rounds": 2,
        "max_fits": None,
        "patience": None,
        "selection_metric": "mae",
    }
    if not isinstance(options, dict) or set(options) - defaults.keys():
        raise ValueError("Unexpected study options")
    result = defaults | options
    for key, allowed in (
        ("llm", {"codex", "api", "simulated"}),
        ("model", {"lightgbm", "ridge"}),
        ("financials_source", {"auto", "sec", "issuer"}),
        ("data", {"daily", "synthetic"}),
        ("selection_metric", {"mae", "sharpe"}),
    ):
        if not isinstance(result[key], str) or result[key] not in allowed:
            raise ValueError("Unsupported provider, model, data mode or selection metric")
    model = result["llm_model"]
    if (
        not isinstance(model, str)
        or len(model) > 120
        or (result["llm"] == "api" and not model.strip())
    ):
        raise ValueError("API mode requires a model name; credentials come from the environment")
    for key in ("news_research", "financials", "only_new_evidence"):
        if type(result[key]) is not bool:
            raise ValueError(key + " must be boolean")
    if result["max_fits"] is None:
        result["max_fits"] = result["rounds"]
    for key in ("rounds", "max_fits", "patience"):
        if result[key] is not None and type(result[key]) is not int:
            raise ValueError("Research limits must be integers")
    validate_limits(result["rounds"], result["max_fits"], result["patience"])
    return result


def watchlist(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 20:
        raise ValueError("A watchlist needs 1 to 20 unique stock symbols")
    symbols = [stock_symbol(item) for item in value]
    if len(set(symbols)) != len(symbols):
        raise ValueError("Watchlist symbols must be unique")
    return symbols


def study_arguments(task, workspace, symbol, options):
    arguments = [
        "research-study",
        symbol,
        "--output",
        str(task / "study"),
        "--no-ui",
        "--factor-root",
        str(Path(__file__).resolve().parents[3] / "instruments"),
        "--history-dir",
        str(workspace / "history" / options["data"]),
    ]
    for key, flag in (
        ("llm", "llm"),
        ("model", "model"),
        ("data", "prices"),
        ("rounds", "rounds"),
        ("max_fits", "max-fits"),
        ("patience", "patience"),
        ("selection_metric", "selection-metric"),
    ):
        if options[key] is not None:
            arguments += ["--" + flag, str(options[key])]
    if options["data"] == "synthetic":
        arguments += ["--news", "simulated", "--related", "simulated"]
    if options["llm_model"]:
        arguments += ["--llm-model=" + options["llm_model"]]
    if options["news_research"]:
        arguments += ["--news-research"]
    if options["financials"]:
        arguments += ["--financials", "--financials-source", options["financials_source"]]
    return arguments


def observe_task(task, manifest):
    """Collect fresh evidence independently of news/LLM availability; never invent maturity."""
    history = task.parent.parent / "history" / manifest["data"]
    view = history_view(history, manifest["symbol"])
    actionable = any(e["status"] in {"pending", "insufficient_evidence"} for e in view["entries"])
    if manifest["data"] == "daily" and actionable:
        snapshot = collection_snapshot(manifest["symbol"], task / "prices")
        view = observe_prices(history, manifest["symbol"], snapshot)
        reason = "fresh_public_prices"
    else:
        reason = (
            "synthetic_not_observed" if manifest["data"] == "synthetic" else "no_pending_forecasts"
        )
    atomic_write(task / "observation.json", canonical(view | {"reason": reason}))


def run_task(task):
    """The launcher passes an inherited lock fd held for this entire process lifetime."""
    from quant_rsi.cli import main

    task = Path(task)
    manifest = json.loads((task / "task.json").read_bytes())
    state = {"status": "running", "started_at": time.time()}
    atomic_write(task / "worker.json", canonical(state))
    try:
        if manifest.get("action") == "observe":
            observe_task(task, manifest)
            code = 0
        else:
            arguments = manifest["arguments"]
            skip = False
            if (
                manifest.get("only_new_evidence")
                and manifest.get("action", "research") == "research"
                and arguments[0] == "research-study"
            ):
                from quant_rsi.research.study_trigger import refresh_evidence

                trigger, arguments = refresh_evidence(task, manifest)
                atomic_write(task / "trigger.json", canonical(trigger))
                skip = bool(trigger["previous_completed_task"]) and not manifest.get("force", False)
            if skip:
                state.update(status="skipped", reason="unchanged_evidence", exit_code=0)
                atomic_write(task / "worker.json", canonical(state | {"finished_at": time.time()}))
                return 0
            code = main(arguments)
        state.update(status="completed" if code == 0 else "failed", exit_code=code)
    except Exception as exc:
        state.update(status="failed", error=str(exc), exit_code=1)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
    atomic_write(task / "worker.json", canonical(state | {"finished_at": time.time()}))
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(run_task(sys.argv[1]))
