"""Serial watchlists reuse the study and observation paths, including failures."""

import json
from copy import deepcopy

import pytest
from test_study_history import prices, result_for

from quant_rsi import cli
from quant_rsi.core.artifacts import atomic_write, canonical
from quant_rsi.research import study, study_history, study_queue, study_workbench
from quant_rsi.research.study_queue import run_task, study_options
from quant_rsi.research.study_workbench import Workbench

OFFLINE = {"llm": "simulated", "data": "synthetic", "model": "ridge", "news_research": False}


@pytest.mark.parametrize("check_inputs", [False, True])
@pytest.mark.parametrize("only_new_evidence", [False, True])
def test_all_daily_batch_entry_points_receive_the_same_external_cache(
    tmp_path, check_inputs, only_new_evidence
):
    workbench = Workbench(tmp_path)
    batch = workbench.batch(
        ["XOM", "KO"],
        OFFLINE
        | {"data": "daily", "external_drivers": True, "only_new_evidence": only_new_evidence},
        check_inputs=check_inputs,
        force=True,
    )
    paths = []
    for row in batch["tasks"]:
        if row["action"] == "observe":
            continue
        manifest = json.loads((workbench.task(row["id"]) / "task.json").read_bytes())
        args = manifest["arguments"]
        paths.append(args[args.index("--external-drivers-cache-dir") + 1])
        assert ("--check-inputs" in args) is check_inputs
    assert paths == [str(tmp_path / "batches" / batch["batch_id"] / "external_drivers")] * 2


def template(**changes):
    return {
        "name": "Two stocks",
        "symbols": ["KO", "AAPL"],
        "options": OFFLINE,
        "interval_hours": 24,
        "enabled": False,
    } | changes


def finish_queue(workbench):
    while any(t["status"] == "queued" for t in workbench.listing()):
        workbench.tick()
        for process in workbench.children.values():
            process.wait(timeout=30)


def test_watchlist_real_workers_keep_failed_stock_and_continue_with_frozen_limits(tmp_path):
    workbench = Workbench(tmp_path)
    plan = workbench.save_plan(
        template(
            symbols=["UNKNOWN", "AAPL"],
            options=OFFLINE
            | {
                "rounds": 3,
                "max_fits": 2,
                "selection_metric": "sharpe",
            },
        )
    )
    batch = workbench.run_plan(plan["id"])
    broken_task = workbench.task(batch["tasks"][1]["id"])
    manifest = json.loads((broken_task / "task.json").read_bytes())
    manifest["arguments"] += ["--catalog", str(tmp_path / "missing-factor-catalog.json")]
    atomic_write(broken_task / "task.json", canonical(manifest))
    # Editing a saved template never changes already queued task inputs.
    workbench.save_plan(template(symbols=["KO"], options=OFFLINE | {"rounds": 4}), plan["id"])
    finish_queue(workbench)
    rows = sorted(workbench.listing(), key=lambda row: row["order"])
    assert [row["status"] for row in rows] == ["completed", "failed", "completed", "completed"]
    assert all(row["batch_id"] == batch["batch_id"] for row in rows)
    assert all(row["rounds"] == 3 and row["max_fits"] == 2 for row in rows)
    failed, completed = [row for row in rows if row["action"] == "research"]
    assert failed["error"]
    result = json.loads((workbench.task(completed["id"]) / "study/result.json").read_bytes())
    assert len(result["iterations"]["attempts"]) == 2
    assert result["configuration"]["selection_metric"] == "sharpe"
    overview = Workbench(tmp_path).overview()
    assert len(overview["tasks"]) == 4
    assert not overview["histories"]["daily"]["entries"]
    assert all(e["status"] != "settled" for e in overview["histories"]["synthetic"]["entries"])


def test_schedule_coalesces_missed_cycles_restart_pause_and_cancel(tmp_path, monkeypatch):
    workbench = Workbench(tmp_path)
    now = study_workbench.time.time()
    monkeypatch.setattr(study_workbench.time, "time", lambda: now)
    plan = workbench.save_plan(template(enabled=True))
    monkeypatch.setattr(Workbench, "_launch", lambda *args: None)
    workbench.tick()
    assert workbench.listing() == []
    now += 10 * 24 * 3600
    restarted = Workbench(tmp_path)
    restarted.tick()
    assert len(restarted.listing()) == 4
    assert restarted.plans()[0]["next_run"] == now + 24 * 3600
    now += 2 * 24 * 3600
    Workbench(tmp_path).tick()
    assert len(restarted.listing()) == 4  # No overlapping batches for the same template.
    with pytest.raises(ValueError, match="unfinished"):
        restarted.run_plan(plan["id"])
    for task in restarted.listing():
        assert restarted.cancel(task["id"])["status"] == "cancelled"
    restarted.save_plan(template(enabled=False), plan["id"])
    now += 10 * 24 * 3600
    restarted.tick()
    assert len(restarted.listing()) == 4
    restarted.run_plan(plan["id"])  # Paused schedules still permit an explicit run.
    assert len(restarted.listing()) == 8


def test_launch_failure_advances_and_queued_cancellation_is_durable(tmp_path, monkeypatch):
    workbench = Workbench(tmp_path)
    batch = workbench.batch(["KO"], OFFLINE)

    def unavailable(*args, **kwargs):
        raise OSError("Cannot start interpreter")

    monkeypatch.setattr(study_workbench.subprocess, "Popen", unavailable)
    workbench.tick()
    first, second = [workbench.task(t["id"]) for t in batch["tasks"]]
    assert workbench.item(first)["status"] == "failed"
    assert "Cannot start" in workbench.item(first)["error"]
    with pytest.raises(ValueError, match="Only queued"):
        workbench.cancel(first.name)
    workbench.cancel(second.name)
    assert Workbench(tmp_path).item(second)["status"] == "cancelled"


@pytest.mark.parametrize(
    "change",
    [
        {"rounds": True},
        {"max_fits": 11},
        {"patience": 0},
        {"selection_metric": []},
        {"news_research": "yes"},
        {"llm": "api"},
        {"output": "/tmp/arbitrary"},
    ],
)
def test_invalid_settings_do_not_create_tasks(tmp_path, change):
    workbench = Workbench(tmp_path)
    with pytest.raises(ValueError):
        workbench.batch(["KO"], OFFLINE | change)
    assert workbench.listing() == []


@pytest.mark.parametrize(
    "change",
    [
        {"symbols": ["KO", "ko"]},
        {"symbols": []},
        {"symbols": ["../KO"]},
        {"symbols": ["KO", None]},
        {"interval_hours": True},
        {"enabled": "true"},
        {"name": " "},
        {"options": {"rounds": 20}},
    ],
)
def test_invalid_templates_are_atomic(tmp_path, change):
    workbench = Workbench(tmp_path)
    with pytest.raises(ValueError):
        workbench.save_plan(template(**change))
    assert workbench.plans() == [] and workbench.listing() == []


def test_independent_observation_survives_llm_failure_and_enters_next_research(
    tmp_path, monkeypatch
):
    """Controlled real-adapter receipts + real Ridge fits, not elapsed market evidence."""
    workspace = tmp_path / "workbench"
    workbench = Workbench(workspace)
    history = workspace / "history/daily"
    _, initial = prices(tmp_path, monkeypatch, "2026-10-23")
    publication = tmp_path / "published"
    publication.mkdir()
    study_history.register_study(history, publication, result_for(initial), initial)
    assert study_history.history_view(history)["entries"][0]["status"] == "pending"
    prices(tmp_path, monkeypatch, "2026-10-30")
    # The queue's observer collects a new receipt in its own directory.
    batch = workbench.batch(["KO"], OFFLINE | {"data": "daily"})
    observe, research = [workbench.task(t["id"]) for t in batch["tasks"]]
    assert run_task(observe) == 0
    observation = json.loads((observe / "observation.json").read_bytes())
    assert observation["reason"] == "fresh_public_prices"
    assert observation["entries"][0]["status"] == "settled"
    assert (observe / "prices/collection_receipt.json").is_file()
    retained = (observe / "observation.json").read_bytes()
    original_main = cli.main

    def unavailable(*args):
        raise ValueError("LLM unavailable")

    monkeypatch.setattr(cli, "main", unavailable)
    assert run_task(research) == 1
    assert (observe / "observation.json").read_bytes() == retained
    assert study_history.history_view(history)["entries"][0]["status"] == "settled"

    # Another cycle uses the ordinary study path and injects the settled outcome.
    next_task = workbench._enqueue("KO", study_options(OFFLINE | {"data": "daily"}))
    manifest = json.loads((next_task / "task.json").read_bytes())
    manifest["arguments"] += [
        "--news",
        "simulated",
        "--related",
        "simulated",
        "--no-related-prices",
    ]
    atomic_write(next_task / "task.json", canonical(manifest))
    monkeypatch.setattr(cli, "main", original_main)
    contexts = []
    original_hypothesis = study.simulated_hypothesis

    def capture(context, *args, **kwargs):
        contexts.append(deepcopy(context))
        return original_hypothesis(context, *args, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", capture)
    assert run_task(next_task) == 0
    assert contexts[0]["prior_outcomes"]["entries"][0]["status"] == "settled"
    assert (observe / "observation.json").read_bytes() == retained


def test_observation_failure_is_recorded_without_fake_outcomes(tmp_path, monkeypatch):
    workbench = Workbench(tmp_path)
    task = workbench._enqueue("KO", study_options(OFFLINE | {"data": "daily"}), action="observe")
    monkeypatch.setattr(
        study_queue, "history_view", lambda *args: {"entries": [{"status": "pending"}]}
    )

    def fail(*args):
        raise ValueError("Price source unavailable")

    monkeypatch.setattr(study_queue, "collection_snapshot", fail)
    assert run_task(task) == 1
    assert not (task / "observation.json").exists()
    assert json.loads((task / "worker.json").read_bytes())["error"] == "Price source unavailable"
