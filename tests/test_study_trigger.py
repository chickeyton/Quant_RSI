"""No paid calls on unchanged evidence; failed attempts never consume a trigger."""

import json
from copy import deepcopy

import pytest
from test_study_financials import snapshot
from test_study_queue import OFFLINE, finish_queue

from quant_rsi import cli
from quant_rsi.core.artifacts import canonical
from quant_rsi.research import study_trigger as trigger
from quant_rsi.research.study_queue import run_task, study_options
from quant_rsi.research.study_resume import run_lock
from quant_rsi.research.study_workbench import Workbench


def test_real_workers_skip_repeat_and_force_manual_cycle(tmp_path):
    workbench = Workbench(tmp_path)
    options = OFFLINE | {"only_new_evidence": True, "financials": True}
    workbench.batch(["KO", "AAPL"], options)
    finish_queue(workbench)
    second = workbench.batch(["KO", "AAPL"], options)
    finish_queue(workbench)
    research = [
        workbench.item(workbench.task(t["id"]))
        for t in second["tasks"]
        if t["action"] == "research"
    ]
    assert [r["status"] for r in research] == ["skipped", "skipped"]
    for row in research:
        assert row["reason"] == "unchanged_evidence"
        assert not (workbench.task(row["id"]) / "study").exists()
    forced = workbench.batch(["KO"], options, force=True)
    finish_queue(workbench)
    row = workbench.item(workbench.task(forced["tasks"][-1]["id"]))
    assert row["status"] == "completed"
    assert (workbench.task(row["id"]) / "study/financial_comparison.json").exists()


def test_requested_history_survives_queue_preparation_and_changes_trigger(
    evidence, tmp_path, monkeypatch
):
    calls = []
    original = trigger.collection_snapshot

    def collect(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(trigger, "collection_snapshot", collect)
    workbench = Workbench(tmp_path)
    fingerprints = []
    for sessions in (252, 445):
        options = OFFLINE | {"data": "daily", "historical_sessions": sessions}
        batch = workbench.batch(["KO"], options)
        task = workbench.task(batch["tasks"][-1]["id"])
        manifest = json.loads((task / "task.json").read_bytes())
        receipt, args = trigger.refresh_evidence(task, manifest)
        assert args[args.index("--historical-sessions") + 1] == str(sessions)
        assert Workbench(tmp_path).item(task)["historical_sessions"] == sessions
        assert any(r["historical_sessions"] == sessions for r in workbench.overview()["tasks"])
        fingerprints.append(receipt["fingerprint"])
    assert calls == [
        {"horizon_sessions": 5, "historical_sessions": 252},
        {"horizon_sessions": 5, "historical_sessions": 445},
    ]
    assert fingerprints[0] != fingerprints[1]


@pytest.mark.parametrize(
    "failure", [None, "collecting_prices", "collecting_news", "collecting_financials"]
)
def test_preparation_progress_survives_reopen_and_failure_then_clears_on_handoff(
    evidence, tmp_path, monkeypatch, failure
):
    workbench = Workbench(tmp_path)
    options = study_options(
        OFFLINE | {"data": "daily", "financials": True, "only_new_evidence": True}
    )
    stages, cli_calls = [], []
    task = None

    for function, stage in (
        ("collection_snapshot", "collecting_prices"),
        ("issuer_news", "collecting_news"),
        ("collect_financials", "collecting_financials"),
        ("mature_feedback", "checking_new_evidence"),
    ):
        original = getattr(trigger, function)

        def collect(*args, _original=original, _stage=stage, **kwargs):
            item = Workbench(tmp_path).item(task)
            assert item["status"] == "running" and item["stage"] == _stage
            assert not (task / "study/runtime.json").exists()
            stages.append(_stage)
            if failure == _stage:
                raise ValueError("Source unavailable")
            return _original(*args, **kwargs)

        monkeypatch.setattr(trigger, function, collect)

    def research(arguments):
        assert Workbench(tmp_path).item(task)["stage"] is None
        cli_calls.append(arguments)
        return 0

    monkeypatch.setattr(cli, "main", research)
    for cycle in range(1 if failure else 2):
        task = workbench._enqueue("KO", options)
        manifest = json.loads((task / "task.json").read_bytes()) | {"queue_state": "dispatched"}
        (task / "task.json").write_bytes(canonical(manifest))
        with run_lock(task):
            assert run_task(task) == (1 if failure else 0)
        item = Workbench(tmp_path).item(task)
        if failure:
            assert item["status"] == "failed" and item["stage"] == failure
            assert item["error"] == "Source unavailable" and not item["resumable"]
            assert not cli_calls and not (task / "trigger.json").exists()
        else:
            assert item["stage"] is None
            assert item["status"] == ("completed" if cycle == 0 else "skipped")
            assert len(cli_calls) == 1
    if not failure:
        assert (
            stages
            == [
                "collecting_prices",
                "collecting_news",
                "collecting_financials",
                "checking_new_evidence",
            ]
            * 2
        )


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    state = {
        "prices": {"dates": ["2024-04-29"], "opens": [10], "closes": [11]},
        "news": {
            "items": [
                {
                    "symbol": "KO",
                    "title": "Demand",
                    "summary": "Grows",
                    "url": "https://example.test/news",
                    "received_at": "first",
                }
            ]
        },
        "financials": snapshot(),
        "entries": [],
    }

    def collect(symbol, directory, **kwargs):
        directory.mkdir(parents=True)
        return deepcopy(state["prices"])

    monkeypatch.setattr(trigger, "collection_snapshot", collect)
    monkeypatch.setattr(trigger, "observe_prices", lambda *a: None)
    monkeypatch.setattr(trigger, "issuer_news", lambda *a: deepcopy(state["news"]))
    monkeypatch.setattr(trigger, "profile_related", lambda *a: {})
    monkeypatch.setattr(trigger, "collect_financials", lambda *a: deepcopy(state["financials"]))
    monkeypatch.setattr(trigger, "mature_feedback", lambda *a, **kw: {"entries": state["entries"]})
    workbench = Workbench(tmp_path)
    options = OFFLINE | {"data": "daily", "financials": True, "only_new_evidence": True}

    def check():
        batch = workbench.batch(["KO"], options)
        task = workbench.task(batch["tasks"][-1]["id"])
        manifest = json.loads((task / "task.json").read_bytes())
        receipt, args = trigger.refresh_evidence(task, manifest)
        assert "--prices-dir" in args and "--financials-dir" in args
        assert args[args.index("--news-snapshot") + 1] == str(task / "preflight/news.json")
        assert json.loads((task / "preflight/news.json").read_bytes()) == state["news"]
        return task, receipt

    return state, check


def test_receipt_changes_do_not_trigger_and_failed_prior_is_not_consumed(evidence):
    state, check = evidence
    task, first = check()
    (task / "trigger.json").write_bytes(canonical(first))
    (task / "worker.json").write_text('{"status":"failed"}')
    assert check()[1]["previous_completed_task"] is None
    (task / "worker.json").write_text('{"status":"completed"}')
    state["news"]["items"][0]["received_at"] = "second"
    state["news"]["items"][0]["id"] = "new-receipt-id"
    assert check()[1]["previous_completed_task"] == task.name


@pytest.mark.parametrize("changed", ["prices", "news", "financials", "maturity"])
def test_each_new_evidence_source_changes_fingerprint(evidence, changed):
    state, check = evidence
    _, first = check()
    if changed == "prices":
        state["prices"]["closes"][0] = 12
    elif changed == "news":
        state["news"]["items"][0]["title"] = "Demand reverses"
    elif changed == "financials":
        state["financials"]["companyfacts"]["facts"]["us-gaap"]["Revenues"]["units"]["USD"][-1][
            "val"
        ] = 130
    else:
        state["entries"] = [{"study_id": "old", "status": "settled", "metrics": {"mae": 1}}]
    assert check()[1]["fingerprint"] != first["fingerprint"]


def test_failed_refresh_does_not_become_unchanged_evidence(evidence, monkeypatch):
    _, check = evidence

    def unavailable(*args):
        raise ValueError("RSS unavailable")

    monkeypatch.setattr(trigger, "issuer_news", unavailable)
    with pytest.raises(ValueError, match="RSS unavailable"):
        check()


def test_diagnostic_preflight_collects_extra_history_before_worker(evidence, tmp_path, monkeypatch):
    original, calls = trigger.collection_snapshot, []

    def collect(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(trigger, "collection_snapshot", collect)
    workbench = Workbench(tmp_path)
    receipts = []
    for folds in (0, 3):
        options = OFFLINE | {"data": "daily", "horizon_sessions": 60, "walk_forward_folds": folds}
        batch = workbench.batch(["KO"], options)
        task = workbench.task(batch["tasks"][-1]["id"])
        manifest = json.loads((task / "task.json").read_bytes())
        receipt, args = trigger.refresh_evidence(task, manifest)
        receipts.append(receipt)
        assert args[args.index("--walk-forward-folds") + 1] == str(folds)
    assert calls == [
        {"horizon_sessions": 60, "historical_sessions": 285},
        {"horizon_sessions": 60, "historical_sessions": 445},
    ]
    assert receipts[0]["fingerprint"] != receipts[1]["fingerprint"]
