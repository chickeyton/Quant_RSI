"""No paid calls on unchanged evidence; failed attempts never consume a trigger."""

import json
from copy import deepcopy

import pytest
from test_study_financials import snapshot
from test_study_queue import OFFLINE, finish_queue

from quant_rsi.core.artifacts import canonical
from quant_rsi.research import study_trigger as trigger
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
