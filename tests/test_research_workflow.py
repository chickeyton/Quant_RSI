"""Single-profile integration; invented evidence does not establish research quality."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import test_forecast_comparison
import test_forecast_features
from market_helpers import account_rules, contract_and_events, write_events

from quant_rsi import cli
from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.factors import discovery
from quant_rsi.factors.discovery_graph import grammar, propose
from quant_rsi.forecast import forecast_settlement
from quant_rsi.market import market_contracts, market_data
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.research import workflow

ROOT = Path(__file__).resolve().parents[1]
prepare_inputs = test_forecast_features.prepare_inputs
dummy_worker = test_forecast_comparison.dummy_worker


def profile(envelope, campaign_id="workflow-test"):
    envelope, leaves = discovery.bind_catalog(
        envelope, ROOT / "instruments", "US", "KO", {"qlib_roc": "target", "qlib_ma": "target"}
    )
    return dict(
        schema_version=1,
        inputs=envelope,
        discovery=dict(
            campaign_id=campaign_id,
            instrument="KO",
            exposure="engineering_fixture",
            leaves=[leaf.model_dump(mode="json") for leaf in leaves],
            fold_origins=[[80], [90]],
            max_rounds=2,
        ),
        account=account_rules(max_positions=1, fill_cap_units={"KO": 1000}).model_dump(mode="json"),
        strategy_threshold=0.0,
    )


def test_one_profile_real_worker_loop_and_terminal_replay(prepare_inputs, tmp_path, capsys):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    value = profile(envelope)
    path = tmp_path / "profile.json"
    path.write_bytes(canonical(value))
    assert cli.main(["research-workflow", "run", str(path), "--max-fits", "1"]) == 0
    first = workflow.inspect("workflow-test")
    assert first["discovery"]["fits_attempted"] == 1
    assert first["report"] is None
    result = workflow.resume("workflow-test", max_fits=9)
    assert result["status"] == "completed"
    assert result["discovery"]["fits_attempted"] == 10
    assert len(result["report"]["trials"]) == 2
    assert result["report"]["feedback_objective"] == "development_forecast_mae_only"
    for trial in result["report"]["trials"]:
        assert len(trial["accounts"]) == 4
        for episode in trial["accounts"]:
            assert len(episode["account"]["valuation_sessions"]) == 6
            assert set(episode["account"]["metrics"]) == {"strategy", "buy_and_hold"}
    before = (workflow.workflow_root("workflow-test") / "journal.sqlite").read_bytes()
    assert workflow.prepare(value) == result
    assert workflow.resume("workflow-test", max_fits=10) == result
    assert workflow.inspect("workflow-test", replay=True) == result
    assert (workflow.workflow_root("workflow-test") / "journal.sqlite").read_bytes() == before
    assert cli.main(["research-workflow", "report", "workflow-test"]) == 0
    output = capsys.readouterr().out
    assert "buy_and_hold" in output and "Costs" in output and "development-only" in output
    # Same grammar and visited candidate, only the permitted development feedback changes.
    plan = workflow.WorkflowProfile.model_validate_json(canonical(value)).discovery
    history = result["discovery"]["history"][:1]
    positive = [history[0] | {"improvement_pp": 1.0}]
    negative = [history[0] | {"improvement_pp": -1.0}]
    assert (
        propose(grammar(plan.leaves), positive).ref != propose(grammar(plan.leaves), negative).ref
    )


def test_profile_identity_and_preexisting_campaign_reject(prepare_inputs):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    value = profile(envelope)
    workflow.prepare(value)
    with pytest.raises(ValueError, match="another frozen profile"):
        workflow.prepare(value | {"strategy_threshold": 0.1})
    other = profile(envelope, "already-started")
    parsed = workflow.WorkflowProfile.model_validate_json(canonical(other))
    discovery.initialize(parsed.inputs, parsed.discovery)
    with pytest.raises(ValueError, match="before starting"):
        workflow.prepare(other)


def test_initialization_recovers_original_profile(prepare_inputs, monkeypatch):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    value = profile(envelope)
    original = workflow.atomic_write

    def fail_manifest(path, raw):
        if path.name == "manifest.json":
            raise OSError("simulated manifest checkpoint failure")
        return original(path, raw)

    with monkeypatch.context() as patch:
        patch.setattr(workflow, "atomic_write", fail_manifest)
        with pytest.raises(OSError):
            workflow.prepare(value)
    recovered = workflow.prepare(value)
    assert recovered["discovery"]["fits_attempted"] == 0
    assert workflow.prepare(value) == recovered


@pytest.fixture
def controlled_workflow(request, monkeypatch, tmp_path):
    prepare = request.getfixturevalue("prepare_inputs")
    worker = request.getfixturevalue("dummy_worker")
    contract, events, _ = contract_and_events(instruments=("KO",), count=105)
    clock = [utc_text(contract.calendar.sessions[99].close_at + timedelta(hours=1))]
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: clock[0])

    def historical_only(**kwargs):
        source, rows, evidence = contract_and_events(**kwargs)
        return (
            source,
            [r for r in rows if timestamp(r["event_at"]) <= timestamp(clock[0])],
            evidence,
        )

    monkeypatch.setattr(test_forecast_features, "contract_and_events", historical_only)
    inputs, envelope = prepare(future_calendar=5, target_instrument="KO")
    workflow.prepare(profile(envelope))
    result = workflow.resume("workflow-test", max_fits=10)
    assert result["status"] == "completed"
    assert all(t["paper"]["forecast"]["forward"] for t in result["report"]["trials"])

    def receive(at, request_id):
        clock[0] = utc_text(at)
        rows = [r for r in events if timestamp(r["event_at"]) <= at]
        return market_data.ingest(
            inputs.target[0].contract,
            write_events(tmp_path / (request_id + ".csv"), inputs.target[0].contract, rows),
            request_id=request_id,
        )["snapshot_ref"]

    return dict(inputs=inputs, clock=clock, worker=worker, receive=receive, contract=contract)


def test_issued_forecast_to_open_fill_valuation_settlement_and_replay(controlled_workflow):
    state = controlled_workflow
    sessions = state["contract"].calendar.sessions
    first_ref = state["receive"](sessions[100].open_at + timedelta(seconds=1), "first-open")
    first = workflow.observe("workflow-test", first_ref, request_id="open")
    for row in first["records"]:
        assert row["outcome"]["outcome"]["status"] == "immature"
        assert row["paper"]["accounts"]["buy_and_hold"]["fills"]
    final_ref = state["receive"](sessions[104].close_at + timedelta(seconds=1), "target-close")
    final = workflow.observe("workflow-test", final_ref, request_id="mature")
    for row in final["records"]:
        assert row["outcome"]["outcome"]["status"] == "settled"
        assert row["paper"]["status"] == "complete"
        assert len(row["paper"]["accounts"]["buy_and_hold"]["valuations"]) == 6
    assert workflow.observe("workflow-test", final_ref, request_id="mature") == final
    with pytest.raises(ValueError, match="another snapshot"):
        workflow.observe("workflow-test", first_ref, request_id="mature")
    before = deepcopy(final)
    assert workflow.inspect("workflow-test", replay=True)["observations"][-1] == before
    assert state["worker"]["fits"] == 10
    # A coherent new blob/reference is still not evidence of the recorded account behavior.
    journal, store, *_ = workflow._open("workflow-test")
    try:
        tampered = deepcopy(final)
        tampered["records"][0]["paper"]["accounts"]["buy_and_hold"]["cash"] = "1"
        with journal.connection:
            journal.connection.execute(
                "UPDATE events SET payload=? WHERE kind=?",
                (
                    canonical({"result_ref": store.put_json(tampered)}).decode(),
                    "observation_completed_" + digest(b"mature"),
                ),
            )
    finally:
        journal.close()
    with pytest.raises(ValueError, match="account/outcome replay differs"):
        workflow.inspect("workflow-test", replay=True)


def test_observation_interruption_resumes_same_request(controlled_workflow, monkeypatch):
    state = controlled_workflow
    sessions = state["contract"].calendar.sessions
    snapshot = state["receive"](sessions[104].close_at + timedelta(seconds=1), "terminal-quotes")
    original = workflow.commit_event

    def fail_completion(journal, kind, payload):
        if kind.startswith("observation_completed_"):
            raise OSError("simulated interruption after forecast settlement")
        return original(journal, kind, payload)

    with monkeypatch.context() as patch:
        patch.setattr(workflow, "commit_event", fail_completion)
        with pytest.raises(OSError):
            workflow.observe("workflow-test", snapshot, request_id="original")
    with pytest.raises(ValueError, match="pending"):
        workflow.observe("workflow-test", snapshot, request_id="different")
    result = workflow.observe("workflow-test", snapshot, request_id="original")
    assert result["sequence"] == 1
    assert state["worker"]["fits"] == 10


def test_invalid_snapshot_clock_does_not_poison_request(controlled_workflow):
    state = controlled_workflow
    initial_clock = state["clock"][0]
    sessions = state["contract"].calendar.sessions
    snapshot = state["receive"](sessions[100].open_at + timedelta(seconds=1), "new-quotes")
    received_clock = state["clock"][0]
    state["clock"][0] = initial_clock
    with pytest.raises(ValueError, match="receipt is later"):
        workflow.observe("workflow-test", snapshot, request_id="bad-clock")
    assert workflow.inspect("workflow-test")["observations"] == []
    state["clock"][0] = received_clock
    assert workflow.observe("workflow-test", snapshot, request_id="valid")["sequence"] == 1


def test_external_child_progress_rejects_without_blocking_or_losing_partial_work(
    controlled_workflow,
):
    state = controlled_workflow
    sessions = state["contract"].calendar.sessions
    older = state["receive"](sessions[100].open_at + timedelta(seconds=1), "first-open")
    newer = state["receive"](sessions[104].close_at + timedelta(seconds=1), "target-close")
    trials = workflow.inspect("workflow-test")["report"]["trials"]
    forecast_settlement.settle(trials[1]["experiment_id"], newer, request_id="external")
    with pytest.raises(ValueError, match="roll back"):
        workflow.observe("workflow-test", older, request_id="stale")
    failed = workflow.inspect("workflow-test")["observations"][-1]
    assert failed["status"] == "rejected" and len(failed["records"]) == 1
    with pytest.raises(ValueError, match="roll back"):
        workflow.observe("workflow-test", older, request_id="stale")
    succeeded = workflow.observe("workflow-test", newer, request_id="current")
    assert succeeded["sequence"] == 2 and succeeded["status"] == "completed"
    assert len(succeeded["records"]) == 2
    assert workflow.inspect("workflow-test", replay=True)["observations"] == [failed, succeeded]
    assert state["worker"]["fits"] == 10


def test_failed_fit_is_costed_and_reported_as_missing_strategy_decision(
    prepare_inputs, dummy_worker
):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    dummy_worker["fail"].add(1)
    workflow.prepare(profile(envelope))
    result = workflow.resume("workflow-test", max_fits=10)
    failed = [
        episode
        for trial in result["report"]["trials"]
        for episode in trial["accounts"]
        if episode["fit_status"] == "failed"
    ]
    assert result["discovery"]["fits_attempted"] == 10
    assert failed
    for episode in failed:
        assert not episode["account"]["accounts"]["strategy"]["fills"]
        assert episode["account"]["accounts"]["buy_and_hold"]["fills"]
    assert workflow.inspect("workflow-test", replay=True) == result
