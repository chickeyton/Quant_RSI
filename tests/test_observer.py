from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import lightgbm as lgb
import pytest
from market_helpers import account_rules, contract_and_events, write_events

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, canonical, digest
from quant_rsi.execution import observer
from quant_rsi.execution.baseline import seal_baseline
from quant_rsi.execution.observer_journal import ObserverJournal
from quant_rsi.market import market_contracts, market_data
from quant_rsi.market.market_contracts import DataContract, parse_observation_plan
from quant_rsi.research import research, researcher
from quant_rsi.research.researcher import ResearchPlan


@pytest.fixture
def live_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    contract, events, evidence = contract_and_events(instruments=("A", "B"), count=38)
    value = contract.model_dump(mode="json")
    for i in range(32, 38):
        date = datetime(2028, 1, 1, 9, tzinfo=timezone.utc) + timedelta(days=i - 32)
        value["calendar"]["sessions"][i].update(
            open_at=date.isoformat(), close_at=(date + timedelta(hours=7)).isoformat()
        )
    contract = DataContract.model_validate_json(canonical(value))
    for event in events:
        if event["session_id"] is not None and int(event["session_id"][1:]) >= 32:
            i = int(event["session_id"][1:])
            session = contract.calendar.sessions[i]
            at = session.open_at if event["kind"] == "OPEN_QUOTE" else session.close_at
            event.update(event_at=at.isoformat(), source_available_at=at.isoformat())
            prices = {
                ("A", "OPEN_QUOTE"): [100, 110, 100, 85, 90, 95],
                ("B", "OPEN_QUOTE"): [50, 40, 50, 45, 50, 55],
                ("A", "CLOSE_QUOTE"): [100, 120, 90, 85, 90, 95],
                ("B", "CLOSE_QUOTE"): [50, 45, 50, 45, 50, 55],
            }
            event["price"] = str(prices[(event["instrument"], event["kind"])][i - 32])
    evidence_path = tmp_path / "evidence.txt"
    evidence_path.write_bytes(evidence)
    historical = [
        event
        for event in events
        if event["session_id"] is None or int(event["session_id"][1:]) < 32
    ]
    imported = market_data.ingest(
        contract,
        write_events(tmp_path / "historical.csv", contract, historical),
        request_id="history",
        evidence_file=evidence_path,
    )
    value = ResearchPlan.model_validate_json(
        Path("examples/research_development.json").read_bytes()
    ).model_dump(mode="json")
    value["development"]["data"] = dict(
        source="local_event_snapshot",
        sessions=32,
        instruments=2,
        train_cutoff=20,
        seed=7,
        snapshot_ref=imported["snapshot_ref"],
        decision_delay_seconds=60,
    )
    value["development"]["portfolio"] = account_rules().model_dump(mode="json")
    value["development"]["proposals"][0]["factor"] = dict(op="momentum", window=2)
    value["development"]["budget"].update(max_attempts=1, max_worker_seconds=20)
    value["policy"].update(window_min=1, window_max=3)
    research.create_research(
        ResearchPlan.model_validate_json(canonical(value)), tmp_path / "research", steps=1
    )
    sealed = seal_baseline(tmp_path / "research")
    clock = ["2027-12-31T12:00:00Z"]
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: clock[0])
    common = dict(
        schema_version=1,
        mode="registered_baseline_observer",
        purpose="developer_visible_observation",
        model_bundle_ref=sealed["bundle_ref"],
        contract_ref=sealed["contract_ref"],
        max_information_age_seconds=315360000,
        max_model_age_seconds=315360000,
        expires_at="2029-01-01T00:00:00Z",
    )
    plan = parse_observation_plan(
        canonical(
            common
            | account_rules().model_dump(mode="json")
            | {"account_mode": "indicative_account"}
        )
    )

    def ingest(session, phase, request, *, instruments=("A", "B"), revision=None):
        selected = [
            event.copy()
            for event in events
            if event["session_id"] == session
            and event["kind"] in phase
            and event["instrument"] in instruments
        ]
        if revision is not None:
            for event in selected:
                event.update(revision=revision, price="999", source_available_at=clock[0])
        return market_data.ingest(
            contract,
            write_events(tmp_path / (request + ".csv"), contract, selected),
            request_id=request,
        )

    return dict(
        clock=clock, plan=plan, common=common, contract=contract, ingest=ingest, root=tmp_path
    )


def start_orders(fixture, *, plan=None):
    observer.register_observer(plan or fixture["plan"], "episode")
    fixture["clock"][0] = "2028-01-01T16:01:00Z"
    fixture["ingest"]("s32", {"OPEN_QUOTE", "CLOSE_QUOTE"}, "day1")
    return observer.advance_observer("episode", request_id="day1", steps=8)


def state():
    with market_data.market_context() as (registry, _):
        journal = ObserverJournal(registry)
        return journal.trace("episode")[2]


def test_durable_submit_fill_no_step_end_liquidation_and_readonly(live_fixture, monkeypatch):
    report = start_orders(live_fixture)
    assert report["pending_order_count"] == 2
    assert Decimal(report["reserved_cash"]) == Decimal("959.50")
    assert report["observations"][0]["submission_status"] == "timely_durable_submission"
    assert report["request"]["actual_steps"] == 2
    assert (
        observer.advance_observer("episode", request_id="day1", steps=8)["checkpoint_ref"]
        == report["checkpoint_ref"]
    )
    with pytest.raises(ValueError, match="fixed step grant"):
        observer.advance_observer("episode", request_id="day1", steps=9)
    live_fixture["clock"][0] = "2028-01-02T09:01:00Z"
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "open2")
    filled = observer.advance_observer("episode", request_id="fill", steps=8)
    assert Decimal(filled["cash"]) == Decimal("505.10")
    assert {key: value["quantity"] for key, value in filled["holdings"].items()} == {"A": 3, "B": 4}
    assert filled["pending_order_count"] == 0
    with monkeypatch.context() as locked:

        def forbidden(*args, **kwargs):
            pytest.fail("status/replay attempted fit/proposal/write")

        locked.setattr(lgb, "train", forbidden)
        locked.setattr(researcher, "propose", forbidden)
        locked.setattr(controller.subprocess, "run", forbidden)
        locked.setattr(ArtifactStore, "put", forbidden)
        root = market_data.market_root()
        before = {
            str(path): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()
        }
        assert observer.inspect_observer("episode")["checkpoint_ref"] == filled["checkpoint_ref"]
        assert observer.inspect_observer("episode", replay=True)["cash"] == filled["cash"]
        assert before == {
            str(path): digest(path.read_bytes()) for path in root.rglob("*") if path.is_file()
        }


def test_missing_open_dual_cursors_durable_stop_and_late_resolution(live_fixture):
    start_orders(live_fixture)
    live_fixture["clock"][0] = "2028-01-02T09:01:00Z"
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "only_B", instruments=("B",))
    partial = observer.advance_observer("episode", request_id="partial", steps=8)
    assert Decimal(partial["cash"]) == Decimal("838.40")
    assert Decimal(partial["reserved_cash"]) == Decimal("505")
    cursor = partial["market_cursor"]
    live_fixture["clock"][0] = "2028-01-02T16:01:00Z"
    live_fixture["ingest"]("s33", {"CLOSE_QUOTE"}, "close2")
    later = observer.advance_observer("episode", request_id="independent_prediction", steps=8)
    assert later["receipt_cursor"] > partial["receipt_cursor"]
    assert later["market_cursor"] == cursor and later["observation_count"] == 2
    assert "account_boundary_pending" in later["observations"][-1]["blocked_reasons"]
    stopped = observer.stop_observer("episode", request_id="stop")
    assert stopped["pending_order_count"] == 1 and Decimal(stopped["reserved_cash"]) == Decimal(
        "505"
    )
    live_fixture["clock"][0] = "2028-01-03T10:00:00Z"
    assert observer.stop_observer("episode", request_id="stop")["stop"] == stopped["stop"]
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "late_A", instruments=("A",))
    resolved = observer.advance_observer("episode", request_id="resolve", steps=8)
    assert Decimal(resolved["cash"]) == Decimal("505.10")
    assert resolved["holdings"]["A"]["quantity"] == 3 and resolved["holdings"]["B"]["quantity"] == 4
    assert len(state()["account"]["fills"]) == 2
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "revised_A", instruments=("A",), revision=1)
    revised = observer.advance_observer("episode", request_id="revision", steps=8)
    assert revised["cash"] == resolved["cash"] and revised["holdings"] == resolved["holdings"]
    assert revised["correction_count"] > resolved["correction_count"]
    assert len(state()["account"]["fills"]) == 2
    observer.inspect_observer("episode", replay=True)


@pytest.mark.parametrize("fault", ["lost_confirmation", "late_commit"])
def test_post_commit_confirmation_loss_or_actual_late_commit_never_fills(
    live_fixture, monkeypatch, fault
):
    observer.register_observer(live_fixture["plan"], "episode")
    live_fixture["clock"][0] = "2028-01-01T16:01:00Z"
    live_fixture["ingest"]("s32", {"OPEN_QUOTE", "CLOSE_QUOTE"}, "day1")
    original = ObserverJournal.commit
    tripped = []

    def interrupted(self, episode, next_state, kind, payload, **kwargs):
        if kind == "STEP" and next_state["pending_submission"] is not None and not tripped:
            tripped.append(True)
            if fault == "late_commit":
                live_fixture["clock"][0] = "2028-01-02T09:00:00Z"
                return original(self, episode, next_state, kind, payload, **kwargs)
            original(self, episode, next_state, kind, payload, **kwargs)
            raise RuntimeError("crashed after intent commit before confirmation")
        return original(self, episode, next_state, kind, payload, **kwargs)

    with monkeypatch.context() as injected:
        injected.setattr(ObserverJournal, "commit", interrupted)
        if fault == "lost_confirmation":
            with pytest.raises(RuntimeError, match="crashed after intent"):
                observer.advance_observer("episode", request_id="window", steps=8)
        else:
            observer.advance_observer("episode", request_id="window", steps=8)
    live_fixture["clock"][0] = "2028-01-02T09:01:00Z"
    observer.advance_observer("episode", request_id="window")
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "open2")
    result = observer.advance_observer("episode", request_id="after", steps=8)
    assert Decimal(result["cash"]) == Decimal("1000") and Decimal(result["reserved_cash"]) == 0
    assert not state()["account"]["fills"]
    assert result["observations"][0]["submission_status"] == "unsubmitted_or_late"
    observer.inspect_observer("episode", replay=True)


def test_prediction_only_backfill_cannot_create_risk(live_fixture):
    plan = parse_observation_plan(
        canonical(live_fixture["common"] | {"account_mode": "prediction_only"})
    )
    observer.register_observer(plan, "episode")
    live_fixture["clock"][0] = "2028-01-01T16:01:00Z"
    # A fresh receipt for an old economic event does not become a forward observation.
    live_fixture["ingest"]("s31", {"CLOSE_QUOTE"}, "backfill")
    backfill = observer.advance_observer("episode", request_id="old", steps=4)
    assert backfill["observation_count"] == 0 and backfill["cash"] is None
    live_fixture["ingest"]("s32", {"OPEN_QUOTE", "CLOSE_QUOTE"}, "day1")
    result = observer.advance_observer("episode", request_id="new", steps=4)
    assert result["observation_count"] == 1 and result["pending_order_count"] == 0
    assert result["holdings"] is None
    observer.inspect_observer("episode", replay=True)


def test_same_plan_alias_or_missing_initialized_ledger_cannot_reset_capital(live_fixture):
    initial = observer.register_observer(live_fixture["plan"], "episode")
    assert (
        observer.register_observer(live_fixture["plan"], "episode")["checkpoint_ref"]
        == initial["checkpoint_ref"]
    )
    with pytest.raises(ValueError, match="aliases cannot reset"):
        observer.register_observer(live_fixture["plan"], "renamed")
    with market_data.market_context() as (registry, _):
        with registry.connection:
            registry.connection.execute("DROP TABLE observer_events")
    with pytest.raises(ValueError, match="missing or inconsistent"):
        observer.inspect_observer("episode")
    with pytest.raises(ValueError, match="missing or inconsistent"):
        observer.register_observer(live_fixture["plan"], "episode")


def test_information_age_gate_does_not_refresh_old_data_when_fit_is_new(live_fixture):
    value = live_fixture["plan"].model_dump(mode="json") | {"max_information_age_seconds": 60}
    report = start_orders(live_fixture, plan=parse_observation_plan(canonical(value)))
    assert report["observation_count"] == 1 and report["pending_order_count"] == 0
    assert "market_information_age_limit" in report["observations"][0]["blocked_reasons"]
    assert not state()["account"]["orders"]


@pytest.mark.parametrize(
    "when,cancelled", [("08:59:59", True), ("09:00:00", False), ("09:02:00", False)]
)
def test_stop_receipt_before_equal_and_after_target(live_fixture, when, cancelled):
    start_orders(live_fixture)
    live_fixture["clock"][0] = "2028-01-02T" + when + "Z"
    report = observer.stop_observer("episode", request_id="stop")
    assert report["pending_order_count"] == (0 if cancelled else 2)
    assert Decimal(report["reserved_cash"]) == (Decimal(0) if cancelled else Decimal("959.50"))
    assert not state()["account"]["fills"]
    assert (
        observer.stop_observer("episode", request_id="stop")["checkpoint_ref"]
        == report["checkpoint_ref"]
    )
    observer.inspect_observer("episode", replay=True)


def test_inference_crosses_open_or_confirmation_persists_later(live_fixture, monkeypatch):
    from quant_rsi.execution import observer_engine

    observer.register_observer(live_fixture["plan"], "episode")
    live_fixture["clock"][0] = "2028-01-01T16:01:00Z"
    live_fixture["ingest"]("s32", {"OPEN_QUOTE", "CLOSE_QUOTE"}, "day1")
    original = observer_engine.predict_view

    def slow_prediction(*args, **kwargs):
        value = original(*args, **kwargs)
        live_fixture["clock"][0] = "2028-01-02T09:00:00Z"
        return value

    with monkeypatch.context() as injected:
        injected.setattr(observer_engine, "predict_view", slow_prediction)
        report = observer.advance_observer("episode", request_id="slow", steps=8)
    assert report["observations"][0]["prediction_timeliness"] == "forward_late"
    assert not state()["account"]["orders"]
    observer.inspect_observer("episode", replay=True)


def test_timely_post_commit_witness_may_be_durably_recorded_after_target(live_fixture, monkeypatch):
    original = ObserverJournal.commit

    def slow_confirmation(self, episode, next_state, kind, payload, **kwargs):
        if kind == "CONFIRM":
            assert payload["at"] == "2028-01-01T16:01:00Z"
            live_fixture["clock"][0] = "2028-01-02T09:00:00Z"
        return original(self, episode, next_state, kind, payload, **kwargs)

    with monkeypatch.context() as injected:
        injected.setattr(ObserverJournal, "commit", slow_confirmation)
        report = start_orders(live_fixture)
    assert report["pending_order_count"] == 2
    assert report["observations"][0]["submission_status"] == "timely_durable_submission"
    live_fixture["clock"][0] = "2028-01-02T09:01:00Z"
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "open2")
    filled = observer.advance_observer("episode", request_id="fill", steps=8)
    assert Decimal(filled["cash"]) == Decimal("505.10")
    observer.inspect_observer("episode", replay=True)


def test_sql_failure_rolls_back_fill_receipt_and_cursors_then_resumes_once(
    live_fixture, monkeypatch
):
    import sqlite3

    start_orders(live_fixture)
    live_fixture["clock"][0] = "2028-01-02T09:01:00Z"
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "open2")
    original, parents = ObserverJournal.commit, []

    class FailedInsert:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def execute(self, sql, *args):
            if sql.startswith("INSERT INTO observer_events"):
                raise sqlite3.OperationalError("injected event-insert failure")
            return self.connection.execute(sql, *args)

    def fault(self, episode, next_state, kind, payload, **kwargs):
        if kind == "STEP" and next_state["account"]["fills"] and not parents:
            parents.append(self.store.json(episode["head_ref"]))
            saved = self.connection
            self.connection = FailedInsert(saved)
            try:
                return original(self, episode, next_state, kind, payload, **kwargs)
            finally:
                self.connection = saved
        return original(self, episode, next_state, kind, payload, **kwargs)

    with monkeypatch.context() as injected:
        injected.setattr(ObserverJournal, "commit", fault)
        with pytest.raises(sqlite3.OperationalError, match="event-insert"):
            observer.advance_observer("episode", request_id="fill", steps=3)
    assert state() == parents[0]
    assert state()["window"]["target"] == state()["window"]["start"] + 3
    with pytest.raises(ValueError, match="fixed step grant"):
        observer.advance_observer("episode", request_id="fill", steps=4)
    result = observer.advance_observer("episode", request_id="fill")
    assert Decimal(result["cash"]) == Decimal("505.10") and len(state()["account"]["fills"]) == 2
    assert (
        observer.advance_observer("episode", request_id="fill")["checkpoint_ref"]
        == result["checkpoint_ref"]
    )
    observer.inspect_observer("episode", replay=True)


def test_response_loss_after_fill_commit_never_repeats_fill(live_fixture, monkeypatch):
    start_orders(live_fixture)
    live_fixture["clock"][0] = "2028-01-02T09:01:00Z"
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "open2")
    original, crashed = ObserverJournal.commit, []

    def fault(self, episode, next_state, kind, payload, **kwargs):
        result = original(self, episode, next_state, kind, payload, **kwargs)
        if kind == "STEP" and next_state["account"]["fills"] and not crashed:
            crashed.append(True)
            raise RuntimeError("lost response after durable fill")
        return result

    with monkeypatch.context() as injected:
        injected.setattr(ObserverJournal, "commit", fault)
        with pytest.raises(RuntimeError, match="durable fill"):
            observer.advance_observer("episode", request_id="fill", steps=3)
    result = observer.advance_observer("episode", request_id="fill")
    assert Decimal(result["cash"]) == Decimal("505.10") and len(state()["account"]["fills"]) == 2
    with market_data.market_context() as (registry, _):
        journal = ObserverJournal(registry)
        current = journal.episode("episode")
        stale = current | {"sequence": current["sequence"] - 1}
        with pytest.raises(ValueError, match="compare-and-swap"):
            journal.commit(stale, journal.trace("episode")[2], "STOP", {})
        assert journal.episode("episode") == current


def test_interrupted_activation_keeps_original_capital_and_start_watermark(
    live_fixture, monkeypatch
):
    original = ObserverJournal.commit

    def fail_activation(self, episode, next_state, kind, payload, **kwargs):
        if kind == "ACTIVATE":
            raise RuntimeError("lost initial activation confirmation")
        return original(self, episode, next_state, kind, payload, **kwargs)

    with monkeypatch.context() as injected:
        injected.setattr(ObserverJournal, "commit", fail_activation)
        with pytest.raises(RuntimeError, match="activation"):
            observer.register_observer(live_fixture["plan"], "episode")
    original_state = state()
    live_fixture["clock"][0] = "2028-01-01T16:01:00Z"
    live_fixture["ingest"]("s32", {"OPEN_QUOTE", "CLOSE_QUOTE"}, "before_activation")
    live_fixture["clock"][0] = "2028-01-01T16:02:00Z"
    activated = observer.register_observer(live_fixture["plan"], "episode")
    assert activated["start_watermark"] == original_state["start_watermark"]
    assert activated["active_at"] == live_fixture["clock"][0]
    result = observer.advance_observer("episode", request_id="old_input", steps=3)
    assert result["observation_count"] == 0 and Decimal(result["cash"]) == Decimal("1000")
    observer.inspect_observer("episode", replay=True)


def test_queue_exit_waits_for_new_legal_open_and_charges_actual_costs(live_fixture):
    value = live_fixture["plan"].model_dump(mode="json") | {
        "termination": "queue_exit_at_next_eligible_open"
    }
    start_orders(live_fixture, plan=parse_observation_plan(canonical(value)))
    live_fixture["clock"][0] = "2028-01-02T09:01:00Z"
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "open2")
    observer.advance_observer("episode", request_id="fill", steps=4)
    stopped = observer.stop_observer("episode", request_id="stop")
    assert Decimal(stopped["cash"]) == Decimal("505.10")
    queued = observer.advance_observer("episode", request_id="queue_exit", steps=4)
    assert Decimal(queued["cash"]) == Decimal("505.10") and queued["pending_order_count"] == 2
    assert len(state()["account"]["fills"]) == 2
    live_fixture["clock"][0] = "2028-01-03T09:01:00Z"
    live_fixture["ingest"]("s34", {"OPEN_QUOTE"}, "exit_quotes")
    result = observer.advance_observer("episode", request_id="exit_fill", steps=8)
    assert Decimal(result["cash"]) == Decimal("990.10")
    assert all(position["quantity"] == 0 for position in result["holdings"].values())
    assert len(state()["account"]["fills"]) == 4
    observer.inspect_observer("episode", replay=True)


def test_first_observed_revision_one_is_not_a_correction_of_prior_decisions(live_fixture):
    start_orders(live_fixture)
    live_fixture["clock"][0] = "2028-01-02T09:01:00Z"
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "first_revision_one", revision=1)
    report = observer.advance_observer("episode", request_id="new_event", steps=4)
    assert report["correction_count"] == 0
    live_fixture["ingest"]("s33", {"OPEN_QUOTE"}, "second_revision", revision=2)
    report = observer.advance_observer("episode", request_id="correction", steps=4)
    assert report["correction_count"] == 2
    observer.inspect_observer("episode", replay=True)


def test_journal_rejects_wrong_state_identity_before_any_artifact_or_ledger_write(live_fixture):
    observer.register_observer(live_fixture["plan"], "episode")
    with market_data.market_context() as (registry, _):
        journal = ObserverJournal(registry)
        episode, initial, current, _ = journal.trace("episode")
        before = {
            str(path): digest(path.read_bytes())
            for path in registry.root.rglob("*")
            if path.is_file()
        }
        with pytest.raises(ValueError, match="initial state episode/plan"):
            journal.register("wrong", episode["plan_ref"], initial)
        for wrong in (current | {"episode_id": "wrong"}, current | {"plan_ref": "f" * 64}):
            with pytest.raises(ValueError, match="next state episode/plan"):
                journal.commit(episode, wrong, "STOP", {})
        assert before == {
            str(path): digest(path.read_bytes())
            for path in registry.root.rglob("*")
            if path.is_file()
        }
