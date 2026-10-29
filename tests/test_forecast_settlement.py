"""Offline clock-controlled outcomes; synthetic evidence is not predictive validation."""

from copy import deepcopy
from datetime import timedelta

import pytest
import test_forecast_features
from market_helpers import contract_and_events, write_events
from test_forecast_comparison import dummy_worker  # noqa: F401
from test_forecast_features import prepare_inputs  # noqa: F401
from test_news_comparison import news_inputs  # noqa: F401

from quant_rsi import cli
from quant_rsi.core import controller
from quant_rsi.core.artifacts import canonical, decode, digest
from quant_rsi.forecast import forecast_settlement as observation
from quant_rsi.forecast.forecast_experiment import (
    experiment_root,
    prepare_experiment,
    resume_experiment,
)
from quant_rsi.market import market_contracts, market_data
from quant_rsi.market.market_contracts import timestamp, utc_text


@pytest.fixture
def published(request, monkeypatch, tmp_path):
    prepare = request.getfixturevalue("prepare_inputs")
    worker_state = request.getfixturevalue("dummy_worker")
    calendar, all_events, _ = contract_and_events(count=105)
    clock = [utc_text(calendar.calendar.sessions[99].close_at + timedelta(hours=1))]
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: clock[0])

    def historical_only(**kwargs):
        contract, events, evidence = contract_and_events(**kwargs)
        return (
            contract,
            [row for row in events if timestamp(row["event_at"]) <= timestamp(clock[0])],
            evidence,
        )

    monkeypatch.setattr(test_forecast_features, "contract_and_events", historical_only)
    inputs, envelope = prepare(future_calendar=5)
    prepared = prepare_experiment(envelope, "observed", fold_origins=((80, 85, 90),))
    report = resume_experiment("observed", protocol_ref=prepared["protocol_ref"], max_fits=4)
    assert report["forecast"]["forward"] is True
    state = dict(inputs=inputs, clock=clock, report=report, worker=worker_state)

    def receive(events, request_id, *, at=None):
        if at is not None:
            clock[0] = utc_text(at)
        contract = inputs.target[0].contract
        return market_data.ingest(
            contract,
            write_events(tmp_path / (request_id + ".csv"), contract, events),
            request_id=request_id,
        )["snapshot_ref"]

    state["receive"] = receive
    state["future_events"] = [row for row in all_events if row["session_id"] == "s104"]
    state["target_close"] = calendar.calendar.sessions[104].close_at
    state["original_snapshot"] = inputs.target[0].spec.snapshot_ref
    observation.register("observed")
    return state


def test_forward_matures_from_actual_receipt_and_repeats_without_side_effects(published):
    original = experiment_root("observed")
    before = {path.name: digest(path.read_bytes()) for path in original.iterdir() if path.is_file()}
    first = observation.settle("observed", published["original_snapshot"], request_id="early")
    assert first["outcome"]["status"] == "immature"
    assert first["outcome"]["metrics"] is None
    published["clock"][0] = utc_text(published["target_close"] + timedelta(hours=1))
    missing = observation.settle("observed", published["original_snapshot"], request_id="missing")
    assert missing["outcome"]["label"]["reason"] == "target_or_anchor_quote_not_known"
    snapshot = published["receive"](published["future_events"], "target")
    mature = observation.settle("observed", snapshot, request_id="mature")
    assert mature["outcome"]["status"] == "settled"
    assert mature["outcome"]["label"]["available_at"] == published["clock"][0]
    metrics = mature["outcome"]["metrics"]
    assert metrics["actual_price"] == 205
    assert metrics["forecast"]["observations"] == 1
    assert metrics["unchanged_price"]["mae_pp"] == pytest.approx(abs(100 * (205 / 203 - 1)))
    assert observation.settle("observed", snapshot, request_id="mature") == mature
    assert observation.register("observed")["latest"] == mature
    assert observation.inspect("observed", replay=True)["observations"] == [first, missing, mature]
    assert published["worker"]["fits"] == 4
    assert before == {
        path.name: digest(path.read_bytes()) for path in original.iterdir() if path.is_file()
    }
    with pytest.raises(ValueError, match="another snapshot"):
        observation.settle("observed", published["original_snapshot"], request_id="mature")


def test_target_revision_appends_and_anchor_revision_abstains(published):
    snapshot = published["receive"](
        published["future_events"], "first", at=published["target_close"] + timedelta(hours=1)
    )
    first = observation.settle("observed", snapshot, request_id="first")
    revision = deepcopy(published["future_events"][-1])
    revision.update(revision=1, price="210")
    revised = published["receive"]([revision], "correction")
    second = observation.settle("observed", revised, request_id="correction")
    assert second["previous_ref"] == first["observation_ref"]
    assert first["outcome"]["metrics"]["actual_price"] == 205
    assert second["outcome"]["metrics"]["actual_price"] == 210
    anchor = next(
        deepcopy(row["event"])
        for row in published["inputs"].target[0].snapshot["events"]
        if row["event"]["kind"] == "CLOSE_QUOTE" and row["event"]["session_id"] == "s99"
    )
    anchor.update(revision=1, price="199")
    corrected = published["receive"]([anchor], "origin-correction")
    third = observation.settle("observed", corrected, request_id="origin-correction")
    assert third["outcome"]["status"] == "insufficient_evidence"
    assert third["outcome"]["reason"] == "published_origin_price_revised"
    assert third["outcome"]["metrics"] is None
    assert observation.inspect("observed", replay=True)["observations"] == [first, second, third]
    with pytest.raises(ValueError, match="roll back"):
        observation.settle("observed", snapshot, request_id="rollback")


def test_wrong_source_backwards_clock_and_runtime_fail_closed(published, monkeypatch):
    source = published["inputs"].contexts["market"][0].spec.snapshot_ref
    with pytest.raises(ValueError, match="target source receipt prefix"):
        observation.settle("observed", source, request_id="wrong-source")
    published["clock"][0] = utc_text(timestamp(published["clock"][0]) - timedelta(hours=1))
    with pytest.raises(ValueError, match="clock moved backwards"):
        observation.settle("observed", published["original_snapshot"], request_id="backwards")
    monkeypatch.setattr(controller, "runtime_identity", lambda: {"changed": True})
    with pytest.raises(ValueError, match="recorded runtime"):
        observation.settle("observed", published["original_snapshot"], request_id="changed")
    assert observation.inspect("observed")["observations"] == []


def test_receipt_cannot_be_backdated_and_cli_replays(published, capsys):
    snapshot = published["receive"](
        published["future_events"], "late", at=published["target_close"] + timedelta(days=3)
    )
    published["clock"][0] = utc_text(published["target_close"] + timedelta(days=1))
    with pytest.raises(ValueError, match="later than the trusted"):
        observation.settle("observed", snapshot, request_id="backdated")
    published["clock"][0] = utc_text(published["target_close"] + timedelta(days=3))
    assert (
        cli.main(
            [
                "forecast-observe",
                "settle",
                "observed",
                "--snapshot-ref",
                snapshot,
                "--request-id",
                "late",
            ]
        )
        == 0
    )
    result = decode(capsys.readouterr().out.encode())
    assert result["outcome"]["label"]["available_at"] == published["clock"][0]
    assert cli.main(["forecast-observe", "replay", "observed"]) == 0
    assert decode(capsys.readouterr().out.encode())["latest"] == result


def test_retrospective_publication_never_gets_prospective_metrics(request):
    prepare = request.getfixturevalue("prepare_inputs")
    request.getfixturevalue("dummy_worker")
    inputs, envelope = prepare(future_calendar=5)
    prepared = prepare_experiment(envelope, "retrospective", fold_origins=((80, 85, 90),))
    resume_experiment("retrospective", protocol_ref=prepared["protocol_ref"], max_fits=4)
    observation.register("retrospective")
    row = observation.settle("retrospective", inputs.target[0].spec.snapshot_ref, request_id="late")
    assert row["outcome"]["status"] == "not_prospective"
    assert row["outcome"]["metrics"] is None
    assert observation.inspect("retrospective", replay=True)["latest"] == row


def test_replay_rejects_changed_outcome_artifact(published):
    row = observation.settle("observed", published["original_snapshot"], request_id="first")
    path = observation.observation_root("observed") / "artifacts" / row["observation_ref"]
    path.write_bytes(canonical({"tampered": True}))
    with pytest.raises(ValueError, match="integrity failure"):
        observation.inspect("observed", replay=True)


def test_registration_recovers_after_manifest_write(published, monkeypatch):
    root = observation.observation_root("observed")
    manifest = (root / "manifest.json").read_bytes()
    (root / "journal.sqlite").unlink()  # Crash before the first Journal initialization.
    monkeypatch.setattr(controller, "runtime_identity", lambda: {"new_code": True})
    report = observation.register("observed")
    assert report["observations"] == []
    assert (root / "manifest.json").read_bytes() == manifest
    with pytest.raises(ValueError, match="recorded runtime"):
        observation.inspect("observed", replay=True)


def test_new_corporate_action_stays_outside_frozen_objective(published):
    notice = dict(
        event_id="unexpected-action",
        revision=0,
        kind="CORPORATE_ACTION_NOTICE",
        instrument="A",
        session_id=None,
        event_at=utc_text(published["target_close"]),
        source_available_at=utc_text(published["target_close"]),
        price=None,
        status=None,
        action="split",
        quality="valid",
    )
    snapshot = published["receive"](
        published["future_events"] + [notice],
        "action",
        at=published["target_close"] + timedelta(hours=1),
    )
    row = observation.settle("observed", snapshot, request_id="action")
    assert row["outcome"]["status"] == "insufficient_evidence"
    assert row["outcome"]["reason"] == "forecast_action_effect_or_type_unknown"
    assert row["outcome"]["metrics"] is None
    assert observation.inspect("observed", replay=True)["latest"] == row


def test_existing_publication_registration_preserves_original_runtime(published, monkeypatch):
    from quant_rsi.core.journal import Journal

    root = observation.observation_root("observed")
    # Remove only this empty observer to emulate its first registration after an upgrade.
    (root / "journal.sqlite").unlink()
    (root / "manifest.json").unlink()
    original = (experiment_root("observed") / "manifest.json").read_bytes()
    monkeypatch.setattr(controller, "runtime_identity", lambda: {"new_code": True})
    report = observation.register("observed")
    assert report["publication"]["forecast"]["forward"] is True
    assert (experiment_root("observed") / "manifest.json").read_bytes() == original
    assert observation.inspect("observed", replay=True) == report
    journal = Journal(root)
    assert journal.attempts() == []
    journal.close()


def test_news_comparison_publication_uses_original_confirmation(request):
    inputs, _, prepared, _ = request.getfixturevalue("news_inputs")
    worker_state = request.getfixturevalue("dummy_worker")
    published = resume_experiment("news-pair", protocol_ref=prepared["protocol_ref"], max_fits=3)
    registered = observation.register("news-pair")
    assert registered["publication"]["forecast"] == published["forecast"]
    row = observation.settle("news-pair", inputs.target[0].spec.snapshot_ref, request_id="news")
    assert observation.inspect("news-pair", replay=True)["latest"] == row
    assert worker_state["fits"] == 3


def test_discovery_publication_binds_allocated_child_without_new_fits(request):
    from test_factor_discovery import plan

    from quant_rsi.factors import discovery

    prepare = request.getfixturevalue("prepare_inputs")
    worker_state = request.getfixturevalue("dummy_worker")
    inputs, envelope = prepare(future_calendar=5, target_instrument="KO")
    discovery.initialize(envelope, plan(max_rounds=1))
    result = discovery.resume("discovery-test", max_fits=5)
    child = result["history"][0]["experiment_id"]
    report = observation.register(child)
    assert report["publication"]["input_ref"] == inputs.ref
    row = observation.settle(child, inputs.target[0].spec.snapshot_ref, request_id="discovery")
    assert observation.inspect(child, replay=True)["latest"] == row
    assert worker_state["fits"] == 5


def test_observation_allowance_is_global_and_repeat_remains_idempotent(published, monkeypatch):
    root = observation.observation_root("observed")
    (root / "journal.sqlite").unlink()
    (root / "manifest.json").unlink()
    monkeypatch.setattr(observation, "MAX_OBSERVATIONS", 1)
    observation.register("observed")
    snapshot = published["original_snapshot"]
    first = observation.settle("observed", snapshot, request_id="last-slot")
    with pytest.raises(ValueError, match="budget exhausted"):
        observation.settle("observed", snapshot, request_id="renamed")
    assert observation.settle("observed", snapshot, request_id="last-slot") == first
    assert observation.register("observed")["latest"] == first
    assert published["worker"]["fits"] == 4
