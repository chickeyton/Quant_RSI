from contextlib import contextmanager

import lightgbm as lgb
import numpy as np
import pytest
from market_helpers import contract_and_events, write_events

from quant_rsi import market_contracts, market_data
from quant_rsi.artifacts import canonical, digest
from quant_rsi.asof import make_feature_view, mature_label, predict_view
from quant_rsi.market_contracts import PredictionOnlyPlan
from quant_rsi.schemas import FactorSpec


@pytest.fixture
def imported(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-24T12:00:00+00:00")
    contract, events, evidence = contract_and_events()
    evidence_path = tmp_path / "evidence.txt"
    evidence_path.write_bytes(evidence)
    source = write_events(tmp_path / "source.csv", contract, events)
    report = market_data.ingest(contract, source, request_id="first", evidence_file=evidence_path)
    snapshot, contract = market_data.load_snapshot(report["snapshot_ref"])
    return contract, events, report, snapshot, source


def factor(op="momentum"):
    return FactorSpec(op=op, window=2)


def test_historical_receipts_and_maturity_are_separate(imported):
    contract, _, report, snapshot, _ = imported
    view = make_feature_view(snapshot, contract, "s2", "2020-10-21T16:00:00Z", factor())
    assert view["rows"][0]["features"] == pytest.approx([0.21, 0.1, 0.1])
    assert snapshot["events"][0]["first_seen_at"] == report["first_seen_at"]
    received = make_feature_view(
        snapshot, contract, "s2", "2020-10-21T16:00:00Z", factor(), basis="observed_receipts"
    )
    assert received["rows"][0]["status"] == "unresolved"
    assert received["rows"][0]["features"] is None
    assert (
        mature_label(snapshot, contract, "s2", "A", "2020-10-23T08:59:00Z")["status"] == "immature"
    )
    mature = mature_label(snapshot, contract, "s2", "A", "2020-10-23T09:00:00Z")
    assert mature["value"] == pytest.approx(-1 / 121)


def test_inference_needs_no_future_open_or_fit(imported, monkeypatch):
    contract, _, _, snapshot, _ = imported
    snapshot = snapshot | {
        "events": [
            item
            for item in snapshot["events"]
            if item["event"]["session_id"] in (None, "s0", "s1", "s2")
        ]
    }
    model = lgb.train(
        {"verbosity": -1, "num_threads": 1, "min_data_in_leaf": 1},
        lgb.Dataset(np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), label=[0.0, 1.0]),
        num_boost_round=2,
    )
    monkeypatch.setattr(lgb, "train", lambda *a, **k: pytest.fail("inference tried to fit"))
    view = make_feature_view(snapshot, contract, "s2", "2020-10-21T16:00:00Z", factor())
    prediction = predict_view(model.model_to_string(), view)
    assert prediction["rows"][0]["score"] is not None
    assert prediction == predict_view(model.model_to_string(), view)
    assert (
        mature_label(snapshot, contract, "s2", "A", "2026-10-24T12:00:00Z")["status"] == "immature"
    )


def test_future_revision_cannot_change_old_features(imported, tmp_path):
    contract, events, _, snapshot, _ = imported
    before = make_feature_view(snapshot, contract, "s2", "2020-10-21T16:00:00Z", factor())
    old = next(event for event in events if event["event_id"] == "A.CLOSE_QUOTE.1")
    correction = old | dict(revision=1, price="999", source_available_at="2020-10-22T12:00:00Z")
    report = market_data.ingest(
        contract,
        write_events(tmp_path / "revision.csv", contract, [correction]),
        request_id="revision",
    )
    revised, _ = market_data.load_snapshot(report["snapshot_ref"])
    after = make_feature_view(revised, contract, "s2", "2020-10-21T16:00:00Z", factor())
    assert after["rows"] == before["rows"]
    assert len(revised["events"]) == len(snapshot["events"]) + 1


def test_import_idempotency_conflicts_and_readonly_replay(imported, tmp_path):
    contract, events, report, snapshot, source = imported
    assert market_data.ingest(contract, source, request_id="first") == report
    duplicate = market_data.ingest(contract, source, request_id="renamed")
    assert duplicate["fresh_events"] == 0
    newer, _ = market_data.load_snapshot(duplicate["snapshot_ref"])
    assert newer["events"] == snapshot["events"]
    conflict = events[1] | {"price": "777"}
    with pytest.raises(ValueError, match="different content"):
        market_data.ingest(
            contract,
            write_events(tmp_path / "conflict.csv", contract, [conflict]),
            request_id="conflict",
        )
    relabel = events[1] | {"event_id": "relabeled"}
    with pytest.raises(ValueError, match="relabel"):
        market_data.ingest(
            contract,
            write_events(tmp_path / "relabel.csv", contract, [relabel]),
            request_id="relabel",
        )
    root = market_data.market_root()
    before = {str(p): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    market_data.inspect_data(duplicate["snapshot_ref"], replay=True)
    after = {str(p): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    assert before == after


def test_receipt_clock_is_sampled_after_lock_and_never_backdated(imported, monkeypatch):
    contract, _, _, _, source = imported
    real_context, nested = market_data.market_context, []
    clock = ["2026-10-24T12:00:01Z"]
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: clock[0])

    @contextmanager
    def interleaving(**kwargs):
        if not nested:
            nested.append(True)
            clock[0] = "2026-10-24T12:00:02Z"
            nested.append(market_data.ingest(contract, source, request_id="wins-lock"))
            clock[0] = "2026-10-24T12:00:03Z"
        with real_context(**kwargs) as context:
            yield context

    monkeypatch.setattr(market_data, "market_context", interleaving)
    report = market_data.ingest(contract, source, request_id="delayed-before-lock")
    assert report["first_seen_at"] == "2026-10-24T12:00:03Z"
    assert report["receipt_seq"] == nested[1]["receipt_seq"] + 1
    market_data.inspect_data(report["snapshot_ref"], replay=True)
    clock[0] = "2026-10-24T12:00:01Z"
    assert market_data.ingest(contract, source, request_id="delayed-before-lock") == report
    with pytest.raises(ValueError, match="clock moved backwards"):
        market_data.ingest(contract, source, request_id="clock-regression")


def test_missing_registered_ledger_is_not_recreated(imported):
    _, _, report, _, _ = imported
    database = market_data.market_root() / "data.sqlite"
    database.unlink()
    with pytest.raises(Exception, match="unable to open"):
        market_data.inspect_data(report["snapshot_ref"])
    assert not database.exists()


def test_prediction_only_does_not_require_fabricated_financial_rules():
    value = dict(
        schema_version=1,
        mode="registered_baseline_observer",
        purpose="developer_visible_observation",
        model_bundle_ref="a" * 64,
        contract_ref="b" * 64,
        account_mode="prediction_only",
        max_information_age_seconds=86400,
        max_model_age_seconds=86400,
        expires_at="2026-10-25T12:00:00Z",
    )
    plan = PredictionOnlyPlan.model_validate_json(canonical(value))
    assert not hasattr(plan, "initial_cash")
    with pytest.raises(ValueError, match="Extra inputs"):
        PredictionOnlyPlan.model_validate_json(canonical(value | {"initial_cash": "1000"}))


def test_grid_retains_outside_unknown_and_new_members(imported, tmp_path):
    _, _, _, _, _ = imported
    contract, events, evidence = contract_and_events(instruments=("A", "B", "C", "D"))
    data = contract.model_dump(mode="json") | {"source_id": "mask"}
    contract = market_contracts.DataContract.model_validate_json(canonical(data))
    events = [
        event
        for event in events
        if not (event["instrument"] == "C" and event["kind"] == "UNIVERSE_STATUS")
    ]
    for event in events:
        if event["event_id"] == "B.membership":
            event.update(
                event_at="2020-10-21T09:00:00Z", source_available_at="2020-10-21T09:00:00Z"
            )
        elif event["event_id"] == "D.membership":
            event["status"] = "outside_universe"
    report = market_data.ingest(
        contract, write_events(tmp_path / "masks.csv", contract, events), request_id="masks"
    )
    snapshot, contract = market_data.load_snapshot(report["snapshot_ref"])
    view = make_feature_view(snapshot, contract, "s2", "2020-10-21T16:00:00Z", factor())
    assert [(row["instrument"], row["status"]) for row in view["rows"]] == [
        ("A", "eligible"),
        ("B", "ineligible"),
        ("C", "unresolved"),
        ("D", "ineligible"),
    ]
    assert "insufficient_asset_warmup" in view["rows"][1]["reasons"]
    assert "membership_outside_universe" in view["rows"][3]["reasons"]


def test_known_action_excludes_mature_label_and_observed_basis_has_correct_qualification(
    imported, tmp_path
):
    contract, _, _, _, _ = imported
    notice = dict(
        event_id="A.split",
        revision=0,
        kind="CORPORATE_ACTION_NOTICE",
        instrument="A",
        session_id=None,
        event_at="2020-10-23T09:00:00Z",
        source_available_at="2020-10-22T10:00:00Z",
        price=None,
        status=None,
        action="split; unsupported accounting",
        quality="valid",
    )
    report = market_data.ingest(
        contract, write_events(tmp_path / "notice.csv", contract, [notice]), request_id="notice"
    )
    snapshot, contract = market_data.load_snapshot(report["snapshot_ref"])
    label = mature_label(snapshot, contract, "s2", "A", "2020-10-23T09:00:00Z")
    assert label["status"] == "ineligible" and label["value"] is None
    view = make_feature_view(
        snapshot, contract, "s2", "2026-10-24T12:00:00Z", factor(), basis="observed_receipts"
    )
    assert view["rows"][0]["status"] == "unresolved"
    assert "historical_exploration_not_observed_forward" not in view["qualifications"]
