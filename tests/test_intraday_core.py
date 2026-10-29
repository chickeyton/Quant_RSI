"""Intraday storage, timing, features, model and simulator; synthetic fixtures only."""

import math
from decimal import Decimal

import numpy as np
import pytest
from intraday_helpers import build_snapshot, contract, plan, sessions
from pydantic import ValidationError

from quant_rsi.intraday.contracts import ExecutionTiming, validate_timing
from quant_rsi.intraday.evaluation import evaluate_development
from quant_rsi.intraday.features import build_features, default_leaves, session_features
from quant_rsi.intraday.model import walk_forward_folds
from quant_rsi.intraday.simulator import commission, simulate
from quant_rsi.intraday.storage import MISSING, VALID, Block, SnapshotStore, encode_column
from quant_rsi.intraday.timing import (
    NO_INDEX,
    entry_allowed,
    fill_bar_start,
    fill_index,
    halted,
    labels,
    session_timing,
)


# Storage ------------------------------------------------------------------------------------
def test_identical_data_gives_identical_snapshot_digest(tmp_path):
    _, first = build_snapshot(tmp_path / "a", count=3)
    _, second = build_snapshot(tmp_path / "b", count=3)
    assert first == second


def test_snapshot_round_trip_and_integrity(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    snapshot = store.load(ref)
    block = snapshot.block("KO", "decision", snapshot.contract.calendar.sessions[0].session_id)
    assert len(block) == 78 and block.columns["status"].dtype == np.uint8
    path = next((tmp_path / "store" / "objects").rglob("*"))
    while path.is_dir():
        path = next(path.iterdir())
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        SnapshotStore(tmp_path / "store").load(ref)


def test_column_encoding_is_canonical_little_endian():
    assert encode_column(np.array([1], dtype=">i8"), "<i8") == (1).to_bytes(8, "little")


def test_missing_bars_cannot_carry_prices():
    with pytest.raises(ValueError, match="fill-forward"):
        Block(
            "KO",
            "execution_quotes",
            "s",
            {
                "start": np.array([1]),
                "bid": np.array([1.0]),
                "ask": np.array([1.1]),
                "status": np.array([MISSING], np.uint8),
            },
        )


# Timing -------------------------------------------------------------------------------------
def test_decision_latency_must_cover_availability_buffer():
    spec = contract(sessions(3), buffer=5)
    with pytest.raises(ValueError, match="decision_latency_seconds"):
        validate_timing(
            spec,
            ExecutionTiming(
                decision_latency_seconds=2, order_latency_seconds=1, max_fill_delay_seconds=60
            ),
        )


def test_fill_bar_never_starts_before_exec_time():
    session = sessions(1)[0]
    timing = session_timing(
        session,
        300,
        ExecutionTiming(
            decision_latency_seconds=10, order_latency_seconds=1, max_fill_delay_seconds=120
        ),
        900,
    )
    assert (timing.fill_start[:-1] >= timing.exec_time[:-1]).all()
    # Bar t ends at the next bar's start; the fill is the next minute boundary after latency.
    assert timing.fill_start[0] == int(session.open_at.timestamp()) + 360
    assert fill_bar_start(session, int(session.close_at.timestamp()) + 1) == NO_INDEX


def test_labels_use_the_same_price_rule_as_fills(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    snapshot = store.load(ref)
    session = snapshot.contract.calendar.sessions[0]
    execution = snapshot.block("KO", "execution", session.session_id)
    timing = session_timing(session, 300, plan().timing, 900)
    y = labels(execution, timing, 6, 120)
    entry = fill_index(execution, int(timing.fill_start[0]), 120)
    leave = fill_index(execution, int(timing.fill_start[6]), 120)
    opens = execution.columns["open"]
    assert y[0] == pytest.approx(opens[leave] / opens[entry] - 1)
    assert np.isnan(y[-6:]).all()  # labels never cross the session close


def test_early_close_flatten_and_entry_window():
    session = sessions(1, early_close=0)[0]
    timing = session_timing(session, 300, plan().timing, 900)
    assert timing.bars == 42 and timing.flatten_at == int(session.close_at.timestamp()) - 900
    allowed = [t for t in range(timing.bars) if entry_allowed(timing, t, 6)]
    assert allowed and all(timing.fill_start[t + 6] <= timing.flatten_at for t in allowed)


def test_halt_is_causal_and_needs_more_than_n_missing_bars():
    status = np.array([VALID, MISSING, MISSING, MISSING, VALID, MISSING, MISSING, MISSING, MISSING])
    flags = halted(status, 3)
    assert not flags[:8].any() and flags[8]


# Features -----------------------------------------------------------------------------------
def test_features_ignore_future_bars(tmp_path):
    store, ref = build_snapshot(tmp_path, count=25)
    snapshot = store.load(ref)
    leaves = default_leaves(snapshot.contract)
    calendar = snapshot.contract.calendar.sessions
    _, before = session_features(snapshot, calendar, 22, "KO", leaves, last_bar=40)

    class Perturbed:
        contract = snapshot.contract

        def block(self, instrument, grid, session_id):
            block = snapshot.block(instrument, grid, session_id)
            if session_id != calendar[22].session_id or block is None:
                return block
            columns = {k: np.array(v, copy=True) for k, v in block.columns.items()}
            for name in ("open", "high", "low", "close"):
                if name in columns:
                    columns[name][41:] *= 1.5
            if "volume" in columns:
                columns["volume"][41:] *= 7
            return Block(block.instrument, block.grid, block.session_id, columns)

    _, after = session_features(Perturbed(), calendar, 22, "KO", leaves, last_bar=40)
    np.testing.assert_array_equal(before, after)


def test_missing_bar_makes_window_features_missing(tmp_path):
    first = sessions(1)[0].session_id
    store, ref = build_snapshot(tmp_path, count=3, missing={("KO", first): range(25, 30)})
    snapshot = store.load(ref)
    table = build_features(snapshot, ("ret_1", "ret_6", "minutes_since_open"), session_indices=[0])
    row = 5  # decision bar 5 covers minutes 25..29, all missing
    assert math.isnan(table.values[row, 0]) and math.isnan(table.values[row + 3, 1])
    assert table.values[row, 2] == 30.0
    assert table.missing_leaves(row) == ("ret_1", "ret_6")


def test_unknown_leaf_rejected(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    with pytest.raises(ValueError, match="unsupported"):
        build_features(store.load(ref), ("ctx:QQQ:ret_1",))


# Model and folds ----------------------------------------------------------------------------
def test_folds_respect_embargo_and_bounds():
    folds = walk_forward_folds(plan().folds, 40)
    assert folds[0].train_sessions[-1] == 22 and folds[0].test_sessions == (24, 25, 26, 27)
    with pytest.raises(ValueError, match="exceeds"):
        walk_forward_folds(plan(folds={"folds": 5}).folds, 40)


def test_plan_rejects_cross_session_lookback():
    with pytest.raises(ValidationError):
        plan(allow_cross_session_lookback=True)


# Simulator ----------------------------------------------------------------------------------
def test_commission_minimum_and_cap():
    costs = plan().costs
    assert commission(10, Decimal("600"), costs) == Decimal("1.00")
    assert commission(1000, Decimal("60000"), costs) == Decimal("5.00")
    assert commission(10, Decimal("50"), costs) == Decimal("0.50")


def _always_long(snapshot, indices, score=1.0):
    return {("KO", i, t): score for i in indices for t in range(78)}


def test_simulator_reconciles_cash_and_ends_flat(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    snapshot = store.load(ref)
    result = simulate(snapshot, plan(), _always_long(snapshot, [0, 1, 2]), [0, 1, 2])
    cash = Decimal("100000")
    for fill in result.fills:
        delta = Decimal(fill["notional"]) + (
            Decimal(fill["commission"]) if fill["side"] == "buy" else -Decimal(fill["commission"])
        )
        cash += -delta if fill["side"] == "buy" else delta
    assert result.sessions[-1].end_equity == cash
    assert not any(s.unresolved for s in result.sessions)
    metrics = result.metrics()
    assert metrics["round_trips"] > 0 and metrics["prediction_coverage"] == 1.0
    assert all(t["holding_bars"] <= 6 for t in result.round_trips)
    sides = [f["side"] for f in result.fills if f["session_id"] == result.sessions[0].session_id]
    assert sides[-1] == "sell"


def test_no_prediction_opens_nothing_and_lowers_coverage(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    snapshot = store.load(ref)
    result = simulate(snapshot, plan(), {}, [0])
    assert result.fills == [] and result.metrics()["prediction_coverage"] == 0.0


def test_fill_quantity_uses_decision_close_not_fill_price(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    snapshot = store.load(ref)
    result = simulate(snapshot, plan(), _always_long(snapshot, [0]), [0])
    session = snapshot.contract.calendar.sessions[0].session_id
    close0 = Decimal(
        repr(round(float(snapshot.block("KO", "decision", session).columns["close"][0]), 6))
    )
    expected = int(Decimal("20000") / close0)
    assert result.fills[0]["quantity"] == expected


def test_missing_quotes_fall_back_conservatively_and_are_counted(tmp_path):
    store, ref = build_snapshot(tmp_path, count=3)
    snapshot = store.load(ref)

    class NoQuotes:
        contract = snapshot.contract

        def block(self, instrument, grid, session_id):
            return (
                None if grid == "execution_quotes" else snapshot.block(instrument, grid, session_id)
            )

    result = simulate(NoQuotes(), plan(), _always_long(snapshot, [0]), [0])
    assert result.spread_fallbacks == len(result.fills) > 0


def test_halted_target_blocks_new_entries(tmp_path):
    first = sessions(1)[0].session_id
    store, ref = build_snapshot(tmp_path, count=3, missing={("KO", first): range(60, 85)})
    snapshot = store.load(ref)
    result = simulate(snapshot, plan(), _always_long(snapshot, [0]), [0])
    halt_time = int(sessions(1)[0].open_at.timestamp()) + 85 * 60
    assert not [f for f in result.fills if f["side"] == "buy" and f["at"] > halt_time]


# Development evaluation ---------------------------------------------------------------------
def test_development_evaluation_runs_end_to_end(tmp_path):
    store, ref = build_snapshot(tmp_path, count=32)
    report = evaluate_development(store.load(ref), plan())
    assert report["evidence_status"] == "development_only"
    assert len(report["folds"]) == 2
    for fold in report["folds"]:
        assert fold["training_rows"] >= 50 and fold["metrics"]["sessions"] == 4
    repeat = evaluate_development(store.load(ref), plan())
    assert [f["model_digest"] for f in repeat["folds"]] == [
        f["model_digest"] for f in report["folds"]
    ]
