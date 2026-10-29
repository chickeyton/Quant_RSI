"""Learnable trading policy: shared entry rules and nested calibration; synthetic only."""

# ruff: noqa: F811
import math
from decimal import Decimal

import pytest
from intraday_helpers import build_snapshot, plan
from pydantic import ValidationError
from test_intraday_paper import _register, _run_day, _runner, world  # noqa: F401

from quant_rsi.intraday.calibration import candidates, choose_policy, split_calibration
from quant_rsi.intraday.contracts import SignalPolicy
from quant_rsi.intraday.evaluation import evaluate_development
from quant_rsi.intraday.features import build_features
from quant_rsi.intraday.model import label_vector
from quant_rsi.intraday.policy import entry_quantity, round_trip_cost, signal_digest
from quant_rsi.intraday.simulator import simulate

PRICE = Decimal("60")
EQUITY = Decimal("100000")


def _signal(**overrides) -> SignalPolicy:
    return plan(signal={"entry_threshold": 0.0, "exit_threshold": -1.0, **overrides}).signal


def _search(**overrides):
    return {
        "entry_thresholds": [-1.0, 10.0],
        "exit_thresholds": [-2.0],
        "calibration_sessions": 5,
        "min_calibration_trades": 1,
        **overrides,
    }


def test_round_trip_cost_is_two_commissions_and_one_spread():
    costs = plan().costs
    # 100 shares at 50: per-share 0.50 is below the 1.00 minimum; 2 x 1 / 5000 + 2 bps.
    assert round_trip_cost(costs, Decimal("50"), 100) == pytest.approx(0.0004 + 0.0002)
    assert round_trip_cost(costs, Decimal("50"), 0) == math.inf


def test_default_policy_keeps_the_fixed_threshold_and_full_size():
    signal, costs = _signal(), plan().costs
    assert entry_quantity(signal, costs, 1e-6, PRICE, EQUITY) == 333  # 20000 / 60
    assert entry_quantity(signal, costs, 0.0, PRICE, EQUITY) is None
    assert entry_quantity(signal, costs, 1.0, PRICE, Decimal("100")) == 0  # under one lot


def test_net_of_cost_declines_entries_whose_edge_does_not_cover_costs():
    signal, costs = _signal(entry_rule="net_of_cost"), plan().costs
    cost = round_trip_cost(costs, PRICE, 333)
    assert 0.0003 < cost < 0.0005
    assert entry_quantity(signal, costs, 0.0003, PRICE, EQUITY) is None
    assert entry_quantity(signal, costs, 0.0005, PRICE, EQUITY) == 333


def test_edge_scaled_sizing_grows_with_the_edge_and_declines_tiny_orders():
    signal = _signal(sizing="edge_scaled", full_size_edge=0.002)
    costs = plan().costs
    assert entry_quantity(signal, costs, 0.001, PRICE, EQUITY) == 166  # half the edge
    assert entry_quantity(signal, costs, 0.01, PRICE, EQUITY) == 333  # capped at full size
    assert entry_quantity(signal, costs, 0.000001, PRICE, EQUITY) is None  # < one lot
    netted = _signal(entry_rule="net_of_cost", sizing="edge_scaled", full_size_edge=0.002)
    # A small order pays the minimum commission, so its own hurdle rejects it.
    assert entry_quantity(netted, costs, 0.00042, PRICE, EQUITY) is None


def test_policy_contracts_are_validated():
    with pytest.raises(ValidationError, match="full_size_edge"):
        _signal(sizing="edge_scaled")
    with pytest.raises(ValidationError, match="full_size_edge"):
        _signal(full_size_edge=0.01)
    with pytest.raises(ValidationError, match="exceeds 64"):
        plan(
            policy_search=_search(
                entry_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                exit_thresholds=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
                full_size_edges=[0.1, 0.2],
            ),
            signal={"sizing": "edge_scaled", "full_size_edge": 0.1},
        )
    with pytest.raises(ValidationError, match="no exit threshold"):
        plan(policy_search=_search(entry_thresholds=[0.0], exit_thresholds=[0.1]))
    with pytest.raises(ValidationError, match="edge_scaled"):
        plan(policy_search=_search(full_size_edges=[0.01]))


def test_candidates_skip_inverted_pairs_and_keep_grid_order():
    grid = plan(policy_search=_search(entry_thresholds=[0.0, 0.1], exit_thresholds=[0.05, -0.1]))
    pairs = [(c.entry_threshold, c.exit_threshold) for c in candidates(grid)]
    assert pairs == [(0.0, -0.1), (0.1, 0.05), (0.1, -0.1)]


def test_calibration_split_holds_out_the_last_training_sessions():
    grid = plan(policy_search=_search())
    inner, calibration = split_calibration(grid, tuple(range(23)))
    # Inner sessions 0-16, session 17 is the embargo, sessions 18-22 calibrate.
    assert calibration == (18, 19, 20, 21, 22) and inner == tuple(range(17))
    with pytest.raises(ValueError, match="cannot hold"):
        split_calibration(grid, tuple(range(6)))


def test_simulator_counts_policy_declined_entries(tmp_path):
    store, ref = build_snapshot(tmp_path, count=4)
    snapshot = store.load(ref)
    bars = snapshot.block("KO", "decision", snapshot.contract.calendar.sessions[0].session_id)
    predictions = {("KO", 0, t): 0.0003 for t in range(len(bars))}
    fixed = plan(signal={"entry_threshold": 0.0})
    netted = plan(signal={"entry_threshold": 0.0, "entry_rule": "net_of_cost"})
    trades = simulate(snapshot, fixed, predictions, [0]).metrics()
    declined = simulate(snapshot, netted, predictions, [0]).metrics()
    assert trades["round_trips"] > 0 and trades["policy_declined_entries"] == 0
    assert declined["round_trips"] == 0 and declined["policy_declined_entries"] > 0


def test_calibration_picks_the_best_eligible_candidate(tmp_path):
    store, ref = build_snapshot(tmp_path, count=32)
    snapshot = store.load(ref)
    grid = plan(policy_search=_search())
    table = build_features(snapshot, grid.leaves)
    targets = label_vector(snapshot, table, grid)
    choice = choose_policy(snapshot, grid, table, targets, tuple(range(23)))
    # entry 10.0 never trades, so only the always-enter candidate is eligible.
    assert choice.source == "calibrated" and choice.signal.entry_threshold == -1.0
    assert [row["eligible"] for row in choice.candidates] == [True, False]
    never = plan(policy_search=_search(entry_thresholds=[10.0, 20.0]))
    fallback = choose_policy(snapshot, never, table, targets, tuple(range(23)))
    assert fallback.source == "declared_fallback" and fallback.signal == never.signal


def test_evaluation_learns_each_fold_policy_before_its_test_sessions(tmp_path):
    store, ref = build_snapshot(tmp_path, count=32)
    snapshot = store.load(ref)
    grid = plan(policy_search=_search())
    report = evaluate_development(snapshot, grid)
    for fold in report["folds"]:
        policy = fold["policy"]
        assert policy["source"] == "calibrated" and len(policy["calibration_sessions"]) == 5
        assert max(policy["calibration_sessions"]) < min(fold["test_sessions"])
        assert max(policy["calibration_sessions"]) <= fold["train_sessions"][-1]
    assert evaluate_development(snapshot, grid) == report  # deterministic
    plain = evaluate_development(snapshot, plan())
    assert all(f["policy"]["source"] == "declared" for f in plain["folds"])


def test_paper_runner_applies_the_same_entry_rule(tmp_path, world):
    signal = {**plan().signal.model_dump(), "entry_threshold": 0.0, "exit_threshold": -1.0}
    for rule, expect_entries in (("fixed_threshold", True), ("net_of_cost", False)):
        folder = tmp_path / rule
        runner, _, journal = _runner(folder, world, signal={**signal, "entry_rule": rule})
        runner._score = lambda target, t: (0.0003, ())
        _register(runner)
        _run_day(runner, world[2])
        entries = [r for r in journal.intents("ko-p1") if r["intent"].purpose == "entry"]
        declined = journal.events("ko-p1", "entry_declined")
        assert bool(entries) is expect_entries and bool(declined) is not expect_entries


def test_fit_command_prints_and_writes_the_learned_signal(tmp_path, capsys):
    import json

    from quant_rsi.intraday.__main__ import main

    store, ref = build_snapshot(tmp_path, count=32)
    plan_path, signal_path = tmp_path / "plan.json", tmp_path / "signal.json"
    plan_path.write_text(plan(policy_search=_search()).model_dump_json())
    args = ["fit", "--store", str(tmp_path / "store"), "--snapshot", ref, "--plan"]
    args += [str(plan_path), "--output", str(tmp_path / "model.txt")]
    assert main([*args, "--policy-output", str(signal_path)]) == 0
    fitted = json.loads(capsys.readouterr().out)
    assert fitted["policy"]["source"] == "calibrated"
    assert fitted["policy"]["calibration_sessions"][-1] == "2026-02-17"  # last session
    written = SignalPolicy.model_validate_json(signal_path.read_bytes())
    assert written.model_dump(mode="json") == fitted["signal"]
    assert written.entry_threshold == -1.0


def test_plan_refuses_a_calibration_block_the_first_fold_cannot_hold():
    # First fold: 24 - 1 embargo = 23 training sessions; 22 + 1 embargo leaves no inner set.
    with pytest.raises(ValidationError, match="cannot hold 22 calibration"):
        plan(policy_search=_search(calibration_sessions=22))
    plan(policy_search=_search(calibration_sessions=21))


def test_paper_plan_refuses_a_signal_that_differs_from_its_digest(world):
    from test_intraday_paper import _plan

    snapshot, model, _ = world
    sealed = plan().signal
    edited = {**sealed.model_dump(), "entry_threshold": 0.5}
    with pytest.raises(ValidationError, match="signal_sha256"):
        _plan(snapshot, model, signal=edited, signal_sha256=signal_digest(sealed))
    assert _plan(snapshot, model, signal=edited).signal.entry_threshold == 0.5
