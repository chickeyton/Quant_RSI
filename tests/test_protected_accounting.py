"""Independent cash-flow oracles, not implementation-derived expected values."""

import numpy as np
import pytest

from quant_rsi.accounting import evaluate, returns_from_equities
from quant_rsi.data import MarketData, synthetic_data
from quant_rsi.promotion import TestRegistry
from quant_rsi.schemas import DataSpec, PortfolioSpec


def fixture_inputs():
    source = synthetic_data(
        DataSpec(source="synthetic", sessions=122, instruments=1, train_cutoff=40, seed=17)
    )
    data = MarketData(
        source.dates, source.instruments, np.full((122, 1), 100.0), np.full((122, 1), 100.0)
    )
    spec = PortfolioSpec(
        kind="long_only_topk",
        initial_cash=1000.0,
        max_positions=1,
        allocation=1.0,
        fee_bps=100.0,
        slippage_bps=0.0,
        max_drawdown=1.0,
        benchmark="cash_zero_interest",
        terminal_policy="liquidate_at_final_open",
    )
    predictions = [
        {
            "decision": t,
            "decision_time": data.timestamp(t, "close"),
            "instrument": data.instruments[0],
            "score": 1.0,
        }
        for t in range(80, 120)
    ]
    return data, spec, predictions


def test_terminal_cost_and_report_only_subdivision():
    data, spec, predictions = fixture_inputs()
    result = evaluate(data, 80, predictions, spec, 252)
    returns = result["net_returns"]
    assert len(returns) == 40
    assert result["fills"][0]["quantity"] == 9
    assert result["fills"][-1]["terminal"] is True
    assert result["fills"][-1]["execution_time"] == data.timestamp(121, "open")
    assert result["metrics"]["fees"] == 18.0
    assert result["metrics"]["final_cash"] == 982.0
    assert returns[0] == pytest.approx(-9 / 1000)
    assert returns[1:39] == [0.0] * 38
    assert returns[-1] == pytest.approx(-9 / 991)
    expected = -1128897 / 9910000
    assert result["metrics"]["annualized_mean_net_active_return"] == pytest.approx(expected)
    # Slicing the report leaves one account path, one final liquidation, same mass.
    split = 0.25 * 252 * np.mean(returns[:10]) + 0.75 * 252 * np.mean(returns[10:])
    assert split == pytest.approx(expected)


def test_registered_total_loss_runs_actual_kernel_and_absorbs():
    data, spec, predictions = fixture_inputs()
    result = evaluate(data, 80, predictions, spec, 252, total_loss_at=82)
    assert result["net_returns"] == [-1.0] + [0.0] * 39
    assert result["metrics"]["annualized_mean_net_active_return"] == pytest.approx(-6.3)
    assert result["metrics"]["final_cash"] == 0.0
    assert result["metrics"]["fees"] == 9.0
    assert result["registered_total_loss_events"][0]["equity_destroyed"] == 991.0
    assert all(s["equity"] == 0 and s["holdings"] == [0] for s in result["snapshots"])
    with pytest.raises(ValueError):
        evaluate(data, 80, predictions, spec, 252, total_loss_at=122)
    with pytest.raises(ValueError):
        returns_from_equities(1000.0, [0.0, 1.0], registered_zero_equity=True)


def test_zero_trade_grid_and_missing_terminal_quote():
    data, spec, predictions = fixture_inputs()
    result = evaluate(data, 80, predictions, spec.model_copy(update={"initial_cash": 1.0}), 252)
    assert result["net_returns"] == [0.0] * 40
    assert result["metrics"]["filled_trades"] == 0
    broken = data.to_json()
    broken["opens"][-1][0] = None
    with pytest.raises(ValueError, match="finite quotes"):
        MarketData.from_json(broken)
    with pytest.raises(ValueError, match="valuation"):
        returns_from_equities(1000.0, [991.0] * 39 + [None], registered_zero_equity=True)


def binding(package="A", revision=0, candidate="B", report="report1"):
    return {
        "expected_package_ref": package,
        "expected_revision": revision,
        "candidate_package_ref": candidate,
        "report_ref": report,
        "plan_ref": "fixture_plan",
    }


@pytest.mark.parametrize("outcome", ["positive", "negative", "inconclusive", "invalid"])
def test_isolated_registry_decision_matrix(outcome):
    registry = TestRegistry("A")
    request = binding()
    result = registry.decide("id", request, outcome)
    assert result["decision"] == ("promote" if outcome == "positive" else "retain")
    assert registry.decide("id", request, outcome) == result
    with pytest.raises(ValueError, match="reused"):
        registry.decide("id", binding(report="changed"), outcome)


def test_registry_aba_revision_and_inflight_reference():
    registry = TestRegistry("A")
    frozen = binding()
    registry.decide("one", frozen, "positive")
    registry.decide(
        "two", binding(package="B", revision=1, candidate="A", report="report2"), "positive"
    )
    assert registry.state["registry"]["package_ref"] == "A"
    assert registry.state["registry"]["revision"] == 2
    assert frozen == binding()
    with pytest.raises(ValueError, match="CAS"):
        registry.decide("stale", frozen, "positive")
