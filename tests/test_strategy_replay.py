"""Pure account oracles; these invented prices establish no predictive advantage."""

from copy import deepcopy
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from market_helpers import account_rules, contract_and_events

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.execution.strategy_replay import advance_paper_strategy, replay_strategy
from quant_rsi.market.market_contracts import utc_text


def market(*, changes=None):
    contract, events, _ = contract_and_events(count=12)
    for event in events:
        if event["kind"].endswith("QUOTE"):
            event["price"] = "100"
    if changes:
        changes(events)
    rows = [
        dict(
            event=event,
            payload_ref=digest(canonical(event)),
            first_seen_at=event["source_available_at"],
            receipt_seq=1,
        )
        for event in events
    ]
    snapshot = dict(
        contract_ref=digest(canonical(contract.model_dump(mode="json"))),
        receipt_watermark=1,
        events=rows,
    )
    return SimpleNamespace(
        contract=contract,
        snapshot=snapshot,
        instruments=("A",),
        dates=tuple(s.session_id for s in contract.calendar.sessions),
        decision_at=lambda index: (
            contract.calendar.sessions[index].close_at + timedelta(seconds=60)
        ),
    )


def rules(**kwargs):
    return account_rules(fill_cap_units={"A": 1000}, max_positions=1, **kwargs)


def prediction(data, index=0, score=0.1):
    at = utc_text(data.decision_at(index))
    return dict(
        origin_session=f"s{index}",
        score=score,
        model_ref="a" * 64,
        result_ref="b" * 64,
        feature_as_of=at,
        available_at=at,
        confirmation_at=at,
    )


def historical(data, scores=(0.1,), **kwargs):
    return replay_strategy(
        data,
        [prediction(data, i * 5, score) for i, score in enumerate(scores)],
        rules(**kwargs),
        origin_sessions=[f"s{i * 5}" for i in range(len(scores))],
        end_session=f"s{len(scores) * 5}",
    )


def test_identical_positive_strategy_and_buyhold_have_independent_cash_oracle():
    result = historical(market())
    strategy = result["accounts"]["strategy"]
    buyhold = result["accounts"]["buy_and_hold"]
    for value in (strategy, buyhold):
        assert value["cash"] == "91.00"
        assert value["holdings"]["A"]["quantity"] == 9
        assert value["cumulative_commission"] == "9.00"
        assert value["valuations"][-1]["nav"] == "991.00"
        assert len(value["valuations"]) == 6
        assert value["orders"][0]["target_session"] == "s1"
        assert value["fills"][0]["price"] == "100"
    assert result["metrics"]["strategy"] == result["metrics"]["buy_and_hold"]
    assert Decimal(result["metrics"]["strategy"]["net_return"]) == Decimal("-0.009")
    assert result["metrics"]["strategy"]["turnover"] == "0.90"
    assert result["evidence_mode"] == "historical_development_simulated_issuance"


@pytest.mark.parametrize("score", [-0.1, 0, None])
def test_nonpositive_and_missing_start_in_cash_without_disabling_buyhold(score):
    result = historical(market(), (score,))
    assert result["accounts"]["strategy"]["fills"] == []
    assert Decimal(result["metrics"]["strategy"]["net_return"]) == 0
    assert result["accounts"]["buy_and_hold"]["fills"]


@pytest.mark.parametrize("second", [-0.1, None])
def test_expired_prediction_exits_at_next_open_on_negative_or_missing(second):
    result = historical(market(), (0.1, second))
    state = result["accounts"]["strategy"]
    assert [fill["side"] for fill in state["fills"]] == ["buy", "sell"]
    assert [order["target_session"] for order in state["orders"]] == ["s1", "s6"]
    assert result["decisions"][0]["signal"]["expiry_session"] == "s5"
    assert state["cash"] == "964.00"  # 1000 - 9 buy commission - 27 sell costs.
    assert state["holdings"]["A"]["quantity"] == 0
    assert len(state["valuations"]) == 11
    assert len(result["accounts"]["buy_and_hold"]["fills"]) == 1


def test_omitted_prediction_is_recorded_and_full_grid_is_preserved():
    data = market()
    result = replay_strategy(
        data, [prediction(data)], rules(), origin_sessions=["s0", "s5"], end_session="s10"
    )
    assert result["decisions"][1]["signal"]["reason"] == "missing_prediction"
    assert result["accounts"]["strategy"]["orders"][-1]["target_session"] == "s6"


def test_fill_caps_lots_and_gap_use_existing_affordability():
    data = market()
    result = replay_strategy(
        data,
        [prediction(data)],
        account_rules(fill_cap_units={"A": 4}, lot_size=2),
        origin_sessions=["s0"],
        end_session="s5",
    )
    assert result["accounts"]["strategy"]["holdings"]["A"]["quantity"] == 4
    assert result["accounts"]["strategy"]["cash"] == "596.00"
    assert result["accounts"]["strategy"]["orders"][0]["cancelled_quantity"] == 4


def alter(events, *, session, phase="CLOSE_QUOTE", **updates):
    for event in events:
        if event["session_id"] == session and event["kind"] == phase:
            event.update(updates)


@pytest.mark.parametrize("phase", ["OPEN_QUOTE", "CLOSE_QUOTE"])
def test_missing_or_late_quote_never_drops_a_session_or_restores_full_return(phase):
    data = market(
        changes=lambda events: alter(
            events, session="s1", phase=phase, source_available_at="2030-01-01T00:00:00Z"
        )
    )
    result = historical(data)
    assert len(result["accounts"]["strategy"]["valuations"]) == 6
    assert result["metrics"]["strategy"]["net_return"] is None
    assert result["metrics"]["strategy"]["max_drawdown"] is None
    assert "s1" in result["metrics"]["strategy"]["unresolved_sessions"]
    if phase == "OPEN_QUOTE":
        assert result["accounts"]["strategy"]["fills"] == []
        assert result["accounts"]["strategy"]["orders"][0]["status"] == "pending_resolution"


def test_losing_strategy_reports_loss_and_risk_stop_without_dropping_inventory():
    data = market(changes=lambda events: alter(events, session="s2", price="50"))
    result = historical(data, max_drawdown="0.2")
    metrics = result["metrics"]["strategy"]
    assert Decimal(metrics["max_drawdown"]) == Decimal("-0.459")
    assert "drawdown_limit" in metrics["suspended_reasons"]
    assert metrics["ending_inventory"]["A"]["quantity"] == 9


def test_unresolved_action_preserves_liability_and_suppresses_return():
    data = market()
    event = dict(
        event_id="action",
        revision=0,
        kind="CORPORATE_ACTION_NOTICE",
        instrument="A",
        session_id=None,
        event_at=utc_text(data.decision_at(2)),
        source_available_at=utc_text(data.decision_at(2)),
        price=None,
        status=None,
        action="split",
        quality="valid",
    )
    data.snapshot["events"].append(
        dict(
            event=event,
            payload_ref=digest(canonical(event)),
            first_seen_at=event["event_at"],
            receipt_seq=1,
        )
    )
    result = historical(data)
    assert result["accounts"]["strategy"]["holdings"]["A"]["quantity"] == 9
    assert result["metrics"]["strategy"]["net_return"] is None
    assert result["metrics"]["strategy"]["unresolved_sessions"] == ["s2", "s3", "s4", "s5"]


def test_bad_grid_and_future_feature_information_are_rejected():
    data = market()
    with pytest.raises(ValueError, match="step-five"):
        replay_strategy(data, [], rules(), origin_sessions=["s0", "s4"], end_session="s9")
    row = prediction(data)
    row["feature_as_of"] = utc_text(data.decision_at(1))
    with pytest.raises(ValueError, match="after its origin"):
        replay_strategy(data, [row], rules(), origin_sessions=["s0"], end_session="s5")


def initialize_paper(data):
    row = prediction(data)
    initial = advance_paper_strategy(data, row, rules(), observed_at=row["confirmation_at"])
    assert initial["accounts"]["strategy"]["orders"][0]["status"] == "prepared"
    confirmed = advance_paper_strategy(
        data,
        row,
        rules(),
        previous=initial,
        observed_at=utc_text(data.decision_at(0) + timedelta(seconds=1)),
    )
    assert confirmed["accounts"]["strategy"]["orders"][0]["status"] == "pending_resolution"
    return row, confirmed


def test_paper_timely_submission_fills_only_after_actual_receipt_and_survives_resume():
    data = market()
    row, state = initialize_paper(data)
    open_at = data.contract.calendar.sessions[1].open_at
    for event in data.snapshot["events"]:
        if event["event"]["session_id"] == "s1" and event["event"]["kind"] == "OPEN_QUOTE":
            event["first_seen_at"] = utc_text(open_at + timedelta(seconds=30))
    early = advance_paper_strategy(
        data, row, rules(), previous=state, observed_at=utc_text(open_at)
    )
    assert early["accounts"]["strategy"]["fills"] == []
    filled = advance_paper_strategy(
        data, row, rules(), previous=early, observed_at=utc_text(open_at + timedelta(seconds=30))
    )
    assert len(filled["accounts"]["strategy"]["fills"]) == 1
    repeated = advance_paper_strategy(
        data, row, rules(), previous=filled, observed_at=utc_text(open_at + timedelta(seconds=30))
    )
    assert repeated == filled
    final = advance_paper_strategy(
        data, row, rules(), previous=repeated, observed_at=utc_text(data.decision_at(5))
    )
    assert final["status"] == "complete"
    assert final["metrics"]["strategy"]["net_return"] == "-0.009"
    assert filled["accounts"]["strategy"]["fills"] == final["accounts"]["strategy"]["fills"]


def test_paper_initialization_and_confirmation_cannot_be_backdated():
    data = market()
    row = prediction(data)
    opening = utc_text(data.contract.calendar.sessions[1].open_at)
    late = advance_paper_strategy(data, row, rules(), observed_at=opening)
    assert late["status"] == "not_prospective_late_initialization"
    assert late["accounts"]["strategy"]["orders"] == []
    initial = advance_paper_strategy(data, row, rules(), observed_at=row["confirmation_at"])
    late = advance_paper_strategy(data, row, rules(), observed_at=opening, previous=initial)
    assert (
        late["accounts"]["strategy"]["orders"][0]["cancellation_reason"]
        == "late_durable_submission"
    )
    assert late["accounts"]["strategy"]["fills"] == []


def test_processed_correction_suspends_without_rewriting_fill_or_valuation():
    data = market()
    row, state = initialize_paper(data)
    state = advance_paper_strategy(
        data, row, rules(), previous=state, observed_at=utc_text(data.decision_at(1))
    )
    before = deepcopy(state["accounts"]["strategy"])
    original = next(
        item
        for item in data.snapshot["events"]
        if item["event"]["session_id"] == "s1" and item["event"]["kind"] == "OPEN_QUOTE"
    )
    revised = deepcopy(original)
    revised["event"].update(revision=1, price="200")
    revised["payload_ref"] = digest(canonical(revised["event"]))
    revised["first_seen_at"] = utc_text(data.decision_at(2))
    data.snapshot["events"].append(revised)
    state = advance_paper_strategy(
        data, row, rules(), previous=state, observed_at=utc_text(data.decision_at(2))
    )
    assert state["accounts"]["strategy"]["fills"] == before["fills"]
    assert state["accounts"]["strategy"]["valuations"][:2] == before["valuations"]
    assert any(
        "processed_evidence_corrected" in value
        for value in state["accounts"]["strategy"]["suspended_reasons"]
    )
    assert state["metrics"]["strategy"]["net_return"] is None


def test_paper_future_calendar_can_extend_beyond_historical_input_rows():
    data = market()
    data.dates = data.dates[:1]
    row, state = initialize_paper(data)
    state = advance_paper_strategy(
        data, row, rules(), previous=state, observed_at=utc_text(data.decision_at(5))
    )
    assert state["target_session"] == "s5"
    assert state["status"] == "complete"


def test_paper_missing_open_stays_unresolved_when_receipt_arrives_after_deadline():
    data = market()
    row, state = initialize_paper(data)
    opening = next(
        item
        for item in data.snapshot["events"]
        if item["event"]["session_id"] == "s1" and item["event"]["kind"] == "OPEN_QUOTE"
    )
    opening["first_seen_at"] = utc_text(data.decision_at(2))
    state = advance_paper_strategy(
        data, row, rules(), previous=state, observed_at=utc_text(data.decision_at(1))
    )
    assert state["accounts"]["strategy"]["fills"] == []
    state = advance_paper_strategy(
        data, row, rules(), previous=state, observed_at=utc_text(data.decision_at(5))
    )
    assert state["accounts"]["strategy"]["fills"] == []
    assert state["accounts"]["strategy"]["orders"][0]["status"] == "pending_resolution"
    assert state["metrics"]["strategy"]["net_return"] is None
    assert len(state["accounts"]["strategy"]["valuations"]) == 6


def test_paper_frozen_policy_and_monotonic_clock_cannot_be_replaced():
    data = market()
    row, state = initialize_paper(data)
    with pytest.raises(ValueError, match="frozen prediction"):
        advance_paper_strategy(
            data,
            row,
            rules(),
            previous=state,
            threshold="0.05",
            observed_at=utc_text(data.decision_at(1)),
        )
    with pytest.raises(ValueError, match="moved backwards"):
        advance_paper_strategy(data, row, rules(), previous=state, observed_at=row["available_at"])


def test_paper_missing_close_cannot_disappear_after_a_later_complete_mark():
    data = market()
    row, state = initialize_paper(data)
    for item in data.snapshot["events"]:
        if item["event"]["session_id"] == "s2" and item["event"]["kind"] == "CLOSE_QUOTE":
            item["first_seen_at"] = "2030-01-01T00:00:00Z"
    state = advance_paper_strategy(
        data, row, rules(), previous=state, observed_at=utc_text(data.decision_at(5))
    )
    assert state["accounts"]["strategy"]["valuations"][2]["nav"] is None
    assert state["accounts"]["strategy"]["valuations"][-1]["nav"] == "991.00"
    assert state["metrics"]["strategy"]["net_return"] is None
