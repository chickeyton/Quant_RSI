from decimal import Decimal

import pytest
from market_helpers import account_rules

from quant_rsi import execution_account as account


def quote(price):
    return dict(price=str(price), payload_ref="a" * 64)


def submit(state, instructions, rules, day):
    state, ids = account.prepare_orders(
        state,
        instructions,
        rules,
        decision_ref=str(day) * 64,
        target_session=f"s{day}",
        target_at=f"2020-10-{day:02d}T09:00:00Z",
        prediction_finished_at=f"2020-10-{day - 1:02d}T16:00:00Z",
    )
    return account.confirm_submission(state, ids, f"2020-10-{day - 1:02d}T16:00:01Z")


def instruction(instrument, side, qty, price):
    return dict(instrument=instrument, side=side, quantity=qty, reference_price=str(price))


def purchased():
    rules = account_rules()
    state = submit(
        account.new_account(rules),
        [instruction("A", "buy", 5, 100), instruction("B", "buy", 10, 50)],
        rules,
        20,
    )
    assert [order["submitted_quantity"] for order in state["orders"]] == [5, 9]
    assert account.reserved_cash(state) == Decimal("959.50")
    state = account.resolve_open(
        state,
        {"A": quote(110), "B": quote(40)},
        rules,
        session_id="s20",
        resolution_at="2020-10-20T09:01:00Z",
    )
    return rules, state


def test_independent_cash_oracle_and_no_end_of_step_liquidation():
    rules, state = purchased()
    assert Decimal(state["cash"]) == Decimal("505.10")
    assert {key: value["quantity"] for key, value in state["holdings"].items()} == {"A": 3, "B": 4}
    assert account.reserved_cash(state) == 0
    state = account.value_close(
        state,
        {"A": quote(120), "B": quote(45)},
        rules,
        session_id="s20",
        close_at="2020-10-20T16:00:00Z",
    )
    assert Decimal(state["valuations"][-1]["nav"]) == Decimal("1045.10")
    assert Decimal(state["valuations"][-1]["return_value"]) == Decimal("0.0451")
    rules = account_rules(fill_cap_units={"A": 2, "B": 4})
    state = submit(
        state, [instruction("A", "sell", 3, 120), instruction("B", "sell", 4, 45)], rules, 21
    )
    state = account.resolve_open(
        state,
        {"A": quote(100), "B": quote(50)},
        rules,
        session_id="s21",
        resolution_at="2020-10-21T09:01:00Z",
    )
    assert Decimal(state["cash"]) == Decimal("893.10")
    assert state["holdings"]["A"]["quantity"] == 1
    state = account.value_close(
        state, {"A": quote(90)}, rules, session_id="s21", close_at="2020-10-21T16:00:00Z"
    )
    assert Decimal(state["valuations"][-1]["nav"]) == Decimal("983.10")
    assert float(state["valuations"][-1]["return_value"]) == pytest.approx(-620 / 10451)
    preserved = account.stop_account(state, "2020-10-21T16:01:00Z")
    assert preserved["cash"] == state["cash"] and preserved["holdings"] == state["holdings"]
    exit_state = submit(state, [instruction("A", "sell", 1, 90)], rules, 22)
    exit_state = account.resolve_open(
        exit_state, {"A": quote(85)}, rules, session_id="s22", resolution_at="2020-10-22T09:01:00Z"
    )
    assert Decimal(exit_state["cash"]) == Decimal("975.55")
    assert sum(
        Decimal(exit_state["cumulative_" + key]) for key in ("commission", "tax", "slippage")
    ) == Decimal("19.45")
    assert Decimal(exit_state["realized_gross_pnl"]) == Decimal("-5")


def test_sale_proceeds_cannot_fund_original_buy_submission():
    rules = account_rules()
    state = account.new_account(rules)
    state.update(cash="0", holdings={"A": dict(quantity=10, cost_basis="1000")})
    state = submit(
        state, [instruction("A", "sell", 10, 100), instruction("B", "buy", 10, 100)], rules, 20
    )
    assert [(order["side"], order["instrument"]) for order in state["orders"]] == [("sell", "A")]
    state = account.resolve_open(
        state,
        {"A": quote(100), "B": quote(100)},
        rules,
        session_id="s20",
        resolution_at="2020-10-20T09:01:00Z",
    )
    assert "B" not in state["holdings"]


def test_missing_open_preserves_reservation_and_blocks_dependent_orders():
    rules = account_rules()
    state = submit(account.new_account(rules), [instruction("A", "buy", 5, 100)], rules, 20)
    delayed = account.resolve_open(
        state, {}, rules, session_id="s20", resolution_at="2020-10-20T09:01:00Z"
    )
    assert delayed == state
    stopped = account.stop_account(delayed, "2020-10-20T10:00:00Z")
    assert stopped["orders"][0]["status"] == "pending_resolution"
    assert account.reserved_cash(stopped) == Decimal("505")
    with pytest.raises(ValueError, match="unresolved orders"):
        submit(stopped, [instruction("B", "buy", 1, 50)], rules, 21)
    resolved = account.resolve_open(
        stopped, {"A": quote(110)}, rules, session_id="s20", resolution_at="2020-10-21T10:00:00Z"
    )
    assert resolved["orders"][0]["filled_quantity"] == 3
    assert (
        account.resolve_open(
            resolved,
            {"A": quote(111)},
            rules,
            session_id="s20",
            resolution_at="2020-10-21T11:00:00Z",
        )
        == resolved
    )


def test_lost_or_late_confirmation_never_becomes_timely():
    rules = account_rules()
    prepared, ids = account.prepare_orders(
        account.new_account(rules),
        [instruction("A", "buy", 5, 100)],
        rules,
        decision_ref="a" * 64,
        target_session="s20",
        target_at="2020-10-20T09:00:00Z",
        prediction_finished_at="2020-10-19T16:00:00Z",
    )
    abandoned = account.abandon_unconfirmed(prepared)
    assert account.reserved_cash(abandoned) == 0
    assert abandoned["orders"][0]["cancellation_reason"] == "missing_post_commit_confirmation"
    late = account.confirm_submission(prepared, ids, "2020-10-20T09:00:00Z")
    assert late["orders"][0]["cancellation_reason"] == "late_durable_submission"
    timely = account.confirm_submission(prepared, ids, "2020-10-20T08:59:59Z")
    stopped = account.stop_account(timely, "2020-10-20T08:59:59.5Z")
    assert stopped["orders"][0]["cancellation_reason"] == "stop_cancel_unelapsed_target"


def test_unknown_mark_or_action_preserves_inventory_and_breaks_return_continuity():
    rules, state = purchased()
    unresolved = account.value_close(
        state, {"A": quote(120)}, rules, session_id="s20", close_at="2020-10-20T16:00:00Z"
    )
    assert unresolved["valuations"][-1]["nav"] is None
    assert unresolved["holdings"] == state["holdings"]
    fresh = account.value_close(
        unresolved,
        {"A": quote(120), "B": quote(45)},
        rules,
        session_id="s21",
        close_at="2020-10-21T16:00:00Z",
    )
    assert fresh["valuations"][-1]["return_value"] is None
    notice = account.apply_action_notice(fresh, "A", "b" * 64)
    blocked = account.value_close(
        notice,
        {"A": quote(60), "B": quote(45)},
        rules,
        session_id="s22",
        close_at="2020-10-22T16:00:00Z",
    )
    assert blocked["valuations"][-1]["nav"] is None
    assert blocked["cash"] == state["cash"] and blocked["holdings"] == state["holdings"]
    assert blocked["cumulative_commission"] == state["cumulative_commission"]


@pytest.mark.parametrize("status", ["ineligible", "unresolved", "eligible"])
def test_retained_inventory_uses_risk_budget_and_slots_before_any_sell_fills(status):
    rules = account_rules(allocation="0.5", max_positions=1)
    state = account.new_account(rules)
    state.update(cash="500", holdings={"A": dict(quantity=5, cost_basis="500")})
    prices = {"A": quote(100), "B": quote(100)}
    state = account.value_close(
        state, prices, rules, session_id="s20", close_at="2020-10-20T16:00:00Z"
    )
    prediction = dict(
        rows=[
            dict(instrument="A", status=status, score=0 if status == "eligible" else None),
            dict(instrument="B", status="eligible", score=1),
        ]
    )
    instructions = account.rebalance_instructions(state, prediction, prices, rules)
    assert not any(item["side"] == "buy" for item in instructions)
    assert state["holdings"]["A"]["quantity"] == 5
    if status == "eligible":
        assert instructions == [instruction("A", "sell", 5, 100)]


def test_retained_inventory_leaves_only_actual_unoccupied_capacity():
    rules = account_rules(allocation="0.5", max_positions=2)
    state = account.new_account(rules)
    state.update(cash="800", holdings={"A": dict(quantity=2, cost_basis="200")})
    prices = {"A": quote(100), "B": quote(100)}
    state = account.value_close(
        state, prices, rules, session_id="s20", close_at="2020-10-20T16:00:00Z"
    )
    prediction = dict(
        rows=[
            dict(instrument="A", status="ineligible", score=None),
            dict(instrument="B", status="eligible", score=1),
        ]
    )
    assert account.rebalance_instructions(state, prediction, prices, rules) == [
        instruction("B", "buy", 3, 100)
    ]


def test_price_gap_with_no_affordable_lot_cancels_without_fake_zero_fill():
    rules = account_rules()
    state = submit(account.new_account(rules), [instruction("A", "buy", 5, 100)], rules, 20)
    resolved = account.resolve_open(
        state, {"A": quote(1000)}, rules, session_id="s20", resolution_at="2020-10-20T09:01:00Z"
    )
    assert resolved["fills"] == [] and resolved["holdings"] == {}
    assert Decimal(resolved["cash"]) == Decimal("1000") and account.reserved_cash(resolved) == 0
    assert resolved["orders"][0]["cancelled_quantity"] == 5
