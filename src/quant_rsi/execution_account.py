"""Pure finite accounting shared by local research and the durable observer.

Prices are raw, fees are explicit, and unresolved inputs preserve liabilities.
The controller owns clocks, persistence, source eligibility and submission proof.
"""

from copy import deepcopy
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal

from quant_rsi.artifacts import canonical, digest
from quant_rsi.market_contracts import decimal_text, timestamp

ACTIVE = {"prepared", "pending_resolution"}


def money(value, rules):
    return Decimal(value).quantize(
        Decimal(1).scaleb(-rules.currency_precision), rounding=ROUND_HALF_EVEN
    )


def amount(value):
    return format(value, "f")


def new_account(rules):
    return dict(
        cash=rules.initial_cash,
        holdings={},
        orders=[],
        fills=[],
        valuations=[],
        cumulative_commission="0",
        cumulative_tax="0",
        cumulative_slippage="0",
        realized_gross_pnl="0",
        last_nav=rules.initial_cash,
        high_water=rules.initial_cash,
        return_continuity=True,
        suspended_reasons=[],
        action_instruments=[],
        stop_receipt=None,
    )


def transaction_cost(side, quantity, price, rules):
    if side not in {"buy", "sell"} or type(quantity) is not int or quantity < 0:
        raise ValueError("invalid derived transaction")
    raw = Decimal(decimal_text(price, positive=True)) * quantity
    costs = rules.buy_costs if side == "buy" else rules.sell_costs
    slip = Decimal(costs.slippage_bps) / 10000
    gross = money(raw * (1 + slip if side == "buy" else 1 - slip), rules)
    commission = money(gross * Decimal(costs.commission_bps) / 10000, rules)
    tax = money(gross * Decimal(costs.tax_bps) / 10000, rules)
    return dict(
        raw_notional=money(raw, rules),
        notional=gross,
        commission=commission,
        tax=tax,
        slippage=abs(gross - money(raw, rules)),
        cash_amount=gross + commission + tax if side == "buy" else gross - commission - tax,
    )


def reserved_cash(state):
    return sum(
        (Decimal(order["reserved_cash"]) for order in state["orders"] if order["status"] in ACTIVE),
        Decimal(0),
    )


def affordable(quantity, price, limit, rules):
    """Monotone binary search preserves lots without per-share loops."""
    low, high = 0, quantity // rules.lot_size
    while low < high:
        middle = (low + high + 1) // 2
        if transaction_cost("buy", middle * rules.lot_size, price, rules)["cash_amount"] <= limit:
            low = middle
        else:
            high = middle - 1
    return low * rules.lot_size


def prepare_orders(
    state, instructions, rules, *, decision_ref, target_session, target_at, prediction_finished_at
):
    result = deepcopy(state)
    if any(order["status"] in ACTIVE for order in result["orders"]):
        raise ValueError("unresolved orders block dependent account trading")
    if timestamp(prediction_finished_at) >= timestamp(target_at):
        return result, []
    keys = [(item["instrument"], item["side"]) for item in instructions]
    if len(set(keys)) != len(keys) or len({key[0] for key in keys}) != len(keys):
        raise ValueError("one derived instruction per instrument is required")
    identifiers = []
    for item in sorted(instructions, key=lambda x: (x["side"] != "sell", x["instrument"])):
        instrument, side, quantity = item["instrument"], item["side"], item["quantity"]
        if (
            side not in {"buy", "sell"}
            or type(quantity) is not int
            or quantity < 0
            or quantity % rules.lot_size
            or instrument not in rules.fill_cap_units
        ):
            raise ValueError("unsupported order identity, quantity or fill cap")
        if instrument in state["action_instruments"]:
            continue
        if side == "buy":
            free = Decimal(result["cash"]) - reserved_cash(result)
            quantity = affordable(quantity, item["reference_price"], free, rules)
            reservation = transaction_cost(side, quantity, item["reference_price"], rules)[
                "cash_amount"
            ]
        else:
            quantity = min(quantity, result["holdings"].get(instrument, {}).get("quantity", 0))
            reservation = Decimal(0)
        if not quantity:
            continue
        order = dict(
            instrument=instrument,
            side=side,
            submitted_quantity=quantity,
            requested_quantity=item["quantity"],
            reference_price=item["reference_price"],
            decision_ref=decision_ref,
            target_session=target_session,
            target_at=target_at,
            prediction_finished_at=prediction_finished_at,
            submission_confirmed_at=None,
            reserved_cash=amount(reservation),
            status="prepared",
            filled_quantity=0,
            cancelled_quantity=0,
            cancellation_reason=None,
        )
        order["order_id"] = digest(canonical(order))
        if any(previous["order_id"] == order["order_id"] for previous in result["orders"]):
            raise ValueError("decision has already submitted this order")
        result["orders"].append(order)
        identifiers.append(order["order_id"])
    return result, identifiers


def confirm_submission(state, identifiers, post_commit_witness):
    result = deepcopy(state)
    selected = [order for order in result["orders"] if order["order_id"] in identifiers]
    if len(selected) != len(set(identifiers)):
        raise ValueError("submission confirmation references unknown orders")
    for order in selected:
        if order["status"] != "prepared":
            raise ValueError("order is not awaiting durable submission confirmation")
        if timestamp(post_commit_witness) < timestamp(order["prediction_finished_at"]):
            raise ValueError("trusted submission clock moved backwards")
        if timestamp(post_commit_witness) >= timestamp(order["target_at"]):
            cancel_order(order, "late_durable_submission")
        else:
            order["submission_confirmed_at"] = post_commit_witness
            order["status"] = "pending_resolution"
    return result


def cancel_order(order, reason):
    order.update(
        status="cancelled",
        cancellation_reason=reason,
        reserved_cash="0",
        cancelled_quantity=order["submitted_quantity"] - order["filled_quantity"],
    )


def abandon_unconfirmed(state):
    result = deepcopy(state)
    for order in result["orders"]:
        if order["status"] == "prepared":
            cancel_order(order, "missing_post_commit_confirmation")
    return result


def stop_account(state, received_at):
    result = deepcopy(state)
    if result["stop_receipt"] is not None:
        return result
    result["stop_receipt"] = received_at
    for order in result["orders"]:
        if order["status"] == "prepared":
            cancel_order(order, "missing_post_commit_confirmation")
        elif order["status"] == "pending_resolution" and timestamp(order["target_at"]) > timestamp(
            received_at
        ):
            cancel_order(order, "stop_cancel_unelapsed_target")
    return result


def apply_action_notice(state, instrument, notice_ref):
    result = deepcopy(state)
    if instrument not in result["action_instruments"]:
        result["action_instruments"].append(instrument)
        result["action_instruments"].sort()
    reason = "unsupported_action:" + instrument + ":" + notice_ref
    if reason not in result["suspended_reasons"]:
        result["suspended_reasons"].append(reason)
    return result


def resolve_open(state, quotes, rules, *, session_id, resolution_at):
    """Only known, eligible raw opens resolve an already timely standing order."""
    result = deepcopy(state)
    orders = sorted(
        result["orders"], key=lambda order: (order["side"] != "sell", order["instrument"])
    )
    for order in orders:
        if order["status"] != "pending_resolution" or order["target_session"] != session_id:
            continue
        if timestamp(resolution_at) < timestamp(order["target_at"]):
            raise ValueError("cannot resolve an open before its market event")
        if order["submission_confirmed_at"] is None or timestamp(
            order["submission_confirmed_at"]
        ) >= timestamp(order["target_at"]):
            raise ValueError("standing order lacks timely durable submission proof")
        instrument, side = order["instrument"], order["side"]
        quote = quotes.get(instrument)
        if instrument in result["action_instruments"] or quote is None:
            continue
        price = quote["price"]
        quantity = min(order["submitted_quantity"], rules.fill_cap_units[instrument])
        if side == "buy":
            limit = min(Decimal(order["reserved_cash"]), Decimal(result["cash"]))
            quantity = affordable(quantity, price, limit, rules)
        else:
            quantity = min(quantity, result["holdings"].get(instrument, {}).get("quantity", 0))
        if not quantity:
            order.update(
                status="resolved",
                filled_quantity=0,
                reserved_cash="0",
                cancelled_quantity=order["submitted_quantity"],
                cancellation_reason="target_resolved_without_affordable_or_available_units",
            )
            continue
        costs = transaction_cost(side, quantity, price, rules)
        position = result["holdings"].setdefault(instrument, dict(quantity=0, cost_basis="0"))
        if side == "buy":
            result["cash"] = amount(Decimal(result["cash"]) - costs["cash_amount"])
            position["quantity"] += quantity
            position["cost_basis"] = amount(Decimal(position["cost_basis"]) + costs["raw_notional"])
        else:
            basis = (
                money(Decimal(position["cost_basis"]) * quantity / position["quantity"], rules)
                if position["quantity"]
                else Decimal(0)
            )
            position["quantity"] -= quantity
            position["cost_basis"] = amount(Decimal(position["cost_basis"]) - basis)
            result["cash"] = amount(Decimal(result["cash"]) + costs["cash_amount"])
            result["realized_gross_pnl"] = amount(
                Decimal(result["realized_gross_pnl"]) + costs["raw_notional"] - basis
            )
        for component in ("commission", "tax", "slippage"):
            key = "cumulative_" + component
            result[key] = amount(Decimal(result[key]) + costs[component])
        order.update(
            status="resolved",
            filled_quantity=quantity,
            reserved_cash="0",
            cancelled_quantity=order["submitted_quantity"] - quantity,
            cancellation_reason="target_resolved_remainder_cancelled"
            if quantity < order["submitted_quantity"]
            else None,
        )
        result["fills"].append(
            dict(
                order_id=order["order_id"],
                instrument=instrument,
                side=side,
                quantity=quantity,
                price=price,
                quote_ref=quote["payload_ref"],
                resolved_at=resolution_at,
                **{key: amount(value) for key, value in costs.items()},
            )
        )
    if Decimal(result["cash"]) < 0 or reserved_cash(result) > Decimal(result["cash"]):
        raise ValueError("cash/reservation invariant violated")
    return result


def value_close(state, quotes, rules, *, session_id, close_at):
    result = deepcopy(state)
    if any(row["session_id"] == session_id for row in result["valuations"]):
        raise ValueError("close was already valued; revisions cannot rewrite past conclusions")
    reasons = []
    marks = {}
    for instrument, holding in result["holdings"].items():
        if not holding["quantity"]:
            continue
        quote = quotes.get(instrument)
        if instrument in result["action_instruments"]:
            reasons.append("unsupported_action:" + instrument)
        elif quote is None:
            reasons.append("missing_fresh_close:" + instrument)
        else:
            marks[instrument] = quote
    if any(
        order["status"] in ACTIVE and timestamp(order["target_at"]) <= timestamp(close_at)
        for order in result["orders"]
    ):
        reasons.append("unresolved_elapsed_order")
    nav = (
        None
        if reasons
        else money(
            Decimal(result["cash"])
            + sum(
                (
                    Decimal(quote["price"]) * result["holdings"][instrument]["quantity"]
                    for instrument, quote in marks.items()
                ),
                Decimal(0),
            ),
            rules,
        )
    )
    previous = Decimal(result["last_nav"]) if result["last_nav"] is not None else None
    daily = (
        None
        if nav is None or previous is None or not result["return_continuity"]
        else (nav / previous - 1 if previous > 0 else Decimal(0))
    )
    drawdown = None
    if nav is not None:
        high = max(Decimal(result["high_water"]), nav)
        result["high_water"] = amount(high)
        drawdown = nav / high - 1
        if (
            drawdown <= -Decimal(rules.max_drawdown)
            and "drawdown_limit" not in result["suspended_reasons"]
        ):
            result["suspended_reasons"].append("drawdown_limit")
    result["last_nav"] = None if nav is None else amount(nav)
    result["return_continuity"] = nav is not None
    result["valuations"].append(
        dict(
            session_id=session_id,
            close_at=close_at,
            status="unresolved" if reasons else "complete",
            nav=None if nav is None else amount(nav),
            return_value=None if daily is None else amount(daily),
            drawdown=None if drawdown is None else amount(drawdown),
            reasons=reasons,
            marks=marks,
            benchmark="cash_zero_interest",
        )
    )
    return result


def rebalance_instructions(state, prediction, quotes, rules):
    if not state["valuations"] or state["valuations"][-1]["status"] != "complete":
        return []
    if state["suspended_reasons"] or state["stop_receipt"] is not None:
        return []
    eligible = {row["instrument"] for row in prediction["rows"] if row["status"] == "eligible"}
    held = {key: position for key, position in state["holdings"].items() if position["quantity"]}
    if any(instrument not in quotes for instrument in held):
        return []
    held_values = {
        key: Decimal(quotes[key]["price"]) * position["quantity"] for key, position in held.items()
    }
    retained = {key for key in held if key not in eligible}
    budget = Decimal(state["valuations"][-1]["nav"]) * Decimal(rules.allocation)
    retained_value = sum((held_values[key] for key in retained), Decimal(0))
    target_budget = max(Decimal(0), budget - retained_value)
    # Submitted sells have not freed exposure or position slots yet.
    available_buy_exposure = max(Decimal(0), budget - sum(held_values.values(), Decimal(0)))
    available_new_slots = max(0, rules.max_positions - len(held))
    ranked = sorted(
        (
            row
            for row in prediction["rows"]
            if row["status"] == "eligible"
            and row["score"] is not None
            and row["instrument"] in quotes
        ),
        key=lambda row: (-row["score"], row["instrument"]),
    )[: max(0, rules.max_positions - len(retained))]
    selected = {row["instrument"] for row in ranked}
    per_asset = target_budget / max(1, len(selected))
    result = []
    for instrument in sorted(eligible):
        if instrument not in quotes:
            continue
        price = quotes[instrument]["price"]
        target = (
            int(
                (per_asset / Decimal(price) / rules.lot_size).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
            * rules.lot_size
            if instrument in selected
            else 0
        )
        current = state["holdings"].get(instrument, {}).get("quantity", 0)
        if target > current:
            if not current and not available_new_slots:
                continue
            allowed = (
                int(
                    (available_buy_exposure / Decimal(price) / rules.lot_size).to_integral_value(
                        rounding=ROUND_FLOOR
                    )
                )
                * rules.lot_size
            )
            target = current + min(target - current, allowed)
            available_buy_exposure -= Decimal(price) * (target - current)
            if target > current and not current:
                available_new_slots -= 1
        if target != current:
            result.append(
                dict(
                    instrument=instrument,
                    side="buy" if target > current else "sell",
                    quantity=abs(target - current),
                    reference_price=price,
                )
            )
    return result
