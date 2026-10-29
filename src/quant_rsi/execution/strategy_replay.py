"""Single-target long/cash replay using the shared finite account.

The controller verifies model/result authority and freezes the policy before fitting.
Historical availability is simulated, not evidence of a timely forecast issuance.
Paper observations retain earlier account entries and never retroactively refill orders.
"""

from copy import deepcopy
from decimal import ROUND_FLOOR, Decimal

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.evaluation.local_research import known_notices, quotes_asof
from quant_rsi.execution import execution_account as account
from quant_rsi.market.asof import session_index
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.market.market_data import visible_revisions

HORIZON = 5


def _validate(data, rules, threshold):
    if len(data.instruments) != 1 or set(rules.fill_cap_units) != set(data.instruments):
        raise ValueError("strategy replay requires exactly one target and its explicit fill cap")
    if rules.currency != data.contract.currency or rules.termination != "preserve_inventory":
        raise ValueError("strategy replay requires matching currency and preserve_inventory")
    threshold = Decimal(str(threshold))
    if not threshold.is_finite():
        raise ValueError("strategy threshold must be finite")
    return data.instruments[0], threshold


def _active(snapshot, contract, instrument, at, basis):
    events = [
        row["event"]
        for row in visible_revisions(snapshot, contract, at, basis=basis)
        if row["event"]["kind"] == "UNIVERSE_STATUS" and row["event"]["instrument"] == instrument
    ]
    if not events:
        return False
    latest = max(timestamp(event["event_at"]) for event in events)
    return {event["status"] for event in events if timestamp(event["event_at"]) == latest} == {
        "active"
    }


def _quotes(data, snapshot, index, phase, at, basis):
    session = data.contract.calendar.sessions[index]
    quotes = quotes_asof(snapshot, data.contract, session.session_id, phase, at, basis=basis)
    if phase == "OPEN":
        # A status effective after the target open must not authorize that open.
        visible = dict(snapshot)
        visible["events"] = [
            row
            for row in snapshot["events"]
            if row["event"]["kind"] != "UNIVERSE_STATUS"
            or timestamp(row["event"]["event_at"]) <= session.open_at
        ]
        quotes = {
            key: value
            for key, value in quotes.items()
            if _active(visible, data.contract, key, at, basis)
        }
    return quotes


def _notices(state, data, snapshot, at, basis):
    for notice in known_notices(snapshot, data.contract, at, basis=basis):
        state = account.apply_action_notice(
            state, notice["event"]["instrument"], notice["payload_ref"]
        )
    return state


def _signal(row, index, data, threshold, *, paper=False):
    session = data.contract.calendar.sessions[index]
    if row is None:
        return dict(origin_session=session.session_id, target="cash", reason="missing_prediction")
    result = deepcopy(row)
    if row["origin_session"] != session.session_id:
        raise ValueError("prediction does not match its frozen origin")
    for field in ("model_ref", "result_ref", "feature_as_of"):
        if not row.get(field):
            raise ValueError("prediction is missing " + field)
    if timestamp(row["feature_as_of"]) > data.decision_at(index):
        raise ValueError("prediction uses information after its origin decision cutoff")
    if paper and timestamp(row["feature_as_of"]) > timestamp(row["available_at"]):
        raise ValueError("paper prediction precedes its feature information cutoff")
    score = None if row.get("score") is None else Decimal(str(row["score"]))
    if score is not None and not score.is_finite():
        raise ValueError("nonfinite score must be represented by an explicit missing prediction")
    result.update(
        target="long" if score is not None and score > threshold else "cash",
        reason="positive_prediction"
        if score is not None and score > threshold
        else row.get("reason", "missing_prediction")
        if score is None
        else "prediction_at_or_below_threshold",
        prediction_horizon_sessions=HORIZON,
        expiry_session=data.contract.calendar.sessions[index + HORIZON].session_id,
    )
    return result


def _submit(state, data, rules, index, signal, quotes, available_at, confirmation_at):
    """Enter once while flat; exit held units for cash. Never top up a capped fill."""
    instrument = data.instruments[0]
    reason = signal["reason"]
    instructions = []
    if state["suspended_reasons"] or any(
        order["status"] in account.ACTIVE for order in state["orders"]
    ):
        reason = "account_suspended_or_unresolved_order"
    elif instrument not in quotes:
        reason = "missing_decision_close"
    elif timestamp(available_at) >= data.contract.calendar.sessions[index + 1].open_at:
        reason = "prediction_not_available_before_next_open"
    else:
        holding = state["holdings"].get(instrument, {}).get("quantity", 0)
        side = "sell" if signal["target"] == "cash" else "buy"
        quantity = holding if side == "sell" else 0
        if side == "buy" and not holding:
            budget = Decimal(state["cash"]) * Decimal(rules.allocation)
            price = Decimal(quotes[instrument]["price"])
            quantity = int((budget / price / rules.lot_size).to_integral_value(ROUND_FLOOR))
            quantity *= rules.lot_size
        if quantity:
            instructions = [
                dict(
                    instrument=instrument,
                    side=side,
                    quantity=quantity,
                    reference_price=quotes[instrument]["price"],
                )
            ]
    if not instructions:
        return state, dict(signal=signal, reason=reason, order_ids=[])
    target = data.contract.calendar.sessions[index + 1]
    state, identifiers = account.prepare_orders(
        state,
        instructions,
        rules,
        decision_ref=digest(canonical(signal)),
        target_session=target.session_id,
        target_at=utc_text(target.open_at),
        prediction_finished_at=available_at,
    )
    if confirmation_at is not None:
        state = account.confirm_submission(state, identifiers, confirmation_at)
    return state, dict(signal=signal, reason=reason, order_ids=identifiers)


def _step(state, data, snapshot, rules, index, at, basis):
    session = data.contract.calendar.sessions[index]
    open_at = utc_text(session.open_at + (data.decision_at(index) - session.close_at))
    state = _notices(state, data, snapshot, open_at, basis)
    state = account.resolve_open(
        state,
        _quotes(data, snapshot, index, "OPEN", open_at, basis),
        rules,
        session_id=data.dates[index],
        resolution_at=open_at,
    )
    state = _notices(state, data, snapshot, at, basis)
    return account.value_close(
        state,
        _quotes(data, snapshot, index, "CLOSE", at, basis),
        rules,
        session_id=data.dates[index],
        close_at=utc_text(data.contract.calendar.sessions[index].close_at),
    )


def _metrics(state, rules):
    rows = state["valuations"]
    complete = bool(rows) and all(row["status"] == "complete" for row in rows)
    complete = (
        complete
        and not state["action_instruments"]
        and not any(
            reason.startswith("processed_evidence_corrected:")
            for reason in state["suspended_reasons"]
        )
    )
    initial = Decimal(rules.initial_cash)
    exposures = [row.get("exposure_fraction") for row in rows[1:]]
    return dict(
        status="complete" if complete else "insufficient_evidence",
        net_return=account.amount(Decimal(rows[-1]["nav"]) / initial - 1) if complete else None,
        max_drawdown=min((row["drawdown"] for row in rows), key=Decimal) if complete else None,
        turnover=account.amount(
            sum((Decimal(fill["raw_notional"]) for fill in state["fills"]), Decimal(0)) / initial
        ),
        turnover_definition="sum_absolute_raw_fill_notional_divided_by_initial_cash",
        fees={key: state["cumulative_" + key] for key in ("commission", "tax", "slippage")},
        mean_close_exposure=account.amount(
            sum((Decimal(value) for value in exposures), Decimal(0)) / len(exposures)
        )
        if complete and exposures and all(value is not None for value in exposures)
        else None,
        valuation_sessions=len(rows),
        unresolved_sessions=[row["session_id"] for row in rows if row["status"] != "complete"],
        suspended_reasons=state["suspended_reasons"],
        ending_inventory=state["holdings"],
        ending_cash=state["cash"],
    )


def _exposure(state):
    row = state["valuations"][-1]
    row["cash"] = state["cash"]
    row["holdings"] = deepcopy(state["holdings"])
    row["exposure_fraction"] = (
        account.amount((Decimal(row["nav"]) - Decimal(state["cash"])) / Decimal(row["nav"]))
        if row["nav"] is not None and Decimal(row["nav"]) > 0
        else "0"
        if row["nav"] == "0"
        else None
    )
    return state


def replay_strategy(data, predictions, rules, *, origin_sessions, end_session, threshold="0"):
    """Replay one verified historical fold on its entire registered session grid.

    The caller binds actual model/result references; simulated origin-time order
    confirmation is explicitly retrospective and cannot qualify as paper evidence.
    """
    _, threshold = _validate(data, rules, threshold)
    indices = [session_index(data.contract, origin) for origin in origin_sessions]
    end = session_index(data.contract, end_session)
    if not indices or indices != list(range(indices[0], indices[-1] + 1, HORIZON)):
        raise ValueError("frozen historical origins must be a contiguous step-five grid")
    if end != indices[-1] + HORIZON or end >= len(data.dates):
        raise ValueError("historical episode must end at the last five-session target close")
    rows = {row["origin_session"]: row for row in predictions}
    if len(rows) != len(predictions) or not set(rows) <= set(origin_sessions):
        raise ValueError("prediction origins duplicate or escape the frozen grid")
    states = {name: account.new_account(rules) for name in ("strategy", "buy_and_hold")}
    decisions = []
    for index in range(indices[0], end + 1):
        at = utc_text(data.decision_at(index))
        for name in states:
            states[name] = _exposure(
                _step(states[name], data, data.snapshot, rules, index, at, None)
            )
        if index not in indices:
            continue
        signal = _signal(rows.get(data.dates[index]), index, data, threshold)
        signal["simulated_available_at"] = at
        quotes = _quotes(data, data.snapshot, index, "CLOSE", at, None)
        states["strategy"], decision = _submit(
            states["strategy"], data, rules, index, signal, quotes, at, at
        )
        decisions.append(decision)
        if index == indices[0]:
            states["buy_and_hold"], _ = _submit(
                states["buy_and_hold"],
                data,
                rules,
                index,
                dict(origin_session=data.dates[index], target="long", reason="buy_and_hold"),
                quotes,
                at,
                at,
            )
    return dict(
        kind="HistoricalStrategyReplay",
        evidence_mode="historical_development_simulated_issuance",
        knowledge_basis=data.contract.knowledge_basis,
        snapshot_ref=digest(canonical(data.snapshot)),
        policy=dict(
            threshold=str(threshold), horizon_sessions=HORIZON, entry="positive_long_else_cash"
        ),
        execution_interval="next_registered_open_to_exit_next_open_or_terminal_close_mark",
        valuation_sessions=list(data.dates[indices[0] : end + 1]),
        end_policy="preserve_inventory",
        benchmark="same_target_buy_and_hold",
        decisions=decisions,
        accounts=states,
        metrics={name: _metrics(state, rules) for name, state in states.items()},
    )


def _remember(state, data, snapshot, at):
    """Bind consumed quote/status evidence so later revisions cannot rewrite fills."""
    processed = {row["session_id"] for row in state["accounts"]["strategy"]["valuations"]}
    processed.update(state["opened_sessions"])
    for row in visible_revisions(snapshot, data.contract, at, basis="observed_receipts"):
        event = row["event"]
        if (
            event["kind"] in {"UNIVERSE_STATUS", "CORPORATE_ACTION_NOTICE"}
            or event["session_id"] in processed
        ):
            state["consumed_evidence"].setdefault(event["event_id"], row["payload_ref"])


def _corrected(state, data, snapshot, at):
    for row in visible_revisions(snapshot, data.contract, at, basis="observed_receipts"):
        event_id = row["event"]["event_id"]
        if event_id in state["consumed_evidence"] and (
            state["consumed_evidence"][event_id] != row["payload_ref"]
        ):
            reason = "processed_evidence_corrected:" + event_id
            for value in state["accounts"].values():
                if reason not in value["suspended_reasons"]:
                    value["suspended_reasons"].append(reason)


def advance_paper_strategy(
    data, prediction, rules, *, observed_at, previous=None, threshold="0", snapshot=None
):
    """Advance one publication's five-session hypothetical account, without fitting.

    Initial state contains prepared orders. The controller must commit it before
    supplying it as ``previous`` with a new trusted clock witness, which confirms
    submission. Late initialization/confirmation never creates a backdated fill.
    Each open/close is processed once; missing opens remain unresolved. The caller
    persists this returned state and verifies appended snapshot/receipt authority.
    """
    _, threshold = _validate(data, rules, threshold)
    observed_at = utc_text(timestamp(observed_at))
    snapshot = data.snapshot if snapshot is None else snapshot
    if snapshot["contract_ref"] != digest(canonical(data.contract.model_dump(mode="json"))):
        raise ValueError("paper snapshot belongs to a different registered contract")
    index = session_index(data.contract, prediction["origin_session"])
    if index + HORIZON >= len(data.contract.calendar.sessions):
        raise ValueError("paper prediction has no registered five-session target")
    signal = _signal(prediction, index, data, threshold, paper=True)
    issued = timestamp(prediction["available_at"])
    confirmed = timestamp(prediction["confirmation_at"])
    if confirmed < issued or confirmed > timestamp(observed_at):
        raise ValueError("paper publication confirmation is unavailable or precedes issuance")
    if issued < data.contract.calendar.sessions[index].close_at:
        raise ValueError("paper publication precedes the forecast origin close")
    binding = digest(
        canonical(
            dict(
                prediction=prediction,
                rules=rules.model_dump(mode="json"),
                contract=data.contract.model_dump(mode="json"),
                threshold=str(threshold),
            )
        )
    )
    if previous is None:
        state = dict(
            kind="PaperStrategyState",
            binding_ref=binding,
            evidence_mode="observed_receipts_paper_simulation",
            origin_session=data.contract.calendar.sessions[index].session_id,
            target_session=data.contract.calendar.sessions[index + HORIZON].session_id,
            signal=signal,
            threshold=str(threshold),
            horizon_sessions=HORIZON,
            execution_interval="next_registered_open_to_terminal_close_mark",
            end_policy="preserve_inventory",
            benchmark="same_target_buy_and_hold",
            initialized_at=observed_at,
            observed_at=observed_at,
            last_snapshot_ref=digest(canonical(snapshot)),
            cursor=index + 1,
            opened_sessions=[],
            consumed_evidence={},
            accounts={name: account.new_account(rules) for name in ("strategy", "buy_and_hold")},
            decisions={},
            status="awaiting_submission_confirmation",
        )
        opening = data.contract.calendar.sessions[index + 1].open_at
        if confirmed >= opening or timestamp(observed_at) >= opening:
            state["status"] = "not_prospective_late_initialization"
        else:
            quotes = _quotes(data, snapshot, index, "CLOSE", observed_at, "observed_receipts")
            for name in state["accounts"]:
                value = _notices(
                    state["accounts"][name], data, snapshot, observed_at, "observed_receipts"
                )
                value = _exposure(
                    account.value_close(
                        value,
                        quotes,
                        rules,
                        session_id=data.contract.calendar.sessions[index].session_id,
                        close_at=utc_text(data.contract.calendar.sessions[index].close_at),
                    )
                )
                selected = (
                    signal
                    if name == "strategy"
                    else dict(
                        origin_session=data.contract.calendar.sessions[index].session_id,
                        target="long",
                        reason="buy_and_hold",
                    )
                )
                value, decision = _submit(
                    value, data, rules, index, selected, quotes, observed_at, None
                )
                state["accounts"][name] = value
                state["decisions"][name] = decision
            _remember(state, data, snapshot, observed_at)
    else:
        state = deepcopy(previous)
        if state["binding_ref"] != binding:
            raise ValueError("paper strategy changed its frozen prediction, policy or contract")
        if timestamp(observed_at) < timestamp(state["observed_at"]):
            raise ValueError("paper observation clock moved backwards")
        state["observed_at"] = observed_at
        state["last_snapshot_ref"] = digest(canonical(snapshot))
        _corrected(state, data, snapshot, observed_at)
        for name, value in state["accounts"].items():
            state["accounts"][name] = _notices(
                value, data, snapshot, observed_at, "observed_receipts"
            )
        if state["status"] == "awaiting_submission_confirmation":
            for name, value in state["accounts"].items():
                ids = [
                    order["order_id"] for order in value["orders"] if order["status"] == "prepared"
                ]
                state["accounts"][name] = account.confirm_submission(value, ids, observed_at)
            state["status"] = (
                "not_prospective_late_confirmation"
                if timestamp(observed_at) >= data.contract.calendar.sessions[index + 1].open_at
                else "observing"
            )
        if state["status"] == "observing":
            while state["cursor"] <= index + HORIZON:
                cursor = state["cursor"]
                session = data.contract.calendar.sessions[cursor]
                if session.open_at > timestamp(observed_at):
                    break
                if session.session_id not in state["opened_sessions"]:
                    quotes = _quotes(
                        data, snapshot, cursor, "OPEN", observed_at, "observed_receipts"
                    )
                    if not quotes and session.close_at > timestamp(observed_at):
                        break
                    for name, value in state["accounts"].items():
                        value = _notices(value, data, snapshot, observed_at, "observed_receipts")
                        if any(
                            reason.startswith("processed_evidence_corrected:")
                            for reason in value["suspended_reasons"]
                        ):
                            state["accounts"][name] = value
                            continue
                        state["accounts"][name] = account.resolve_open(
                            value,
                            quotes,
                            rules,
                            session_id=session.session_id,
                            resolution_at=observed_at,
                        )
                    state["opened_sessions"].append(session.session_id)
                    _remember(state, data, snapshot, observed_at)
                if session.close_at > timestamp(observed_at):
                    break
                quotes = _quotes(data, snapshot, cursor, "CLOSE", observed_at, "observed_receipts")
                deadline = (
                    data.contract.calendar.sessions[cursor + 1].open_at
                    if cursor + 1 < len(data.contract.calendar.sessions)
                    else data.decision_at(cursor)
                )
                if not quotes and timestamp(observed_at) < deadline:
                    break
                for name, value in state["accounts"].items():
                    value = _notices(value, data, snapshot, observed_at, "observed_receipts")
                    state["accounts"][name] = _exposure(
                        account.value_close(
                            value,
                            quotes,
                            rules,
                            session_id=session.session_id,
                            close_at=utc_text(session.close_at),
                        )
                    )
                state["cursor"] += 1
                _remember(state, data, snapshot, observed_at)
            if state["cursor"] > index + HORIZON:
                state["status"] = "complete"
    state["metrics"] = {name: _metrics(value, rules) for name, value in state["accounts"].items()}
    if state["status"] != "complete":
        for metrics in state["metrics"].values():
            metrics.update(status=state["status"], net_return=None, max_drawdown=None)
    return state
