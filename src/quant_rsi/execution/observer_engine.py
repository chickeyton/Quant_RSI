"""Deterministic observer transitions; trusted clocks are supplied by the controller.

Replay supplies the clocks already committed in the event chain. Source inputs never
contain controller events, money, orders, fills or checkpoint state.
"""

from copy import deepcopy

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.evaluation.local_research import known_notices, quotes_asof
from quant_rsi.execution import execution_account as account
from quant_rsi.market.asof import make_feature_view, predict_view, session_index
from quant_rsi.market.market_contracts import ObservationPlan, timestamp, utc_text
from quant_rsi.market.market_data import visible_revisions


def initial_state(episode_id, plan, runtime_ref, created_at, watermark):
    return dict(
        kind="ObserverCheckpoint",
        episode_id=episode_id,
        plan_ref=digest(canonical(plan.model_dump(mode="json"))),
        runtime_ref=runtime_ref,
        created_at=created_at,
        active_at=None,
        start_watermark=watermark,
        receipt_cursor=watermark,
        market_cursor=0,
        processed_steps=0,
        last_control_at=created_at,
        observations=[],
        pending_closes=[],
        receipts=[],
        correction_flags=[],
        seen_notices=[],
        account=account.new_account(plan) if isinstance(plan, ObservationPlan) else None,
        pending_submission=None,
        window=None,
        requests={},
        stop=None,
    )


def clock_after(state, value):
    if timestamp(value) < timestamp(state["last_control_at"]):
        raise ValueError("trusted observer clock moved backwards")


def timeline(contract, active_at):
    return [
        (session.session_id, phase, utc_text(getattr(session, phase.lower() + "_at")))
        for session in contract.calendar.sessions
        for phase in ("OPEN", "CLOSE")
        if getattr(session, phase.lower() + "_at") > timestamp(active_at)
    ]


def model_reasons(plan, bundle, now):
    value = timestamp(now)
    reasons = []
    if value < timestamp(bundle["model_effective_at"]):
        reasons.append("model_not_effective")
    if value >= plan.expires_at:
        reasons.append("episode_expired")
    if (
        value - timestamp(bundle["actual_fit_completed_at"])
    ).total_seconds() > plan.max_model_age_seconds:
        reasons.append("actual_model_age_limit")
    if (
        value - timestamp(bundle["training_information_cutoff"])
    ).total_seconds() > plan.max_information_age_seconds:
        reasons.append("market_information_age_limit")
    return reasons


def source_inputs(state, snapshot, contract, now):
    changed = False
    new_receipts = [row for row in snapshot["receipts"] if row["seq"] > state["receipt_cursor"]]
    if new_receipts:
        if any(timestamp(row["first_seen_at"]) > timestamp(now) for row in new_receipts):
            raise ValueError("observer clock precedes committed input receipts")
        fresh = [row for row in snapshot["events"] if row["receipt_seq"] > state["receipt_cursor"]]
        already = {row["session_id"] for row in state["observations"]} | set(
            state["pending_closes"]
        )
        for receipt in new_receipts:
            classifications = []
            for row in fresh:
                if row["receipt_seq"] != receipt["seq"]:
                    continue
                event = row["event"]
                forward = (
                    timestamp(event["event_at"]) > timestamp(state["active_at"])
                    and timestamp(row["first_seen_at"]) > timestamp(state["active_at"])
                    and row["receipt_seq"] > state["start_watermark"]
                )
                classifications.append(
                    dict(
                        payload_ref=row["payload_ref"],
                        event_id=event["event_id"],
                        revision=event["revision"],
                        classification="forward_input" if forward else "historical_backfill",
                        receipt_delay_seconds=(
                            timestamp(row["first_seen_at"]) - timestamp(event["event_at"])
                        ).total_seconds(),
                    )
                )
                if (
                    forward
                    and event["kind"] == "CLOSE_QUOTE"
                    and event["session_id"] not in already
                ):
                    state["pending_closes"].append(event["session_id"])
                    already.add(event["session_id"])
                earlier = [
                    item["event"]["revision"]
                    for item in snapshot["events"]
                    if item["event"]["event_id"] == event["event_id"]
                    and item["receipt_seq"] <= state["receipt_cursor"]
                ]
                if earlier and event["revision"] > max(earlier):
                    state["correction_flags"].append(
                        dict(
                            kind="later_revision_does_not_rewrite",
                            payload_ref=row["payload_ref"],
                            potentially_affected_observation_refs=[
                                item["observation_ref"] for item in state["observations"]
                            ],
                        )
                    )
            state["receipts"].append(receipt | {"inputs": classifications})
        state["receipt_cursor"] = new_receipts[-1]["seq"]
        state["pending_closes"].sort(key=lambda name: session_index(contract, name))
        changed = True
    for notice in known_notices(snapshot, contract, now, basis="observed_receipts"):
        if notice["payload_ref"] in state["seen_notices"]:
            continue
        state["seen_notices"].append(notice["payload_ref"])
        state["correction_flags"].append(
            dict(
                kind="unsupported_action_notice",
                payload_ref=notice["payload_ref"],
                potentially_affected_observation_refs=[
                    row["observation_ref"] for row in state["observations"]
                ],
            )
        )
        if state["account"] is not None:
            state["account"] = account.apply_action_notice(
                state["account"], notice["event"]["instrument"], notice["payload_ref"]
            )
        changed = True
    return changed


def tradable_open_quotes(snapshot, contract, session_id, now):
    quotes = quotes_asof(snapshot, contract, session_id, "OPEN", now, basis="observed_receipts")
    opening = contract.calendar.sessions[session_index(contract, session_id)].open_at
    visible = visible_revisions(snapshot, contract, now, basis="observed_receipts")
    for instrument in list(quotes):
        membership = [
            row["event"]
            for row in visible
            if row["event"]["kind"] == "UNIVERSE_STATUS"
            and row["event"]["instrument"] == instrument
            and timestamp(row["event"]["event_at"]) <= opening
        ]
        if not membership:
            del quotes[instrument]
            continue
        latest = max(timestamp(event["event_at"]) for event in membership)
        if {event["status"] for event in membership if timestamp(event["event_at"]) == latest} != {
            "active"
        }:
            del quotes[instrument]
    return quotes


def advance_market(state, plan, snapshot, contract, now):
    if state["account"] is None:
        return False, None
    phases = timeline(contract, state["active_at"])
    if state["market_cursor"] >= len(phases):
        return False, None
    session_id, phase, event_at = phases[state["market_cursor"]]
    if timestamp(event_at) > timestamp(now):
        return False, None
    if phase == "OPEN":
        state["account"] = account.resolve_open(
            state["account"],
            tradable_open_quotes(snapshot, contract, session_id, now),
            plan,
            session_id=session_id,
            resolution_at=now,
        )
        if any(
            order["status"] in account.ACTIVE and order["target_session"] == session_id
            for order in state["account"]["orders"]
        ):
            return False, "pending_resolution"
    else:
        quotes = quotes_asof(
            snapshot, contract, session_id, "CLOSE", now, basis="observed_receipts"
        )
        index = session_index(contract, session_id)
        next_open = (
            contract.calendar.sessions[index + 1].open_at
            if index + 1 < len(contract.calendar.sessions)
            else None
        )
        waiting = not quotes or any(
            position["quantity"]
            and instrument not in quotes
            and instrument not in state["account"]["action_instruments"]
            for instrument, position in state["account"]["holdings"].items()
        )
        if waiting and (next_open is None or timestamp(now) < next_open):
            return False, "awaiting_fresh_close_information"
        state["account"] = account.value_close(
            state["account"], quotes, plan, session_id=session_id, close_at=event_at
        )
        state["account"]["valuations"][-1]["observed_at"] = now
    state["market_cursor"] += 1
    return True, None


def account_at_close(state, contract, session_id):
    if state["account"] is None:
        return True
    phases = timeline(contract, state["active_at"])
    return state["market_cursor"] > 0 and phases[state["market_cursor"] - 1][:2] == (
        session_id,
        "CLOSE",
    )


def transition_step(previous, plan, bundle, model_text, contract, snapshot, now, finish_clock):
    clock_after(previous, now)
    if previous["active_at"] is None or previous["pending_submission"] is not None:
        raise ValueError("episode activation/submission recovery must precede the next step")
    state, objects, clocks = deepcopy(previous), {}, []
    changed = source_inputs(state, snapshot, contract, now)
    old_account = deepcopy(state["account"])
    progressed, boundary = advance_market(state, plan, snapshot, contract, now)
    changed = changed or progressed or old_account != state["account"]
    if state["pending_closes"]:
        session_id = state["pending_closes"][0]
        ready = account_at_close(state, contract, session_id)
        phases = timeline(contract, state["active_at"])
        close_position = next(
            (i for i, phase in enumerate(phases) if phase[:2] == (session_id, "CLOSE")), None
        )
        passed = close_position is not None and state["market_cursor"] > close_position + 1
        # A missing old open cannot prevent the independent prediction recorder.
        if ready or passed or boundary == "pending_resolution" or state["account"] is None:
            factor = FactorSpec.model_validate_json(canonical(bundle["factor"]))
            view = make_feature_view(
                snapshot, contract, session_id, now, factor, basis="observed_receipts"
            )
            prediction = predict_view(model_text, view)
            finished = finish_clock()
            if timestamp(finished) < timestamp(now):
                raise ValueError("trusted prediction clock moved backwards")
            clocks.append(finished)
            index = session_index(contract, session_id)
            target = (
                contract.calendar.sessions[index + 1]
                if index + 1 < len(contract.calendar.sessions)
                else None
            )
            on_time = target is not None and timestamp(finished) < target.open_at
            gate = model_reasons(plan, bundle, finished)
            if state["account"] is not None:
                gate.extend(state["account"]["suspended_reasons"])
            if state["stop"] is not None:
                gate.append("durable_stop_intent")
            if not ready:
                gate.append("account_boundary_pending")
            if not on_time:
                gate.append("late_or_unregistered_target_open")
            observation = dict(
                kind="RegisteredObservation",
                episode_id=state["episode_id"],
                plan_ref=state["plan_ref"],
                model_bundle_ref=plan.model_bundle_ref,
                snapshot_ref=digest(canonical(snapshot)),
                feature_view_ref=digest(canonical(view)),
                prediction=prediction,
                started_at=now,
                prediction_finished_at=finished,
                target_session=None if target is None else target.session_id,
                target_at=None if target is None else utc_text(target.open_at),
                prediction_timeliness="forward_on_time" if on_time else "forward_late",
                new_risk_blocked_reasons=gate,
                purpose="developer_visible_observation",
            )
            ref = digest(canonical(observation))
            objects[ref], objects[digest(canonical(view))] = observation, view
            state["observations"].append(
                dict(
                    session_id=session_id,
                    observation_ref=ref,
                    prediction_timeliness=observation["prediction_timeliness"],
                    submission_status="no_orders",
                    blocked_reasons=gate,
                )
            )
            state["pending_closes"].pop(0)
            if state["account"] is not None and not gate:
                quotes = quotes_asof(
                    snapshot, contract, session_id, "CLOSE", now, basis="observed_receipts"
                )
                instructions = account.rebalance_instructions(
                    state["account"], prediction, quotes, plan
                )
                if not any(
                    order["status"] in account.ACTIVE for order in state["account"]["orders"]
                ):
                    state["account"], ids = account.prepare_orders(
                        state["account"],
                        instructions,
                        plan,
                        decision_ref=ref,
                        target_session=target.session_id,
                        target_at=utc_text(target.open_at),
                        prediction_finished_at=finished,
                    )
                    if ids:
                        state["pending_submission"] = dict(order_ids=ids, observation_ref=ref)
                        state["observations"][-1]["submission_status"] = (
                            "prepared_without_timely_proof"
                        )
            changed = True
    # Explicit termination can only queue risk-reducing orders at a future legal open.
    if (
        state["account"] is not None
        and plan.termination == "queue_exit_at_next_eligible_open"
        and state["pending_submission"] is None
        and (
            state["stop"] is not None
            or model_reasons(plan, bundle, now)
            or state["account"]["suspended_reasons"]
        )
        and not any(order["status"] in account.ACTIVE for order in state["account"]["orders"])
    ):
        future = [
            session for session in contract.calendar.sessions if session.open_at > timestamp(now)
        ]
        if future:
            visible = visible_revisions(snapshot, contract, now, basis="observed_receipts")
            instructions = []
            for instrument, position in state["account"]["holdings"].items():
                membership = [
                    row["event"]
                    for row in visible
                    if row["event"]["kind"] == "UNIVERSE_STATUS"
                    and row["event"]["instrument"] == instrument
                ]
                if (
                    not position["quantity"]
                    or not membership
                    or instrument in state["account"]["action_instruments"]
                ):
                    continue
                latest = max(timestamp(event["event_at"]) for event in membership)
                if {
                    event["status"]
                    for event in membership
                    if timestamp(event["event_at"]) == latest
                } == {"active"}:
                    instructions.append(
                        dict(
                            instrument=instrument,
                            side="sell",
                            quantity=position["quantity"],
                            reference_price=None,
                        )
                    )
            if instructions:
                finished = finish_clock()
                if timestamp(finished) < timestamp(clocks[-1] if clocks else now):
                    raise ValueError("trusted exit controller clock moved backwards")
                clocks.append(finished)
                decision = dict(
                    kind="ControllerExitIntent",
                    episode_id=state["episode_id"],
                    at=finished,
                    target_session=future[0].session_id,
                )
                ref = digest(canonical(decision))
                objects[ref] = decision
                state["account"], ids = account.prepare_orders(
                    state["account"],
                    instructions,
                    plan,
                    decision_ref=ref,
                    target_session=future[0].session_id,
                    target_at=utc_text(future[0].open_at),
                    prediction_finished_at=finished,
                )
                if ids:
                    state["pending_submission"] = dict(order_ids=ids, observation_ref=None)
                    changed = True
    if not changed:
        return None
    state["processed_steps"] += 1
    state["last_control_at"] = clocks[-1] if clocks else now
    payload = dict(
        started_at=now,
        finish_clocks=clocks,
        snapshot_ref=digest(canonical(snapshot)),
        boundary=boundary,
        output_refs=sorted(objects),
    )
    return state, payload, objects


def confirm_submission(previous, at, *, recover=False):
    clock_after(previous, at)
    state = deepcopy(previous)
    pending = state["pending_submission"]
    if pending is None:
        raise ValueError("no pending submission to confirm/recover")
    state["account"] = (
        account.abandon_unconfirmed(state["account"])
        if recover
        else account.confirm_submission(state["account"], pending["order_ids"], at)
    )
    if pending["observation_ref"] is not None:
        selected = [
            order
            for order in state["account"]["orders"]
            if order["order_id"] in pending["order_ids"]
        ]
        status = (
            "timely_durable_submission"
            if all(order["status"] == "pending_resolution" for order in selected)
            else "unsubmitted_or_late"
        )
        for observation in state["observations"]:
            if observation["observation_ref"] == pending["observation_ref"]:
                observation["submission_status"] = status
    state["pending_submission"] = None
    state["last_control_at"] = at
    return state
