"""Fixed local models scored on a registered Decimal account grid after OPENED."""

import fcntl
import os
from decimal import Decimal

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.evaluation.local_research import amount_costs, known_notices, quotes_asof
from quant_rsi.execution import execution_account as account
from quant_rsi.market.asof import make_feature_view, predict_view
from quant_rsi.market.market_contracts import AccountRules, trusted_now, utc_text


def require_score_owner(root, owner_fd):
    lock_path = root / ".controller.lock"
    try:
        if not os.path.samestat(os.fstat(owner_fd), lock_path.stat()):
            raise ValueError("local scorer requires the actual authority lock")
        with lock_path.open("r+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("local scorer authority lock is not held")
        fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, TypeError) as error:
        raise ValueError("local scorer requires the actual authority lock") from error


def consumption_evidence(data, task, selection):
    start, end = task["grid"]["valuations"]
    factor = (
        FactorSpec.model_validate_json(canonical(selection["factor"]))
        if selection["role"] == "active"
        else None
    )
    rows = []
    for index in range(start, end + 1):
        as_of = utc_text(data.decision_at(index))
        rows.append(
            dict(
                session_id=data.dates[index],
                as_of=as_of,
                close_at=data.timestamp(index, "close"),
                next_session_id=data.dates[index + 1] if index < end else None,
                next_open_at=data.timestamp(index + 1, "open") if index < end else None,
                feature_view=None
                if factor is None
                or not task["grid"]["decisions"][0] <= index <= task["grid"]["decisions"][1]
                else make_feature_view(
                    data.snapshot, data.contract, data.dates[index], data.decision_at(index), factor
                ),
                close_quotes=quotes_asof(
                    data.snapshot, data.contract, data.dates[index], "CLOSE", as_of
                ),
                open_quotes={
                    data.dates[j]: quotes_asof(
                        data.snapshot, data.contract, data.dates[j], "OPEN", as_of
                    )
                    for j in range(start + 1, index + 1)
                },
                notices=[
                    dict(instrument=r["event"]["instrument"], payload_ref=r["payload_ref"])
                    for r in known_notices(data.snapshot, data.contract, as_of)
                ],
            )
        )
    return dict(
        kind="LocalScoreConsumptionEvidence",
        source_ref=task["source_ref"],
        episode_ref=task["episode_ref"],
        task_ref=digest(canonical(task)),
        grid=task["grid"],
        instruments=task["instruments"],
        knowledge_basis=task["knowledge_basis"],
        qualifications=task["qualifications"],
        rows=rows,
    )


def evaluate_consumed(evidence, selection, model_text, annualization):
    """Replay already consumed inputs without accessing a registry or protected source."""
    rules = AccountRules.model_validate_json(canonical(selection["portfolio"]))
    grid = evidence["grid"]
    if rules.termination != grid["terminal_policy"] or rules.initial_cash != grid["nav0"]:
        raise ValueError("score account/termination differs from registered grid")
    if selection["role"] not in {"active", "cash_fallback"}:
        raise ValueError("unknown selected strategy role")
    rows = evidence["rows"]
    if len(rows) != grid["return_count"]:
        raise ValueError("consumed score evidence omits a registered valuation")
    state, predictions, coverage_failures = account.new_account(rules), [], []
    for ordinal, row in enumerate(rows):
        for notice in row["notices"]:
            state = account.apply_action_notice(state, notice["instrument"], notice["payload_ref"])
        pending = sorted(
            {o["target_session"] for o in state["orders"] if o["status"] == "pending_resolution"}
        )
        for session_id in pending:
            state = account.resolve_open(
                state,
                row["open_quotes"].get(session_id, {}),
                rules,
                session_id=session_id,
                resolution_at=row["as_of"],
            )
        state = account.value_close(
            state,
            row["close_quotes"],
            rules,
            session_id=row["session_id"],
            close_at=row["close_at"],
        )
        prediction = None
        is_decision = (
            grid["decisions"][0] <= grid["valuations"][0] + ordinal <= grid["decisions"][1]
        )
        if selection["role"] == "active" and is_decision:
            view = row["feature_view"]
            if (
                view["session_id"] != row["session_id"]
                or view["as_of"] != row["as_of"]
                or view["expected_instruments"] != evidence["instruments"]
            ):
                raise ValueError("consumed feature evidence differs from its registered grid")
            prediction = predict_view(model_text, view)
            predictions.append(prediction)
            if any(r["status"] != "eligible" for r in prediction["rows"]):
                coverage_failures.append("incomplete_prediction_coverage:" + row["session_id"])
        elif selection["role"] == "active":
            if row["feature_view"] is not None:
                raise ValueError("feature evidence exists outside the registered decision grid")
        elif row["feature_view"] is not None or model_text is not None:
            raise ValueError("cash fallback cannot substitute model evidence")
        if (
            ordinal == len(rows) - 1
            or prediction is None
            or any(o["status"] in account.ACTIVE for o in state["orders"])
        ):
            continue
        exit_now = (
            rules.termination == "queue_exit_at_next_eligible_open" and ordinal == len(rows) - 2
        )
        if exit_now:
            instructions = [
                dict(
                    instrument=symbol,
                    side="sell",
                    quantity=holding["quantity"],
                    reference_price=row["close_quotes"][symbol]["price"],
                )
                for symbol, holding in sorted(state["holdings"].items())
                if holding["quantity"]
                and symbol in row["close_quotes"]
                and symbol not in state["action_instruments"]
            ]
        else:
            instructions = account.rebalance_instructions(
                state, prediction, row["close_quotes"], rules
            )
        state, ids = account.prepare_orders(
            state,
            instructions,
            rules,
            decision_ref=digest(canonical(prediction)),
            target_session=row["next_session_id"],
            target_at=row["next_open_at"],
            prediction_finished_at=row["as_of"],
        )
        state = account.confirm_submission(state, ids, row["as_of"])
    reasons = list(state["suspended_reasons"]) + coverage_failures
    complete = len(state["valuations"]) == grid["return_count"] and all(
        r["status"] == "complete" and r["return_value"] is not None for r in state["valuations"]
    )
    if not complete:
        reasons.append("incomplete_registered_valuation_path")
    if any(o["status"] in account.ACTIVE for o in state["orders"]):
        reasons.append("unresolved_terminal_orders")
    remaining = {s: h for s, h in state["holdings"].items() if h["quantity"]}
    if rules.termination == "queue_exit_at_next_eligible_open" and remaining:
        reasons.append("registered_terminal_exit_incomplete")
    returns = [r["return_value"] for r in state["valuations"]]
    utility = (
        float(sum((Decimal(r) for r in returns), Decimal(0)) / grid["return_count"] * annualization)
        if complete
        else None
    )
    drawdowns = [Decimal(r["drawdown"]) for r in state["valuations"] if r["drawdown"] is not None]
    accounting = dict(
        account=state,
        net_returns=returns,
        nav0=rules.initial_cash,
        return_count=grid["return_count"],
        valid_return_count=sum(r is not None for r in returns),
        benchmark="cash_zero_interest",
        terminal_policy=rules.termination,
        terminal_inventory=remaining,
        unexecuted_exit_cost="not_incurred_not_estimated_or_deducted",
        eligibility_reasons=reasons,
        metrics=dict(
            annualized_mean_net_active_return=utility,
            total_return=float(Decimal(state["last_nav"]) / Decimal(rules.initial_cash) - 1)
            if complete
            else None,
            max_drawdown=float(min(drawdowns)) if complete else None,
            costs=amount_costs(state),
            final_cash=state["cash"],
            final_nav=state["last_nav"],
            terminal_marked_inventory_value=None
            if state["last_nav"] is None
            else format(Decimal(state["last_nav"]) - Decimal(state["cash"]), "f"),
        ),
    )
    return dict(
        status="completed" if not reasons else "risk_failure",
        error=None,
        predictions=predictions,
        accounting=accounting,
        return_count=grid["return_count"],
        utility=utility,
        eligible=not reasons,
        consumed_evidence=evidence,
    )


def local_score(authority, attempt_id, *, replay=False):
    if replay:
        return replay_local_score(authority, attempt_id)
    from lightgbm.basic import LightGBMError

    from quant_rsi.evaluation.protected_scoring import authorize_score, read_episode

    _, binding = authorize_score(authority, attempt_id, replay=replay)
    envelope = dict(
        kind="ProtectedScoreResult", attempt_id=attempt_id, binding_ref=digest(canonical(binding))
    )
    try:
        data = read_episode(authority, attempt_id, replay=replay)
        plan = authority.store.json(binding["plan_ref"])
        task = next(t for t in plan["tasks"] if digest(canonical(t)) == binding["task_ref"])
        selection = authority.store.json(binding["selection_ref"])
        model = (
            authority.store.get(selection["model_ref"]).decode()
            if selection["role"] == "active"
            else None
        )
        consumed = consumption_evidence(data, task, selection)
        return envelope | evaluate_consumed(
            consumed, selection, model, plan["utility"]["annualization"]
        )
    except (ValueError, OSError, TypeError, KeyError, LightGBMError) as error:
        return envelope | dict(
            status="invalid",
            error=f"{type(error).__name__}: {error}"[:500],
            predictions=None,
            accounting=None,
            return_count=binding["grid"]["return_count"],
            utility=None,
            eligible=False,
            consumed_evidence=None,
        )


def replay_local_score(authority, attempt_id):
    """A replay capability cannot reopen or create a score input/result."""
    from quant_rsi.evaluation.protected_scoring import authorize_score

    attempt, _ = authorize_score(authority, attempt_id, replay=True)
    if attempt["result_ref"] is None:
        raise ValueError("local replay requires an already committed score result")
    result = authority.store.json(attempt["result_ref"])
    validate_accepted_score(authority, attempt_id, result)
    return result


def validate_accepted_score(authority, attempt_id, result, *, first_acceptance=False, raw_ref=None):
    from quant_rsi.evaluation.protected_scoring import authorize_score

    attempt, binding = authorize_score(authority, attempt_id, replay=not first_acceptance)
    result_ref = digest(canonical(result))
    evidence_ref = digest(canonical(result.get("consumed_evidence")))
    identity = dict(
        kind="LocalScoreSourceValidationReceipt",
        attempt_id=attempt_id,
        binding_ref=attempt["binding_ref"],
        result_ref=result_ref,
        consumed_evidence_ref=evidence_ref,
        source_ref=binding["source_ref"],
        episode_ref=binding["episode_ref"],
        task_ref=binding["task_ref"],
        selection_ref=binding["selection_ref"],
        plan_ref=binding["plan_ref"],
        authority_id=binding["authority_id"],
        batch_id=binding["batch_id"],
        charged_seconds=attempt["seconds"],
        outbox=attempt["outbox"],
        raw_ref=raw_ref if first_acceptance else attempt["raw_ref"],
    )
    if first_acceptance:
        # This is the only acceptance validation that may reopen registered originals.
        # It runs under the already debited, dispatched, OPENED score capability.
        if result != local_score(authority, attempt_id):
            raise ValueError(
                "local scorer output differs from registered original source/model/grid"
            )
        return identity | {"validated_at": trusted_now()}
    receipt_ref = attempt.get("source_validation_receipt_ref")
    if receipt_ref is None:
        raise ValueError("local score lacks its committed source validation receipt")
    receipt = authority.store.json(receipt_ref)
    if result["status"] != attempt["status"]:
        raise ValueError("local score status differs from its committed attempt")
    if set(receipt) != set(identity) | {"validated_at"} or any(
        receipt[k] != v for k, v in identity.items()
    ):
        raise ValueError("local score source validation receipt binding mismatch")
    if result["consumed_evidence"] is not None:
        selection = authority.store.json(binding["selection_ref"])
        model = (
            authority.store.get(selection["model_ref"]).decode()
            if selection["role"] == "active"
            else None
        )
        plan = authority.store.json(binding["plan_ref"])
        envelope = dict(
            kind="ProtectedScoreResult", attempt_id=attempt_id, binding_ref=attempt["binding_ref"]
        )
        if result != envelope | evaluate_consumed(
            result["consumed_evidence"], selection, model, plan["utility"]["annualization"]
        ):
            raise ValueError("local sealed score evidence replay mismatch")
    return receipt
