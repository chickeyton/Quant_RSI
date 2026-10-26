"""Causal price-task action checks and labels, without account cash or order effects."""

import json
import math
from decimal import Decimal, InvalidOperation

from quant_rsi.artifacts import canonical, digest
from quant_rsi.asof import make_feature_view, session_index
from quant_rsi.market_contracts import timestamp, utc_text
from quant_rsi.market_data import knowledge_time, visible_revisions


def target_session(contract, origin, objective):
    index = session_index(contract, origin)
    target = index + objective.horizon_sessions
    if target >= len(contract.calendar.sessions):
        return None
    return contract.calendar.sessions[target]


def cash_interpretation_matches(item, notice):
    """Recognize only exact versioned cash formats bound to the original payload."""
    event = item["event"]
    if digest(canonical(event)) != notice.notice_ref:
        return False
    try:
        value = json.loads(event["action"], parse_float=Decimal)
        if notice.source_format == "cash_dividend_v1":
            if set(value) != {
                "schema_version",
                "action_kind",
                "amount",
                "currency",
                "effective_at",
            }:
                return False
            return (
                type(value["schema_version"]) is int
                and value["schema_version"] == 1
                and value["action_kind"] == "cash_dividend"
                and value["currency"] == notice.currency
                and timestamp(value["effective_at"]) == notice.effective_at
                and Decimal(str(value["amount"])) == Decimal(notice.amount)
            )
        if notice.source_format == "yahoo_chart_dividends_v1":
            if set(value) != {"amount", "date", "event_kind", "provider"}:
                return False
            return (
                value["provider"] == "Yahoo"
                and value["event_kind"] == "dividends"
                and type(value["date"]) is int
                and notice.effective_at.timestamp() == value["date"]
                and timestamp(event["event_at"]) == notice.effective_at
                and Decimal(str(value["amount"])) == Decimal(notice.amount)
            )
    except (ValueError, TypeError, KeyError, InvalidOperation):
        return False
    return False


def action_assessment(visible, instrument, start_at, end_at, objective):
    interpretations = {item.notice_ref: item for item in objective.notices}
    reasons, evidence = [], []
    start, end = timestamp(start_at), timestamp(end_at)
    for item in visible:
        event = item["event"]
        if event["kind"] != "CORPORATE_ACTION_NOTICE" or event["instrument"] != instrument:
            continue
        notice = interpretations.get(item["payload_ref"])
        reason = None
        if notice is None or notice.effective_at is None:
            # An event timestamp alone is not evidence of a notice's effective time.
            reason = "forecast_action_effect_or_type_unknown"
        elif notice.action_kind != "cash_dividend":
            # No noncash source schema is supported yet. An annotation cannot prove
            # that an original split/unknown notice lies outside this interval.
            reason = "forecast_unsupported_action_effect_unverified"
        elif not cash_interpretation_matches(item, notice):
            reason = "forecast_cash_interpretation_mismatch"
        elif start < notice.effective_at <= end and notice.currency != objective.currency:
            reason = "forecast_cash_currency_mismatch"
        evidence.append(
            {
                "notice_ref": item["payload_ref"],
                "interpretation_ref": None
                if notice is None
                else digest(canonical(notice.model_dump(mode="json"))),
                "effective_at": None
                if notice is None or notice.effective_at is None
                else utc_text(notice.effective_at),
                "reason": reason,
            }
        )
        if reason:
            reasons.append(reason)
    return sorted(set(reasons)), evidence


def forecast_feature_view(
    snapshot, contract, session_id, as_of, factor, objective, *, basis=None, watermark=None
):
    if contract.currency != objective.currency or contract.price_basis != "raw":
        raise ValueError("forecast quote currency/basis differs from the registered source")
    return make_feature_view(
        snapshot,
        contract,
        session_id,
        as_of,
        factor,
        basis=basis,
        watermark=watermark,
        forecast_objective=objective,
    )


def mature_forecast_label(
    snapshot,
    contract,
    origin,
    instrument,
    training_cutoff,
    objective,
    *,
    basis=None,
    watermark=None,
):
    if instrument not in contract.instruments:
        raise ValueError("forecast label instrument is outside the registered grid")
    end = target_session(contract, origin, objective)
    if end is None:
        return {"status": "immature", "reason": "future_label_calendar_absent", "value": None}
    first_session = contract.calendar.sessions[session_index(contract, origin)]
    interval = [utc_text(first_session.close_at), utc_text(end.close_at)]
    fixed = {"origin_session": origin, "target_session": end.session_id, "interval": interval}
    if timestamp(training_cutoff) < end.close_at:
        return fixed | {"status": "immature", "reason": "target_close_not_elapsed", "value": None}
    basis = basis or contract.knowledge_basis
    visible = visible_revisions(
        snapshot, contract, training_cutoff, basis=basis, watermark=watermark
    )
    reasons, actions = action_assessment(
        visible, instrument, first_session.close_at, end.close_at, objective
    )
    if reasons:
        return fixed | {
            "status": "ineligible",
            "reason": reasons[0],
            "reasons": reasons,
            "action_evidence": actions,
            "value": None,
        }
    sessions = {origin, end.session_id}
    quotes = {
        item["event"]["session_id"]: item
        for item in visible
        if item["event"]["kind"] == "CLOSE_QUOTE"
        and item["event"]["instrument"] == instrument
        and item["event"]["session_id"] in sessions
    }
    if set(quotes) != sessions or any(
        item["event"]["quality"] != "valid" for item in quotes.values()
    ):
        return fixed | {
            "status": "immature",
            "reason": "target_or_anchor_quote_not_known",
            "value": None,
        }
    left, right = quotes[origin], quotes[end.session_id]
    value = math.log(float(right["event"]["price"])) - math.log(float(left["event"]["price"]))
    if not math.isfinite(value):
        raise ValueError("nonfinite five-session log-return label")
    refs = [left["payload_ref"], right["payload_ref"]] + [row["notice_ref"] for row in actions]
    return fixed | {
        "status": "mature",
        "reason": None,
        "value": value,
        "anchor_price": left["event"]["price"],
        "target_price": right["event"]["price"],
        "source_refs": sorted(set(refs)),
        "action_evidence": actions,
        "available_at": utc_text(
            max(
                [end.close_at]
                + [knowledge_time(item, basis) for item in visible if item["payload_ref"] in refs]
            )
        ),
    }


def forecast_errors(predicted, actual):
    """Positive errors and their explicit maximization score; never clip invalid output."""
    if not predicted or len(predicted) != len(actual):
        raise ValueError("forecast scoring requires the complete fixed label grid")
    errors = []
    for pred, label in zip(predicted, actual, strict=True):
        if not math.isfinite(pred) or not math.isfinite(label):
            raise ValueError("nonfinite forecast or label")
        try:
            predicted_ratio, actual_ratio = math.exp(pred), math.exp(label)
        except OverflowError as error:
            raise ValueError("forecast has a nonfinite implied price") from error
        if not (math.isfinite(predicted_ratio) and predicted_ratio > 0):
            raise ValueError("forecast has an invalid implied positive price")
        if not (math.isfinite(actual_ratio) and actual_ratio > 0):
            raise ValueError("label has an invalid implied positive price")
        error = 100 * (math.expm1(pred) - math.expm1(label))
        if not math.isfinite(error):
            raise ValueError("nonfinite percentage-point forecast error")
        errors.append(error)
    mae = math.fsum(abs(value) for value in errors) / len(errors)
    try:
        rmse = math.sqrt(math.fsum(value * value for value in errors) / len(errors))
    except OverflowError as error:
        raise ValueError("nonfinite forecast squared error") from error
    if not math.isfinite(mae) or not math.isfinite(rmse):
        raise ValueError("nonfinite forecast aggregate error")
    return {"mae_pp": mae, "rmse_pp": rmse, "negative_mae_pp": -mae, "observations": len(errors)}
