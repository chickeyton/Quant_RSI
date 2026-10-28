"""Publication from the separately fitted model; no collection or training."""

import math

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.forecast.forecast_kernel import action_assessment, target_session
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.market.market_data import visible_revisions


def publication_report(inputs, plan, protocol, store, result_ref, selection_ref, issued_at):
    data, objective = inputs.target
    result = store.json(result_ref)
    metadata = store.json(result["worker_metadata_ref"])
    selection = store.json(selection_ref)
    group = selection["selected_recipe"]
    section = protocol["publication"]
    if (
        metadata["purpose"] != "publication"
        or metadata["selection_ref"] != selection_ref
        or metadata["recipe_group"] != group
        or metadata["input_ref"] != inputs.ref
        or metadata["protocol_ref"] != plan.protocol_ref
        or timestamp(issued_at) < timestamp(metadata["model_effective_at"])
        or timestamp(issued_at) < timestamp(section["as_of"])
    ):
        raise ValueError("publication model/selection/timing binding mismatch")
    origin = data.contract.calendar.sessions[data.spec.sessions - 1]
    target = target_session(data.contract, origin.session_id, objective)
    if target is None:
        raise ValueError("publication needs five registered future sessions")
    next_open = data.contract.calendar.sessions[data.spec.sessions].open_at
    view_ref = section["feature_view_refs"][group]
    view = store.json(view_ref)
    visible = visible_revisions(
        data.snapshot, data.contract, section["as_of"], basis="observed_receipts"
    )
    reasons, actions = action_assessment(
        visible, data.instruments[0], origin.close_at, target.close_at, objective
    )
    reasons.extend(view["reasons"])
    quotes = [
        item
        for item in visible
        if item["event"]["kind"] == "CLOSE_QUOTE"
        and item["event"]["instrument"] == data.instruments[0]
        and item["event"]["session_id"] == origin.session_id
        and item["event"]["quality"] == "valid"
    ]
    anchor = float(quotes[0]["event"]["price"]) if len(quotes) == 1 else None
    if anchor is None:
        reasons.append("origin_close_missing_or_conflicting")
    predictions = store.json(result["predictions_ref"])
    if len(predictions) != 1 or predictions[0]["origin_session"] != origin.session_id:
        raise ValueError("publication prediction origin differs")
    score = predictions[0]["score"]
    price = simple_return = None
    if not reasons:
        if score is None or not math.isfinite(score):
            raise ValueError("publication has no finite model prediction")
        try:
            price, simple_return = anchor * math.exp(score), math.expm1(score)
        except OverflowError as exc:
            raise ValueError("nonfinite implied publication price") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError("invalid implied publication price")
    return dict(
        kind="ContextFiveSessionForecast",
        schema_version=1,
        input_ref=inputs.ref,
        protocol_ref=plan.protocol_ref,
        selection_ref=selection_ref,
        publication_result_ref=result_ref,
        model_ref=result["model_ref"],
        model_effective_at=metadata["model_effective_at"],
        recipe_group=group,
        recipe_ref=metadata["recipe_ref"],
        feature_names=metadata["feature_names"],
        feature_view_ref=view_ref,
        instrument=data.instruments[0],
        currency=objective.currency,
        price_unit=objective.price_unit,
        price_unit_claim=objective.price_unit_claim,
        origin_session=origin.session_id,
        origin_close_at=utc_text(origin.close_at),
        target_session=target.session_id,
        target_close_at=utc_text(target.close_at),
        forward_deadline=utc_text(next_open),
        horizon_sessions=5,
        frozen_input_asof=section["as_of"],
        issued_at=issued_at,
        status="abstain" if reasons else "forecast",
        reasons=sorted(set(reasons)),
        origin_price=anchor,
        predicted_log_return=score if not reasons else None,
        forecast_price=price,
        forecast_return=simple_return,
        action_evidence=actions,
        source_refs=view["source_refs"],
        training_information_cutoff=metadata["training_information_cutoff"],
        training_rows=metadata["training_rows"],
        historical_recipe_comparison=selection["candidates"],
        publication_model_validation_metrics=None,
        publication_model_inherits_evaluation_scores=False,
        model_inputs=["target_prices"]
        + (["industry_prices", "market_prices"] if group == "B2" else [])
        + (["typed_news_factor"] if group == "N1" else [])
        + (["discovered_factor"] if group == "D1" else []),
        sidecar_inputs_used_in_training=metadata["sidecar_inputs_used_in_training"],
        qualifications=sorted(
            set(
                protocol["qualifications"]
                + list(objective.source_qualifications)
                + ["uncalibrated_point_forecast_no_probability_interval"]
            )
        ),
        economic_or_statistical_advantage_claim=False,
    )


def confirmed_report(report, confirmation):
    ref = digest(canonical(report))
    if confirmation["forecast_ref"] != ref or timestamp(confirmation["confirmed_at"]) < timestamp(
        report["issued_at"]
    ):
        raise ValueError("publication confirmation does not bind the committed report")
    forward = (
        report["status"] == "forecast"
        and timestamp(report["origin_close_at"])
        <= timestamp(report["issued_at"])
        < timestamp(report["forward_deadline"])
        and timestamp(confirmation["confirmed_at"]) < timestamp(report["forward_deadline"])
    )
    return report | dict(
        forecast_ref=ref,
        post_commit_confirmed_at=confirmation["confirmed_at"],
        forward=forward,
        mode="forward" if forward else "retrospective_or_abstained",
    )
