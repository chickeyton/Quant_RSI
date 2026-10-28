"""Durable, idempotent five-session forecasts with replay and post-commit timing."""

import math

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.execution.baseline import read_forecast_bundle
from quant_rsi.forecast.forecast_contracts import ForecastObjective
from quant_rsi.forecast.forecast_kernel import (
    action_assessment,
    forecast_feature_view,
    target_session,
)
from quant_rsi.market import market_contracts
from quant_rsi.market.asof import predict_view, session_index
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.market.market_data import market_context, visible_revisions

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_requests (
    request_id TEXT PRIMARY KEY, binding_ref TEXT NOT NULL, request_ref TEXT NOT NULL,
    result_ref TEXT, confirmation_ref TEXT
);
CREATE TABLE IF NOT EXISTS forecast_failures (
    seq INTEGER PRIMARY KEY, request_id TEXT NOT NULL, failure_ref TEXT NOT NULL
);
"""


def compute_forecast(registry, request):
    """Pure recomputation from a frozen knowledge boundary; never trains or publishes."""
    bundle, model, contract, factor = read_forecast_bundle(registry, request["bundle_ref"])
    snapshot, input_contract = registry.snapshot(request["snapshot_ref"])
    if contract != input_contract or snapshot["contract_ref"] != bundle["contract_ref"]:
        raise ValueError("forecast snapshot differs from the sealed model input contract")
    objective = ForecastObjective.model_validate_json(canonical(bundle["objective"]))
    origin = contract.calendar.sessions[session_index(contract, request["origin_session"])]
    target = target_session(contract, origin.session_id, objective)
    if target is None:
        raise ValueError("five future registered trading sessions are required")
    next_open = contract.calendar.sessions[session_index(contract, origin.session_id) + 1].open_at
    if (
        request["target_session"] != target.session_id
        or request["target_close_at"] != utc_text(target.close_at)
        or request["forward_deadline"] != utc_text(next_open)
        or request["objective_ref"] != bundle["objective_ref"]
        or request["watermark"] != snapshot["receipt_watermark"]
        or request["knowledge_basis"] != "observed_receipts"
    ):
        raise ValueError("forecast request target/knowledge binding mismatch")
    issued = timestamp(request["issued_at"])
    if issued < origin.close_at or issued < timestamp(bundle["model_effective_at"]):
        raise ValueError("forecast cannot precede the origin close or actual model registration")
    view = forecast_feature_view(
        snapshot,
        contract,
        origin.session_id,
        request["issued_at"],
        factor,
        objective,
        basis="observed_receipts",
        watermark=request["watermark"],
    )
    prediction = predict_view(model, view)
    visible = visible_revisions(
        snapshot,
        contract,
        issued,
        basis="observed_receipts",
        watermark=request["watermark"],
    )
    historical = registry.store.json(registry.store.json(bundle["kernel_result_ref"])["report_ref"])
    mean = historical["naive_baselines"]["training_mature_label_mean"]["predicted_log_return"]
    rows = []
    for predicted in prediction["rows"]:
        instrument = predicted["instrument"]
        reasons, actions = action_assessment(
            visible,
            instrument,
            origin.close_at,
            target.close_at,
            objective,
        )
        reasons = list(predicted["reasons"]) + reasons
        quotes = [
            item
            for item in visible
            if item["event"]["kind"] == "CLOSE_QUOTE"
            and item["event"]["instrument"] == instrument
            and item["event"]["session_id"] == origin.session_id
            and item["event"]["quality"] == "valid"
        ]
        anchor = float(quotes[0]["event"]["price"]) if len(quotes) == 1 else None
        if anchor is None:
            reasons.append("origin_close_missing_or_conflicting")
        price = simple_return = None
        if not reasons:
            score = predicted["score"]
            try:
                if not math.isfinite(score):
                    raise ValueError("non-finite forecast score")
                simple_return = math.expm1(score)
                price = anchor * math.exp(score)
                if not math.isfinite(price) or price <= 0:
                    raise ValueError("invalid implied forecast price")
            except (OverflowError, TypeError) as exc:
                raise ValueError(
                    "invalid forecast; no clipping or replacement is permitted"
                ) from exc
        rows.append(
            dict(
                instrument=instrument,
                status="abstain" if reasons else "forecast",
                reasons=sorted(set(reasons)),
                origin_price=anchor,
                forecast_price=price,
                forecast_return=simple_return,
                predicted_log_return=predicted["score"] if not reasons else None,
                unchanged_price_benchmark=anchor,
                training_mean_price_benchmark=anchor * math.exp(mean)
                if anchor is not None
                else None,
                source_refs=predicted["source_refs"],
                action_evidence=actions,
            )
        )
    return dict(
        kind="FiveSessionPriceForecast",
        schema_version=1,
        request_ref=digest(canonical(request)),
        bundle_ref=request["bundle_ref"],
        model_version=bundle["model_ref"],
        objective_ref=bundle["objective_ref"],
        snapshot_ref=request["snapshot_ref"],
        origin_session=origin.session_id,
        data_cutoff=utc_text(origin.close_at),
        issued_at=request["issued_at"],
        target_session=target.session_id,
        target_close_at=utc_text(target.close_at),
        horizon_sessions=5,
        currency=contract.currency,
        price_unit=objective.price_unit,
        price_unit_claim=objective.price_unit_claim,
        knowledge_basis="observed_receipts",
        training_knowledge_basis=contract.knowledge_basis,
        training_information_cutoff=bundle["training_information_cutoff"],
        qualifications=sorted(
            set(
                view["qualifications"]
                + list(objective.source_qualifications)
                + [
                    "historical_exploration_not_independent_performance_evidence",
                    "uncalibrated_point_forecast_no_probability_interval",
                    "price_units_" + objective.price_unit_claim,
                ]
            )
        ),
        feature_view_ref=digest(canonical(view)),
        prediction=prediction,
        rows=rows,
        historical_evaluation=historical["metrics"],
        historical_benchmarks=historical["naive_baselines"],
        economic_or_statistical_advantage_claim=False,
    )


def issue_forecast(bundle_ref, snapshot_ref, origin_session, *, request_id):
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("forecast request ID must contain 1..128 characters")
    binding = dict(bundle_ref=bundle_ref, snapshot_ref=snapshot_ref, origin_session=origin_session)
    binding_ref = digest(canonical(binding))
    with market_context() as (registry, _):
        connection, store = registry.connection, registry.store
        connection.executescript(SCHEMA)
        row = connection.execute(
            "SELECT * FROM forecast_requests WHERE request_id=?", (request_id,)
        ).fetchone()
        if row is not None and row["binding_ref"] != binding_ref:
            raise ValueError("forecast request ID already belongs to different inputs")
        if row is None:
            bundle, _, contract, _ = read_forecast_bundle(registry, bundle_ref)
            snapshot, input_contract = registry.snapshot(snapshot_ref)
            if contract != input_contract:
                raise ValueError("forecast source differs from model contract")
            objective = ForecastObjective.model_validate_json(canonical(bundle["objective"]))
            target = target_session(contract, origin_session, objective)
            if target is None:
                raise ValueError("five future registered trading sessions are required")
            index = session_index(contract, origin_session)
            request = dict(
                kind="ForecastRequest",
                schema_version=1,
                request_id=request_id,
                **binding,
                objective_ref=bundle["objective_ref"],
                issued_at=market_contracts.trusted_now(),
                watermark=snapshot["receipt_watermark"],
                knowledge_basis="observed_receipts",
                target_session=target.session_id,
                target_close_at=utc_text(target.close_at),
                forward_deadline=utc_text(contract.calendar.sessions[index + 1].open_at),
            )
            request_ref = store.put_json(request)
            with connection:
                connection.execute(
                    "INSERT INTO forecast_requests VALUES (?,?,?,NULL,NULL)",
                    (request_id, binding_ref, request_ref),
                )
            row = connection.execute(
                "SELECT * FROM forecast_requests WHERE request_id=?", (request_id,)
            ).fetchone()
        request = store.json(row["request_ref"])
        try:
            if row["result_ref"] is None:
                forecast = compute_forecast(registry, request)
                generated_at = market_contracts.trusted_now()
                if timestamp(generated_at) < timestamp(request["issued_at"]):
                    raise ValueError("trusted forecast clock moved backwards")
                result_ref = store.put_json(dict(forecast=forecast, generated_at=generated_at))
                with connection:
                    connection.execute(
                        "UPDATE forecast_requests SET result_ref=? WHERE request_id=?",
                        (result_ref, request_id),
                    )
            else:
                result_ref = row["result_ref"]
            if row["confirmation_ref"] is None:
                # This clock sample happens strictly after the result ledger commit.
                confirmed_at = market_contracts.trusted_now()
                generated_at = store.json(result_ref)["generated_at"]
                if timestamp(confirmed_at) < timestamp(generated_at):
                    raise ValueError("trusted commit confirmation clock moved backwards")
                confirmation_ref = store.put_json(
                    dict(
                        request_ref=row["request_ref"],
                        result_ref=result_ref,
                        result_commit_confirmed_at=confirmed_at,
                    )
                )
                with connection:
                    connection.execute(
                        "UPDATE forecast_requests SET confirmation_ref=? WHERE request_id=?",
                        (confirmation_ref, request_id),
                    )
        except Exception as exc:
            failure_ref = store.put_json(
                dict(
                    request_ref=row["request_ref"],
                    failed_at=market_contracts.trusted_now(),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
            with connection:
                connection.execute(
                    "INSERT INTO forecast_failures(request_id,failure_ref) VALUES (?,?)",
                    (request_id, failure_ref),
                )
            raise
        return _inspect(registry, request_id, replay=True)


def _inspect(registry, request_id, *, replay):
    row = registry.connection.execute(
        "SELECT * FROM forecast_requests WHERE request_id=?", (request_id,)
    ).fetchone()
    if row is None:
        raise ValueError("forecast request is not registered")
    store = registry.store
    request = store.json(row["request_ref"])
    binding = {key: request[key] for key in ("bundle_ref", "snapshot_ref", "origin_session")}
    if request["request_id"] != request_id or digest(canonical(binding)) != row["binding_ref"]:
        raise ValueError("forecast request ledger binding mismatch")
    failures = [
        item[0]
        for item in registry.connection.execute(
            "SELECT failure_ref FROM forecast_failures WHERE request_id=? ORDER BY seq",
            (request_id,),
        )
    ]
    for ref in failures:
        store.get(ref)
    if row["result_ref"] is None:
        return dict(request_id=request_id, status="pending", request=request, failures=failures)
    result = store.json(row["result_ref"])
    forecast = result["forecast"]
    if forecast["request_ref"] != row["request_ref"] or timestamp(
        result["generated_at"]
    ) < timestamp(request["issued_at"]):
        raise ValueError("forecast result request/time binding mismatch")
    if replay and compute_forecast(registry, request) != forecast:
        raise ValueError("forecast semantic replay differs from the persisted result")
    confirmation = store.json(row["confirmation_ref"]) if row["confirmation_ref"] else None
    if confirmation is not None and (
        confirmation["request_ref"] != row["request_ref"]
        or confirmation["result_ref"] != row["result_ref"]
        or timestamp(confirmation["result_commit_confirmed_at"]) < timestamp(result["generated_at"])
    ):
        raise ValueError("forecast durable confirmation mismatch")
    status = (
        "unconfirmed"
        if confirmation is None
        else "late"
        if timestamp(confirmation["result_commit_confirmed_at"])
        >= timestamp(request["forward_deadline"])
        else "forward"
    )
    if any(item["status"] == "abstain" for item in forecast["rows"]):
        status = "abstain"
    return dict(
        request_id=request_id,
        status=status,
        forecast=forecast,
        generated_at=result["generated_at"],
        confirmation=confirmation,
        forward_deadline=request["forward_deadline"],
        result_ref=row["result_ref"],
        failures=failures,
        replay_verified=replay,
    )


def inspect_forecast(request_id, *, replay=False):
    with market_context() as (registry, _):
        return _inspect(registry, request_id, replay=replay)
