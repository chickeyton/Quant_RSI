"""Five-session forecasts in the existing local worker and C2 result/replay path."""

import math

import numpy as np

from quant_rsi.artifacts import canonical, digest
from quant_rsi.asof import predict_view, qualifications
from quant_rsi.forecast_kernel import (
    forecast_errors,
    forecast_feature_view,
    mature_forecast_label,
    target_session,
)
from quant_rsi.market_contracts import timestamp, utc_text
from quant_rsi.schemas import FactorSpec


def evaluation_origins(plan):
    return range(
        plan.data.train_cutoff,
        plan.data.sessions - plan.objective.horizon_sessions,
        plan.evaluation_stride,
    )


def training_view(data, plan, factor):
    if data.spec != plan.data:
        raise ValueError("forecast data differs from the bound plan")
    cutoff = data.decision_at(plan.data.train_cutoff)
    rows, x, y, used, available = [], [], [], set(), []
    for index in range(factor.window, plan.data.train_cutoff):
        view = forecast_feature_view(
            data.snapshot,
            data.contract,
            data.dates[index],
            data.decision_at(index),
            factor,
            plan.objective,
        )
        for feature in view["rows"]:
            label = mature_forecast_label(
                data.snapshot,
                data.contract,
                data.dates[index],
                feature["instrument"],
                cutoff,
                plan.objective,
            )
            eligible = feature["status"] == "eligible" and label["status"] == "mature"
            # Return intervals are (origin close, target close]; touching the first
            # evaluation origin is allowed, crossing it is not.
            if eligible and timestamp(label["interval"][1]) > timestamp(cutoff):
                raise ValueError("training label overlaps the registered evaluation interval")
            rows.append(
                dict(
                    session_id=data.dates[index],
                    instrument=feature["instrument"],
                    feature_view_ref=digest(canonical(view)),
                    feature=feature,
                    label=label,
                    training_eligible=eligible,
                )
            )
            if eligible:
                x.append(feature["features"])
                y.append(label["value"])
                used.update(feature["source_refs"] + label["source_refs"])
                available.append(timestamp(label["available_at"]))
    if len(x) < 12:
        raise ValueError("insufficient mature, eligible five-session training observations")
    market_times = [
        timestamp(row["event"]["event_at"])
        for row in data.snapshot["events"]
        if row["payload_ref"] in used
    ]
    view = dict(
        kind="ForecastTrainingView",
        objective_ref=digest(canonical(plan.objective.model_dump(mode="json"))),
        knowledge_basis=data.contract.knowledge_basis,
        snapshot_ref=data.spec.snapshot_ref,
        training_view_asof=utc_text(cutoff),
        rows=rows,
        interval_policy="(origin_close,target_close]; target_close<=first_evaluation_origin",
    )
    metadata = dict(
        training_view_ref=digest(canonical(view)),
        training_view_asof=utc_text(cutoff),
        training_information_cutoff=utc_text(max(market_times)),
        last_training_label_available=utc_text(max(available)),
        training_rows=len(x),
        knowledge_basis=data.contract.knowledge_basis,
        qualifications=qualifications(data.contract),
        forecast_objective_ref=view["objective_ref"],
    )
    return np.asarray(x), np.asarray(y), metadata, view


def development_predictions(data, plan, factor, model_text):
    result = []
    objective_ref = digest(canonical(plan.objective.model_dump(mode="json")))
    for index in evaluation_origins(plan):
        view = forecast_feature_view(
            data.snapshot,
            data.contract,
            data.dates[index],
            data.decision_at(index),
            factor,
            plan.objective,
        )
        prediction = predict_view(model_text, view)
        target = target_session(data.contract, data.dates[index], plan.objective)
        prediction.update(
            kind="FiveSessionForecastPrediction",
            factor=factor.model_dump(mode="json"),
            forecast_objective_ref=objective_ref,
            target_session=target.session_id,
            target_close_at=utc_text(target.close_at),
        )
        result.append(prediction)
    return result


def training_mean_baseline(data, plan, factor):
    """No fit: use exactly the model's eligible training labels and feature/label mask."""
    _, labels, metadata, view = training_view(data, plan, factor)
    selected = [row for row in view["rows"] if row["training_eligible"]]
    refs = {
        ref
        for row in selected
        for ref in row["feature"]["source_refs"] + row["label"]["source_refs"]
    }
    return math.fsum(labels) / len(labels), dict(
        method="fixed_mean_of_same_model_training_labels",
        label_count=len(labels),
        as_of=metadata["training_view_asof"],
        training_view_ref=metadata["training_view_ref"],
        eligible_grid=[
            dict(session_id=row["session_id"], instrument=row["instrument"]) for row in selected
        ],
        source_refs=sorted(refs),
        additional_ML_fits=0,
    )


def evaluate(data, plan, predictions):
    origins = list(evaluation_origins(plan))
    if len(predictions) != len(origins):
        raise ValueError("forecast cannot omit a fixed evaluation origin")
    factor = FactorSpec.model_validate_json(canonical(predictions[0]["factor"]))
    if factor.model_dump(mode="json") not in [proposal.factor for proposal in plan.proposals]:
        raise ValueError("forecast factor is outside the bound proposals")
    objective_ref = digest(canonical(plan.objective.model_dump(mode="json")))
    labels_asof = data.decision_at(plan.data.sessions - 1)
    rows, pred_values, actual_values, reasons = [], [], [], []
    for index, prediction in zip(origins, predictions, strict=True):
        target = target_session(data.contract, data.dates[index], plan.objective)
        if (
            prediction["factor"] != factor.model_dump(mode="json")
            or prediction["session_id"] != data.dates[index]
            or prediction["as_of"] != utc_text(data.decision_at(index))
            or prediction["forecast_objective_ref"] != objective_ref
            or prediction["target_session"] != target.session_id
            or prediction["target_close_at"] != utc_text(target.close_at)
            or [r["instrument"] for r in prediction["rows"]] != list(data.instruments)
        ):
            raise ValueError("forecast origin/target/objective/instrument grid mismatch")
        for predicted in prediction["rows"]:
            label = mature_forecast_label(
                data.snapshot,
                data.contract,
                data.dates[index],
                predicted["instrument"],
                labels_asof,
                plan.objective,
            )
            eligible = predicted["status"] == "eligible" and label["status"] == "mature"
            rows.append(
                dict(
                    origin_session=data.dates[index],
                    target_session=target.session_id,
                    instrument=predicted["instrument"],
                    label=label,
                    prediction=predicted,
                    eligible=eligible,
                )
            )
            if not eligible:
                reasons.append("incomplete_fixed_forecast_grid")
            else:
                if predicted["score"] is None:
                    raise ValueError("eligible forecast is missing its score")
                pred_values.append(predicted["score"])
                actual_values.append(label["value"])
    if len(origins) < plan.min_development_sessions:
        reasons.append("insufficient_fixed_forecast_origins")
    complete = not reasons
    metrics = (
        forecast_errors(pred_values, actual_values)
        if complete
        else {"mae_pp": None, "rmse_pp": None, "negative_mae_pp": None, "observations": len(rows)}
    )
    mean, mean_evidence = training_mean_baseline(data, plan, factor)
    baselines = {
        "unchanged_price": {
            "predicted_log_return": 0.0,
            "additional_ML_fits": 0,
            "metrics": forecast_errors([0.0] * len(actual_values), actual_values)
            if complete
            else None,
        },
        "training_mature_label_mean": {
            "predicted_log_return": mean,
            "evidence": mean_evidence,
            "metrics": forecast_errors([mean] * len(actual_values), actual_values)
            if complete
            else None,
        },
    }
    return dict(
        mode="forecast_historical_exploration",
        forward=False,
        forecast_objective_ref=objective_ref,
        knowledge_basis=data.contract.knowledge_basis,
        qualifications=qualifications(data.contract),
        label_asof=utc_text(labels_asof),
        origin_indices=origins,
        rows=rows,
        eligible=complete,
        eligibility_reasons=sorted(set(reasons)),
        metrics=metrics,
        naive_baselines=baselines,
        fixed_grid_incomplete_means_no_selection_score=True,
        metric_unit="simple_return_percentage_points",
        prior_exposure=plan.objective.prior_exposure,
        economic_or_statistical_advantage_claim=False,
    )
