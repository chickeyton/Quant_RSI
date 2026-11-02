"""One post-selection fit for publication, separate from every research score."""

from copy import deepcopy

import numpy as np

from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.research.study_estimators import StandardizedRidge

NOTICE = (
    "Selected factors and model settings are frozen. One publication fit uses retained "
    "rows whose labels have matured by the latest price close. Development and terminal "
    "scores belong to the original research model, not this refit. The published forecast "
    "has no observed outcome or accuracy score."
)


def publication_prediction(result):
    """Old studies retain their selected-model forecast; failed refits never fall back."""
    if "latest_fit" in result.get("study", {}):
        return result["study"]["latest_fit"]
    selected = result["iterations"]["selected_round"]
    return result["iterations"]["attempts"][selected - 1]["prediction"]


def refit_latest(model, selected_round, selected):
    _, matrix, _, ids = model._fits[selected_round - 1]
    configuration = deepcopy(model.model_configuration) | {
        "training_policy": "available_mature_labels_through_latest_close"
    }
    requested = np.arange(30, len(model.data.dates) - model.horizon)
    labels = np.log(
        model.data.closes[requested + model.horizon, 0] / model.data.closes[requested, 0]
    )
    available = np.isfinite(matrix[requested]).all(axis=1) & np.isfinite(labels)
    training, labels = requested[available], labels[available]
    result = {
        "status": "unavailable",
        "selected_round": selected_round,
        "factor_ids": deepcopy(ids),
        "model_configuration": configuration,
        "fit_budget": 1,
        "extra_fits": 0,
        "training_sessions": [model.data.dates[i] for i in training],
        "excluded_training_sessions": [model.data.dates[i] for i in requested[~available]],
        "notice": NOTICE,
    }
    missing = [identity for j, identity in enumerate(ids) if not np.isfinite(matrix[-1, j])]
    if missing or len(training) < 12:
        result["latest_forecast_unavailable"] = {
            "origin_session": model.data.dates[-1],
            "reasons": ["latest_selected_features_unavailable"]
            if missing
            else ["insufficient_mature_training_rows"],
            "missing_factor_ids": missing,
        }
        return result
    x = matrix[training]

    def fit():
        if model.estimator == "ridge":
            return StandardizedRidge(x, labels, configuration["parameters"]["alpha"]).state()
        import lightgbm as lgb

        fitted = lgb.train(
            MODEL_PARAMETERS | {"seed": 7},
            lgb.Dataset(x, label=labels),
            num_boost_round=BOOST_ROUNDS,
        )
        return {"model_string": fitted.model_to_string()}

    inputs = {
        "x": x.tolist(),
        "labels": labels.tolist(),
        "training_sessions": result["training_sessions"],
        "configuration": configuration,
        "factor_ids": ids,
    }
    state = model.fit_cache.call("latest_fit", inputs, fit) if model.fit_cache else fit()
    if model.estimator == "ridge":
        predicted = StandardizedRidge.from_state(state).predict(matrix[-1:])
    else:
        import lightgbm as lgb

        predicted = lgb.Booster(model_str=state["model_string"]).predict(matrix[-1:], num_threads=1)
    price = float(model.data.closes[-1, 0] * np.exp(float(predicted[0])))
    if not np.isfinite(price) or price <= 0:
        raise ValueError("Publication model produced an invalid latest forecast")
    forecast = deepcopy(selected.get("latest_forecast", {})) | {
        "origin_session": model.data.dates[-1],
        "reference_price": float(model.data.closes[-1, 0]),
        "predicted_price": price,
        "target_session": model.price_info["latest_target_session"],
        "horizon_sessions": model.horizon,
        "training_information_cutoff": model.data.dates[training[-1] + model.horizon],
        "training_rows": len(training),
        "fit_mode": "refit_selected_factors",
        "outcome": "not_observed",
        "feature_values": {identity: float(matrix[-1, j]) for j, identity in enumerate(ids)},
    }
    result.update(status="completed", extra_fits=1, latest_forecast=forecast)
    return result
