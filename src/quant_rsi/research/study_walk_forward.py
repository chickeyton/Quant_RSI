"""Post-selection chronological refits using one frozen factor matrix and model policy."""

from copy import deepcopy

import numpy as np

from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.research.demo_features import price_metrics
from quant_rsi.research.demo_strategy import evaluate_strategy
from quant_rsi.research.study_estimators import StandardizedRidge

NOTICE = (
    "Post-selection retrospective diagnostic: factors and model settings are frozen after "
    "research, then refitted on expanding mature-label training windows. Current news, LLM "
    "knowledge and recipe selection were not historically isolated. These are not independent "
    "out-of-sample research results. No feedback to selection or latest forecast. Each fold "
    "starts a separate account; returns and Sharpe are not a continuous portfolio track record."
)


def extra_history(folds, sessions, horizon):
    if type(folds) is not int or folds not in (0, 2, 3):
        raise ValueError("Walk-forward diagnostic needs 0, 2 or 3 folds")
    return max(0, folds - 1) * (sessions + horizon)


def price_history_sessions(horizon, sessions, folds, requested=None):
    """One collection requirement for direct studies and workbench preparation."""
    minimum = 120 + 3 * (horizon - 5) + extra_history(folds, sessions, horizon)
    if requested is None:
        return minimum
    if type(requested) is not int or not minimum <= requested <= 445:
        raise ValueError(
            f"Historical sessions must be an integer from {minimum} through 445 "
            "for this horizon and diagnostic plan"
        )
    return requested


def freeze_windows(data, last_origin, horizon, sessions, folds):
    """Choose dates before any hypotheses; keep fold valuation intervals disjoint."""
    extra_history(folds, sessions, horizon)
    windows = []
    for number in range(folds):
        end = last_origin - (folds - number - 1) * (sessions + horizon)
        start = end - sessions + 1
        if start - horizon - 30 < 12 or end + horizon >= len(data.dates):
            raise ValueError("Insufficient retained history for the walk-forward plan")
        windows.append(
            {
                "fold": number + 1,
                "origin_indices": list(range(start, end + 1)),
                "first_origin": data.dates[start],
                "last_origin": data.dates[end],
                "last_target": data.dates[end + horizon],
                "training_label_boundary": data.dates[start],
            }
        )
    return {"folds": windows, "fit_budget": folds, "notice": NOTICE}


def evaluate_windows(model, selected_round, policy):
    """Refit only the already selected features; retain any unavailable fold explicitly."""
    _, matrix, _, ids = model._fits[selected_round - 1]
    h = model.horizon
    configuration = deepcopy(model.model_configuration) | {
        "training_policy": "expanding_mature_label_window_per_diagnostic_fold"
    }
    results = []
    for window in model.walk_forward_plan["folds"]:
        origins = np.asarray(window["origin_indices"])
        requested = np.arange(30, origins[0] - h)
        labels = np.log(model.data.closes[requested + h, 0] / model.data.closes[requested, 0])
        available = np.isfinite(matrix[requested]).all(axis=1) & np.isfinite(labels)
        training, labels = requested[available], labels[available]
        missing = origins[~np.isfinite(matrix[origins]).all(axis=1)]
        result = deepcopy(window) | {
            "training_sessions": [model.data.dates[i] for i in training],
            "excluded_training_sessions": [model.data.dates[i] for i in requested[~available]],
            "missing_evaluation_sessions": [model.data.dates[i] for i in missing],
        }
        if len(training) < 12 or len(missing):
            results.append(
                result
                | {
                    "status": "unavailable",
                    "extra_fits": 0,
                    "reason": "insufficient_mature_training_rows"
                    if len(training) < 12
                    else "missing_evaluation_inputs",
                }
            )
            continue
        x = matrix[training]

        def fit(x=x, labels=labels):
            if model.estimator == "ridge":
                return StandardizedRidge(
                    x, labels, model.model_configuration["parameters"]["alpha"]
                ).state()
            import lightgbm as lgb

            estimator = lgb.train(
                MODEL_PARAMETERS | {"seed": 7},
                lgb.Dataset(x, label=labels),
                num_boost_round=BOOST_ROUNDS,
            )
            return {"model_string": estimator.model_to_string()}

        if model.fit_cache:
            state = model.fit_cache.call(
                f"walk_forward_fit_{window['fold']}",
                {
                    "x": x.tolist(),
                    "labels": labels.tolist(),
                    "training_sessions": result["training_sessions"],
                    "configuration": configuration,
                    "factor_ids": ids,
                },
                fit,
            )
        else:
            state = fit()
        if model.estimator == "ridge":
            returns = StandardizedRidge.from_state(state).predict(matrix[origins])
        else:
            import lightgbm as lgb

            returns = lgb.Booster(model_str=state["model_string"]).predict(
                matrix[origins], num_threads=1
            )
        evaluation = model._development(
            np.asarray(returns), float(np.mean(labels)), origins=origins
        )
        results.append(
            result
            | {
                "status": "completed",
                "extra_fits": 1,
                "last_training_label_session": model.data.dates[training[-1] + h],
                "evaluation": evaluation,
                "strategy": evaluate_strategy(
                    model.data, model.data.instruments[0], evaluation["rows"], **policy
                ),
            }
        )
    complete = bool(results) and all(r["status"] == "completed" for r in results)
    pooled = None
    if complete:
        pooled = {
            "model": price_metrics([row for r in results for row in r["evaluation"]["rows"]]),
            **{
                name: price_metrics(
                    [row for r in results for row in r["evaluation"]["baselines"][name]["rows"]]
                )
                for name in ("unchanged_price", "training_label_mean")
            },
        }
    return {
        "status": "completed" if complete else "incomplete",
        "selected_round": selected_round,
        "factor_ids": deepcopy(ids),
        "model_configuration": configuration,
        "horizon_sessions": h,
        "fit_budget": len(results),
        "extra_fits": sum(r["extra_fits"] for r in results),
        "folds": results,
        "pooled_forecast_metrics": pooled,
        "notice": NOTICE,
    }
