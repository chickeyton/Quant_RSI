"""One diagnostic refit without financial features, after research selection is frozen."""

import numpy as np

from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.research.demo_strategy import evaluate_strategy
from quant_rsi.research.study_estimators import StandardizedRidge


def compare_financials(model, selected_round, prediction, policy):
    _, matrix, labels, ids = model._fits[selected_round - 1]
    financial_ids = {d["factor_id"] for d in model.financial_features["definitions"]}
    removed = [i for i, identity in enumerate(ids) if identity in financial_ids]
    kept = [i for i in range(len(ids)) if i not in removed]
    notice = (
        "One additional diagnostic fit on identical training/development rows and model settings; "
        "only financial columns are removed. This result does not alter factor/model selection "
        "or the latest forecast. Retrospective development evidence, not independent gain."
    )
    if not removed or not kept:
        return {
            "status": "not_applicable",
            "extra_fits": 0,
            "notice": notice,
            "reason": "selected_model_needs_both_financial_and_nonfinancial_columns",
        }
    x = matrix[model.train][:, kept]
    if model.estimator == "ridge":
        estimator = StandardizedRidge(x, labels, model.model_configuration["parameters"]["alpha"])
        returns = estimator.predict(matrix[model.origins][:, kept])
    else:
        import lightgbm as lgb

        estimator = lgb.train(
            MODEL_PARAMETERS | {"seed": 7},
            lgb.Dataset(x, label=labels),
            num_boost_round=BOOST_ROUNDS,
        )
        returns = estimator.predict(matrix[model.origins][:, kept], num_threads=1)
    evaluation = model._development(np.asarray(returns), float(np.mean(labels)))
    accounts = evaluate_strategy(
        model.data, model.data.instruments[0], evaluation["rows"], **policy
    )
    return {
        "status": "completed",
        "extra_fits": 1,
        "selected_round": selected_round,
        "removed_factor_ids": [ids[i] for i in removed],
        "kept_factor_ids": [ids[i] for i in kept],
        "notice": notice,
        "with_financials": {
            "forecast": prediction["development"]["metrics"],
            "strategy": prediction["strategy"]["strategy"]["metrics"],
        },
        "without_financials": {
            "forecast": evaluation["metrics"],
            "strategy": accounts["strategy"]["metrics"],
        },
        "training_sessions": [model.data.dates[i] for i in model.train],
        "development_sessions": [model.data.dates[i] for i in model.origins],
    }
