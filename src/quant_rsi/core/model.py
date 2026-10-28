import numpy as np

from quant_rsi.core.artifacts import digest
from quant_rsi.core.data import MarketData, decision_indices, features
from quant_rsi.core.schemas import DevelopmentPlan, FactorSpec


def predict_fixed(data: MarketData, factor: FactorSpec, model_text: str, cutoff: int) -> list:
    """Inference only; the serialized model and preprocessing are fixed before exposure."""
    import lightgbm as lgb

    indices = list(decision_indices(data, cutoff))
    x = features(data, factor)[indices]
    if not np.isfinite(x).all():
        raise ValueError("non-finite fixed-model features")
    model = lgb.Booster(model_str=model_text)
    if model.num_feature() != x.shape[-1]:
        raise ValueError("sealed model feature interface differs")
    scores = model.predict(x.reshape(-1, x.shape[-1]), num_threads=1).reshape(
        len(indices), len(data.instruments)
    )
    if not np.isfinite(scores).all():
        raise ValueError("non-finite fixed-model predictions")
    return [
        {
            "decision": t,
            "decision_time": data.timestamp(t, "close"),
            "instrument": instrument,
            "score": float(scores[row, col]),
        }
        for row, t in enumerate(indices)
        for col, instrument in enumerate(data.instruments)
    ]


def fit_predict(data: MarketData, plan: DevelopmentPlan, factor: FactorSpec) -> dict:
    if plan.data.source == "local_event_snapshot":
        from quant_rsi.evaluation.local_research import fit_predict_local

        return fit_predict_local(data, plan, factor)
    import lightgbm as lgb

    x = features(data, factor)
    cutoff = plan.data.train_cutoff
    # Label for t matures at open t+2; it must be known by close cutoff.
    train = np.arange(factor.window, cutoff - 1)
    if len(train) * len(data.instruments) < 12:
        raise ValueError("insufficient mature training observations for this factor")
    y = data.opens[train + 2] / data.opens[train + 1] - 1
    train_x = x[train].reshape(-1, x.shape[-1])
    if not np.isfinite(train_x).all():
        raise ValueError("non-finite training features")
    params = {
        "objective": "regression",
        "verbosity": -1,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": plan.data.seed,
        "num_leaves": 7,
        "min_data_in_leaf": 5,
        "learning_rate": 0.05,
        "feature_pre_filter": False,
    }
    model = lgb.train(params, lgb.Dataset(train_x, label=y.reshape(-1)), num_boost_round=24)
    indices = list(decision_indices(data, cutoff))
    eval_x = x[indices].reshape(-1, x.shape[-1])
    if not np.isfinite(eval_x).all():
        raise ValueError("non-finite prediction features")
    scores = model.predict(eval_x).reshape(len(indices), len(data.instruments))
    model_text = model.model_to_string()
    return {
        "model_text": model_text,
        "model_digest": digest(model_text.encode()),
        "training_information_cutoff": data.timestamp(cutoff, "close"),
        "last_training_label_available": data.timestamp(int(train[-1]) + 2, "open"),
        "training_rows": len(train_x),
        "predictions": [
            {
                "decision": t,
                "decision_time": data.timestamp(t, "close"),
                "instrument": instrument,
                "score": float(scores[row, col]),
            }
            for row, t in enumerate(indices)
            for col, instrument in enumerate(data.instruments)
        ],
    }
