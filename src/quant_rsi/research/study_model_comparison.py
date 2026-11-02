"""One alternate estimator on the selected study's frozen features and training labels."""

from copy import deepcopy

import numpy as np

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.research.demo_features import price_metrics
from quant_rsi.research.demo_strategy import evaluate_strategy
from quant_rsi.research.study_estimators import StandardizedRidge, model_configuration

NOTICE = (
    "One additional model fit on identical selected factors, mature training labels and "
    "evaluation sessions. LightGBM uses the fixed study settings; alternate Ridge uses "
    "alpha=1 with training-only standardization. No parameter search or feedback to "
    "selection, publication or the LLM. Factors were already selected during research: "
    "this is a post-selection model comparison, not independent research-policy evidence."
)


def compare_models(bundle, data, selected, terminal, *, policy):
    """Fit without evaluation outcomes, then score both models on the same retained rows."""
    primary = bundle["model_configuration"]["name"]
    alternate = "ridge" if primary == "lightgbm" else "lightgbm"
    configuration = model_configuration(alternate, horizon_sessions=bundle["horizon_sessions"])
    train = bundle["segments"]["train"]
    development = bundle["segments"]["development"]
    if not train["available"] or not development["available"]:
        raise ValueError("Model comparison requires complete common training/development inputs")
    if max(train["label_sessions"]) > min(development["sessions"]):
        raise ValueError("Model comparison requires mature training labels")
    x, labels = np.asarray(train["features"], dtype=float), np.asarray(train["labels"], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(labels).all():
        raise ValueError("Model comparison requires finite training inputs")
    if alternate == "ridge":
        fitted = StandardizedRidge(x, labels, 1.0)
        predict = fitted.predict
    else:
        import lightgbm as lgb

        fitted = lgb.train(
            MODEL_PARAMETERS | {"seed": 7},
            lgb.Dataset(x, label=labels),
            num_boost_round=BOOST_ROUNDS,
        )

        def predict(values):
            return fitted.predict(values, num_threads=1)

    result = {
        "status": "completed",
        "notice": NOTICE,
        "selected_round": bundle["selected_round"],
        "factor_ids": deepcopy(bundle["factor_ids"]),
        "inputs_sha256": digest(canonical(bundle)),
        "account_policy": deepcopy(policy),
        "comparison_fits": 1,
        "selected_model": primary,
        "models": {primary: deepcopy(bundle["model_configuration"]), alternate: configuration},
        "segments": {},
    }
    for name in ("development", "terminal"):
        segment = bundle["segments"][name]
        if not segment["available"] or (
            name == "terminal" and (terminal or {}).get("status") != "completed"
        ):
            result["segments"][name] = {
                "status": "unavailable",
                "reason": "disabled_or_unavailable_common_evaluation; no rows dropped",
            }
            continue
        native = selected["development"] if name == "development" else terminal["evaluation"]
        account = selected["strategy"] if name == "development" else terminal["strategy"]
        rows = deepcopy(native["rows"])
        if [row["origin_session"] for row in rows] != segment["sessions"]:
            raise ValueError("Model comparison differs from the frozen evaluation population")
        values = np.asarray(segment["features"], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Model comparison cannot drop incomplete evaluation rows")
        for row, value in zip(rows, predict(values), strict=True):
            row["predicted_log_return"] = float(value)
            row["predicted_price"] = float(row["reference_price"] * np.exp(value))
            if not np.isfinite(row["predicted_price"]) or row["predicted_price"] <= 0:
                raise ValueError("Alternate model produced an invalid price forecast")
        alternate_account = evaluate_strategy(data, bundle["symbol"], rows, **policy)
        if name == "terminal":
            alternate_account["evidence_mode"] = "historical_terminal_hypothetical_account"
        baselines = {
            key: {
                "metrics": deepcopy(value["metrics"]),
                "strategy": evaluate_strategy(data, bundle["symbol"], value["rows"], **policy),
            }
            for key, value in native["baselines"].items()
        }
        if name == "terminal":
            for value in baselines.values():
                value["strategy"]["evidence_mode"] = "historical_terminal_hypothetical_account"
        result["segments"][name] = {
            "status": "completed",
            "sample_count": len(rows),
            "sessions": list(segment["sessions"]),
            "baselines": baselines,
            "buy_and_hold": deepcopy(account["buy_and_hold"]),
            "models": {
                primary: {"metrics": deepcopy(native["metrics"]), "strategy": deepcopy(account)},
                alternate: {
                    "metrics": price_metrics(rows),
                    "rows": rows,
                    "strategy": alternate_account,
                },
            },
        }
    return result


def comparison_rows(comparison):
    """Shared compact table for the HTML report and live workbench."""
    rows = []
    for window, segment in comparison["segments"].items():
        if segment["status"] != "completed":
            continue
        candidates = segment["models"] | segment["baselines"]
        candidates["buy_and_hold"] = {
            "metrics": {},
            "strategy": {"strategy": segment["buy_and_hold"]},
        }
        for name, value in candidates.items():
            metrics = value["strategy"]["strategy"]["metrics"]
            rows.append(
                {
                    "window": window,
                    "model": name.replace("_", " ")
                    + (" (selected)" if name == comparison["selected_model"] else ""),
                    "sample_count": None if name == "buy_and_hold" else segment["sample_count"],
                    "daily_return_count": metrics.get("risk_statistics", {}).get(
                        "daily_return_count"
                    ),
                    "mae": value["metrics"].get("mae_price"),
                    "direction_hit_rate": value["metrics"].get("direction_hit_rate"),
                    **{
                        key: metrics.get(key)
                        for key in ("net_return", "sharpe_ratio", "max_drawdown")
                    },
                }
            )
    return rows
