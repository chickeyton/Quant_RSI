"""Numerical features and LightGBM fits on synthetic or retained public daily prices."""

from copy import deepcopy

import numpy as np

from quant_rsi.core.data import synthetic_data
from quant_rsi.core.schemas import DataSpec
from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.research.demo_features import (
    PRICE_RECIPE,
    compact_development,
    price_feature,
    price_metrics,
)


class LightGBMDemo:
    """Two development fits share fixed origins and one mature-label training window."""

    def __init__(self, library, *, prices=None, evaluation_sessions=1):
        if type(evaluation_sessions) is not int or not 1 <= evaluation_sessions <= 20:
            raise ValueError("evaluation_sessions must be an integer from 1 through 20")
        self.evaluation_sessions = evaluation_sessions
        self.library = deepcopy(library)
        self.library["definitions"] = [
            item for item in self.library["definitions"] if item["kind"] == "numeric"
        ]
        self.library["excluded_factor_ids"] = [
            item["factor_id"] for item in library["definitions"] if item["kind"] != "numeric"
        ]
        self.library["execution_note"] = (
            "Only numeric price recipes can enter this model. News guides hypothesis selection; "
            "no historical news features or event_signal columns are constructed."
        )
        recipes = [
            PRICE_RECIPE.validate_python(item["recipe"]) for item in self.library["definitions"]
        ]
        if len(recipes) < 2:
            raise ValueError("LightGBM demo needs at least two numeric factors for one revision")
        self.horizon = 5
        self.price_info = None
        if prices is None:
            self.origin = 100
            self.data = synthetic_data(
                DataSpec(
                    source="synthetic",
                    sessions=128,
                    instruments=1,
                    train_cutoff=self.origin,
                    seed=7,
                )
            )
        else:
            self.data, self.price_info = prices
            self.origin = len(self.data.dates) - 1 - self.horizon
        self.price_provider = "public_daily_prices" if self.price_info else "synthetic_price_series"
        self.origins = np.arange(self.origin - evaluation_sessions + 1, self.origin + 1)
        if evaluation_sessions == 1:
            # Keep the original demonstration's window and output unchanged.
            self.train = np.arange(
                max(recipe.window for recipe in recipes), self.origin - self.horizon + 1
            )
        else:
            # Reserve enough history for every safe proposed window (1..30).
            # No training label reaches even the first evaluation origin.
            self.train = np.arange(30, self.origins[0] - self.horizon)
        if len(self.train) < 12:
            raise ValueError("Insufficient mature training rows for the selected price window")

    def context(self, symbol):
        if self.price_info and self.data.instruments != (symbol,):
            raise ValueError("Price data does not match the target symbol")
        result = {
            "provider": self.price_provider,
            "reference_price": float(self.data.closes[self.origin, 0]),
            "origin_session": self.data.dates[self.origin],
            "model_mode": "lightgbm",
            "price_evidence": deepcopy(self.price_info),
            "model_note": (
                (
                    "Real training on retained public daily prices. "
                    if self.price_info
                    else f"Real training on synthetic prices, not actual prices of {symbol}. "
                )
                + "Both development estimates share one historical origin. Current news is "
                "research context only, not historically available input. Text factors are "
                "unavailable. A separate latest_forecast, when present, has no observed outcome."
            ),
        }
        if self.evaluation_sessions > 1:
            result["evaluation_sessions"] = self.evaluation_sessions
            result["first_evaluation_session"] = self.data.dates[self.origins[0]]
            result["model_note"] = (
                "Retrospective development study on "
                + ("retained public daily prices. " if self.price_info else "synthetic prices. ")
                + "Both revisions use the same training window and evaluation origins. "
                "Current news guides retrospective factor selection, not historically available "
                "features. The second revision reuses disclosed development outcomes, so this "
                "is not independent holdout validation. Five-session outcomes overlap."
            )
        return result

    def predict(self, context, factors, hypothesis):
        import lightgbm as lgb

        if not factors or any(item["kind"] != "numeric" for item in factors):
            raise ValueError("LightGBM demo accepts only selected numeric price factors")
        matrix = np.column_stack([price_feature(self.data, item["recipe"]) for item in factors])
        x = matrix[self.train]
        y = np.log(self.data.closes[self.train + self.horizon, 0] / self.data.closes[self.train, 0])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite numerical training inputs")
        if not np.isfinite(matrix[self.origins]).all():
            raise ValueError("Non-finite numerical evaluation inputs")
        model = lgb.train(
            MODEL_PARAMETERS | {"seed": 7}, lgb.Dataset(x, label=y), num_boost_round=BOOST_ROUNDS
        )
        predicted_returns = np.asarray(model.predict(matrix[self.origins], num_threads=1))
        value = float(predicted_returns[-1])
        price = float(self.data.closes[self.origin, 0] * np.exp(value))
        if not np.isfinite(price) or price <= 0:
            raise ValueError("LightGBM demo produced an invalid price")
        result = {
            "provider": "lightgbm",
            "price_provider": self.price_provider,
            "predicted_price": price,
            "predicted_log_return": value,
            "factor_ids": [item["factor_id"] for item in factors],
            "feature_values": {
                item["factor_id"]: float(matrix[self.origin, index])
                for index, item in enumerate(factors)
            },
            "horizon_sessions": self.horizon,
            "origin_session": self.data.dates[self.origin],
            "training_rows": len(self.train),
            "training_feature_columns": x.shape[1],
            "last_training_label_session": self.data.dates[self.train[-1] + self.horizon],
            "training_target": "five_session_close_log_return",
            "boost_rounds": BOOST_ROUNDS,
        }
        if self.evaluation_sessions > 1:
            result["development"] = self._development(predicted_returns, float(y.mean()))
        if self.price_info:
            if not np.isfinite(matrix[-1:]).all():
                raise ValueError("Non-finite numerical latest-forecast inputs")
            latest_return = float(model.predict(matrix[-1:], num_threads=1)[0])
            latest_price = float(self.data.closes[-1, 0] * np.exp(latest_return))
            if not np.isfinite(latest_price) or latest_price <= 0:
                raise ValueError("LightGBM produced an invalid latest forecast")
            result["latest_forecast"] = {
                "origin_session": self.data.dates[-1],
                "reference_price": float(self.data.closes[-1, 0]),
                "predicted_price": latest_price,
                "target_session": self.price_info["latest_target_session"],
                "horizon_sessions": self.horizon,
                "training_information_cutoff": self.data.dates[
                    self.origin if self.evaluation_sessions == 1 else self.train[-1] + self.horizon
                ],
                "outcome": "not_observed",
            }
        return result

    def _development(self, predicted_returns, training_mean):
        def rows(returns):
            result = []
            for origin, value in zip(self.origins, returns, strict=True):
                reference = float(self.data.closes[origin, 0])
                predicted = float(reference * np.exp(value))
                if not np.isfinite(predicted) or predicted <= 0:
                    raise ValueError("LightGBM study produced an invalid development price")
                result.append(
                    {
                        "origin_session": self.data.dates[origin],
                        "target_session": self.data.dates[origin + self.horizon],
                        "reference_price": reference,
                        "predicted_price": predicted,
                        "observed_price": float(self.data.closes[origin + self.horizon, 0]),
                        "predicted_log_return": float(value),
                    }
                )
            return result

        predicted = rows(predicted_returns)
        baselines = {}
        for name, value in (("unchanged_price", 0.0), ("training_label_mean", training_mean)):
            baseline_rows = rows(np.full(self.evaluation_sessions, value))
            baselines[name] = {
                "prediction": "zero_log_return"
                if name == "unchanged_price"
                else "training_mean_log_return",
                "predicted_log_return": value,
                "rows": baseline_rows,
                "metrics": price_metrics(baseline_rows),
            }
        return {
            "rows": predicted,
            "metrics": price_metrics(predicted),
            "baselines": baselines,
            "metric_units": {
                "mae_price": "quote_price",
                "rmse_price": "quote_price",
                "mape_percent": "percent",
                "direction_hit_rate": "fraction",
                "sample_count": "observations",
            },
            "currency": self.price_info.get("currency") if self.price_info else None,
            "first_evaluation_session": self.data.dates[self.origins[0]],
            "last_evaluation_session": self.data.dates[self.origins[-1]],
            "evaluation_note": "retrospective_development_same_data_feedback_reuse",
        }

    def observe(self, context, prediction):
        result = {
            "provider": "public_daily_observation"
            if self.price_info
            else "synthetic_price_observation",
            "observed_price": float(self.data.closes[self.origin + self.horizon, 0]),
            "observed_session": self.data.dates[self.origin + self.horizon],
            "origin_session": self.data.dates[self.origin],
        }
        if self.evaluation_sessions > 1:
            result["development"] = deepcopy(compact_development(prediction["development"]))
        return result

    def summarize(self, context, first, second, feedback):
        second_error = second["predicted_price"] - feedback["observed_price"]
        source = "retained public daily" if self.price_info else "synthetic"
        result = {
            "provider": "development_model_summary",
            "text": (
                f"{context['symbol']}: two real LightGBM fits on {source} prices. "
                f"Initial price error {feedback['error']:+.4f}; revised price error "
                f"{second_error:+.4f} against the same disclosed {source} observation. "
                "The revised estimate has used this feedback; it is not independent validation "
                "or evidence of predictive gain."
            ),
            "factors_changed": first["factor_ids"] != second["factor_ids"],
            "revised_development_error": second_error,
            "second_outcome": "reused_development_observation",
            "improvement": "not_evaluated",
        }
        if self.price_info:
            result["latest_forecast"] = second["latest_forecast"]
            result["text"] += (
                " The separate latest forecast uses the same fitted model; outcome unobserved."
            )
        if self.evaluation_sessions > 1:
            result["development"] = {
                "first_metrics": deepcopy(first["development"]["metrics"]),
                "revised_metrics": deepcopy(second["development"]["metrics"]),
                "baselines": deepcopy(compact_development(first["development"])["baselines"]),
                "evaluation_note": "retrospective_development_same_data_feedback_reuse",
            }
            result["text"] += (
                f" Metrics describe the same {self.evaluation_sessions} matured origins; "
                "news informs retrospective selection and the revised fit reuses these disclosed "
                "outcomes. Overlapping five-session labels are not independent samples."
            )
        return result
