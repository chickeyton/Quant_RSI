"""Numerical features and LightGBM fits on synthetic or retained public daily prices."""

from copy import deepcopy

import numpy as np

from quant_rsi.core.data import features, synthetic_data
from quant_rsi.core.schemas import DataSpec, FactorSpec
from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS


class LightGBMDemo:
    """Two development fits share one origin and mature-label training window."""

    def __init__(self, library, *, prices=None):
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
            FactorSpec.model_validate(item["recipe"]) for item in self.library["definitions"]
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
        self.train = np.arange(
            max(recipe.window for recipe in recipes), self.origin - self.horizon + 1
        )
        if len(self.train) < 12:
            raise ValueError("Insufficient mature training rows for the selected price window")

    def context(self, symbol):
        if self.price_info and self.data.instruments != (symbol,):
            raise ValueError("Price data does not match the target symbol")
        return {
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

    def predict(self, context, factors, hypothesis):
        import lightgbm as lgb

        if not factors or any(item["kind"] != "numeric" for item in factors):
            raise ValueError("LightGBM demo accepts only selected numeric price factors")
        matrix = np.column_stack(
            [
                features(self.data, FactorSpec.model_validate(item["recipe"]))[:, 0, 0]
                for item in factors
            ]
        )
        x = matrix[self.train]
        y = np.log(self.data.closes[self.train + self.horizon, 0] / self.data.closes[self.train, 0])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite numerical training inputs")
        model = lgb.train(
            MODEL_PARAMETERS | {"seed": 7}, lgb.Dataset(x, label=y), num_boost_round=BOOST_ROUNDS
        )
        value = float(model.predict(matrix[self.origin : self.origin + 1], num_threads=1)[0])
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
        if self.price_info:
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
                "training_information_cutoff": self.data.dates[self.origin],
                "outcome": "not_observed",
            }
        return result

    def observe(self, context, prediction):
        return {
            "provider": "public_daily_observation"
            if self.price_info
            else "synthetic_price_observation",
            "observed_price": float(self.data.closes[self.origin + self.horizon, 0]),
            "observed_session": self.data.dates[self.origin + self.horizon],
            "origin_session": self.data.dates[self.origin],
        }

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
        return result
