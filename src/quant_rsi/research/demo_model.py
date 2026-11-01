"""Shared numerical study path with LightGBM or standardized ridge fits."""

from copy import deepcopy
from datetime import datetime

import numpy as np

from quant_rsi.core.artifacts import canonical
from quant_rsi.core.data import synthetic_data
from quant_rsi.core.schemas import DataSpec
from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS
from quant_rsi.research.demo_features import (
    PRICE_RECIPE,
    compact_development,
    price_feature,
    price_metrics,
)
from quant_rsi.research.study_estimators import StandardizedRidge, model_configuration
from quant_rsi.research.study_factor_profiles import prepare_bound_features


class LightGBMDemo:
    """Development fits share fixed origins and one mature-label training window."""

    def __init__(
        self,
        library,
        *,
        prices=None,
        evaluation_sessions=1,
        terminal_sessions=0,
        news_history=None,
        news_cutoffs=None,
        news_symbol=None,
        estimator="lightgbm",
        ridge_alpha=None,
        fit_cache=None,
    ):
        self.fit_cache = fit_cache
        self.model_configuration = model_configuration(estimator, ridge_alpha)
        self.estimator = estimator
        self.model_label = "LightGBM" if estimator == "lightgbm" else "Ridge"
        if type(evaluation_sessions) is not int or not 1 <= evaluation_sessions <= 20:
            raise ValueError("evaluation_sessions must be an integer from 1 through 20")
        self.evaluation_sessions = evaluation_sessions
        if type(terminal_sessions) is not int or not 0 <= terminal_sessions <= 20:
            raise ValueError("terminal_sessions must be an integer from 0 through 20")
        if terminal_sessions and evaluation_sessions == 1:
            raise ValueError("Terminal evaluation requires a multi-origin study")
        self._fits = []
        self._terminal_started = False
        self._news_configuration_open = True
        self._news_history = None
        self._guidance_history = None
        self.guidance_support = []
        self._news_columns = {}
        self.news_feature_support = None
        self.peer_features = None
        self.profile_features = None
        self.financial_features = None
        self._has_predicted = False
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
        self.terminal_origins = np.arange(self.origin - terminal_sessions + 1, self.origin + 1)
        if terminal_sessions:
            # Development labels AND account valuations end before the terminal window.
            self.origin = int(self.terminal_origins[0]) - self.horizon - 1
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
        self._planned_development_end = self.data.dates[self.origins[-1] + self.horizon]
        if news_history is not None:
            self.configure_news(news_history, news_cutoffs, news_symbol)

    def configure_news(self, news_history, news_cutoffs, news_symbol):
        """Freeze one target's evidence and common rows before hypothesis selection."""
        if not self._news_configuration_open or self._news_history is not None:
            raise ValueError("News must be configured once, before context or prediction")
        if self.evaluation_sessions <= 1:
            raise ValueError("News features require a multi-origin development study")
        if not isinstance(news_symbol, str) or not news_symbol:
            raise ValueError("News features require an explicit target symbol")
        if news_cutoffs is None or len(news_cutoffs) != len(self.data.dates):
            raise ValueError("News cutoffs must align with every price session")
        times = [datetime.fromisoformat(value) for value in news_cutoffs]
        if any(value.tzinfo is None for value in times) or any(
            left >= right for left, right in zip(times, times[1:], strict=False)
        ):
            raise ValueError("News cutoffs must be aware, chronological timestamps")
        if self.price_info and (
            tuple(news_cutoffs) != tuple(self.price_info.get("session_close_times", ()))
            or self.data.instruments != (news_symbol,)
        ):
            raise ValueError("News requires the retained target's actual session close times")
        self._news_history = deepcopy(news_history)
        self._news_cutoffs = tuple(news_cutoffs)
        self._news_symbol = news_symbol
        self._news_configuration_open = False
        _, evidence = self._news_column(
            {"op": "headline_keyword_count", "keywords": ["earnings"], "window_days": 30}
        )
        available = np.array([item["value"] is not None for item in evidence])
        training = self.train[available[self.train]]
        origins = self.origins[available[self.origins]]
        runnable = len(training) >= 12 and len(origins) >= 2
        reasons = {
            reason
            for index in np.concatenate((self.train, self.origins))
            for reason in evidence[index]["reasons"]
        }
        if len(training) < 12:
            reasons.add("insufficient_news_training_rows")
        if len(origins) < 2:
            reasons.add("insufficient_news_evaluation_rows")
        self.news_feature_support = {
            "runnable": runnable,
            "original_training_rows": len(self.train),
            "original_evaluation_rows": len(self.origins),
            "available_training_rows": len(training),
            "available_evaluation_rows": len(origins),
            "retained_training_rows": len(training) if runnable else len(self.train),
            "retained_evaluation_rows": len(origins) if runnable else len(self.origins),
            "reasons": sorted(reasons),
            "source_basis": "retained_target_issuer_headline_snapshots",
            "providers": list(getattr(self._news_history, "providers", ())),
            "mask_policy": "shared_before_factor_selection",
        }
        if runnable:
            self.train, self.origins = training, origins
            self.origin = int(origins[-1])
        self.library["news_feature_support"] = deepcopy(self.news_feature_support)
        self.library["execution_note"] = (
            "Numeric price recipes and retained target-issuer headline counts use the same frozen "
            "training and evaluation rows. Missing news observations are unavailable, not zero."
            if runnable
            else "Only numeric price recipes can execute: news history has insufficient available "
            "training or evaluation rows. News proposals remain specifications."
        )
        return deepcopy(self.news_feature_support)

    def configure_guidance(self, history):
        if self._has_predicted or self._guidance_history is not None or self._news_history is None:
            raise ValueError("Guidance must be configured once after news and before prediction")
        self._guidance_history = deepcopy(history)
        indices = np.concatenate((self.train, self.origins))
        for days in (7, 30):
            recipe = {"op": "guidance_direction", "window_days": days}
            values, evidence = self._news_column(recipe)
            reasons = sorted({reason for i in indices for reason in evidence[i]["reasons"]})
            definition = {
                "factor_id": f"news.guidance_direction_{days}",
                "kind": "news",
                "input_kind": "retained_issuer_guidance",
                "recipe": recipe,
                "description": "Latest explicit issuer guidance: raised +1, lowered -1, "
                "reaffirmed 0; no statement in the inspected window also 0. "
                "Not a stock-return forecast. Missing extraction is unavailable.",
                "runnable": bool(np.isfinite(values[indices]).all()),
                "reasons": reasons,
            }
            self.guidance_support.append(definition)
        self.library["definitions"].extend(deepcopy(self.guidance_support))
        self.library["execution_note"] += (
            " Guidance-direction definitions marked runnable use source-bound extractions "
            "on the same frozen rows; they can be selected or omitted each round."
        )

    def _news_column(self, recipe):
        key = canonical(recipe)
        if key not in self._news_columns:
            guidance = recipe.get("op") == "guidance_direction"
            history = self._guidance_history if guidance else self._news_history
            if history is None:
                raise ValueError("News feature history is not configured")
            evidence = [
                deepcopy(history.evaluate(recipe, self._news_symbol, cutoff))
                for cutoff in self._news_cutoffs
            ]
            values = np.array(
                [np.nan if item["value"] is None else float(item["value"]) for item in evidence]
            )
            if any(
                item["value"] is not None
                and (not np.isfinite(value) or (value not in {-1, 0, 1} if guidance else value < 0))
                for item, value in zip(evidence, values, strict=True)
            ):
                raise ValueError("News executor returned an invalid numeric feature")
            self._news_columns[key] = values, evidence
        return self._news_columns[key]

    def configure_peers(self, bundle):
        if self._has_predicted or self.peer_features is not None:
            raise ValueError("Related prices must be configured once before prediction")
        self.peer_features = deepcopy(bundle)
        self.library["definitions"].extend(deepcopy(bundle["definitions"]))
        self.library["execution_note"] += (
            " Related-price recipes marked runnable share the fixed training/evaluation rows. "
            "Current company suggestions inform retrospective selection, not causal evidence."
        )

    def configure_financials(self, bundle):
        if self._has_predicted or self.financial_features is not None:
            raise ValueError("Financial features must be configured once before prediction")
        existing = {item["factor_id"] for item in self.library["definitions"]}
        if existing.intersection(item["factor_id"] for item in bundle["definitions"]):
            raise ValueError("Financial factor IDs overlap existing definitions")
        self.financial_features = deepcopy(bundle)
        self.library["definitions"].extend(deepcopy(bundle["definitions"]))
        self.library["execution_note"] += (
            " Financial ratios are computed from filing-dated retained facts; "
            "missing rows are not filled or dropped."
        )

    def configure_profile(self, profile):
        if self._has_predicted or self.profile_features is not None:
            raise ValueError("Factor profile must be configured once before prediction")
        bundle = prepare_bound_features(
            profile, self.data, self.peer_features or {}, np.concatenate((self.train, self.origins))
        )
        existing = {item["factor_id"] for item in self.library["definitions"]}
        if existing.intersection(item["factor_id"] for item in bundle["definitions"]):
            raise ValueError("Bound factor conflicts with an existing study definition")
        self.profile_features = bundle
        self.library["definitions"].extend(deepcopy(bundle["definitions"]))
        self.library.pop("instrument_profile", None)
        self.library["profile_resolution"] = {
            "market": profile["market"],
            "instrument": profile["instrument"],
            "status": "runtime_resolved_on_training_and_development_rows",
        }
        self.library["execution_note"] += (
            " Bound factors retain exact source and ownership metadata; composites use explicit "
            "binding leaves. Pending sources cannot be replaced with target share prices."
        )

    def context(self, symbol):
        self._news_configuration_open = False
        if self._news_history is not None and symbol != self._news_symbol:
            raise ValueError("News data does not match the target symbol")
        if self.price_info and self.data.instruments != (symbol,):
            raise ValueError("Price data does not match the target symbol")
        result = {
            "provider": self.price_provider,
            "reference_price": float(self.data.closes[self.origin, 0]),
            "origin_session": self.data.dates[self.origin],
            "model_mode": self.estimator,
            "model_configuration": deepcopy(self.model_configuration),
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
                + "All revisions use the same training window and evaluation origins. "
                "Current news guides retrospective factor selection, not historically available "
                "features. Each revision reuses disclosed development outcomes, so this "
                "is not independent holdout validation. Five-session outcomes overlap."
            )
        if self.news_feature_support is not None:
            result["news_feature_support"] = deepcopy(self.news_feature_support)
            result["model_note"] += (
                " Retained issuer-headline features are optional and use only snapshots available "
                "at each supplied session cutoff. Missing coverage is not a zero count."
            )
        if self.peer_features is not None:
            result["related_price_support"] = deepcopy(self.peer_features["support"])
        if self._guidance_history is not None:
            result["guidance_feature_support"] = deepcopy(self.guidance_support)
        if self.profile_features is not None:
            result["factor_profile"] = deepcopy(self.library["profile_resolution"])
        if self.financial_features is not None:
            result["financial_features"] = {
                "definitions": deepcopy(self.financial_features["definitions"]),
                "latest_visible_facts": deepcopy(self.financial_features.get("evidence", [{}])[-1]),
                "notice": self.financial_features["notice"],
            }
        return result

    def predict(self, context, factors, hypothesis):
        import lightgbm as lgb

        if self._terminal_started:
            raise ValueError("Research fitting is closed after terminal evaluation starts")
        self._news_configuration_open = False
        self._has_predicted = True
        if not factors or any(item["kind"] not in {"numeric", "news"} for item in factors):
            raise ValueError(
                "LightGBM demo accepts only selected numeric or supported news factors"
            )
        news_factors = [item for item in factors if item["kind"] == "news"]
        headline_factors = [
            f for f in news_factors if f["recipe"].get("op") != "guidance_direction"
        ]
        if headline_factors and not (self.news_feature_support or {}).get("runnable"):
            raise ValueError("News factor execution is unavailable for this study")
        for factor in news_factors:
            if factor["recipe"].get("op") == "guidance_direction" and not any(
                d["runnable"]
                and d["factor_id"] == factor["factor_id"]
                and d["recipe"] == factor["recipe"]
                for d in self.guidance_support
            ):
                raise ValueError("Guidance factor lacks its exact usable retained recipe")
        columns, news_inputs, news_evidence = [], [], {}
        bound = {
            item["factor_id"]: item for item in (self.profile_features or {}).get("definitions", [])
        }
        effective = {item["factor_id"]: item for item in factors}
        bound_inputs = []
        for item in factors:
            if item.get("recipe", {}).get("op") == "financial":
                retained = next(
                    (
                        d
                        for d in (self.financial_features or {}).get("definitions", [])
                        if d["factor_id"] == item["factor_id"]
                    ),
                    None,
                )
                if retained is None or not retained["runnable"] or retained != item:
                    raise ValueError("Financial factor lacks its exact retained usable source")
                columns.append(
                    np.asarray(self.financial_features["columns"][item["factor_id"]], dtype=float)
                )
                continue
            if item["factor_id"] in bound:
                retained = bound[item["factor_id"]]
                if (
                    not retained["runnable"]
                    or retained["recipe"] != item.get("recipe")
                    or retained.get("source") != item.get("source")
                ):
                    raise ValueError("Bound factor lacks its exact usable retained source/recipe")
                columns.append(self.profile_features["columns"][item["factor_id"]])
                dependencies = [bound[key] for key in retained["input_factor_ids"]]
                bound_inputs.append(
                    {"factor_id": item["factor_id"], "inputs": deepcopy(dependencies)}
                )
                for dependency in dependencies:
                    effective[dependency["factor_id"]] = dependency
                    peer_id = dependency.get("runtime_dependency_id")
                    if peer_id:
                        effective[peer_id] = next(
                            d
                            for d in self.peer_features["definitions"]
                            if d["factor_id"] == peer_id
                        )
                continue
            if item["kind"] == "numeric":
                if item["recipe"].get("op") in {"peer_return", "relative_return"}:
                    definitions = {
                        d["factor_id"]: d for d in (self.peer_features or {}).get("definitions", [])
                    }
                    retained = definitions.get(item["factor_id"])
                    if (
                        not retained
                        or not retained["runnable"]
                        or retained["recipe"] != item["recipe"]
                    ):
                        raise ValueError(
                            "Related factor lacks a usable retained recipe on the fixed sample"
                        )
                    columns.append(self.peer_features["columns"][item["factor_id"]])
                    continue
                columns.append(price_feature(self.data, item["recipe"]))
                continue
            values, evidence = self._news_column(item["recipe"])
            columns.append(values)
            news_evidence[item["factor_id"]] = deepcopy(evidence[self.origin]) | {
                "cutoff": self._news_cutoffs[self.origin]
            }
            for role, indices in (("training", self.train), ("evaluation", self.origins)):
                news_inputs.extend(
                    deepcopy(evidence[index])
                    | {
                        "factor_id": item["factor_id"],
                        "role": role,
                        "origin_session": self.data.dates[index],
                        "cutoff": self._news_cutoffs[index],
                    }
                    for index in indices
                )
        matrix = np.column_stack(columns)
        x = matrix[self.train]
        y = np.log(self.data.closes[self.train + self.horizon, 0] / self.data.closes[self.train, 0])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Non-finite numerical training inputs")
        if not np.isfinite(matrix[self.origins]).all():
            raise ValueError("Non-finite numerical evaluation inputs")

        def fit():
            if self.estimator == "ridge":
                return StandardizedRidge(x, y, self.model_configuration["parameters"]["alpha"])
            return lgb.train(
                MODEL_PARAMETERS | {"seed": 7},
                lgb.Dataset(x, label=y),
                num_boost_round=BOOST_ROUNDS,
            )

        model = None

        def fit_state():
            nonlocal model
            model = fit()
            return (
                model.state()
                if self.estimator == "ridge"
                else {"model_string": model.model_to_string()}
            )

        if self.fit_cache:
            state = self.fit_cache.call(
                f"fit_{len(self._fits) + 1}",
                {
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "configuration": self.model_configuration,
                    "factors": factors,
                },
                fit_state,
            )
            if model is None:
                model = (
                    StandardizedRidge.from_state(state)
                    if self.estimator == "ridge"
                    else lgb.Booster(model_str=state["model_string"])
                )
        else:
            model = fit()
        if self.estimator == "ridge":
            predict = model.predict
        else:

            def predict(rows):
                return model.predict(rows, num_threads=1)

        predicted_returns = np.asarray(predict(matrix[self.origins]))
        value = float(predicted_returns[-1])
        price = float(self.data.closes[self.origin, 0] * np.exp(value))
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Study model produced an invalid price")
        result = {
            "provider": self.estimator,
            "model_configuration": deepcopy(self.model_configuration),
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
        }
        if self.estimator == "ridge":
            result["fitted_model"] = model.state()
        else:
            result["boost_rounds"] = BOOST_ROUNDS
        if self.evaluation_sessions > 1:
            result["development"] = self._development(predicted_returns, float(y.mean()))
        self._fits.append((predict, matrix, y.copy(), deepcopy(result["factor_ids"])))
        if self.news_feature_support is not None:
            result["news_feature_support"] = deepcopy(self.news_feature_support)
        if news_factors:
            result["news_feature_inputs"] = news_inputs
            result["news_feature_evidence"] = news_evidence
        if self.peer_features is not None:
            result["input_usage"] = self._input_usage(list(effective.values()))
        if bound_inputs:
            result["bound_factor_inputs"] = bound_inputs
        if self.price_info:
            if not np.isfinite(matrix[-1:]).all():
                peer_factors = [
                    item
                    for item in factors
                    if item["recipe"].get("op") in {"peer_return", "relative_return"}
                ]
                if not news_factors and not peer_factors and not bound_inputs:
                    raise ValueError("Non-finite numerical latest-forecast inputs")
                result["latest_forecast_unavailable"] = {
                    "origin_session": self.data.dates[-1],
                    "reasons": ["latest_news_features_unavailable"]
                    if not peer_factors and not bound_inputs
                    else ["latest_selected_features_unavailable"],
                    "missing_factor_ids": [
                        item["factor_id"]
                        for index, item in enumerate(factors)
                        if not np.isfinite(matrix[-1, index])
                    ],
                    "news_feature_evidence": {
                        item["factor_id"]: deepcopy(self._news_column(item["recipe"])[1][-1])
                        | {"cutoff": self._news_cutoffs[-1]}
                        for item in news_factors
                    },
                }
                return result
            latest_return = float(predict(matrix[-1:])[0])
            latest_price = float(self.data.closes[-1, 0] * np.exp(latest_return))
            if not np.isfinite(latest_price) or latest_price <= 0:
                raise ValueError("Study model produced an invalid latest forecast")
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
            if news_factors or self.peer_features is not None:
                result["latest_forecast"]["feature_values"] = {
                    item["factor_id"]: float(matrix[-1, index])
                    for index, item in enumerate(factors)
                }
            if news_factors:
                result["latest_forecast"]["news_feature_evidence"] = {
                    item["factor_id"]: deepcopy(self._news_column(item["recipe"])[1][-1])
                    | {"cutoff": self._news_cutoffs[-1]}
                    for item in news_factors
                }
        return result

    def _input_usage(self, factors):
        selected = {item["factor_id"] for item in factors}
        usage = [
            {
                "source": "target_prices",
                "status": "used",
                "factor_ids": [
                    item["factor_id"]
                    for item in factors
                    if item["kind"] == "numeric"
                    and item["recipe"].get("op") not in {"peer_return", "financial"}
                ],
                "reason": "target_labels_and_reference_prices",
            }
        ]
        if self.financial_features is not None:
            ids = sorted(
                selected.intersection(
                    d["factor_id"] for d in self.financial_features["definitions"]
                )
            )
            usage.append(
                {
                    "source": "filing_dated_financials",
                    "status": "used" if ids else "not_selected",
                    "factor_ids": ids,
                    "reason": self.financial_features["notice"],
                }
            )
        for peer in self.peer_features["support"]:
            used = sorted(selected.intersection(peer["factor_ids"]))
            usage.append(
                {
                    "source": peer["symbol"],
                    "status": "used"
                    if used
                    else "not_selected"
                    if peer["status"] == "available"
                    else peer["status"],
                    "factor_ids": used,
                    "reason": peer["reason"],
                    "provider": peer.get("provider"),
                    "relationship_status": peer["relationship_status"],
                }
            )
        guidance_ids = [
            item["factor_id"]
            for item in factors
            if item.get("recipe", {}).get("op") == "guidance_direction"
        ]
        if self._guidance_history is not None:
            usage.append(
                {
                    "source": "issuer_guidance",
                    "factor_ids": guidance_ids,
                    "status": "used"
                    if guidance_ids
                    else "not_selected"
                    if any(d["runnable"] for d in self.guidance_support)
                    else "unavailable",
                    "reason": "source_bound_extraction_on_fixed_sample",
                }
            )
        news_ids = [
            item["factor_id"]
            for item in factors
            if item["kind"] == "news" and item["factor_id"] not in guidance_ids
        ]
        support = self.news_feature_support or {}
        usage.append(
            {
                "source": "archived_headlines",
                "status": "used"
                if news_ids
                else "not_selected"
                if support.get("runnable")
                else "unavailable",
                "factor_ids": news_ids,
                "reason": ", ".join(support.get("reasons", [])),
            }
        )
        usage.append(
            {
                "source": "current_news",
                "status": "context_only",
                "factor_ids": [],
                "reason": "hypothesis_context_not_backdated_training_data",
            }
        )
        return usage

    def comparison_inputs(self, selected_round):
        """Copy the selected fit's exact inputs; evaluation labels are deliberately absent."""
        if type(selected_round) is not int or not 1 <= selected_round <= len(self._fits):
            raise ValueError("Comparison requires a completed selected fit")
        _, matrix, training_labels, factor_ids = self._fits[selected_round - 1]
        segments = {}
        for name, indices in (
            ("train", self.train),
            ("development", self.origins),
            ("terminal", self.terminal_origins),
        ):
            values = matrix[indices]
            segments[name] = {
                "sessions": [self.data.dates[i] for i in indices],
                "features": np.where(np.isfinite(values), values, None).tolist(),
                "available": bool(len(indices) and np.isfinite(values).all()),
            }
        segments["train"]["labels"] = training_labels.tolist()
        segments["train"]["label_sessions"] = [
            self.data.dates[i + self.horizon] for i in self.train
        ]
        return {
            "selected_round": selected_round,
            "symbol": self.data.instruments[0],
            "factor_ids": deepcopy(factor_ids),
            "model_configuration": deepcopy(self.model_configuration),
            "segments": segments,
        }

    def terminal_plan(self):
        """Dates only: frozen before research, with no terminal values or outcomes."""
        return {
            "status": "reserved" if len(self.terminal_origins) else "disabled",
            "sample_count": len(self.terminal_origins),
            "origin_sessions": [self.data.dates[i] for i in self.terminal_origins],
            "target_sessions": [self.data.dates[i + self.horizon] for i in self.terminal_origins],
            "development_last_target_session": self._planned_development_end,
            "rule": "select_on_development_then_evaluate_selected_fitted_model_once_no_refit",
            "limitation": "Reserved from research feedback and selection, but retrospective. "
            "Current news and pretrained LLM knowledge are not historically isolated. "
            "This is not prospective validation or calibrated evidence of predictive gain.",
        }

    def evaluate_terminal(self, selected_round):
        if not len(self.terminal_origins):
            raise ValueError("No terminal observations were reserved")
        if self._terminal_started:
            raise ValueError("Terminal evaluation can only run once")
        if type(selected_round) is not int or not 1 <= selected_round <= len(self._fits):
            raise ValueError("Selected round must identify a completed fitted model")
        self._terminal_started = True
        predict, matrix, training_labels, factor_ids = self._fits[selected_round - 1]
        result = self.terminal_plan() | {
            "selected_round": selected_round,
            "factor_ids": factor_ids,
            "refitted": False,
        }
        missing = [
            {
                "origin_session": self.data.dates[origin],
                "factor_ids": [
                    factor_ids[j]
                    for j in range(len(factor_ids))
                    if not np.isfinite(matrix[origin, j])
                ],
            }
            for origin in self.terminal_origins
            if not np.isfinite(matrix[origin]).all()
        ]
        if missing:
            return result | {
                "status": "unavailable",
                "reason": "missing_selected_features",
                "missing_inputs": missing,
            }
        evaluation = self._development(
            np.asarray(predict(matrix[self.terminal_origins])),
            float(training_labels.mean()),
            origins=self.terminal_origins,
        )
        evaluation["evaluation_note"] = "retrospective_terminal_reserved_from_feedback_selection"
        return result | {"status": "completed", "evaluation": evaluation}

    def _development(self, predicted_returns, training_mean, *, origins=None):
        origins = self.origins if origins is None else origins

        def rows(returns):
            result = []
            for origin, value in zip(origins, returns, strict=True):
                reference = float(self.data.closes[origin, 0])
                predicted = float(reference * np.exp(value))
                if not np.isfinite(predicted) or predicted <= 0:
                    raise ValueError("Study model produced an invalid development price")
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
            baseline_rows = rows(np.full(len(origins), value))
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
            "first_evaluation_session": self.data.dates[origins[0]],
            "last_evaluation_session": self.data.dates[origins[-1]],
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
                f"{context['symbol']}: two real {self.model_label} fits on {source} prices. "
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
        if self.price_info and "latest_forecast" in second:
            result["latest_forecast"] = second["latest_forecast"]
            result["text"] += (
                " The separate latest forecast uses the same fitted model; outcome unobserved."
            )
        elif self.price_info and "latest_forecast_unavailable" in second:
            result["latest_forecast_unavailable"] = deepcopy(second["latest_forecast_unavailable"])
            result["text"] += (
                " The latest forecast is unavailable because its news inputs are missing."
            )
        if self.evaluation_sessions > 1:
            result["development"] = {
                "first_metrics": deepcopy(first["development"]["metrics"]),
                "revised_metrics": deepcopy(second["development"]["metrics"]),
                "baselines": deepcopy(compact_development(first["development"])["baselines"]),
                "evaluation_note": "retrospective_development_same_data_feedback_reuse",
            }
            result["text"] += (
                f" Metrics describe the same {len(self.origins)} matured origins; "
                "news informs retrospective selection and the revised fit reuses these disclosed "
                "outcomes. Overlapping five-session labels are not independent samples."
            )
        return result
