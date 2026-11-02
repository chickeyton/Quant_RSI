"""Two explicit study estimators; preprocessing is fitted on training rows only."""

import math

import numpy as np

from quant_rsi.forecast.forecast_protocol import BOOST_ROUNDS, MODEL_PARAMETERS


def model_configuration(name="lightgbm", ridge_alpha=None, *, horizon_sessions=5):
    if type(horizon_sessions) is not int or horizon_sessions not in (5, 20, 60):
        raise ValueError("Prediction horizon must be 5, 20 or 60 trading sessions")
    if name not in {"lightgbm", "ridge"}:
        raise ValueError("Study model must be lightgbm or ridge")
    if name != "ridge" and ridge_alpha is not None:
        raise ValueError("--ridge-alpha requires --model ridge")
    alpha = 1.0 if ridge_alpha is None else ridge_alpha
    if name == "ridge" and (isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("Ridge alpha must be finite and positive")
    return {
        "schema_version": 1,
        "name": name,
        "parameters": {"alpha": float(alpha), "fit_intercept": True}
        if name == "ridge"
        else MODEL_PARAMETERS | {"seed": 7, "num_boost_round": BOOST_ROUNDS},
        "preprocessing": "training_mean_population_std_constant_scale_one"
        if name == "ridge"
        else "none",
        "objective": "sum_squared_error_plus_alpha_times_squared_coefficients"
        if name == "ridge"
        else "regression",
        "training_policy": "fixed_mature_label_window_shared_between_attempts",
        "target": "five_session_close_log_return"
        if horizon_sessions == 5
        else f"{horizon_sessions}_session_close_log_return",
    }


def column_share(weights, factor_ids):
    """Share of one fit's column usage. Not evidence that a column predicts returns."""
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1
        or len(weights) != len(factor_ids)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        return {}
    total = float(weights.sum())
    if total <= 0:
        return {}
    return {
        factor_id: float(weight) / total
        for factor_id, weight in zip(factor_ids, weights, strict=True)
    }


class StandardizedRidge:
    """L2-regularized slopes with an unpenalized intercept; no estimator search."""

    def __init__(self, x, y, alpha):
        model_configuration("ridge", alpha)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (
            x.ndim != 2
            or not all(x.shape)
            or y.shape != (x.shape[0],)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("Ridge requires finite nonempty training rows and aligned targets")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0, ddof=0)
        # Decimal constants can have a tiny nonzero std from mean rounding.
        # Center them exactly so roundoff cannot become a spurious predictor.
        constant = np.all(x == x[0], axis=0)
        self.mean[constant] = x[0, constant]
        self.scale[constant | (self.scale == 0)] = 1.0
        self.intercept = float(y.mean())
        standardized = (x - self.mean) / self.scale
        design = np.vstack((standardized, np.sqrt(alpha) * np.eye(x.shape[1])))
        target = np.concatenate((y - self.intercept, np.zeros(x.shape[1])))
        self.coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        if not np.isfinite(self.coefficients).all():
            raise ValueError("Ridge produced non-finite coefficients")

    @classmethod
    def from_state(cls, state):
        model = object.__new__(cls)
        model.mean = np.asarray(state["mean"], dtype=float)
        model.scale = np.asarray(state["scale"], dtype=float)
        model.coefficients = np.asarray(state["coefficients"], dtype=float)
        model.intercept = float(state["intercept"])
        if (
            model.mean.ndim != 1
            or not len(model.mean)
            or model.scale.shape != model.mean.shape
            or model.coefficients.shape != model.mean.shape
            or not all(np.isfinite(x).all() for x in (model.mean, model.scale, model.coefficients))
            or not (model.scale > 0).all()
            or not math.isfinite(model.intercept)
        ):
            raise ValueError("Invalid retained ridge state")
        return model

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("Ridge requires finite prediction rows with the trained columns")
        result = self.intercept + ((x - self.mean) / self.scale) @ self.coefficients
        if not np.isfinite(result).all():
            raise ValueError("Ridge produced non-finite predictions")
        return result

    def state(self):
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "coefficient_basis": "standardized_features_in_factor_ids_order",
        }
