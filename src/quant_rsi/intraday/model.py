"""Fixed-recipe LightGBM for intraday labels, session-boundary folds and safe inference.

Rows with any missing feature are never passed to the model: they become `no_prediction`
(NaN score). Training uses only complete rows whose label exists.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from quant_rsi.core.artifacts import digest
from quant_rsi.intraday.contracts import FoldSpec, IntradayDevelopmentPlan
from quant_rsi.intraday.features import BarSource, FeatureTable
from quant_rsi.intraday.timing import labels, session_timing

BOOST_ROUNDS = 24


def lightgbm_params(seed: int) -> dict:
    """Same fixed recipe as `core/model.py`."""
    return {
        "objective": "regression",
        "verbosity": -1,
        "num_threads": 1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": seed,
        "num_leaves": 7,
        "min_data_in_leaf": 5,
        "learning_rate": 0.05,
        "feature_pre_filter": False,
    }


def label_vector(
    source: BarSource, table: FeatureTable, plan: IntradayDevelopmentPlan
) -> np.ndarray:
    sessions = source.contract.calendar.sessions
    bar_seconds = source.contract.grid.bar_seconds
    cache: dict[tuple[str, int], np.ndarray] = {}
    result = np.full(len(table.rows), np.nan)
    for row, (target, session_id, index, t) in enumerate(table.rows):
        key = (target, index)
        if key not in cache:
            execution = source.block(target, "execution", session_id)
            if execution is None:
                cache[key] = np.zeros(0)
            else:
                timing = session_timing(
                    sessions[index], bar_seconds, plan.timing, plan.signal.flatten_offset_seconds
                )
                cache[key] = labels(
                    execution, timing, plan.signal.horizon_bars, plan.timing.max_fill_delay_seconds
                )
        values = cache[key]
        result[row] = values[t] if t < len(values) else np.nan
    return result


def model_feature_names(names: Sequence[str]) -> list[str]:
    """LightGBM rejects JSON-special characters such as ':' in feature names."""
    return [name.replace(":", "__") for name in names]


@dataclass(frozen=True)
class FittedModel:
    model_text: str
    feature_names: tuple[str, ...]
    training_rows: int

    @property
    def model_digest(self) -> str:
        return digest(self.model_text.encode())


def fit(
    values: np.ndarray, targets: np.ndarray, names: Sequence[str], *, seed: int, min_rows: int
) -> FittedModel:
    import lightgbm as lgb

    usable = np.isfinite(values).all(axis=1) & np.isfinite(targets)
    if usable.sum() < min_rows:
        raise ValueError(f"insufficient complete training rows: {int(usable.sum())} < {min_rows}")
    dataset = lgb.Dataset(
        values[usable], label=targets[usable], feature_name=model_feature_names(names)
    )
    booster = lgb.train(lightgbm_params(seed), dataset, num_boost_round=BOOST_ROUNDS)
    return FittedModel(booster.model_to_string(), tuple(names), int(usable.sum()))


def predict(model_text: str, values: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Scores for complete rows; NaN (no_prediction) for rows with a missing feature."""
    import lightgbm as lgb

    scores = np.full(len(values), np.nan)
    complete = np.isfinite(values).all(axis=1) if len(values) else np.zeros(0, bool)
    if not complete.any():
        return scores
    booster = lgb.Booster(model_str=model_text)
    if booster.num_feature() != values.shape[1] or booster.feature_name() != model_feature_names(
        names
    ):
        raise ValueError("sealed model feature interface differs")
    result = booster.predict(values[complete], num_threads=1)
    if not np.isfinite(result).all():
        raise ValueError("non-finite fixed-model predictions")
    scores[complete] = result
    return scores


@dataclass(frozen=True)
class Fold:
    train_sessions: tuple[int, ...]
    test_sessions: tuple[int, ...]


def walk_forward_folds(spec: FoldSpec, session_count: int) -> tuple[Fold, ...]:
    """Expanding-window folds on session boundaries with an embargo before each test block."""
    folds = []
    for number in range(spec.folds):
        start = spec.first_test_session + number * spec.test_sessions
        stop = start + spec.test_sessions
        if stop > session_count:
            raise ValueError("fold specification exceeds the registered sessions")
        train = tuple(range(0, start - spec.embargo_sessions))
        if not train:
            raise ValueError("fold has no training sessions after the embargo")
        folds.append(Fold(train, tuple(range(start, stop))))
    return tuple(folds)
