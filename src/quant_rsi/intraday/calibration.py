"""Learning the trading policy from a registered grid on calibration sessions (design 5.7).

Given a block of training sessions, the last `calibration_sessions` of them are held out.
A model is fitted on the sessions before them, minus the fold embargo, and predicts the
calibration sessions. Every grid candidate is replayed on those predictions with the
trusted simulator, and the eligible candidate with the highest mean net return per session
wins; ties go to the earlier candidate in grid order. Nothing after the training block is
read, so the chosen policy can be scored honestly on the sessions that follow it.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from quant_rsi.intraday.contracts import IntradayDevelopmentPlan, SignalPolicy
from quant_rsi.intraday.features import BarSource, FeatureTable
from quant_rsi.intraday.model import FittedModel, fit, predict
from quant_rsi.intraday.simulator import PriceKey, simulate

POLICY_FIELDS = (
    "entry_rule",
    "entry_threshold",
    "exit_threshold",
    "sizing",
    "full_size_edge",
)


@dataclass(frozen=True)
class PolicyChoice:
    signal: SignalPolicy
    source: Literal["declared", "calibrated", "declared_fallback"]
    calibration_sessions: tuple[int, ...] = ()
    candidates: list[dict] = field(default_factory=list)

    def report(self, session_ids: Sequence[str]) -> dict:
        return {
            "source": self.source,
            "signal": policy_fields(self.signal),
            "calibration_sessions": [session_ids[i] for i in self.calibration_sessions],
            "candidates": self.candidates,
        }


def policy_fields(signal: SignalPolicy) -> dict:
    return {name: getattr(signal, name) for name in POLICY_FIELDS}


def candidates(plan: IntradayDevelopmentPlan) -> list[SignalPolicy]:
    search = plan.policy_search
    if search is None:
        return [plan.signal]
    edges = search.full_size_edges or (None,)
    return [
        plan.signal.model_copy(
            update={"entry_threshold": entry, "exit_threshold": exit_, "full_size_edge": edge}
        )
        for entry, exit_, edge in itertools.product(
            search.entry_thresholds, search.exit_thresholds, edges
        )
        if exit_ <= entry
    ]


def split_calibration(
    plan: IntradayDevelopmentPlan, train_sessions: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(inner training sessions, calibration sessions) inside one training block."""
    count = plan.policy_search.calibration_sessions
    inner_count = len(train_sessions) - count - plan.folds.embargo_sessions
    if inner_count <= 0:
        raise ValueError(
            f"{len(train_sessions)} training sessions cannot hold {count} calibration sessions "
            f"plus a {plan.folds.embargo_sessions}-session embargo"
        )
    return tuple(train_sessions[:inner_count]), tuple(train_sessions[-count:])


def row_predictions(
    model: FittedModel, table: FeatureTable, mask: np.ndarray
) -> dict[PriceKey, float]:
    scores = predict(model.model_text, table.values[mask], table.names)
    rows = (row for row, keep in zip(table.rows, mask, strict=True) if keep)
    return {(row[0], row[2], row[3]): float(s) for row, s in zip(rows, scores, strict=True)}


def choose_policy(
    source: BarSource,
    plan: IntradayDevelopmentPlan,
    table: FeatureTable,
    targets: np.ndarray,
    train_sessions: Sequence[int],
) -> PolicyChoice:
    """Learn the policy using only `train_sessions`; the declared signal without a search."""
    search = plan.policy_search
    if search is None:
        return PolicyChoice(plan.signal, "declared")
    inner, calibration = split_calibration(plan, train_sessions)
    session_of = np.array([row[2] for row in table.rows])
    train = np.isin(session_of, inner)
    model = fit(
        table.values[train],
        targets[train],
        table.names,
        seed=plan.seed,
        min_rows=plan.min_training_rows,
    )
    predictions = row_predictions(model, table, np.isin(session_of, calibration))
    rows, best, best_score = [], None, None
    for candidate in candidates(plan):
        metrics = simulate(source, plan, predictions, calibration, signal=candidate).metrics(
            plan.sessions_per_year
        )
        score = metrics["mean_net_return_bps"]
        eligible = (
            score is not None
            and not metrics["unresolved_sessions"]
            and metrics["round_trips"] >= search.min_calibration_trades
        )
        rows.append(
            {
                **policy_fields(candidate),
                "mean_net_return_bps": score,
                "round_trips": metrics["round_trips"],
                "eligible": eligible,
            }
        )
        if eligible and (best_score is None or score > best_score):
            best, best_score = candidate, score
    if best is None:
        return PolicyChoice(plan.signal, "declared_fallback", calibration, rows)
    return PolicyChoice(best, "calibrated", calibration, rows)
