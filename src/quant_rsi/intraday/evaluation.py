"""Development evaluation of one intraday leaf set (the C2 inner-loop scoring unit).

Each fold fits the fixed LightGBM recipe on complete, labelled rows from its training
sessions only, predicts its test sessions without touching their labels, and replays them
through the trusted simulator. Ranking is the median across folds of mean net return per
session; eligibility (minimum trades, prediction coverage, no unresolved sessions) is a
candidate-validity outcome, not an evidence status.
"""

from decimal import Decimal

import numpy as np

from quant_rsi.intraday.contracts import IntradayDevelopmentPlan, validate_timing
from quant_rsi.intraday.features import BarSource, build_features, validate_leaves
from quant_rsi.intraday.model import fit, label_vector, predict, walk_forward_folds
from quant_rsi.intraday.simulator import simulate


def evaluate_development(source: BarSource, plan: IntradayDevelopmentPlan) -> dict:
    contract = source.contract
    validate_timing(contract, plan.timing)
    validate_leaves(contract, plan.leaves)
    sessions = contract.calendar.sessions
    folds = walk_forward_folds(plan.folds, len(sessions))
    table = build_features(source, plan.leaves, context_tolerance=plan.context_missing_tolerance)
    targets = label_vector(source, table, plan)
    session_of = np.array([row[2] for row in table.rows])
    reports = []
    for number, fold in enumerate(folds):
        train = np.isin(session_of, fold.train_sessions)
        test = np.isin(session_of, fold.test_sessions)
        model = fit(
            table.values[train],
            targets[train],
            table.names,
            seed=plan.seed,
            min_rows=plan.min_training_rows,
        )
        scores = predict(model.model_text, table.values[test], table.names)
        predictions = {
            (row[0], row[2], row[3]): float(score)
            for row, score in zip(
                (r for r, keep in zip(table.rows, test, strict=True) if keep), scores, strict=True
            )
        }
        result = simulate(source, plan, predictions, fold.test_sessions)
        metrics = result.metrics(plan.sessions_per_year)
        reasons = []
        if metrics["round_trips"] < plan.min_trades_per_fold:
            reasons.append("too_few_trades")
        coverage = metrics["prediction_coverage"]
        if coverage is None or coverage < plan.min_prediction_coverage:
            reasons.append("prediction_coverage_below_minimum")
        if metrics["unresolved_sessions"]:
            reasons.append("unresolved_sessions")
        reports.append(
            {
                "fold": number,
                "train_sessions": [
                    sessions[i].session_id
                    for i in (fold.train_sessions[0], fold.train_sessions[-1])
                ],
                "test_sessions": [sessions[i].session_id for i in fold.test_sessions],
                "model_digest": model.model_digest,
                "training_rows": model.training_rows,
                "metrics": metrics,
                "ineligible_reasons": reasons,
            }
        )
    eligible = all(not report["ineligible_reasons"] for report in reports)
    means = [report["metrics"]["mean_net_return_bps"] for report in reports]
    ranking = float(np.median(means)) if eligible and None not in means else None
    return {
        "kind": "IntradayDevelopmentEvaluation",
        "evidence_status": "development_only",
        "leaves": list(plan.leaves),
        "folds": reports,
        "eligible": eligible,
        "ranking_metric_median_net_bps": ranking,
        "initial_capital": format(Decimal(plan.initial_capital), "f"),
    }
