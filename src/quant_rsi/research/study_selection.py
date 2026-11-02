"""Preselected development-only adoption rule for the two existing study attempts."""

import math


def select_attempt(first, second, metric):
    if metric == "mae":
        scores = [attempt["development"]["metrics"]["mae_price"] for attempt in (first, second)]
        direction, label = "lower", "price MAE"
    elif metric == "sharpe":
        scores = [
            attempt["strategy"]["strategy"]["metrics"]["sharpe_ratio"]
            for attempt in (first, second)
        ]
        direction, label = "higher", "net strategy Sharpe"
    else:
        raise ValueError("Selection metric must be mae or sharpe")
    scores = [
        float(value) if value is not None and math.isfinite(value) else None for value in scores
    ]
    comparable = all(value is not None for value in scores)
    accepted = comparable and (scores[1] < scores[0] if metric == "mae" else scores[1] > scores[0])
    if not comparable:
        reason = f"Retain initial: {label} is unavailable for one or both attempts."
    elif accepted:
        reason = f"Adopt revised: {label} is strictly {direction} on the same development window."
    else:
        reason = f"Retain initial: revised {label} did not strictly improve (ties retain initial)."
    selected = second if accepted else first
    return {
        "metric": metric,
        "direction": direction,
        "initial_score": scores[0],
        "revised_score": scores[1],
        "selected_attempt": "revised" if accepted else "initial",
        "revision_accepted": accepted,
        "reason": reason,
        "selected_factor_ids": list(selected["factor_ids"]),
        "evidence_scope": "development_selection_after_feedback_not_independent_validation",
        "forecast_note": (
            "The latest forecast follows the selected attempt without another fit or a silent "
            "switch when its inputs are unavailable. Both original attempts remain in the trace."
        ),
    }
