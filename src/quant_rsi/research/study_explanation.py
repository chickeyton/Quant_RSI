"""Read-only explanations from recorded hypotheses, fitted inputs and prior receipts."""

from collections import Counter

NEWS_NOTICE = (
    "Citations support the LLM's hypothesis, not a measured contribution to prediction. "
    "Current article text is research context only. Fitted news features use retained issuer "
    "snapshots available at each historical cutoff; unavailable history is not a zero signal."
)


def question_coverage(hypothesis, fitted_ids=None):
    """LLM claims stay attributed; execution labels come from recorded fit columns."""
    rows = []
    for claim in hypothesis.get("question_map", []):
        ids = claim["factor_ids"]
        status = "no_measured_input"
        if ids:
            status = "proposed_inputs" if fitted_ids is None else "inputs_not_fitted"
            if fitted_ids is not None and set(ids).issubset(fitted_ids):
                status = "inputs_fitted"
        rows.append({**claim, "status": status})
    return rows


def explain_prediction(prediction, hypothesis, factors, catalog, news):
    sources = {item["id"]: item for item in news.get("items", [])}
    definitions = {item["factor_id"]: item for item in catalog.get("definitions", [])}
    definitions.update({item["factor_id"]: item for item in factors})
    fitted = set(prediction.get("factor_ids", []))
    rows = []
    for identity in prediction.get("factor_ids", []):
        definition = definitions.get(identity, {})
        rows.append(
            {
                "factor_id": identity,
                "input_kind": definition.get("input_kind", definition.get("kind", "unknown")),
                "status": "fitted",
                "reason": "Present in this completed fit's recorded feature columns",
            }
        )
    support = prediction.get("news_feature_support") or {}
    for identity, definition in definitions.items():
        unavailable = definition.get("runnable") is False
        if identity in fitted or (definition.get("kind") != "news" and not unavailable):
            continue
        reasons = definition.get("reasons") or [definition.get("availability_reason")]
        if not any(reasons) and unavailable and definition.get("kind") == "news":
            reasons = support.get("reasons", [])
        rows.append(
            {
                "factor_id": identity,
                "input_kind": definition.get("input_kind", definition.get("kind", "unknown")),
                "status": "unavailable" if unavailable else "not_selected",
                "reason": "; ".join(str(r) for r in reasons if r)
                or ("Insufficient historical input coverage" if unavailable else "Not chosen"),
            }
        )
    return {
        "hypothesis": hypothesis.get("hypothesis"),
        "question_map": question_coverage(hypothesis, fitted),
        "citations": [
            {
                "id": identity,
                "title": sources.get(identity, {}).get("title", "Source absent from trace"),
                "url": sources.get(identity, {}).get("url"),
                "role": "hypothesis_context",
            }
            for identity in hypothesis.get("evidence_news_ids", [])
        ],
        "factors": rows,
        "news_columns_fitted": sum(
            definitions.get(identity, {}).get("kind") == "news" for identity in fitted
        ),
        "training_rows": prediction.get("training_rows"),
        "news_coverage": support,
        "notice": NEWS_NOTICE,
    }


def evidence_path(steps):
    """One path per completed fit, including rejected attempts and all later rounds."""
    paths = []
    for number in range(1, 11):
        prefix = {1: "initial", 2: "revised"}.get(number, f"round_{number}")
        stage = {1: "first_prediction", 2: "second_prediction"}.get(
            number, f"round_{number}_prediction"
        )
        if stage not in steps:
            continue
        paths.append(
            {
                "round": number,
                **explain_prediction(
                    steps[stage],
                    steps.get(f"{prefix}_hypothesis", {}),
                    steps.get(f"{prefix}_factors", []),
                    steps.get(f"{prefix}_factor_library", steps.get("factor_library", {})),
                    steps.get("news", {}),
                ),
            }
        )
    return paths


def continuity(prior, selected_ids=None):
    """Separate supplied mature outcomes from pending history and current model adoption."""
    if prior is None:
        return None
    entries = prior.get("history", {}).get("entries", [])
    feedback = {entry["study_id"]: entry for entry in prior.get("entries", [])}
    latest = entries[-1] if entries else None
    previous_ids = set(latest["factor_ids"]) if latest else set()
    current_ids = set(selected_ids or [])
    return {
        "counts": dict(Counter(entry["status"] for entry in entries)),
        "entries": [
            {
                "study_id": entry["study_id"],
                "target_session": (entry.get("forecast") or {}).get("target_session"),
                "fit_mode": (entry.get("forecast") or {}).get(
                    "fit_mode", "selected_research_model"
                ),
                "training_cutoff": (entry.get("forecast") or {}).get("training_information_cutoff"),
                "status": entry["status"],
                "reason": entry.get("reason"),
                "predicted_price": (entry.get("forecast") or {}).get("predicted_price"),
                "actual_price": (feedback.get(entry["study_id"], {}).get("metrics") or {}).get(
                    "actual_price"
                ),
                "price_error": (feedback.get(entry["study_id"], {}).get("metrics") or {}).get(
                    "price_error"
                ),
                "provided_as_feedback": entry["study_id"] in feedback,
            }
            for entry in entries
        ],
        "previous_study_id": latest["study_id"] if latest else None,
        "previous_factor_ids": sorted(previous_ids),
        "selected_factor_ids": selected_ids,
        "added": sorted(current_ids - previous_ids) if latest and selected_ids is not None else [],
        "removed": sorted(previous_ids - current_ids)
        if latest and selected_ids is not None
        else [],
        "notice": "History at this run's start (latest 50 receipts). Only the latest 20 settled "
        "outcomes are supplied as outcome feedback; pending forecasts are not errors. "
        "Providing feedback does not prove the LLM used it. Factor changes compare this run's "
        "selected model with the previous run's selected model, not their predictive quality.",
    }
