"""Remember every attempted factor set and its development measurements across studies."""

from copy import deepcopy

from quant_rsi.core.artifacts import canonical, digest

NOTICE = (
    "Every attempted round of previous studies for this symbol and horizon, including "
    "rejected revisions. Metrics are development measurements on each study's own window "
    "after adaptive feedback; they are not independent evidence that a factor works and are "
    "not comparable as a ranking across different periods or inputs. Rejected recipes are "
    "not restored as catalog candidates; propose them again explicitly to retry them. "
    "Different executable definitions, source bindings and prediction horizons have separate "
    "ledger entries even when their display factor_id is the same. Metrics describe the whole "
    "attempted model containing the factor, not that factor's isolated contribution."
)


def _number(value):
    return value if isinstance(value, int | float) and not isinstance(value, bool) else None


def attempt_records(result):
    """Small per-round measurements; the full predictions stay in the retained result."""
    iterations = result["iterations"]
    records = []
    for number, attempt in enumerate(iterations["attempts"], start=1):
        number = attempt.get("round", number)
        prediction = attempt.get("prediction", {})
        development = prediction.get("development", {})
        metrics = development.get("metrics", {})
        baseline = development.get("baselines", {}).get("unchanged_price", {}).get("metrics", {})
        strategy = prediction.get("strategy", {}).get("strategy", {}).get("metrics", {})
        records.append(
            {
                "round": number,
                "factor_ids": list(prediction.get("factor_ids", [])),
                "selected": number == iterations["selected_round"],
                "accepted": attempt.get("accepted"),
                "mae_price": _number(metrics.get("mae_price")),
                "baseline_mae_price": _number(baseline.get("mae_price")),
                "direction_hit_rate": _number(metrics.get("direction_hit_rate")),
                "sharpe_ratio": _number(strategy.get("sharpe_ratio")),
                "net_return": strategy.get("net_return"),
            }
        )
    return records


def attempted_definitions(result):
    """Attempted factors and their input definitions from the frozen catalogs."""
    ids = {key for attempt in attempt_records(result) for key in attempt["factor_ids"]}
    definitions = {}
    for event in result.get("trace", []):
        if event["stage"].endswith("factor_library"):
            definitions.update({d["factor_id"]: d for d in event["output"]["definitions"]})
    pending = list(ids)
    while pending:
        key = pending.pop()
        for source in definitions.get(key, {}).get("input_factor_ids") or []:
            if source not in ids:
                ids.add(source)
                pending.append(source)
    return [deepcopy(definitions[key]) for key in sorted(ids) if key in definitions]


def factor_identity(key, definitions, horizon):
    """Fingerprint the recorded executable recipe and its transitive source bindings."""
    pending, execution = [key], {}
    while pending:
        identity = pending.pop()
        if identity in execution:
            continue
        definition = definitions.get(identity, {})
        execution[identity] = {
            field: deepcopy(definition[field])
            for field in (
                "kind",
                "input_kind",
                "recipe",
                "source",
                "input_factor_ids",
                "reuse_lifetime",
            )
            if field in definition
        }
        pending.extend(definition.get("input_factor_ids") or [])
    definition_sha = digest(canonical(execution))
    return {
        "ledger_id": digest(canonical([key, horizon, definition_sha])),
        "definition_sha256": definition_sha,
        "horizon_sessions": horizon,
    }


def _better(current, value, lower):
    if value is None:
        return current
    if current is None:
        return value
    return min(current, value) if lower else max(current, value)


def summarize(studies, *, limit=None):
    """Per-factor counts and bounds over chronologically ordered study records."""
    factors = {}
    for study in studies:
        definitions = {d["factor_id"]: d for d in study["definitions"]}
        horizon = study.get("horizon_sessions", 5)
        for attempt in study["attempts"]:
            for key in attempt["factor_ids"]:
                definition = definitions.get(key, {})
                identity = factor_identity(key, definitions, horizon)
                item = factors.setdefault(
                    identity["ledger_id"],
                    {
                        **identity,
                        "factor_id": key,
                        "kind": definition.get("kind"),
                        "description": definition.get("description"),
                        "recipe": deepcopy(definition.get("recipe")),
                        "first_seen": study["issued_at"],
                        "studies": [],
                        "questions": [],
                        "rounds_tried": 0,
                        "rounds_selected": 0,
                        "rounds_beating_unchanged_price": 0,
                        "best_mae_price": None,
                        "best_sharpe_ratio": None,
                        "settled_selected_forecasts": 0,
                        "settled_studies": [],
                    },
                )
                if definition:
                    # Descriptions may change; executable definitions never share a row.
                    item.update(
                        kind=definition.get("kind"),
                        description=definition.get("description"),
                        recipe=deepcopy(definition.get("recipe")),
                    )
                if study["study_id"] not in item["studies"]:
                    item["studies"].append(study["study_id"])
                if (
                    attempt["selected"]
                    and study["outcome_status"] == "settled"
                    and study["study_id"] not in item["settled_studies"]
                ):
                    item["settled_studies"].append(study["study_id"])
                    item["settled_selected_forecasts"] += 1
                if study["question"] not in item["questions"]:
                    item["questions"].append(study["question"])
                item["rounds_tried"] += 1
                item["rounds_selected"] += int(attempt["selected"])
                mae, baseline = attempt["mae_price"], attempt["baseline_mae_price"]
                if mae is not None and baseline is not None and mae < baseline:
                    item["rounds_beating_unchanged_price"] += 1
                item["best_mae_price"] = _better(item["best_mae_price"], mae, True)
                item["best_sharpe_ratio"] = _better(
                    item["best_sharpe_ratio"], attempt["sharpe_ratio"], False
                )
                item["last_seen"] = study["issued_at"]
                item["last_mae_price"] = mae
                item["last_selected"] = attempt["selected"]
    rows = sorted(factors.values(), key=lambda item: item["last_seen"], reverse=True)
    for item in rows:
        item["study_count"] = len(item.pop("studies"))
        del item["settled_studies"]
    return {"notice": NOTICE, "factors": rows[:limit] if limit else rows}
