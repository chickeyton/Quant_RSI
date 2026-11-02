"""A browsable star map of remembered factors and their recorded relationships."""

import json
from importlib.resources import files
from itertools import combinations

from quant_rsi.market.market_contracts import trusted_now
from quant_rsi.research.study_factor_ledger import summarize
from quant_rsi.research.study_history import ledger_studies

CATEGORIES = {
    "research": "Price recipes",
    "news": "News & guidance",
    "financial": "Financial statements",
    "business": "Business expressions",
    "external": "Industry drivers",
    "peer": "Related companies",
    "bound": "Instrument profile",
    "other": "Other",
}
NOTICE = (
    "Star size counts selections, development rounds beating the unchanged-price baseline "
    "and settled selected forecasts. Brightness fades with time since a factor was last "
    "tried. Lines are recorded relationships: derived inputs, factors selected together, "
    "and revisions replacing one set with another. None of these establish that a factor "
    "or relationship predicts returns."
)
REPLACEMENT_PAIRS = 9


def category(factor_id):
    prefix = factor_id.split(".", 1)[0]
    return prefix if prefix in CATEGORIES else "other"


def _round_view(attempt):
    share = attempt.get("column_share")
    if not isinstance(share, dict):
        share = {}
    kept = {}
    for key in attempt["factor_ids"]:
        value = share.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            kept[key] = value
    hypothesis = attempt.get("hypothesis")
    return {
        "round": attempt["round"],
        "factor_ids": list(attempt["factor_ids"]),
        "accepted": bool(attempt.get("accepted")),
        "selected": bool(attempt.get("selected")),
        "mae_price": attempt.get("mae_price"),
        "baseline_mae_price": attempt.get("baseline_mae_price"),
        "column_share": kept,
        "hypothesis": hypothesis if isinstance(hypothesis, str) else "",
    }


def constellation(directory, symbol):
    studies = ledger_studies(directory, symbol)
    stars = {item["factor_id"]: item for item in summarize(studies)["factors"]}
    relations = {}

    def relate(source, target, kind, study):
        if source == target:
            return
        if kind == "co_selected":
            source, target = sorted((source, target))
        item = relations.setdefault(
            (source, target, kind),
            {"source": source, "target": target, "kind": kind, "weight": 0, "studies": []},
        )
        item["weight"] += 1
        item["last_seen"] = study["issued_at"]
        if study["study_id"] not in item["studies"]:
            item["studies"].append(study["study_id"])

    for study in studies:
        definitions = {d["factor_id"]: d for d in study["definitions"]}
        for definition in definitions.values():
            for source in definition.get("input_factor_ids") or []:
                if source not in stars:
                    stars[source] = {
                        "factor_id": source,
                        "kind": None,
                        "description": "Input of a derived factor; not tried directly.",
                        "recipe": None,
                        "first_seen": study["issued_at"],
                        "last_seen": study["issued_at"],
                        "rounds_tried": 0,
                        "rounds_selected": 0,
                        "rounds_beating_unchanged_price": 0,
                        "settled_selected_forecasts": 0,
                        "study_count": 0,
                        "input_only": True,
                    }
                relate(source, definition["factor_id"], "derived", study)
        previous = None
        for attempt in study["attempts"]:
            ids = attempt["factor_ids"]
            if attempt["selected"]:
                for source, target in combinations(ids, 2):
                    relate(source, target, "co_selected", study)
            if previous is not None:
                removed = [key for key in previous if key not in ids]
                added = [key for key in ids if key not in previous]
                if 0 < len(removed) * len(added) <= REPLACEMENT_PAIRS:
                    for source in removed:
                        for target in added:
                            relate(source, target, "replaced", study)
            previous = ids
    for item in stars.values():
        item["category"] = category(item["factor_id"])
        item["importance"] = (
            item["rounds_selected"]
            + item["rounds_beating_unchanged_price"]
            + 2 * item["settled_selected_forecasts"]
        )
    return {
        "symbol": symbol,
        "generated_at": trusted_now(),
        "notice": NOTICE,
        "categories": CATEGORIES,
        "studies": [
            {key: study[key] for key in ("study_id", "issued_at", "question", "horizon_sessions")}
            | {
                "outcome_status": study["outcome_status"],
                "selected_factor_ids": next(
                    (a["factor_ids"] for a in study["attempts"] if a["selected"]), []
                ),
                "rounds": len(study["attempts"]),
                "round_stack": [_round_view(attempt) for attempt in study["attempts"]],
            }
            for study in studies
        ],
        "stars": sorted(stars.values(), key=lambda item: item["factor_id"]),
        "relations": list(relations.values()),
    }


def render(data):
    payload = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    template = files("quant_rsi.research").joinpath("study_constellation.html").read_text()
    return template.replace("/*CONSTELLATION_DATA*/null", payload)
