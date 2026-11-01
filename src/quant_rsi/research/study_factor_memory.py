"""Reuse selected recipes from study history without importing old feature values."""

from copy import deepcopy

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research.demo_proposals import FactorProposal, NewsFactorProposal

NOTICE = (
    "Recipes from the latest five studies of this symbol, including unobserved forecasts. "
    "Prior selection is not evidence that a factor works. Only settled forecast outcomes "
    "supply observed error feedback. Availability is checked against current inputs; "
    "old feature values and historical runnable flags are never reused. "
    "Unselected recipes remain historical candidates, not globally deleted factors."
)


def selected_definitions(result):
    """The frozen catalogs preserve definitions for every round, including the incumbent."""
    selected = result["iterations"]["selected_round"]
    ids = result["iterations"]["attempts"][selected - 1]["prediction"]["factor_ids"]
    definitions = {}
    for event in result.get("trace", []):
        if event["stage"].endswith("factor_library"):
            definitions.update({d["factor_id"]: d for d in event["output"]["definitions"]})
    return [deepcopy(definitions[key]) for key in ids if key in definitions]


def reuse_recipes(library, memories):
    library = deepcopy(library)
    definitions = {d["factor_id"]: d for d in library["definitions"]}
    entries, seen = [], set()
    for memory in memories:
        for old in memory["definitions"]:
            identity = old["factor_id"]
            if identity in seen:
                continue  # Most recently selected version wins; no historical metric ranking.
            seen.add(identity)
            entry = {
                "factor_id": identity,
                "recipe": deepcopy(old.get("recipe")),
                "recipe_sha256": digest(canonical(old.get("recipe"))),
                "source_study_id": memory["study_id"],
                "source_result_ref": memory["result_ref"],
                "source_issued_at": memory["issued_at"],
                "source_outcome_status": memory["outcome_status"],
                "source_input_mode": memory.get("input_mode", "unknown"),
                "status": "requires_current_binding",
            }
            current = definitions.get(identity)
            if current is not None:
                entry["status"] = (
                    "definition_conflict"
                    if current.get("recipe") != old.get("recipe")
                    else "unavailable_current_inputs"
                    if current.get("runnable") is False
                    else "available_in_current_catalog"
                )
            else:
                try:
                    recipe = old.get("recipe", {})
                    if identity.startswith("research."):
                        current = FactorProposal.model_validate(
                            recipe | {"description": old["description"]}
                        ).definition()
                    elif (
                        identity.startswith("news.")
                        and recipe.get("op") == "headline_keyword_count"
                    ):
                        current = NewsFactorProposal.model_validate(
                            {
                                "name": identity.removeprefix("news.").rsplit("_", 1)[0],
                                "keywords": recipe["keywords"],
                                "window_days": recipe["window_days"],
                                "description": old["description"],
                            }
                        ).definition(library.get("news_feature_support", {}).get("runnable", False))
                    if current is not None:
                        if current["factor_id"] != identity or current["kind"] != old["kind"]:
                            raise ValueError("Remembered identity differs from its bounded recipe")
                        definitions[identity] = current
                        entry["status"] = (
                            "unavailable_current_inputs"
                            if current.get("runnable") is False
                            else "restored_candidate"
                        )
                except (KeyError, TypeError, ValueError):
                    entry["status"] = "unsupported_or_invalid_recipe"
            entries.append(entry)
    library["definitions"] = list(definitions.values())
    return library, {"notice": NOTICE, "entries": entries}
