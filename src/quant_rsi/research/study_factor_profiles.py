"""Execute explicitly bound profile factors using the study's existing retained inputs."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import canonical
from quant_rsi.factors.catalog import resolve_instrument_factors
from quant_rsi.factors.expressions import FactorExpression, evaluate_expression
from quant_rsi.research.demo import load_catalog
from quant_rsi.research.demo_features import price_feature


def load_study_catalog(symbol, catalog, root):
    if catalog is not None:
        return load_catalog(catalog)
    root = Path(root)
    if not (root / "US" / symbol / "factors.json").exists():
        return load_catalog(root / "shared/equities.json") | {
            "profile_note": f"No US/{symbol} factor profile; using the shared equities catalog."
        }
    return {
        "provider": "instrument_factor_profile",
        "definitions": [],
        "instrument_profile": resolve_instrument_factors(root, "US", symbol),
    }


def prepare_bound_features(profile, data, peers, required):
    """No collection, row dropping, implicit target substitution or terminal eligibility."""
    definitions, columns = {}, {}
    symbol = data.instruments[0]
    if profile["instrument"] != symbol:
        raise ValueError("Factor profile must match the study target")
    peer_definitions = {item["factor_id"]: item for item in peers.get("definitions", [])}
    for binding in profile["factors"]:
        definition, source = binding["definition"], binding["source"]
        identity = f"bound.{binding['binding_id']}"
        item = deepcopy(definition) | {
            "factor_id": identity,
            "definition_id": definition["factor_id"],
            "definition_scope": binding["definition_scope"],
            "definition_owner": binding["definition_owner"],
            "binding_id": binding["binding_id"],
            "source": deepcopy(source),
            "input_factor_ids": [identity],
        }
        values, reason = None, "unsupported_source_or_field"
        if definition["kind"] != "numeric":
            reason = "text_extraction_not_executable_in_this_profile"
        elif source == {"source_id": "target_daily_prices", "entity": symbol, "field": "close"}:
            values = price_feature(data, definition["recipe"])
        elif source["source_id"] == "related_daily_prices" and source["field"] == "close":
            recipe = definition["recipe"]
            peer_id = f"peer.{source['entity']}.peer_return_{recipe['window']}"
            retained = peer_definitions.get(peer_id)
            if recipe["op"] != "momentum":
                reason = "related_source_recipe_not_supported"
            elif retained is None or retained["recipe"] != {
                "op": "peer_return",
                "symbol": source["entity"],
                "window": recipe["window"],
            }:
                reason = "related_price_column_not_available"
            else:
                values = peers["columns"][peer_id].copy()
                item["runtime_dependency_id"] = peer_id
        if values is None:
            values = np.full(len(data.dates), np.nan)
        else:
            reason = (
                "complete_common_sample"
                if np.isfinite(values[required]).all()
                else ("missing_bound_input_on_fixed_sample")
            )
        item.update(
            runnable=bool(np.isfinite(values[required]).all()),
            availability_reason=reason,
        )
        definitions[identity], columns[identity] = item, values

    for composition in profile.get("compositions", []):
        identity = composition["factor_id"]
        if identity in definitions:
            raise ValueError("Composition shadows a bound factor")
        expression = FactorExpression.model_validate_json(canonical(composition["expression"]))
        leaves = {node.leaf: f"bound.{node.leaf}" for node in expression.nodes if node.op == "leaf"}
        if not set(leaves.values()) <= definitions.keys():
            raise ValueError("Composition refers to an unknown source binding")
        values = np.full(len(data.dates), np.nan)
        for index in range(len(data.dates)):
            sources = {
                name: float(columns[key][index]) if np.isfinite(columns[key][index]) else None
                for name, key in leaves.items()
            }
            value = evaluate_expression(expression, sources)["value"]
            if value is not None:
                values[index] = value
        runnable = bool(np.isfinite(values[required]).all())
        definitions[identity] = {
            "factor_id": identity,
            "definition_id": identity,
            "definition_scope": "instrument",
            "definition_owner": f"{profile['market']}/{symbol}",
            "description": composition["description"],
            "kind": "numeric",
            "input_kind": "bound_factor_values",
            "recipe": {"op": "factor_expression", "expression": composition["expression"]},
            "input_factor_ids": list(leaves.values()),
            "runnable": runnable,
            "availability_reason": "complete_common_sample"
            if runnable
            else ("missing_or_nonfinite_composition_inputs"),
        }
        columns[identity] = values
    return {"definitions": list(definitions.values()), "columns": columns}
