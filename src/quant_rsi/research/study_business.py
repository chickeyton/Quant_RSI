"""Run-local business hypotheses over existing publication-dated financial columns."""

from copy import deepcopy

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_rsi.core.artifacts import canonical
from quant_rsi.factors.expressions import FactorExpression, Name, evaluate_expression


class BusinessInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Name
    factor_id: str = Field(pattern=r"^financial\.[a-z][a-z0-9_]*$")


class BusinessFactorProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    name: Name
    inputs: list[BusinessInput] = Field(min_length=2, max_length=4)
    expression: FactorExpression
    description: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def bindings(self):
        names = [item.name for item in self.inputs]
        leaves = {node.leaf for node in self.expression.nodes if node.op == "leaf"}
        if len(set(names)) != len(names) or set(names) != leaves:
            raise ValueError("Business inputs must uniquely bind every expression leaf")
        return self

    def definition(self, definitions):
        sources = {item["factor_id"]: item for item in definitions}
        ids = [item.factor_id for item in self.inputs]
        for identity in ids:
            if (
                identity not in sources
                or sources[identity].get("recipe", {}).get("op") != "financial"
            ):
                raise ValueError(
                    "Business expressions require supplied financial factor definitions"
                )
        runnable = all(sources[identity].get("runnable") is True for identity in ids)
        return {
            "factor_id": f"business.{self.name}",
            "kind": "numeric",
            "input_kind": "retained_financial_factors",
            "description": self.description,
            "recipe": {
                "op": "financial_expression",
                "inputs": [item.model_dump(mode="json") for item in self.inputs],
                "expression": self.expression.model_dump(mode="json"),
            },
            "input_factor_ids": ids,
            "runnable": runnable,
            "availability_reason": "complete_common_sample"
            if runnable
            else "missing_financial_expression_inputs",
        }


def proposal_from_definition(definition):
    recipe = definition["recipe"]
    if recipe.get("op") != "financial_expression":
        raise ValueError("Expected a financial expression")
    return BusinessFactorProposal.model_validate_json(
        canonical(
            {
                "name": definition["factor_id"].removeprefix("business."),
                "inputs": recipe["inputs"],
                "expression": recipe["expression"],
                "description": definition["description"],
            }
        )
    )


def business_column(definition, bundle):
    """Rebuild from retained columns; missing values propagate without row dropping."""
    proposal = proposal_from_definition(definition)
    expected = proposal.definition((bundle or {}).get("definitions", []))
    if expected != definition or not expected["runnable"]:
        raise ValueError("Business factor lacks its exact usable financial inputs")
    inputs = {item.name: np.asarray(bundle["columns"][item.factor_id]) for item in proposal.inputs}
    lengths = {len(values) for values in inputs.values()}
    if len(lengths) != 1:
        raise ValueError("Business input columns must use the same fixed sample")
    values = np.full(lengths.pop(), np.nan)
    for index in range(len(values)):
        row = {
            name: float(column[index]) if np.isfinite(column[index]) else None
            for name, column in inputs.items()
        }
        result = evaluate_expression(proposal.expression, row)["value"]
        if result is not None:
            values[index] = result
    sources = {item["factor_id"]: item for item in bundle["definitions"]}
    return values, [deepcopy(sources[identity]) for identity in expected["input_factor_ids"]]
