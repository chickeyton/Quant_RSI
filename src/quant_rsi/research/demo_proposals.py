"""Executable, bounded recipe proposals; no generated Python or catalog mutation."""

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quant_rsi.factors.contracts import NumericFactorDefinition


class FactorProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    op: Literal["momentum", "mean_return", "qlib_roc_v1", "qlib_ma_v1"]
    window: int = Field(ge=1, le=30)
    description: str = Field(min_length=1, max_length=1000)

    def definition(self):
        return NumericFactorDefinition.model_validate(
            {
                "factor_id": f"research.{self.op}_{self.window}",
                "kind": "numeric",
                "input_kind": "positive_price_series",
                "description": self.description,
                "recipe": {"op": self.op, "window": self.window},
            }
        ).model_dump(mode="json")


def extend_library(library, hypothesis):
    """Return a per-run catalog, preserving definitions behind past decisions."""
    result = deepcopy(library)
    definitions = {item["factor_id"]: item for item in result["definitions"]}
    proposals = hypothesis.get("proposed_factors", [])
    if not isinstance(proposals, list) or not 1 <= len(proposals) <= 3:
        raise ValueError("Study hypotheses require one to three bounded factor proposals")
    ids = []
    for proposal in proposals:
        definition = FactorProposal.model_validate(proposal).definition()
        identity = definition["factor_id"]
        if identity in ids:
            raise ValueError("Duplicate proposed factor recipe")
        ids.append(identity)
        if identity in definitions and definitions[identity]["recipe"] != definition["recipe"]:
            raise ValueError("Proposed factor conflicts with an existing definition")
        definitions.setdefault(identity, definition)
    if not set(ids).intersection(hypothesis["factor_ids"]):
        raise ValueError("Select at least one proposed factor for execution")
    result["definitions"] = list(definitions.values())
    return result
