"""Executable, bounded recipe proposals; no generated Python or catalog mutation."""

from copy import deepcopy
from typing import Annotated, Literal

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


class NewsFactorProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    keywords: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        min_length=1, max_length=5
    )
    window_days: int = Field(ge=1, le=30)
    description: str = Field(min_length=1, max_length=1000)

    def definition(self, runnable):
        return {
            "factor_id": f"news.{self.name}_{self.window_days}",
            "kind": "news",
            "input_kind": "retained_issuer_headlines",
            "description": self.description,
            "runnable": runnable,
            "recipe": {
                "op": "headline_keyword_count",
                "keywords": self.keywords,
                "window_days": self.window_days,
            },
        }


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
    news_proposals = hypothesis.get("proposed_news_factors", [])
    if not isinstance(news_proposals, list) or len(news_proposals) > 2:
        raise ValueError("At most two news factor proposals are supported per round")
    news_ids = set()
    for proposal in news_proposals:
        definition = NewsFactorProposal.model_validate(proposal).definition(
            library.get("news_feature_support", {}).get("runnable", False)
        )
        identity = definition["factor_id"]
        if identity in news_ids:
            raise ValueError("Duplicate news factor proposal")
        news_ids.add(identity)
        if identity in definitions and definitions[identity]["recipe"] != definition["recipe"]:
            raise ValueError("Use a new news factor name when changing an existing recipe")
        definitions.setdefault(identity, definition)
    result["definitions"] = list(definitions.values())
    return result
