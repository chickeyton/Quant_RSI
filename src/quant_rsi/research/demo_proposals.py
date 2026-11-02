"""Executable, bounded recipe proposals; no generated Python or catalog mutation."""

from copy import deepcopy
from datetime import timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_rsi.core.artifacts import canonical
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.factors.contracts import NumericFactorDefinition
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.research.study_business import BusinessFactorProposal


class StudyVolatilityRecipe(StrictModel):
    """Unannualized sample deviation of N daily log returns, requiring N+1 closes."""

    op: Literal["log_return_volatility"]
    window: int = Field(ge=2, le=30)


class FactorProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    op: Literal["momentum", "mean_return", "qlib_roc_v1", "qlib_ma_v1", "log_return_volatility"]
    window: int = Field(ge=1, le=30)
    description: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_volatility_window(self):
        if self.op == "log_return_volatility" and self.window < 2:
            raise ValueError("Log-return volatility needs at least two daily returns")
        return self

    def definition(self):
        definition = {
            "factor_id": f"research.{self.op}_{self.window}",
            "kind": "numeric",
            "input_kind": "positive_price_series",
            "description": self.description,
            "recipe": {"op": self.op, "window": self.window},
        }
        if self.op == "log_return_volatility":
            # Run-local study recipe; the protected worker/catalog contract is unchanged.
            return definition
        return NumericFactorDefinition.model_validate(definition).model_dump(mode="json")


class NewsFactorProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    keywords: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(
        min_length=1, max_length=5
    )
    window_days: int = Field(ge=1, le=30)
    description: str = Field(min_length=1, max_length=1000)
    reuse_for_hours: int | None = Field(default=None, ge=1, le=720)

    def definition(self, runnable):
        definition = {
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
        if self.reuse_for_hours is not None:
            definition["reuse_lifetime"] = {"hours": self.reuse_for_hours}
        return definition


def news_recipe_expired(definition, as_of):
    """Check study admission time, never the historical timestamps of feature rows."""
    lifetime = definition.get("reuse_lifetime")
    if lifetime is None:
        return False
    if not isinstance(lifetime, dict):
        raise ValueError("Invalid news recipe lifetime")
    hours = lifetime["hours"]
    if type(hours) is not int or not 1 <= hours <= 720:
        raise ValueError("Invalid news recipe lifetime")
    created, expires = timestamp(lifetime["created_at"]), timestamp(lifetime["expires_at"])
    evidence = lifetime["evidence_news_ids"]
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(key, str) or not key.strip() for key in evidence)
    ):
        raise ValueError("Event recipe requires its original news evidence IDs")
    if expires != created + timedelta(hours=hours) or timestamp(as_of) < created:
        raise ValueError("News recipe lifetime differs from its original admission clock")
    return timestamp(as_of) >= expires


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
        if identity in definitions:
            original = definitions[identity].get("reuse_lifetime") or {}
            if original.get("hours") != definition.get("reuse_lifetime", {}).get("hours"):
                raise ValueError("Use a new news factor name when changing its reuse lifetime")
        elif "reuse_lifetime" in definition:
            # The study freezes this clock in its existing operation cache. Re-selection
            # and resume retain the first binding instead of extending the deadline.
            created = library.get("recipe_as_of")
            if created is None:
                raise ValueError("Event news recipes require a frozen study admission clock")
            definition["reuse_lifetime"].update(
                created_at=created,
                expires_at=utc_text(
                    timestamp(created) + timedelta(hours=definition["reuse_lifetime"]["hours"])
                ),
                evidence_news_ids=deepcopy(hypothesis.get("evidence_news_ids", [])),
            )
            news_recipe_expired(definition, created)
        definitions.setdefault(identity, definition)
    business_proposals = hypothesis.get("proposed_business_factors", [])
    if not isinstance(business_proposals, list) or len(business_proposals) > 2:
        raise ValueError("At most two business factor proposals are supported per round")
    business_ids = set()
    for proposal in business_proposals:
        definition = BusinessFactorProposal.model_validate_json(canonical(proposal)).definition(
            library["definitions"]
        )
        identity = definition["factor_id"]
        if identity in business_ids:
            raise ValueError("Duplicate business factor proposal")
        business_ids.add(identity)
        if identity in definitions and definitions[identity]["recipe"] != definition["recipe"]:
            raise ValueError("Use a new business factor name when changing an existing recipe")
        definitions.setdefault(identity, definition)
    result["definitions"] = list(definitions.values())
    return result
