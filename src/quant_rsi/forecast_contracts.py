"""Explicit five-session price tasks; legacy account plan serialization is unchanged."""

import json
from decimal import Decimal
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from quant_rsi.contracts_base import StrictModel
from quant_rsi.market_contracts import Hash, Identifier, decimal_text
from quant_rsi.schemas import BudgetSpec, LocalDataSpec, Proposal


class ForecastNotice(StrictModel):
    schema_version: Literal[1]
    notice_ref: Hash
    action_kind: Literal["cash_dividend", "split", "unsupported"]
    effective_at: AwareDatetime | None
    currency: Identifier | None
    amount: str | None
    unit: Literal["currency_per_share"] | None
    source_format: Literal["cash_dividend_v1", "yahoo_chart_dividends_v1", "unsupported"]
    evidence_ref: Hash

    @model_validator(mode="after")
    def cash_fields(self) -> Self:
        if self.action_kind == "cash_dividend":
            if (
                self.effective_at is None
                or self.currency is None
                or self.amount is None
                or self.unit != "currency_per_share"
                or self.source_format == "unsupported"
            ):
                raise ValueError(
                    "cash interpretation needs effective time, currency and share unit"
                )
            decimal_text(self.amount)
            if Decimal(self.amount) <= 0:
                raise ValueError("cash amount must be strictly positive")
        return self


class ForecastObjective(StrictModel):
    schema_version: Literal[1]
    kind: Literal["five_session_raw_close"]
    horizon_sessions: Literal[5]
    target: Literal["log_close_ratio"]
    currency: Identifier
    price_unit: Literal["raw_currency_per_share"]
    price_unit_claim: Literal["source_documented_not_independently_verified", "explicit_assumption"]
    price_unit_evidence_ref: Hash
    calendar_evidence_ref: Hash
    notices: tuple[ForecastNotice, ...]
    unknown_action_policy: Literal["affected_or_unknown_interval_abstain"]
    prior_exposure: Literal["previously_exposed_historical_exploration"]
    source_qualifications: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_notices(self) -> Self:
        refs = [item.notice_ref for item in self.notices]
        if len(refs) != len(set(refs)):
            raise ValueError("notice interpretations must bind unique source payloads")
        if any(
            item.action_kind == "cash_dividend" and item.currency != self.currency
            for item in self.notices
        ):
            raise ValueError("cash interpretation currency differs from forecast quote units")
        return self


class ForecastDevelopmentPlan(StrictModel):
    schema_version: Literal[1]
    plan_type: Literal["forecast_development"]
    training_mode: Literal["M0"]
    data: LocalDataSpec
    objective: ForecastObjective
    budget: BudgetSpec
    primary_metric: Literal["negative_mae_pp"]
    evaluation_stride: Literal[5]
    min_development_sessions: int = Field(ge=3)
    baseline: str
    proposals: tuple[Proposal, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        names = [p.name for p in self.proposals]
        if len(names) != len(set(names)) or self.baseline != names[0]:
            raise ValueError("unique proposals required; the first is the C2 initial model")
        if self.budget.max_worker_seconds < self.budget.worker_timeout_seconds:
            raise ValueError("worker budget cannot fund one reservation")
        origins = range(
            self.data.train_cutoff,
            self.data.sessions - self.objective.horizon_sessions,
            self.evaluation_stride,
        )
        if len(origins) < self.min_development_sessions:
            raise ValueError("insufficient fixed five-session evaluation origins")
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self


def parse_development_plan(raw):
    # Dispatch does not inject defaults or change serialization of a legacy plan.
    from quant_rsi.schemas import DevelopmentPlan

    value = json.loads(raw)
    if value.get("plan_type") == "forecast_development":
        return ForecastDevelopmentPlan.model_validate_json(raw)
    return DevelopmentPlan.model_validate_json(raw)
