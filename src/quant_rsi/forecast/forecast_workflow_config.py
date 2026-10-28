"""Strict instrument-role configuration for managed enhanced forecasts."""

from typing import Literal

from pydantic import Field, model_validator

from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.forecast import forecast_context_data as context


class ForecastWorkflowConfig(StrictModel):
    schema_version: Literal[1]
    kind: Literal["ForecastWorkflowConfig"]
    target: str
    industry: str
    market: str
    history_sessions: int = Field(default=756, ge=756, le=1000)
    horizon_sessions: Literal[5] = 5
    source: Literal["yahoo_chart_5y_daily"] = "yahoo_chart_5y_daily"
    calendar: Literal["published_nyse_group"] = "published_nyse_group"
    currency: Literal["USD"] = "USD"
    price_unit: Literal["raw_currency_per_share"] = "raw_currency_per_share"
    historical_availability: Literal["explicit_assumption"] = "explicit_assumption"
    max_source_age_seconds: int = Field(default=86400, ge=300, le=86400)

    @model_validator(mode="after")
    def explicit_roles(self):
        context.validate_roles(self.roles)
        return self

    @property
    def roles(self):
        return {role: getattr(self, role) for role in ("target", "industry", "market")}
