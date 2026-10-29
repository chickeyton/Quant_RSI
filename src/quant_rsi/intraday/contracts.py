"""Typed intraday data contracts and plans.

The data contract describes only the data. Latency and fill-window choices are execution
parameters (`ExecutionTiming`) that live in plans and sealed bundles, so changing them never
changes a snapshot's identity.
"""

from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from quant_rsi.brokers.contracts import InstrumentBinding
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.market.market_contracts import CalendarSpec, Identifier, KnowledgeBasis, decimal_text

EXECUTION_BAR_SECONDS = 60
SESSIONS_PER_YEAR = 252


class IntradayGrid(StrictModel):
    bar_seconds: Literal[60, 300]
    execution_bar_seconds: Literal[60]
    session_scope: Literal["regular_trading_hours"]
    bar_timestamp: Literal["bar_start"]
    final_bar_policy: Literal["clip_to_session_close"]
    missing_bar_policy: Literal["explicit_missing_no_fill_forward"]


class IntradayDataContract(StrictModel):
    schema_version: Literal[1]
    source_id: Identifier
    provider: Literal["ibkr_tws_api", "engineering_fixture"]
    provider_statement: str = Field(min_length=1, max_length=1000)
    frequency: Literal["intraday_equities"]
    grid: IntradayGrid
    calendar: CalendarSpec
    instruments: tuple[InstrumentBinding, ...] = Field(min_length=1, max_length=10)
    targets: tuple[Identifier, ...] = Field(min_length=1, max_length=10)
    knowledge_basis: KnowledgeBasis
    availability_buffer_seconds: int = Field(ge=0, le=3600)
    data_entitlement: Literal["realtime", "delayed"]
    quote_fields: Literal["trades", "trades_and_bid_ask"]
    corporate_action_coverage: Literal["unknown", "sidecar_notices"]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        names = [binding.instrument for binding in self.instruments]
        if len(set(names)) != len(names):
            raise ValueError("instrument identities must be unique")
        if len(set(self.targets)) != len(self.targets) or not set(self.targets) <= set(names):
            raise ValueError("targets must be unique registered instruments")
        if len(self.calendar.sessions) > 750:
            raise ValueError("intraday contracts support at most 750 sessions")
        return self

    @property
    def context(self) -> tuple[str, ...]:
        return tuple(b.instrument for b in self.instruments if b.instrument not in self.targets)


class ExecutionTiming(StrictModel):
    decision_latency_seconds: int = Field(ge=0, le=600)
    order_latency_seconds: int = Field(ge=0, le=600)
    max_fill_delay_seconds: int = Field(gt=0, le=900)


def validate_timing(contract: IntradayDataContract, timing: ExecutionTiming) -> None:
    if timing.decision_latency_seconds < contract.availability_buffer_seconds:
        raise ValueError(
            "decision_latency_seconds must be >= availability_buffer_seconds; otherwise the "
            "bar that just closed is never available at the decision"
        )


def _fraction(value: str, *, allow_zero: bool = False) -> str:
    result = decimal_text(value, positive=not allow_zero)
    if Decimal(result) > 1:
        raise ValueError("fraction must not exceed one")
    return result


class CostModel(StrictModel):
    policy: Literal["ibkr_fixed_per_share"]
    commission_per_share: str
    min_commission: str
    max_commission_fraction: str
    spread_policy: Literal["mid_plus_half_spread", "trade_open_plus_half_spread_conservative"]
    assumed_spread_bps: str
    participation_cap: str
    currency: Identifier
    currency_precision: int = Field(ge=0, le=8)

    @field_validator("commission_per_share", "min_commission", "assumed_spread_bps")
    @classmethod
    def amount(cls, value):
        return decimal_text(value)

    @field_validator("max_commission_fraction", "participation_cap")
    @classmethod
    def fraction(cls, value):
        return _fraction(value)


class SignalPolicy(StrictModel):
    kind: Literal["threshold_long_only"]
    entry_threshold: float
    exit_threshold: float
    horizon_bars: int = Field(ge=1, le=78)
    notional_fraction: str
    max_positions: int = Field(ge=1, le=10)
    lot_size: int = Field(ge=1, le=1_000_000)
    flatten_offset_seconds: int = Field(ge=0, le=7200)
    halt_missing_bars: int = Field(ge=1, le=78)
    overnight_policy: Literal["flat"]

    @field_validator("notional_fraction")
    @classmethod
    def fraction(cls, value):
        return _fraction(value)

    @model_validator(mode="after")
    def thresholds(self) -> Self:
        if self.exit_threshold > self.entry_threshold:
            raise ValueError("exit threshold must not exceed the entry threshold")
        return self


class FoldSpec(StrictModel):
    first_test_session: int = Field(ge=5)
    test_sessions: int = Field(ge=1)
    folds: int = Field(ge=1, le=12)
    embargo_sessions: int = Field(ge=0, le=20)


class IntradayDevelopmentPlan(StrictModel):
    schema_version: Literal[1]
    plan_type: Literal["intraday_development"]
    training_mode: Literal["M0"]
    timing: ExecutionTiming
    signal: SignalPolicy
    costs: CostModel
    leaves: tuple[str, ...] = Field(min_length=1, max_length=64)
    folds: FoldSpec
    seed: int = Field(ge=0, le=2**31 - 1)
    initial_capital: str
    min_training_rows: int = Field(ge=12)
    min_trades_per_fold: int = Field(ge=0)
    min_prediction_coverage: float = Field(ge=0, le=1)
    context_missing_tolerance: int = Field(ge=0, le=12)
    allow_cross_session_lookback: Literal[False]
    sessions_per_year: Literal[252]

    @field_validator("initial_capital")
    @classmethod
    def capital(cls, value):
        return decimal_text(value, positive=True)

    @model_validator(mode="after")
    def unique_leaves(self) -> Self:
        if len(set(self.leaves)) != len(self.leaves):
            raise ValueError("duplicate feature leaf")
        return self
