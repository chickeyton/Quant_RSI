"""Instrument profile for intraday work (`instruments/<MARKET>/<SYMBOL>/intraday.json`)."""

from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.brokers.contracts import InstrumentBinding, InstrumentSpec
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.intraday.contracts import IntradayDataContract
from quant_rsi.market.market_contracts import CalendarSpec, Identifier


class ProfileInstrument(StrictModel):
    symbol: Identifier
    primary_exchange: Identifier
    currency: Identifier


class IntradayWorkflowConfig(StrictModel):
    schema_version: Literal[1]
    kind: Literal["IntradayWorkflowConfig"]
    target: ProfileInstrument
    context: tuple[ProfileInstrument, ...] = Field(max_length=9)
    source: Literal["ibkr_tws_api"]
    bar_seconds: Literal[60, 300]
    execution_bar_seconds: Literal[60]
    horizon_bars: int = Field(ge=1, le=78)
    history_sessions: int = Field(ge=20, le=750)
    availability_buffer_seconds: int = Field(ge=0, le=3600)
    decision_latency_seconds: int = Field(ge=0, le=600)
    order_latency_seconds: int = Field(ge=0, le=600)
    max_fill_delay_seconds: int = Field(gt=0, le=900)
    halt_missing_bars: int = Field(ge=1, le=78)
    flatten_offset_seconds: int = Field(ge=0, le=7200)
    forward_schedule: Literal["block_4w_dev_4w_reserved_v1"]
    overnight_policy: Literal["flat"]
    quote_fields: Literal["trades", "trades_and_bid_ask"]
    historical_availability: Literal["explicit_assumption"]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.decision_latency_seconds < self.availability_buffer_seconds:
            raise ValueError("decision latency must cover the availability buffer")
        symbols = [self.target.symbol, *(c.symbol for c in self.context)]
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate instrument in profile")
        return self

    def specs(self) -> list[InstrumentSpec]:
        return [
            InstrumentSpec(
                instrument=item.symbol,
                symbol=item.symbol,
                sec_type="STK",
                currency=item.currency,
                exchange="SMART",
                primary_exchange=item.primary_exchange,
            )
            for item in (self.target, *self.context)
        ]


def build_contract(
    profile: IntradayWorkflowConfig,
    bindings: tuple[InstrumentBinding, ...],
    calendar: CalendarSpec,
    *,
    data_entitlement: Literal["realtime", "delayed"],
) -> IntradayDataContract:
    return IntradayDataContract(
        schema_version=1,
        source_id=f"ibkr_intraday_{profile.target.symbol}_{profile.bar_seconds}s",
        provider="ibkr_tws_api",
        provider_statement=(
            "IBKR TWS API historical TRADES (and optional BID_ASK) bars, regular trading hours; "
            "raw prices; historical availability is an explicit assumption"
        ),
        frequency="intraday_equities",
        grid={
            "bar_seconds": profile.bar_seconds,
            "execution_bar_seconds": profile.execution_bar_seconds,
            "session_scope": "regular_trading_hours",
            "bar_timestamp": "bar_start",
            "final_bar_policy": "clip_to_session_close",
            "missing_bar_policy": "explicit_missing_no_fill_forward",
        },
        calendar=calendar,
        instruments=bindings,
        targets=(profile.target.symbol,),
        knowledge_basis="assumed_historical_availability",
        availability_buffer_seconds=profile.availability_buffer_seconds,
        data_entitlement=data_entitlement,
        quote_fields=profile.quote_fields,
        corporate_action_coverage="unknown",
    )
