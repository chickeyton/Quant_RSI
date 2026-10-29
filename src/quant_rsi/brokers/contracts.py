"""Broker-neutral typed records shared by data collection and paper execution."""

from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.market.market_contracts import Identifier, decimal_text

Side = Literal["buy", "sell"]


class InstrumentSpec(StrictModel):
    """What an instrument profile asks for; never an identity by itself."""

    instrument: Identifier
    symbol: Identifier
    sec_type: Literal["STK"]
    currency: Identifier
    exchange: Literal["SMART"]
    primary_exchange: Identifier


class InstrumentBinding(StrictModel):
    """A provider-qualified instrument: exactly one contract ID, with recorded hours evidence."""

    instrument: Identifier
    symbol: Identifier
    sec_type: Literal["STK"]
    currency: Identifier
    exchange: Literal["SMART"]
    primary_exchange: Identifier
    con_id: int = Field(gt=0)
    time_zone_id: str = Field(max_length=64)
    trading_hours: str = Field(max_length=4000)
    liquid_hours: str = Field(max_length=4000)


class OrderIntent(StrictModel):
    """A persisted decision to trade; submission happens only inside its scheduled window."""

    intent_id: Identifier
    episode_id: Identifier
    instrument: Identifier
    side: Side
    purpose: Literal["entry", "exit"]
    quantity: int = Field(gt=0)
    limit_price: str
    decision_index: int = Field(ge=0)
    session_id: Identifier
    submit_at: AwareDatetime
    submit_deadline: AwareDatetime

    @model_validator(mode="after")
    def coherent(self) -> Self:
        decimal_text(self.limit_price, positive=True)
        if self.submit_deadline <= self.submit_at:
            raise ValueError("submission deadline must follow the scheduled submission")
        return self


class BrokerEvent(StrictModel):
    """One broker-side fact, keyed so duplicates after reconnect are idempotent."""

    kind: Literal["submitted", "status", "fill", "fee", "cancelled", "rejected", "notice"]
    intent_id: str | None
    order_id: int | None
    exec_id: str | None
    status: str | None
    quantity: int | None
    price: str | None
    amount: str | None
    at: AwareDatetime
    detail: str = Field(default="", max_length=2000)
