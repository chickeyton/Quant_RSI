"""Finite local-data and baseline-observer contracts; no implicit market defaults."""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, Field, TypeAdapter, field_validator, model_validator

from quant_rsi.core.contracts_base import StrictModel

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
KnowledgeBasis = Literal[
    "documented_historical_availability", "assumed_historical_availability", "observed_receipts"
]


@lru_cache(maxsize=4096)
def _timestamp_text(value: str) -> datetime:
    return timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))


def timestamp(value: str | datetime) -> datetime:
    # Cache only bounded immutable text, never live clocks, tzinfo objects or evidence.
    if type(value) is str and len(value) <= 128:
        return _timestamp_text(value)
    result = (
        datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    )
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("an explicit timezone is required")
    return result.astimezone(timezone.utc)


def utc_text(value: str | datetime) -> str:
    return timestamp(value).isoformat(timespec="microseconds")


def trusted_now() -> str:
    """Production callers cannot override this clock through configuration or CLI."""
    return utc_text(datetime.now(timezone.utc))


def decimal_text(value: str, *, positive=False) -> str:
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError("amounts must be bounded decimal strings")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("invalid decimal amount") from error
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError("amount must be finite and nonnegative (price/capital must be positive)")
    if number.adjusted() > 15 or number.as_tuple().exponent < -12:
        raise ValueError("unsupported decimal magnitude/precision")
    return format(number.normalize(), "f") if number else "0"


class MarketSession(StrictModel):
    session_id: Identifier
    open_at: AwareDatetime
    close_at: AwareDatetime

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.open_at >= self.close_at:
            raise ValueError("session open must precede close")
        return self


class CalendarSpec(StrictModel):
    calendar_id: Identifier
    timezone: str
    sessions: tuple[MarketSession, ...] = Field(min_length=3, max_length=10000)

    @model_validator(mode="after")
    def valid(self) -> Self:
        try:
            zone = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("unknown calendar timezone") from error
        if len({s.session_id for s in self.sessions}) != len(self.sessions):
            raise ValueError("duplicate calendar session")
        for index, session in enumerate(self.sessions):
            for instant in (session.open_at, session.close_at):
                if instant.utcoffset() != instant.astimezone(zone).utcoffset():
                    raise ValueError("session offset disagrees with the registered timezone")
            if index and self.sessions[index - 1].close_at >= session.open_at:
                raise ValueError("calendar sessions must be chronological and nonoverlapping")
        return self


CSV_FIELDS = {
    "event_id",
    "revision",
    "kind",
    "instrument",
    "session_id",
    "event_at",
    "source_available_at",
    "price",
    "status",
    "action",
    "quality",
}
CSV_REQUIRED = {"event_id", "revision", "kind", "instrument", "event_at", "quality"}


class DataContract(StrictModel):
    schema_version: Literal[1]
    source_id: Identifier
    provider_statement: str = Field(min_length=1, max_length=1000)
    currency: Identifier
    price_basis: Literal["raw"]
    frequency: Literal["daily_equities"]
    calendar: CalendarSpec
    instruments: tuple[Identifier, ...] = Field(min_length=1, max_length=200)
    knowledge_basis: KnowledgeBasis
    availability_evidence_ref: Hash | None
    availability_claim: Literal[
        "source_documented_not_independently_verified", "explicit_assumption", "local_receipt"
    ]
    historical_assumption: Literal["available_at_event_time"] | None
    universe_provenance: Literal[
        "source_documented_not_independently_verified", "unknown_exploration"
    ]
    corporate_action_coverage: Literal["source_declares_complete_notices", "unknown"]
    columns: dict[str, str]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if len(set(self.instruments)) != len(self.instruments):
            raise ValueError("instrument identities must be unique")
        if not CSV_REQUIRED <= set(self.columns) <= CSV_FIELDS:
            raise ValueError("unsupported or missing input field mapping")
        if len(set(self.columns.values())) != len(self.columns) or any(
            not x for x in self.columns.values()
        ):
            raise ValueError("column mappings must be nonempty and one-to-one")
        if self.knowledge_basis == "documented_historical_availability":
            if (
                self.availability_evidence_ref is None
                or self.availability_claim != "source_documented_not_independently_verified"
                or self.historical_assumption is not None
                or "source_available_at" not in self.columns
            ):
                raise ValueError("documented history needs frozen evidence and source availability")
        elif self.knowledge_basis == "assumed_historical_availability":
            if (
                self.availability_claim != "explicit_assumption"
                or self.historical_assumption != "available_at_event_time"
            ):
                raise ValueError("exploration requires an explicit frozen historical assumption")
        elif self.availability_claim != "local_receipt" or self.historical_assumption is not None:
            raise ValueError("observed receipts cannot borrow historical availability assumptions")
        return self


class SourceEvent(StrictModel):
    event_id: Identifier
    revision: int = Field(ge=0, le=1_000_000)
    kind: Literal["OPEN_QUOTE", "CLOSE_QUOTE", "UNIVERSE_STATUS", "CORPORATE_ACTION_NOTICE"]
    instrument: Identifier
    session_id: Identifier | None
    event_at: AwareDatetime
    source_available_at: AwareDatetime | None
    price: str | None
    status: Literal["active", "outside_universe", "suspended", "delisted"] | None
    action: str | None = Field(max_length=1000)
    quality: Literal["valid", "missing", "invalid"]

    @field_validator("price")
    @classmethod
    def price_decimal(cls, value):
        return None if value is None else decimal_text(value, positive=True)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"}:
            if self.session_id is None or self.status is not None or self.action is not None:
                raise ValueError("quote needs its session and cannot contain status/action")
            if (self.quality == "valid") != (self.price is not None):
                raise ValueError(
                    "valid quote needs a price; missing/invalid quote cannot supply one"
                )
            if self.source_available_at is not None and self.source_available_at < self.event_at:
                raise ValueError("quote source availability precedes its market event")
        elif self.kind == "UNIVERSE_STATUS":
            if (
                self.status is None
                or self.price is not None
                or self.action is not None
                or self.quality != "valid"
            ):
                raise ValueError("universe event requires one explicit valid status")
        elif (
            not self.action
            or self.price is not None
            or self.status is not None
            or self.quality != "valid"
        ):
            raise ValueError("action notice requires an explicit action description")
        return self


class SideCosts(StrictModel):
    commission_bps: str
    tax_bps: str
    slippage_bps: str

    @field_validator("commission_bps", "tax_bps", "slippage_bps")
    @classmethod
    def bounded(cls, value):
        result = decimal_text(value)
        if Decimal(result) > 1000:
            raise ValueError("cost above supported 1000 bps bound")
        return result


class RecorderPlan(StrictModel):
    schema_version: Literal[1]
    mode: Literal["registered_baseline_observer"]
    purpose: Literal["developer_visible_observation"]
    model_bundle_ref: Hash
    contract_ref: Hash
    max_information_age_seconds: int = Field(gt=0, le=315_360_000)
    max_model_age_seconds: int = Field(gt=0, le=315_360_000)
    expires_at: AwareDatetime


class PredictionOnlyPlan(RecorderPlan):
    account_mode: Literal["prediction_only"]


class AccountRules(StrictModel):
    currency: Identifier
    currency_precision: int = Field(ge=0, le=8)
    rounding_policy: Literal["half_even_per_component"]
    initial_cash: str
    cash_settlement: Literal["immediate_simulation"]
    inventory_unlock: Literal["immediate_simulation"]
    sizing: Literal["long_only_topk"]
    allocation: str
    max_positions: int = Field(ge=1, le=200)
    lot_size: int = Field(ge=1, le=1_000_000)
    cost_policy: Literal["linear_side_bps"]
    buy_costs: SideCosts
    sell_costs: SideCosts
    cost_claim: Literal["explicit_observation_assumption"]
    fill_policy: Literal["next_registered_open_market"]
    fill_cap_units: dict[Identifier, int]
    funding_policy: Literal["pre_funded_no_unfilled_sale_credit"]
    valuation: Literal["close_fresh_quote_only"]
    benchmark: Literal["cash_zero_interest"]
    action_policy: Literal["notice_suspend_unresolved"]
    termination: Literal["preserve_inventory", "queue_exit_at_next_eligible_open"]
    stop_policy: Literal["cancel_unelapsed_targets"]
    max_drawdown: str

    @field_validator("initial_cash")
    @classmethod
    def capital(cls, value):
        return decimal_text(value, positive=True)

    @field_validator("allocation", "max_drawdown")
    @classmethod
    def fraction(cls, value):
        result = decimal_text(value, positive=True)
        if Decimal(result) > 1:
            raise ValueError("allocation/risk fraction must not exceed one")
        return result

    @model_validator(mode="after")
    def explicit_caps_and_cash(self) -> Self:
        if not self.fill_cap_units or any(
            type(cap) is not int or cap <= 0 or cap % self.lot_size
            for cap in self.fill_cap_units.values()
        ):
            raise ValueError("each fill cap must be an explicit positive multiple of the lot")
        quantum = Decimal(1).scaleb(-self.currency_precision)
        if Decimal(self.initial_cash).quantize(quantum) != Decimal(self.initial_cash):
            raise ValueError("initial cash exceeds registered currency precision")
        return self


class ObservationPlan(RecorderPlan, AccountRules):
    account_mode: Literal["indicative_account"]


def parse_observation_plan(raw):
    return TypeAdapter(
        Annotated[PredictionOnlyPlan | ObservationPlan, Field(discriminator="account_mode")]
    ).validate_json(raw)
