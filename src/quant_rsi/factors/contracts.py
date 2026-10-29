"""Factor scope and input type are independent of the existing worker recipe."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.core.schemas import FactorSpec
from quant_rsi.factors.price_adapters import CandidatePriceRecipe

Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
FactorId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$", max_length=160)]
InstrumentId = Annotated[str, Field(pattern=r"^[A-Z0-9][A-Z0-9._-]{0,31}$")]


class FactorDefinition(StrictModel):
    """Reusable meaning and processing recipe, without an instrument or source binding."""

    factor_id: FactorId
    description: str = Field(min_length=1)


class NumericFactorDefinition(FactorDefinition):
    kind: Literal["numeric"]
    input_kind: Literal["positive_price_series"]
    recipe: FactorSpec | CandidatePriceRecipe


class TextFactorDefinition(FactorDefinition):
    kind: Literal["text"]
    extraction_instruction: str = Field(min_length=1)
    output_fields: tuple[Name, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_fields(self) -> Self:
        if len(set(self.output_fields)) != len(self.output_fields):
            raise ValueError("text output fields must be unique")
        return self


Definition = Annotated[NumericFactorDefinition | TextFactorDefinition, Field(discriminator="kind")]


class FactorCatalog(StrictModel):
    schema_version: Literal[1]
    kind: Literal["shared_factor_catalog"]
    catalog_id: Name
    definitions: tuple[Definition, ...] = Field(min_length=1)


class SourceBinding(StrictModel):
    """An explicit proposed input, not a registered snapshot or proof of availability."""

    source_id: Name
    entity: str = Field(min_length=1)
    field: Name


class FactorBinding(StrictModel):
    binding_id: Name
    definition_id: FactorId
    source: SourceBinding


class InstrumentFactors(StrictModel):
    schema_version: Literal[1]
    kind: Literal["instrument_factor_config"]
    market: InstrumentId
    instrument: InstrumentId
    catalogs: tuple[Name, ...]
    definitions: tuple[Definition, ...]
    bindings: tuple[FactorBinding, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_references(self) -> Self:
        if len(set(self.catalogs)) != len(self.catalogs):
            raise ValueError("catalog references must be unique")
        ids = [binding.binding_id for binding in self.bindings]
        if len(set(ids)) != len(ids):
            raise ValueError("binding IDs must be unique within an instrument")
        return self
