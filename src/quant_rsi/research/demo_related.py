"""Small, source-backed business relationship profiles for the research demo."""

from datetime import date
from pathlib import Path

from pydantic import Field, HttpUrl, TypeAdapter, model_validator

from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.factors.contracts import InstrumentId


class Relationship(StrictModel):
    symbol: InstrumentId
    relationship: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    source_urls: list[HttpUrl] = Field(min_length=1, max_length=3)


class RelationshipProfile(StrictModel):
    instrument: InstrumentId
    reviewed_on: date
    relationships: list[Relationship] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def distinct_related_stocks(self):
        symbols = [item.symbol for item in self.relationships]
        if self.instrument in symbols or len(symbols) != len(set(symbols)):
            raise ValueError("Related stocks must be distinct and exclude the target")
        return self


def profile_related(symbol):
    """Read checked-in relationships; no network discovery or return correlation."""
    TypeAdapter(InstrumentId).validate_python(symbol)
    path = (
        Path(__file__).resolve().parents[3] / "instruments" / "US" / symbol / "relationships.json"
    )
    try:
        profile = RelationshipProfile.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        raise ValueError(
            f"No relationship profile for {symbol}; currently configured: KO"
        ) from None
    if profile.instrument != symbol:
        raise ValueError("Relationship profile binding differs from selected instrument")
    return {
        "provider": "instrument_relationship_profile",
        "stocks": [item.symbol for item in profile.relationships],
        "reviewed_on": profile.reviewed_on.isoformat(),
        "relationships": [item.model_dump(mode="json") for item in profile.relationships],
        "selection": "configured_not_discovered",
        "reason": "Source-backed business relationships, not measured return correlations. "
        "Sources are reviewed configuration, not reverified during this run.",
    }
