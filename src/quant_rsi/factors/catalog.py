"""Resolve explicit shared catalogs plus local definitions without collecting or fitting."""

from pathlib import Path

from pydantic import TypeAdapter

from quant_rsi.factors.contracts import FactorCatalog, InstrumentFactors, InstrumentId


def _read(root: Path, relative: Path) -> bytes:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("factor configuration escapes the instruments root")
    return path.read_bytes()


def resolve_instrument_factors(root: Path, market: str, instrument: str) -> dict:
    """Return detached definitions with their ownership and explicit source bindings.

    Catalogs cannot import other catalogs or single-stock definitions. Local definitions
    cannot shadow shared ones. No catalog is implicitly applied to an instrument.
    """
    adapter = TypeAdapter(InstrumentId)
    market, instrument = adapter.validate_python(market), adapter.validate_python(instrument)
    root = root.resolve()
    config = InstrumentFactors.model_validate_json(
        _read(root, Path(market) / instrument / "factors.json")
    )
    if (config.market, config.instrument) != (market, instrument):
        raise ValueError("instrument factor configuration does not match its directory")

    definitions = {}

    def add(items, scope, owner):
        for definition in items:
            if definition.factor_id in definitions:
                raise ValueError(f"duplicate or shadowed factor definition: {definition.factor_id}")
            definitions[definition.factor_id] = (definition, scope, owner)

    for name in config.catalogs:
        catalog = FactorCatalog.model_validate_json(_read(root, Path("shared") / f"{name}.json"))
        if catalog.catalog_id != name:
            raise ValueError("shared catalog ID does not match its filename")
        add(catalog.definitions, "shared", name)
    add(config.definitions, "instrument", f"{market}/{instrument}")

    factors = []
    for binding in config.bindings:
        if binding.definition_id not in definitions:
            raise ValueError(f"unknown factor definition: {binding.definition_id}")
        definition, scope, owner = definitions[binding.definition_id]
        factors.append(
            dict(
                binding_id=binding.binding_id,
                definition_scope=scope,
                definition_owner=owner,
                definition=definition.model_dump(mode="json"),
                source=binding.source.model_dump(mode="json"),
            )
        )
    return dict(
        schema_version=1,
        kind="resolved_instrument_factor_config",
        status="specification_only",
        runnable=False,
        market=market,
        instrument=instrument,
        factors=factors,
    )
