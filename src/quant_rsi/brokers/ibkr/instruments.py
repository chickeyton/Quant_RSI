"""Contract qualification: an instrument profile must resolve to exactly one contract ID."""

from quant_rsi.brokers.contracts import InstrumentBinding, InstrumentSpec
from quant_rsi.brokers.ibkr.connection import IbkrSession


class AmbiguousInstrument(ValueError):
    pass


def contract_params(spec: InstrumentSpec | InstrumentBinding) -> dict:
    params = {
        "symbol": spec.symbol,
        "sec_type": spec.sec_type,
        "currency": spec.currency,
        "exchange": spec.exchange,
        "primary_exchange": spec.primary_exchange,
    }
    if isinstance(spec, InstrumentBinding):
        params["con_id"] = spec.con_id
    return params


def resolve(session: IbkrSession, spec: InstrumentSpec) -> InstrumentBinding:
    messages = session.call(
        "contract_details",
        {"contract": contract_params(spec)},
        end_kinds={"contract_details_end"},
        collect_kinds={"contract_details"},
    )
    details = [m.payload for m in messages if m.kind == "contract_details"]
    matches = [
        d
        for d in details
        if d["symbol"] == spec.symbol
        and d["currency"] == spec.currency
        and d["primary_exchange"] == spec.primary_exchange
        and d["sec_type"] == spec.sec_type
    ]
    if len({int(d["con_id"]) for d in matches}) != 1:
        raise AmbiguousInstrument(
            f"{spec.instrument}: expected exactly one contract, received {len(matches)}"
        )
    found = matches[0]
    return InstrumentBinding(
        instrument=spec.instrument,
        symbol=spec.symbol,
        sec_type=spec.sec_type,
        currency=spec.currency,
        exchange=spec.exchange,
        primary_exchange=spec.primary_exchange,
        con_id=int(found["con_id"]),
        time_zone_id=str(found["time_zone_id"]),
        trading_hours=str(found["trading_hours"]),
        liquid_hours=str(found["liquid_hours"]),
    )
