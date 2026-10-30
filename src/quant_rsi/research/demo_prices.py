"""Reuse a retained public daily collection without creating a research campaign."""

import json
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import digest
from quant_rsi.core.data import MarketData
from quant_rsi.market.daily_data import HISTORY_SESSIONS, collect_daily
from quant_rsi.market.market_contracts import DataContract
from quant_rsi.market.market_data import parse_csv


def load_daily_prices(symbol, directory):
    root = Path(directory).resolve()
    if not root.exists() or not any(root.iterdir()):
        collect_daily(symbol, root)
    try:
        return _read_collection(symbol, root)
    except (KeyError, TypeError, IndexError):
        raise ValueError(
            "Incomplete daily collection; use a fresh --prices-dir to collect"
        ) from None


def _read_collection(symbol, root):
    receipt = json.loads((root / "collection_receipt.json").read_bytes())
    if (receipt.get("kind"), receipt.get("status"), receipt.get("ticker")) != (
        "PublicDailyCollection",
        "collected",
        symbol,
    ):
        raise ValueError("Daily collection must be successful and match the requested symbol")
    contract_raw = (root / "data_contract.json").read_bytes()
    events_raw = (root / "events.csv").read_bytes()
    if receipt["normalized_inputs"] != {
        "contract_file_sha256": digest(contract_raw),
        "events_file_sha256": digest(events_raw),
    }:
        raise ValueError("Daily collection input hashes differ from the retained receipt")
    contract = DataContract.model_validate_json(contract_raw)
    if contract.instruments != (symbol,) or receipt["historical_sessions"] != HISTORY_SESSIONS:
        raise ValueError("Daily collection universe or historical window differs")
    sessions = contract.calendar.sessions[:HISTORY_SESSIONS]
    if len(contract.calendar.sessions) < HISTORY_SESSIONS + 5:
        raise ValueError("Daily collection needs a published five-session future calendar")
    events = parse_csv(contract, events_raw, receipt["completed_at"])
    quotes = {}
    for event in events:
        if event.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"}:
            key = (event.kind, event.session_id)
            if event.revision or event.quality != "valid" or key in quotes:
                raise ValueError("Daily demo needs one valid unrevised quote per session and phase")
            quotes[key] = float(event.price)
        elif event.kind == "CORPORATE_ACTION_NOTICE" and event.event_at >= sessions[0].open_at:
            if json.loads(event.action).get("event_kind") != "dividends":
                raise ValueError("Unresolved corporate action in the daily price window")
    expected = {
        (kind, session.session_id) for session in sessions for kind in ("OPEN_QUOTE", "CLOSE_QUOTE")
    }
    if quotes.keys() != expected or receipt["origin_session_id"] != sessions[-1].session_id:
        raise ValueError("Daily collection has incomplete or unexpected price sessions")
    data = MarketData(
        tuple(session.session_id for session in sessions),
        (symbol,),
        np.array([[quotes["OPEN_QUOTE", session.session_id]] for session in sessions]),
        np.array([[quotes["CLOSE_QUOTE", session.session_id]] for session in sessions]),
    )
    return data, {
        "provider": "public_daily_prices",
        "collection_directory": str(root),
        "received_at": receipt["completed_at"],
        "source_id": contract.source_id,
        "input_hashes": receipt["normalized_inputs"],
        "price_basis": "raw",
        "currency": contract.currency,
        "qualifications": receipt["qualifications"],
        "latest_target_session": contract.calendar.sessions[HISTORY_SESSIONS - 1 + 5].session_id,
        "session_close_times": [session.close_at.isoformat() for session in sessions],
    }
