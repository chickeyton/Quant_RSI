"""Bounded related-price inputs for the connected study, without changing its sample."""

import hashlib
import re
from datetime import datetime
from urllib.error import URLError

import numpy as np

from quant_rsi.research.demo_prices import load_daily_prices

PEER_WINDOWS = (5, 10, 20)
PEER_OPERATORS = ("peer_return", "relative_return")
MAX_PEERS = 3


def aligned_closes(dates, cutoffs, peer_data, peer_info):
    """Match exact session dates; a later peer close is not known at the target close."""
    peer_times = peer_info["session_close_times"]
    if len(peer_times) != len(peer_data.dates) or len(cutoffs) != len(dates):
        raise ValueError("Related price timestamps must match their sessions")
    lookup = {}
    for index, (day, timestamp) in enumerate(zip(peer_data.dates, peer_times, strict=True)):
        closed = datetime.fromisoformat(timestamp)
        if closed.tzinfo is None:
            raise ValueError("Related price close times must have timezones")
        lookup[day] = (closed, peer_data.closes[index, 0])
    values = np.full(len(dates), np.nan)
    for index, (day, cutoff) in enumerate(zip(dates, cutoffs, strict=True)):
        decision = datetime.fromisoformat(cutoff)
        if decision.tzinfo is None:
            raise ValueError("Target close times must have timezones")
        if day in lookup and lookup[day][0] <= decision:
            values[index] = lookup[day][1]
    return values


def peer_feature(target, peer, op, window):
    """Trailing simple return, or target minus peer return; gaps invalidate the window."""
    if op not in PEER_OPERATORS or type(window) is not int or window not in PEER_WINDOWS:
        raise ValueError("Unsupported bounded related-price recipe")
    values = np.full(len(target), np.nan)
    for index in range(window, len(target)):
        own, other = target[index - window : index + 1], peer[index - window : index + 1]
        if not np.isfinite(other).all() or not (other > 0).all():
            continue
        value = other[-1] / other[0] - 1
        if op == "relative_return":
            if not np.isfinite(own).all() or not (own > 0).all():
                continue
            value = own[-1] / own[0] - 1 - value
        values[index] = value
    return values


def prepare_related_prices(
    data,
    cutoffs,
    required,
    related,
    *,
    mode,
    directory,
    enabled=True,
    simulated_suggestions=False,
):
    """Collect at most three symbols once; failures remain visible and never imputed."""
    candidates = [(symbol, "configured_context") for symbol in related.get("stocks", [])]
    candidates += [
        (item["symbol"], "unverified_suggestion") for item in related.get("research_candidates", [])
    ]
    support, definitions, columns, inputs = [], [], {}, {}
    seen = {data.instruments[0]}
    attempted = 0
    for symbol, source in candidates:
        if symbol in seen:
            continue
        seen.add(symbol)
        entry = {
            "symbol": symbol,
            "relationship_status": source,
            "status": "unavailable",
            "reason": "",
            "factor_ids": [],
        }
        support.append(entry)
        if not enabled:
            entry.update(status="context_only", reason="related_prices_disabled")
            continue
        if mode == "daily" and (
            related.get("provider") == "simulated_relationships"
            or (source == "unverified_suggestion" and simulated_suggestions)
        ):
            entry.update(status="context_only", reason="simulated_relationship_not_collected")
            continue
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,31}", symbol):
            entry["reason"] = "invalid_symbol"
            continue
        if attempted >= MAX_PEERS:
            entry.update(status="context_only", reason="related_price_collection_limit")
            continue
        attempted += 1
        try:
            if mode == "synthetic":
                seed = int.from_bytes(hashlib.sha256(symbol.encode()).digest()[:4], "big")
                rng = np.random.default_rng(seed)
                aligned = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(data.dates))))
                evidence = {
                    "provider": "synthetic_peer_prices",
                    "seed": seed,
                    "note": "Invented prices; not market evidence or measured relationships.",
                }
            else:
                peer_data, evidence = load_daily_prices(
                    symbol, directory / symbol, historical_sessions=len(data.dates)
                )
                if peer_data.instruments != (symbol,):
                    raise ValueError("Related collection symbol mismatch")
                aligned = aligned_closes(data.dates, cutoffs, peer_data, evidence)
            entry["provider"] = evidence["provider"]
        except (ValueError, OSError, URLError) as exc:
            entry.update(reason=str(exc), error_type=type(exc).__name__)
            continue
        inputs[symbol] = {
            "dates": list(data.dates),
            "target_close_cutoffs": list(cutoffs),
            "aligned_closes": [float(x) if np.isfinite(x) else None for x in aligned],
            "evidence": evidence,
        }
        for op in PEER_OPERATORS:
            for window in PEER_WINDOWS:
                identity = f"peer.{symbol}.{op}_{window}"
                values = peer_feature(data.closes[:, 0], aligned, op, window)
                runnable = bool(np.isfinite(values[required]).all())
                definition = {
                    "factor_id": identity,
                    "kind": "numeric",
                    "runnable": runnable,
                    "input_kind": "aligned_target_and_peer_closes",
                    "description": "Trailing peer simple return."
                    if op == "peer_return"
                    else "Target simple return minus peer simple return over the same sessions.",
                    "recipe": {"op": op, "symbol": symbol, "window": window},
                    "availability_reason": "complete_common_sample"
                    if runnable
                    else "missing_peer_window_on_fixed_sample",
                }
                definitions.append(definition)
                columns[identity] = values
                if runnable:
                    entry["factor_ids"].append(identity)
        entry.update(
            status="available" if entry["factor_ids"] else "unavailable",
            reason="complete_common_sample"
            if entry["factor_ids"]
            else "missing_peer_window_on_fixed_sample",
            aligned_sessions=int(np.isfinite(aligned).sum()),
        )
    return {"support": support, "definitions": definitions, "columns": columns, "inputs": inputs}
