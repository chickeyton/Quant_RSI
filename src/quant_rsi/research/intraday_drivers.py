"""Daily study inputs summarised from a retained intraday snapshot's regular-hours bars."""

import math
from datetime import datetime

import numpy as np

from quant_rsi.intraday.storage import VALID, SnapshotStore

PROVIDER = "intraday_snapshot_summary"
NOTICE = (
    "Intraday summaries use only the session's own regular-hours decision bars, so they are "
    "known with its closing price; sessions with any non-valid bar stay missing."
)
METRICS = {
    "intraday_realized_vol": "square root of the summed squared decision-bar log returns, "
    "from the session open",
    "intraday_first_hour_return": "log return from the open to the close of the first hour",
    "intraday_last_hour_return": "log return over the last hour to the session close",
    "intraday_close_location": "(close - session low) / (session high - session low)",
    "intraday_close_vwap_gap": "close / volume-weighted typical price - 1",
    "intraday_last_hour_volume_share": "volume in the last hour / session volume",
}


def summarise_session(block, bar_seconds):
    """Return the metrics of one complete session, or None when any bar is not valid."""
    hour = 3600 // bar_seconds
    columns = block.columns
    if len(block) < 2 * hour or not (columns["status"] == VALID).all():
        return None
    names = ("open", "high", "low", "close", "volume")
    opens, highs, lows, closes, volume = (np.asarray(columns[n], dtype=float) for n in names)
    close, high, low, total = closes[-1], highs.max(), lows.min(), volume.sum()
    log_path = np.log(np.concatenate(([opens[0]], closes)))
    typical = (highs + lows + closes) / 3
    return {
        "intraday_realized_vol": float(math.sqrt(np.square(np.diff(log_path)).sum())),
        "intraday_first_hour_return": float(math.log(closes[hour - 1] / opens[0])),
        "intraday_last_hour_return": float(math.log(close / closes[-hour - 1])),
        "intraday_close_location": None if high == low else float((close - low) / (high - low)),
        "intraday_close_vwap_gap": (
            None if total <= 0 else float(close / ((typical * volume).sum() / total) - 1)
        ),
        "intraday_last_hour_volume_share": (
            None if total <= 0 else float(volume[-hour:].sum() / total)
        ),
    }


def prepare_intraday(snapshot, instrument, session_ids, cutoffs, required):
    """Align each study row with the same session's summary, closed by that row's cutoff."""
    if instrument not in {binding.instrument for binding in snapshot.contract.instruments}:
        raise ValueError(f"Intraday snapshot does not contain {instrument}")
    if len(session_ids) != len(cutoffs):
        raise ValueError("Intraday summaries need one cutoff per study row")
    sessions = {s.session_id: s for s in snapshot.contract.calendar.sessions}
    bar_seconds = snapshot.contract.grid.bar_seconds
    evidence = []
    for session_id, cutoff in zip(session_ids, cutoffs, strict=True):
        cutoff = datetime.fromisoformat(cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("Intraday factor cutoffs require timezones")
        session = sessions.get(session_id)
        block = snapshot.block(instrument, "decision", session_id)
        summary = None
        if session is not None and block is not None and session.close_at <= cutoff:
            summary = summarise_session(block, bar_seconds)
        evidence.append({"session_id": session_id, "summary": summary})
    columns, definitions = {}, []
    for metric, description in METRICS.items():
        identity = f"external.{metric}"
        values = [None if row["summary"] is None else row["summary"][metric] for row in evidence]
        missing = sum(values[int(i)] is None for i in required)
        columns[identity] = values
        definitions.append(
            {
                "factor_id": identity,
                "kind": "numeric",
                "input_kind": "session_intraday_bars",
                "description": f"{instrument} regular-hours {bar_seconds // 60}-minute bars: "
                f"{description}.",
                "recipe": {"op": "external", "metric": metric},
                "runnable": missing == 0,
                "missing_required_rows": missing,
                "source": {"provider": PROVIDER, "snapshot_sha256": snapshot.ref},
                "reason": "available" if missing == 0 else "missing_or_incomplete_session",
            }
        )
    return {
        "provider": PROVIDER,
        "definitions": definitions,
        "columns": columns,
        "evidence": evidence,
        "snapshot_sha256": snapshot.ref,
        "failures": [],
        "notice": NOTICE,
    }


def load_intraday(store, ref, instrument, session_ids, cutoffs, required):
    """Verify the retained snapshot before summarising it."""
    return prepare_intraday(
        SnapshotStore(store).load(ref), instrument, session_ids, cutoffs, required
    )


def combine(bundles):
    """One external bundle for the model; each provider keeps its own evidence."""
    if len(bundles) == 1:
        return bundles[0]
    return {
        "provider": "+".join(b["provider"] for b in bundles),
        "definitions": [d for b in bundles for d in b["definitions"]],
        "columns": {k: v for b in bundles for k, v in b["columns"].items()},
        "evidence": {b["provider"]: b.get("evidence") for b in bundles},
        "snapshot_sha256": {b["provider"]: b.get("snapshot_sha256") for b in bundles},
        "failures": [f for b in bundles for f in b["failures"]],
        "notice": " ".join(b["notice"] for b in bundles),
    }
