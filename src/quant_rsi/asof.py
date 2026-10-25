"""Causal feature/mature-label views and inference with no future-open requirement."""

import math

import numpy as np

from quant_rsi.artifacts import canonical, digest
from quant_rsi.market_contracts import timestamp, utc_text
from quant_rsi.market_data import knowledge_time, visible_revisions
from quant_rsi.schemas import FactorSpec


def session_index(contract, session_id):
    for index, session in enumerate(contract.calendar.sessions):
        if session.session_id == session_id:
            return index
    raise ValueError("decision session is absent from the registered calendar")


def qualifications(contract, basis=None):
    result = []
    if (basis or contract.knowledge_basis) != "observed_receipts":
        result.append("historical_exploration_not_observed_forward")
    if contract.universe_provenance == "unknown_exploration":
        result.append("historical_universe_not_PIT_verified")
    if contract.corporate_action_coverage == "unknown":
        result.append("corporate_action_coverage_unknown")
    return result


def make_feature_view(
    snapshot, contract, session_id, as_of, factor: FactorSpec, *, basis=None, watermark=None
):
    basis = basis or contract.knowledge_basis
    index = session_index(contract, session_id)
    session = contract.calendar.sessions[index]
    visible = visible_revisions(snapshot, contract, as_of, basis=basis, watermark=watermark)
    quotes = {
        (item["event"]["kind"], item["event"]["instrument"], item["event"]["session_id"]): item
        for item in visible
        if item["event"]["kind"] in {"OPEN_QUOTE", "CLOSE_QUOTE"}
    }
    rows = []
    for instrument in contract.instruments:
        reasons, source_refs, values = [], [], None
        membership = [
            item
            for item in visible
            if item["event"]["kind"] == "UNIVERSE_STATUS"
            and item["event"]["instrument"] == instrument
        ]
        if not membership:
            reasons.append("membership_unknown")
        else:
            effective = max(timestamp(item["event"]["event_at"]) for item in membership)
            latest = [
                item for item in membership if timestamp(item["event"]["event_at"]) == effective
            ]
            source_refs.extend(item["payload_ref"] for item in latest)
            statuses = {item["event"]["status"] for item in latest}
            if len(statuses) != 1:
                reasons.append("membership_conflict")
            elif statuses != {"active"}:
                reasons.append("membership_" + next(iter(statuses)))
            elif index >= factor.window:
                warmup_at = contract.calendar.sessions[index - factor.window].close_at
                prior = [
                    item for item in membership if timestamp(item["event"]["event_at"]) <= warmup_at
                ]
                if not prior:
                    reasons.append("insufficient_asset_warmup")
                else:
                    prior_at = max(timestamp(item["event"]["event_at"]) for item in prior)
                    if any(
                        item["event"]["status"] != "active"
                        for item in prior
                        if timestamp(item["event"]["event_at"]) == prior_at
                    ):
                        reasons.append("insufficient_asset_warmup")
        notices = [
            item
            for item in visible
            if item["event"]["kind"] == "CORPORATE_ACTION_NOTICE"
            and item["event"]["instrument"] == instrument
        ]
        if notices:
            reasons.append("unsupported_corporate_action_notice")
            source_refs.extend(item["payload_ref"] for item in notices)
        if timestamp(as_of) < session.close_at:
            reasons.append("decision_close_not_yet_available")
        if index < factor.window:
            reasons.append("insufficient_warmup")
        else:
            required = [
                ("CLOSE_QUOTE", instrument, contract.calendar.sessions[t].session_id)
                for t in range(index - factor.window, index + 1)
            ]
            required.append(("OPEN_QUOTE", instrument, session_id))
            inputs = [quotes.get(key) for key in required]
            if any(item is None or item["event"]["quality"] != "valid" for item in inputs):
                reasons.append("missing_or_invalid_asof_feature_quote")
            else:
                source_refs.extend(item["payload_ref"] for item in inputs)
                prices = [float(item["event"]["price"]) for item in inputs]
                close, opening = prices[:-1], prices[-1]
                daily = [close[j] / close[j - 1] - 1 for j in range(1, len(close))]
                factor_value = (
                    close[-1] / close[0] - 1 if factor.op == "momentum" else float(np.mean(daily))
                )
                values = [factor_value, daily[-1], close[-1] / opening - 1]
                if not all(math.isfinite(value) for value in values):
                    reasons.append("nonfinite_feature")
                    values = None
        unresolved = any(
            reason
            in {
                "membership_unknown",
                "membership_conflict",
                "missing_or_invalid_asof_feature_quote",
                "decision_close_not_yet_available",
                "unsupported_corporate_action_notice",
            }
            for reason in reasons
        )
        if (
            any(
                reason
                in {
                    "insufficient_asset_warmup",
                    "insufficient_warmup",
                    "membership_outside_universe",
                    "membership_suspended",
                    "membership_delisted",
                }
                for reason in reasons
            )
            and "unsupported_corporate_action_notice" not in reasons
        ):
            unresolved = False
        rows.append(
            {
                "instrument": instrument,
                "status": "eligible"
                if not reasons
                else ("unresolved" if unresolved else "ineligible"),
                "reasons": reasons,
                "features": values if not reasons else None,
                "source_refs": sorted(set(source_refs)),
            }
        )
    return {
        "kind": "AsOfFeatureView",
        "contract_ref": snapshot["contract_ref"],
        "snapshot_ref": digest(canonical(snapshot)),
        "knowledge_basis": basis,
        "receipt_watermark": snapshot["receipt_watermark"] if watermark is None else watermark,
        "session_id": session_id,
        "decision_at": utc_text(session.close_at),
        "as_of": utc_text(as_of),
        "factor": factor.model_dump(mode="json"),
        "expected_instruments": list(contract.instruments),
        "rows": rows,
        "qualifications": qualifications(contract, basis),
    }


def mature_label(snapshot, contract, session_id, instrument, training_cutoff, *, basis=None):
    index = session_index(contract, session_id)
    if instrument not in contract.instruments:
        raise ValueError("label instrument is outside the registered identity grid")
    if index + 2 >= len(contract.calendar.sessions):
        return {"status": "immature", "reason": "future_label_sessions_absent", "value": None}
    basis = basis or contract.knowledge_basis
    visible = visible_revisions(snapshot, contract, training_cutoff, basis=basis)
    label_end = contract.calendar.sessions[index + 2].open_at
    if any(
        item["event"]["kind"] == "CORPORATE_ACTION_NOTICE"
        and item["event"]["instrument"] == instrument
        and timestamp(item["event"]["event_at"]) <= label_end
        for item in visible
    ):
        return {
            "status": "ineligible",
            "reason": "unsupported_action_in_label_history",
            "value": None,
        }
    sessions = {contract.calendar.sessions[index + j].session_id for j in (1, 2)}
    rows = {
        item["event"]["session_id"]: item
        for item in visible
        if item["event"]["kind"] == "OPEN_QUOTE"
        and item["event"]["instrument"] == instrument
        and item["event"]["session_id"] in sessions
    }
    if set(rows) != sessions or any(item["event"]["quality"] != "valid" for item in rows.values()):
        return {"status": "immature", "reason": "label_quotes_not_mature_and_known", "value": None}
    first, second = [rows[contract.calendar.sessions[index + j].session_id] for j in (1, 2)]
    value = float(second["event"]["price"]) / float(first["event"]["price"]) - 1
    if not math.isfinite(value):
        raise ValueError("nonfinite mature label")
    return {
        "status": "mature",
        "reason": None,
        "value": value,
        "source_refs": [first["payload_ref"], second["payload_ref"]],
        "available_at": utc_text(max(knowledge_time(item, basis) for item in (first, second))),
        "interval": [first["event"]["event_at"], second["event"]["event_at"]],
    }


def predict_view(model_text, view):
    """Inference-only primitive. Every controller-bound key remains in the output."""
    import lightgbm as lgb

    if [row["instrument"] for row in view["rows"]] != view["expected_instruments"]:
        raise ValueError("inference view does not cover the controller-bound instrument grid")
    model = lgb.Booster(model_str=model_text)
    if model.num_feature() != 3:
        raise ValueError("model preprocessing interface mismatch")
    rows = [row | {"score": None} for row in view["rows"]]
    eligible = [index for index, row in enumerate(rows) if row["status"] == "eligible"]
    if eligible:
        values = np.asarray([rows[index]["features"] for index in eligible], dtype=float)
        if values.shape != (len(eligible), 3) or not np.isfinite(values).all():
            raise ValueError("invalid eligible inference features")
        scores = model.predict(values, num_threads=1)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite inference output")
        for index, score in zip(eligible, scores, strict=True):
            rows[index]["score"] = float(score)
    return {
        "kind": "AsOfPrediction",
        "feature_view_ref": digest(canonical(view)),
        "model_ref": digest(model_text.encode()),
        "session_id": view["session_id"],
        "as_of": view["as_of"],
        "knowledge_basis": view["knowledge_basis"],
        "receipt_watermark": view["receipt_watermark"],
        "rows": rows,
        "qualifications": view["qualifications"],
    }
