"""Pure target-only projection of a complete frozen context preparation.

Collection, original registry provenance, source freshness, authority and actual
export parity remain caller responsibilities. No history is truncated: the full
original envelope, including unused contexts/signals, remains bound as evidence.
"""

from typing import Literal

from quant_rsi.artifacts import canonical, decode, digest
from quant_rsi.contracts_base import StrictModel
from quant_rsi.forecast_inputs import ForecastInputs
from quant_rsi.local_research import LocalResearchData
from quant_rsi.market_contracts import Hash


class ForecastRecursiveMarketEvidence(StrictModel):
    kind: Literal["ForecastRecursiveMarketEvidence"]
    schema_version: Literal[1]
    inputs_ref: Hash
    preparation_receipt_ref: Hash
    target_data_ref: Hash
    projection: Literal["target_only_v1"]


def _history_grid(data):
    calendar = [session.session_id for session in data.contract.calendar.sessions]
    phases = {"OPEN_QUOTE": set(), "CLOSE_QUOTE": set()}
    for row in data.snapshot["events"]:
        event = row["event"]
        if event["kind"] in phases:
            phases[event["kind"]].add(event["session_id"])
    recorded = phases["CLOSE_QUOTE"]
    count = len(recorded)
    if (
        not recorded
        or phases["OPEN_QUOTE"] != recorded
        or set(calendar[:count]) != recorded
        or len(calendar) != count + 5
        or data.spec.sessions != count
    ):
        raise ValueError(
            "target history must match its recorded open/close session prefix "
            "and have exactly five unquoted future calendar sessions"
        )
    return count, calendar[:count], calendar[count:]


def _verify_import(role, data, item):
    """Compare existing receipt fields; never open or reconstruct a registry."""
    receipt = item.get("receipt") if isinstance(item, dict) else None
    required = {
        "source_id",
        "registry_id",
        "snapshot_ref",
        "receipt_seq",
        "raw_ref",
        "first_seen_at",
    }
    if (
        not isinstance(receipt, dict)
        or not required <= set(receipt)
        or type(receipt["receipt_seq"]) is not int
    ):
        raise ValueError(f"{role} preparation import receipt is incomplete")
    matches = [
        row
        for row in data.snapshot["receipts"]
        if row["seq"] == receipt["receipt_seq"]
        and row["raw_ref"] == receipt["raw_ref"]
        and row["first_seen_at"] == receipt["first_seen_at"]
    ]
    if (
        receipt["source_id"] != data.contract.source_id
        or receipt["registry_id"] != data.snapshot["registry_id"]
        or receipt["snapshot_ref"] != data.spec.snapshot_ref
        or receipt["receipt_seq"] != data.snapshot["receipt_watermark"]
        or len(matches) != 1
    ):
        raise ValueError(f"{role} preparation import differs from frozen input receipt")


def project_market_target(inputs: ForecastInputs | dict, preparation_receipt: bytes) -> dict:
    """Derive cutoff N-35 and seed 0, changing only those specification fields.

    N is independently checked against recorded target quote sessions, not the
    full calendar length. Small validated N is useful for offline verification;
    the caller's registered production profile must enforce its chosen N (756
    for the initial profile). This function grants no execution allowance.

    The preparation receipt is hashed byte-for-byte. Its decoded fields bind the
    already-frozen inputs; this is not a claim that the receipt's original
    registry or HTTP provenance has been authenticated by this pure function.
    """
    if type(preparation_receipt) is not bytes:
        raise TypeError("preparation receipt must be its original bytes")
    original = ForecastInputs(inputs.value if isinstance(inputs, ForecastInputs) else inputs)
    receipt = decode(preparation_receipt)
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("kind") != "forecast_context_preparation_receipt"
        or receipt.get("status") != "complete"
        or receipt.get("inputs_ref") != original.ref
        or set(original.contexts) != {"industry", "market"}
        or not isinstance(receipt.get("imports"), dict)
        or set(receipt["imports"]) != {"target", "industry", "market"}
        or receipt.get("audit_manifest_ref") not in original.evidence
    ):
        raise ValueError("complete original context preparation and evidence are required")
    target = original.target[0]
    count, history, future = _history_grid(target)
    if (
        type(receipt.get("historical_sessions")) is not int
        or receipt["historical_sessions"] != count
        or receipt.get("origin_session_id") != history[-1]
        or receipt.get("future_sessions") != future
    ):
        raise ValueError("preparation history/origin/future calendar binding mismatch")
    for role, (data, _) in {"target": original.target, **original.contexts}.items():
        if data.value["kind"] != "LocalResearchData" or data.spec.sessions != count:
            raise ValueError("original preparation must retain each full source history")
        _verify_import(role, data, receipt["imports"][role])

    derived = decode(canonical(target.to_json()))
    derived["specification"].update(train_cutoff=count - 35, seed=0)
    # Reuse the ordinary data validation, including raw receipt reconstruction.
    # No contract, snapshot, action, evidence or first-seen clock is reconstructed.
    LocalResearchData.from_json(derived)
    target_ref = digest(canonical(derived))
    receipt_ref = digest(preparation_receipt)
    market_evidence = ForecastRecursiveMarketEvidence(
        kind="ForecastRecursiveMarketEvidence",
        schema_version=1,
        inputs_ref=original.ref,
        preparation_receipt_ref=receipt_ref,
        target_data_ref=target_ref,
        projection="target_only_v1",
    ).model_dump(mode="json")
    return {
        "inputs": original.value,
        "inputs_ref": original.ref,
        "preparation_receipt": preparation_receipt,
        "preparation_receipt_ref": receipt_ref,
        "target_data": derived,
        "target_data_ref": target_ref,
        "market_evidence": market_evidence,
        "market_evidence_ref": digest(canonical(market_evidence)),
        "quoted_sessions": count,
        "training_boundary": count - 35,
        "development_origins": list(range(count - 35, count - 5, 5)),
    }
