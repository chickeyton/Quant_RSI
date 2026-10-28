"""Receipt-preserving development projections; origin hashes are not read capabilities."""

import csv
import io
import re
from copy import deepcopy

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.market.market_contracts import timestamp, trusted_now, utc_text
from quant_rsi.market.market_data import knowledge_time


def permitted(row, sessions, cutoff, basis):
    event = row["event"]
    known_notice = event["kind"] == "CORPORATE_ACTION_NOTICE"
    return (
        (known_notice or event["session_id"] in sessions or event["session_id"] is None)
        and (known_notice or timestamp(event["event_at"]) <= cutoff)
        and knowledge_time(row, basis) <= cutoff
    )


def development_projection(data, sessions, *, created_at=None):
    """The authority calls this with verified originals, never inside a developer worker."""
    from quant_rsi.evaluation.local_research import LocalResearchData

    if data.value["kind"] != "LocalResearchData" or not 24 <= sessions < data.spec.sessions:
        raise ValueError("projection requires original local data and a strict development prefix")
    allowed = set(data.dates[:sessions])
    cutoff = data.decision_at(sessions - 1)
    rows = [
        deepcopy(row)
        for row in data.snapshot["events"]
        if permitted(row, allowed, cutoff, data.contract.knowledge_basis)
    ]
    identities = {(row["event"]["event_id"], row["event"]["revision"]) for row in rows}
    raw_inputs, receipts, lineage_receipts = {}, [], []
    for original in data.snapshot["receipts"]:
        reader = csv.DictReader(io.StringIO(data.value["raw_inputs"][original["raw_ref"]]))
        selected = [
            row
            for row in reader
            if (
                row[data.contract.columns["event_id"]].strip(),
                int(row[data.contract.columns["revision"]]),
            )
            in identities
        ]
        if not selected:
            continue
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
        raw = buffer.getvalue()
        derived_ref = digest(raw.encode())
        raw_inputs[derived_ref] = raw
        receipt = dict(original, raw_ref=derived_ref, kind="DerivedReceipt")
        receipts.append(receipt)
        lineage_receipts.append(
            dict(
                seq=original["seq"],
                original_raw_ref=original["raw_ref"],
                original_receipt_ref=digest(canonical(original)),
                first_seen_at=original["first_seen_at"],
                derived_raw_ref=derived_ref,
            )
        )
    snapshot = dict(
        data.snapshot,
        kind="DerivedLocalSnapshot",
        events=rows,
        receipts=receipts,
    )
    specification = data.spec.model_dump(mode="json") | {
        "snapshot_ref": digest(canonical(snapshot)),
        "sessions": sessions,
    }
    value = dict(
        kind="DerivedLocalResearchData",
        specification=specification,
        contract=data.contract.model_dump(mode="json"),
        snapshot=snapshot,
        raw_inputs=raw_inputs,
        availability_evidence_hex=None,
        projection_lineage=dict(
            kind="LocalDevelopmentProjectionLineage",
            original_snapshot_ref=data.spec.snapshot_ref,
            original_contract_ref=data.snapshot["contract_ref"],
            availability_evidence_ref=data.contract.availability_evidence_ref,
            allowed_sessions=list(data.dates[:sessions]),
            allowed_instruments=list(data.instruments),
            information_cutoff=utc_text(cutoff),
            derived_created_at=created_at or trusted_now(),
            receipts=lineage_receipts,
            original_event_rows_ref=digest(canonical(rows)),
            qualification="derived_developer_visible_not_independent_evidence",
        ),
    )
    LocalResearchData(value)
    return value


def verify_projection(value, contract):
    """Check the derivative internally; actual origin equality is an authority responsibility."""
    lineage, snapshot = value["projection_lineage"], value["snapshot"]
    expected_keys = {
        "kind",
        "original_snapshot_ref",
        "original_contract_ref",
        "availability_evidence_ref",
        "allowed_sessions",
        "allowed_instruments",
        "information_cutoff",
        "receipts",
        "original_event_rows_ref",
        "qualification",
        "derived_created_at",
    }
    if (
        set(lineage) != expected_keys
        or lineage["kind"] != "LocalDevelopmentProjectionLineage"
        or lineage["qualification"] != "derived_developer_visible_not_independent_evidence"
        or snapshot["kind"] != "DerivedLocalSnapshot"
        or value["availability_evidence_hex"] is not None
        or lineage["original_contract_ref"] != snapshot["contract_ref"]
        or lineage["availability_evidence_ref"] != contract.availability_evidence_ref
        or lineage["allowed_sessions"]
        != [s.session_id for s in contract.calendar.sessions[: value["specification"]["sessions"]]]
        or lineage["allowed_instruments"] != list(contract.instruments)
        or lineage["original_event_rows_ref"] != digest(canonical(snapshot["events"]))
        or len(lineage["receipts"]) != len(snapshot["receipts"])
    ):
        raise ValueError("local projection identity/coverage mismatch")
    cutoff = timestamp(lineage["information_cutoff"])
    from datetime import timedelta

    if cutoff != timestamp(
        contract.calendar.sessions[value["specification"]["sessions"] - 1].close_at
    ) + timedelta(seconds=value["specification"]["decision_delay_seconds"]):
        raise ValueError("projection information cutoff differs from its registered grid")
    if any(
        timestamp(r["first_seen_at"]) > timestamp(lineage["derived_created_at"])
        for r in lineage["receipts"]
    ):
        raise ValueError("derived creation time precedes its actual original receipts")
    if not re.fullmatch(r"[0-9a-f]{64}", lineage["original_snapshot_ref"]):
        raise ValueError("invalid projection origin identity")
    if not all(
        permitted(row, set(lineage["allowed_sessions"]), cutoff, contract.knowledge_basis)
        for row in snapshot["events"]
    ):
        raise ValueError("projection includes a protected event or unavailable revision")
    for receipt, origin in zip(snapshot["receipts"], lineage["receipts"], strict=True):
        if set(origin) != {
            "seq",
            "original_raw_ref",
            "original_receipt_ref",
            "first_seen_at",
            "derived_raw_ref",
        }:
            raise ValueError("invalid projection receipt lineage")
        original = {key: origin[key] for key in ("seq", "first_seen_at")}
        original["raw_ref"] = origin["original_raw_ref"]
        if (
            not re.fullmatch(r"[0-9a-f]{64}", origin["original_raw_ref"])
            or digest(canonical(original)) != origin["original_receipt_ref"]
            or receipt
            != {
                "kind": "DerivedReceipt",
                "seq": origin["seq"],
                "raw_ref": origin["derived_raw_ref"],
                "first_seen_at": origin["first_seen_at"],
            }
        ):
            raise ValueError("projection changed original receipt/first-seen lineage")
