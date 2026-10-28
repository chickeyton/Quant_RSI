"""Immutable local event imports and the two distinct historical/receipt knowledge axes."""

import csv
import io
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.evaluation.evidence import authority_root
from quant_rsi.market import market_contracts as contracts
from quant_rsi.market.market_contracts import DataContract, SourceEvent, timestamp, utc_text


def market_root():
    return authority_root() / "market"


SCHEMA = (
    "CREATE TABLE metadata (anchor_ref TEXT NOT NULL)",
    "CREATE TABLE sources (source_id TEXT PRIMARY KEY, contract_ref TEXT NOT NULL UNIQUE)",
    "CREATE TABLE receipts (seq INTEGER PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, "
    "binding_ref TEXT NOT NULL, source_id TEXT NOT NULL, raw_ref TEXT NOT NULL, "
    "first_seen_at TEXT NOT NULL, snapshot_ref TEXT NOT NULL, fresh_events INTEGER NOT NULL)",
    "CREATE TABLE revisions (source_id TEXT NOT NULL, event_id TEXT NOT NULL, "
    "revision INTEGER NOT NULL, identity_ref TEXT NOT NULL, payload_ref TEXT NOT NULL, "
    "first_seen_at TEXT NOT NULL, receipt_seq INTEGER NOT NULL, "
    "PRIMARY KEY(source_id,event_id,revision), UNIQUE(source_id,identity_ref,revision))",
    "CREATE TABLE bundles (bundle_ref TEXT PRIMARY KEY, source_run_ref TEXT NOT NULL, "
    "checkpoint_ref TEXT NOT NULL, registered_at TEXT NOT NULL)",
)


def finish_bootstrap(root):
    anchor_path, marker = root / "identity.json", root / "bootstrap.json"
    if anchor_path.exists():
        if marker.exists():
            saved = decode(marker.read_bytes())
            if decode(anchor_path.read_bytes()) != saved:
                raise ValueError("market bootstrap/anchor conflict")
            with DataRegistry(root):
                marker.unlink()
        return
    if marker.exists():
        anchor = decode(marker.read_bytes())
    else:
        if any(
            p.name != ".controller.lock" and not p.name.startswith(".pending-")
            for p in root.iterdir()
        ):
            raise ValueError("market identity missing; refusing a new first-seen history")
        anchor = {"kind": "LocalDataRegistry", "version": 1, "registry_id": uuid.uuid4().hex}
        atomic_write(marker, canonical(anchor))
    database = root / "data.sqlite"
    connection = sqlite3.connect(database)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if not tables:
                for statement in SCHEMA:
                    connection.execute(statement)
                connection.execute("INSERT INTO metadata VALUES (?)", (digest(canonical(anchor)),))
            elif tables != {
                "metadata",
                "sources",
                "receipts",
                "revisions",
                "bundles",
            } or connection.execute("SELECT anchor_ref FROM metadata").fetchall() != [
                (digest(canonical(anchor)),)
            ]:
                raise ValueError("market bootstrap ledger conflict")
    finally:
        connection.close()
    atomic_write(anchor_path, canonical(anchor))
    marker.unlink()
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def market_context(*, initialize=False):
    root = market_root()
    if initialize:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    elif not root.is_dir():
        raise ValueError("local data registry is not initialized")
    with controller.controller_lock(root) as owner_fd:
        if initialize:
            finish_bootstrap(root)
        with DataRegistry(root) as registry:
            yield registry, owner_fd


class DataRegistry:
    def __init__(self, root):
        self.root = root
        self.store = ArtifactStore(root / "artifacts")
        self.anchor = decode((root / "identity.json").read_bytes())
        self.connection = sqlite3.connect(f"{(root / 'data.sqlite').as_uri()}?mode=rw", uri=True)
        self.connection.row_factory = sqlite3.Row
        try:
            rows = self.connection.execute("SELECT anchor_ref FROM metadata").fetchall()
            if len(rows) != 1 or rows[0][0] != digest(canonical(self.anchor)):
                raise ValueError("market identity/ledger mismatch")
        except Exception:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def contract(self, contract_ref):
        contract = DataContract.model_validate_json(self.store.get(contract_ref))
        row = self.connection.execute(
            "SELECT contract_ref FROM sources WHERE source_id=?", (contract.source_id,)
        ).fetchone()
        if row is None or row[0] != contract_ref:
            raise ValueError("data contract is not registered")
        if contract.availability_evidence_ref:
            self.store.get(contract.availability_evidence_ref)
        return contract

    def snapshot_document(self, source_id, contract_ref, watermark):
        events = []
        for row in self.connection.execute(
            "SELECT * FROM revisions WHERE source_id=? AND receipt_seq<=? "
            "ORDER BY event_id,revision",
            (source_id, watermark),
        ):
            payload = self.store.json(row["payload_ref"])
            events.append(
                {
                    "payload_ref": row["payload_ref"],
                    "event": payload,
                    "first_seen_at": row["first_seen_at"],
                    "receipt_seq": row["receipt_seq"],
                }
            )
        receipts = [
            {key: row[key] for key in ("seq", "raw_ref", "first_seen_at")}
            for row in self.connection.execute(
                "SELECT seq,raw_ref,first_seen_at FROM receipts WHERE source_id=? AND seq<=? "
                "ORDER BY seq",
                (source_id, watermark),
            )
        ]
        return {
            "kind": "LocalMarketSnapshot",
            "schema_version": 1,
            "registry_id": self.anchor["registry_id"],
            "source_id": source_id,
            "contract_ref": contract_ref,
            "receipt_watermark": watermark,
            "events": events,
            "receipts": receipts,
        }

    def snapshot(self, snapshot_ref):
        value = self.store.json(snapshot_ref)
        row = self.connection.execute(
            "SELECT * FROM receipts WHERE snapshot_ref=?", (snapshot_ref,)
        ).fetchone()
        if row is None or value["registry_id"] != self.anchor["registry_id"]:
            raise ValueError("snapshot is not a committed registry receipt")
        contract = self.contract(value["contract_ref"])
        expected = self.snapshot_document(contract.source_id, value["contract_ref"], row["seq"])
        if value != expected:
            raise ValueError("snapshot differs from its immutable source revisions/receipts")
        for item in value["events"]:
            event = SourceEvent.model_validate_json(canonical(item["event"]))
            validate_event(contract, event, item["first_seen_at"])
        return value, contract

    def latest_snapshot(self, contract_ref):
        contract = self.contract(contract_ref)
        row = self.connection.execute(
            "SELECT snapshot_ref FROM receipts WHERE source_id=? ORDER BY seq DESC LIMIT 1",
            (contract.source_id,),
        ).fetchone()
        if row is None:
            raise ValueError("source has no committed input receipt")
        return row[0]


def validate_event(contract, event, received_at):
    if event.instrument not in contract.instruments:
        raise ValueError("event instrument lies outside the registered identity grid")
    sessions = {session.session_id: session for session in contract.calendar.sessions}
    if event.kind in {"OPEN_QUOTE", "CLOSE_QUOTE"}:
        if event.session_id not in sessions:
            raise ValueError("quote session is absent from the registered calendar")
        expected = (
            sessions[event.session_id].open_at
            if event.kind == "OPEN_QUOTE"
            else sessions[event.session_id].close_at
        )
        if event.event_at != expected:
            raise ValueError("quote event time differs from the registered session phase")
        if event.event_at > timestamp(received_at):
            raise ValueError("future market prices cannot be imported as received observations")
    elif event.session_id is not None and event.session_id not in sessions:
        raise ValueError("status/action references an unknown calendar session")
    if (
        contract.knowledge_basis == "documented_historical_availability"
        and event.source_available_at is None
    ):
        raise ValueError("documented history requires source availability on every revision")


def parse_csv(contract, raw, received_at):
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("local import exceeds the supported 64 MiB batch limit")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if (
        reader.fieldnames is None
        or len(set(reader.fieldnames)) != len(reader.fieldnames)
        or set(reader.fieldnames) != set(contract.columns.values())
    ):
        raise ValueError("CSV headers must exactly match the explicit one-to-one mapping")
    events, seen = [], set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row has a missing or extra column")
        value = {name: (row[column].strip() or None) for name, column in contract.columns.items()}
        revision = value.get("revision")
        if revision is None or not re.fullmatch(r"[0-9]+", revision):
            raise ValueError("revision must be an explicit nonnegative integer")
        value["revision"] = int(revision)
        for field in ("session_id", "source_available_at", "price", "status", "action"):
            value.setdefault(field, None)
        event = SourceEvent.model_validate_json(canonical(value))
        validate_event(contract, event, received_at)
        key = (event.event_id, event.revision)
        if key in seen:
            raise ValueError("duplicate event revision within an import")
        seen.add(key)
        events.append(event)
    if not events:
        raise ValueError("input contains no source events")
    return events


def economic_identity(event):
    return {key: event[key] for key in ("kind", "instrument", "session_id", "event_at")}


def ingest(
    contract: DataContract, input_path: Path, *, request_id: str, evidence_file: Path | None = None
):
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("an import request ID of 1..128 characters is required")
    raw = input_path.read_bytes()
    body = contract.model_dump(mode="json")
    contract_ref, raw_ref = digest(canonical(body)), digest(raw)
    binding_ref = digest(canonical({"contract_ref": contract_ref, "raw_ref": raw_ref}))
    with market_context(initialize=True) as (registry, _):
        store, connection = registry.store, registry.connection
        previous = connection.execute(
            "SELECT * FROM receipts WHERE request_id=?", (request_id,)
        ).fetchone()
        if previous is not None:
            if previous["binding_ref"] != binding_ref:
                raise ValueError("import request ID belongs to different bytes/contract")
            registry.snapshot(previous["snapshot_ref"])
            return receipt_report(dict(previous), registry.anchor["registry_id"])
        received_at = contracts.trusted_now()
        latest = connection.execute(
            "SELECT first_seen_at FROM receipts ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        if latest is not None and timestamp(received_at) < timestamp(latest[0]):
            raise ValueError("trusted receipt clock moved backwards; import was not committed")
        events = parse_csv(contract, raw, received_at)
        if contract.availability_evidence_ref:
            if evidence_file is not None:
                supplied = evidence_file.read_bytes()
                if digest(supplied) != contract.availability_evidence_ref:
                    raise ValueError("historical availability evidence hash mismatch")
                store.put(supplied)
            store.get(contract.availability_evidence_ref)
        store.put(raw)
        store.put_json(body)
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            source = connection.execute(
                "SELECT contract_ref FROM sources WHERE source_id=?", (contract.source_id,)
            ).fetchone()
            if source is not None and source[0] != contract_ref:
                raise ValueError(
                    "source contract is immutable; "
                    "a new contract needs an explicit new source design"
                )
            connection.execute(
                "INSERT OR IGNORE INTO sources VALUES (?,?)", (contract.source_id, contract_ref)
            )
            sequence = connection.execute("SELECT COALESCE(MAX(seq),0)+1 FROM receipts").fetchone()[
                0
            ]
            fresh = 0
            for parsed in events:
                payload = parsed.model_dump(mode="json")
                payload["event_at"] = utc_text(parsed.event_at)
                payload["source_available_at"] = (
                    None
                    if parsed.source_available_at is None
                    else utc_text(parsed.source_available_at)
                )
                payload_ref = store.put_json(payload)
                identity_ref = digest(canonical(economic_identity(payload)))
                prior = connection.execute(
                    "SELECT * FROM revisions WHERE source_id=? AND (event_id=? OR identity_ref=?)",
                    (contract.source_id, parsed.event_id, identity_ref),
                ).fetchall()
                duplicate = False
                for existing in prior:
                    if (
                        existing["event_id"] != parsed.event_id
                        or existing["identity_ref"] != identity_ref
                    ):
                        raise ValueError(
                            "event ID/economic identity conflict; cannot relabel first-seen history"
                        )
                    if existing["revision"] == parsed.revision:
                        if existing["payload_ref"] != payload_ref:
                            raise ValueError("same event/revision has different content")
                        duplicate = True
                    previous_payload = store.json(existing["payload_ref"])
                    previous_source_time = previous_payload["source_available_at"]
                    if previous_source_time is not None and parsed.source_available_at is not None:
                        difference = (
                            parsed.source_available_at - timestamp(previous_source_time)
                        ).total_seconds()
                        if difference * (parsed.revision - existing["revision"]) < 0:
                            raise ValueError("source availability contradicts revision order")
                if not duplicate:
                    connection.execute(
                        "INSERT INTO revisions VALUES (?,?,?,?,?,?,?)",
                        (
                            contract.source_id,
                            parsed.event_id,
                            parsed.revision,
                            identity_ref,
                            payload_ref,
                            received_at,
                            sequence,
                        ),
                    )
                    fresh += 1
            connection.execute(
                "INSERT INTO receipts VALUES (?,?,?,?,?,?,?,?)",
                (
                    sequence,
                    request_id,
                    binding_ref,
                    contract.source_id,
                    raw_ref,
                    received_at,
                    "pending",
                    fresh,
                ),
            )
            snapshot_ref = store.put_json(
                registry.snapshot_document(contract.source_id, contract_ref, sequence)
            )
            connection.execute(
                "UPDATE receipts SET snapshot_ref=? WHERE seq=?", (snapshot_ref, sequence)
            )
        row = dict(connection.execute("SELECT * FROM receipts WHERE seq=?", (sequence,)).fetchone())
        return receipt_report(row, registry.anchor["registry_id"])


def receipt_report(row, registry_id):
    return {
        "registry_id": registry_id,
        "receipt_seq": row["seq"],
        "request_id": row["request_id"],
        "source_id": row["source_id"],
        "snapshot_ref": row["snapshot_ref"],
        "raw_ref": row["raw_ref"],
        "first_seen_at": row["first_seen_at"],
        "fresh_events": row["fresh_events"],
        "purpose": "developer_visible_observation",
        "forward_qualification": "requires_preexisting_registered_episode_and_actual_receipts",
    }


def load_snapshot(snapshot_ref):
    with market_context() as (registry, _):
        return registry.snapshot(snapshot_ref)


def inspect_data(snapshot_ref, *, replay=False):
    with market_context() as (registry, _):
        snapshot, contract = registry.snapshot(snapshot_ref)
        if replay:
            for receipt in snapshot["receipts"]:
                events = parse_csv(
                    contract, registry.store.get(receipt["raw_ref"]), receipt["first_seen_at"]
                )
                expected = {
                    (item["event"]["event_id"], item["event"]["revision"]): item
                    for item in snapshot["events"]
                }
                for event in events:
                    payload = event.model_dump(mode="json")
                    payload["event_at"] = utc_text(event.event_at)
                    payload["source_available_at"] = (
                        None
                        if event.source_available_at is None
                        else utc_text(event.source_available_at)
                    )
                    saved = expected[(event.event_id, event.revision)]
                    if payload != saved["event"] or timestamp(saved["first_seen_at"]) > timestamp(
                        receipt["first_seen_at"]
                    ):
                        raise ValueError("raw import replay differs from committed source revision")
        return {
            "snapshot_ref": snapshot_ref,
            "registry_id": snapshot["registry_id"],
            "contract_ref": snapshot["contract_ref"],
            "receipt_watermark": snapshot["receipt_watermark"],
            "source_id": contract.source_id,
            "revisions": len(snapshot["events"]),
            "knowledge_basis": contract.knowledge_basis,
            "universe_provenance": contract.universe_provenance,
            "provenance_status": contract.availability_claim,
            "evidence_status": "historical_exploration"
            if contract.knowledge_basis != "observed_receipts"
            else "recorded_inputs_not_forward_evidence_by_themselves",
        }


def knowledge_time(item, basis):
    event = item["event"]
    if basis == "observed_receipts":
        times = [timestamp(item["first_seen_at"])]
        if event["source_available_at"] is not None:
            times.append(timestamp(event["source_available_at"]))
        return max(times)
    if basis == "documented_historical_availability":
        if event["source_available_at"] is None:
            raise ValueError("missing documented source availability")
        return timestamp(event["source_available_at"])
    if basis == "assumed_historical_availability":
        return timestamp(event["event_at"])
    raise ValueError("unknown knowledge basis")


def visible_revisions(snapshot, contract, as_of, *, basis=None, watermark=None):
    basis = basis or contract.knowledge_basis
    if basis not in {contract.knowledge_basis, "observed_receipts"}:
        raise ValueError("view cannot substitute another historical knowledge basis")
    boundary = timestamp(as_of)
    limit = snapshot["receipt_watermark"] if watermark is None else watermark
    if type(limit) is not int or not 0 <= limit <= snapshot["receipt_watermark"]:
        raise ValueError("view watermark is outside the committed snapshot")
    latest = {}
    for item in snapshot["events"]:
        event = item["event"]
        if item["receipt_seq"] > limit or knowledge_time(item, basis) > boundary:
            continue
        # Future notices can be known now, but future quotes/status effectiveness cannot.
        if event["kind"] != "CORPORATE_ACTION_NOTICE" and timestamp(event["event_at"]) > boundary:
            continue
        previous = latest.get(event["event_id"])
        if previous is None or previous["event"]["revision"] < event["revision"]:
            latest[event["event_id"]] = item
    return list(latest.values())
