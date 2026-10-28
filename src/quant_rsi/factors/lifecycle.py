"""Observed event-instance lifecycle with append-only, replayable transitions.

These are research hypotheses, not verified real-world causal claims. Evidence
admission/extraction remains the caller's registered source task's responsibility.
"""

import re
from datetime import timedelta
from typing import Literal

from pydantic import Field

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.core.journal import Journal
from quant_rsi.evaluation.evidence import authority_root
from quant_rsi.forecast.forecast_experiment import commit_event, event_payload
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import timestamp, utc_text


class EventTemplate(StrictModel):
    schema_version: Literal[1] = 1
    template_id: str = Field(pattern=r"^[a-z][a-z0-9_.]{0,95}$")
    description: str = Field(min_length=1, max_length=4096)
    ttl_hours: int = Field(ge=1, le=720)
    decay: Literal["none", "linear"] = "none"
    output_unit: Literal["dimensionless"] = "dimensionless"


STATES = {
    "proposed": {"validated", "rejected"},
    "validated": {"experimental", "rejected"},
    "experimental": {"active_for_registered_research", "inactive", "suspended"},
    "active_for_registered_research": {"inactive", "suspended"},
    "suspended": {"archived"},
    "inactive": {"archived"},
    "rejected": {"archived"},
    "archived": set(),
}


def instance_root(instance_id):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", instance_id):
        raise ValueError("invalid factor instance identity")
    return authority_root() / "factor_instances" / instance_id


def _open(instance_id):
    root = instance_root(instance_id)
    raw = (root / "manifest.json").read_bytes()
    manifest = decode(raw)
    journal = Journal(root)
    if manifest["root"] != str(root.resolve()) or dict(
        journal.connection.execute("SELECT key,value FROM metadata")
    ) != {"manifest_digest": digest(raw)}:
        journal.close()
        raise ValueError("factor instance authority differs")
    return manifest, journal, ArtifactStore(root / "artifacts")


def _evidence(store, evidence):
    if not evidence:
        raise ValueError("lifecycle transition requires supporting evidence")
    if sum(map(len, evidence.values())) > 1048576:
        raise ValueError("lifecycle evidence exceeds one MiB")
    for ref, raw in evidence.items():
        if ref != digest(raw):
            raise ValueError("lifecycle evidence hash differs")
        store.put(raw)
    return sorted(evidence)


def create(instance_id, instrument, template, evidence):
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,31}", instrument):
        raise ValueError("invalid instance instrument")
    root = instance_root(instance_id)
    root.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(root):
        binding = dict(
            instance_id=instance_id,
            instrument=instrument,
            template=template.model_dump(mode="json"),
            evidence_refs=sorted(evidence),
        )
        if (root / "manifest.json").exists():
            manifest, journal, store = _open(instance_id)
            try:
                if manifest["binding"] != binding:
                    raise ValueError("instance ID already binds different evidence/template")
                _evidence(store, evidence)
                return _snapshot(manifest, journal, store)
            finally:
                journal.close()
        marker = root / "initializing.json"
        if marker.exists():
            manifest = decode(marker.read_bytes())
            if manifest["binding"] != binding or manifest["root"] != str(root.resolve()):
                raise ValueError("instance initialization identity changed")
        else:
            if any(p.name != ".controller.lock" for p in root.iterdir()):
                raise ValueError("instance authority missing")
            now = timestamp(market_contracts.trusted_now())
            manifest = dict(
                kind="FactorInstance",
                schema_version=1,
                binding=binding,
                root=str(root.resolve()),
                created_at=utc_text(now),
                expires_at=utc_text(now + timedelta(hours=template.ttl_hours)),
            )
            atomic_write(marker, canonical(manifest))
        store = ArtifactStore(root / "artifacts")
        _evidence(store, evidence)
        journal = Journal(root, digest(canonical(manifest)))
        try:
            commit_event(
                journal,
                "transition_created",
                dict(
                    event_id="created",
                    state="proposed",
                    observed_at=manifest["created_at"],
                    reason="observed_event_hypothesis",
                    evidence_refs=sorted(evidence),
                ),
            )
            atomic_write(root / "manifest.json", canonical(manifest))
            marker.unlink()
            return _snapshot(manifest, journal, store)
        finally:
            journal.close()


def transition(instance_id, event_id, state, reason, evidence):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", event_id) or not reason or len(reason) > 4096:
        raise ValueError("transition requires bounded identity and reason")
    root = instance_root(instance_id)
    with controller.controller_lock(root):
        manifest, journal, store = _open(instance_id)
        try:
            existing = event_payload(journal, "transition_" + event_id)
            intent = dict(
                event_id=event_id,
                state=state,
                reason=reason,
                evidence_refs=_evidence(store, evidence),
            )
            if existing:
                if existing != intent | {"observed_at": existing["observed_at"]}:
                    raise ValueError("transition ID already binds a different decision")
            else:
                events = _snapshot(manifest, journal, store)["transitions"]
                if state not in STATES[events[-1]["state"]]:
                    raise ValueError("illegal factor lifecycle transition")
                now = timestamp(market_contracts.trusted_now())
                if now < timestamp(events[-1]["observed_at"]):
                    raise ValueError("lifecycle clock moved backwards")
                if state == "active_for_registered_research" and now >= timestamp(
                    manifest["expires_at"]
                ):
                    raise ValueError("expired event cannot be activated; register a new instance")
                commit_event(
                    journal, "transition_" + event_id, intent | {"observed_at": utc_text(now)}
                )
            return _snapshot(manifest, journal, store)
        finally:
            journal.close()


def _snapshot(manifest, journal, store):
    events = [row["payload"] for row in journal.events() if row["kind"].startswith("transition_")]
    for ref in {ref for e in events for ref in e["evidence_refs"]}:
        store.get(ref)
    value = dict(
        kind="FactorInstanceSnapshot", schema_version=1, manifest=manifest, transitions=events
    )
    store.put_json(value)
    return value


def snapshot(instance_id):
    with controller.controller_lock(instance_root(instance_id)):
        manifest, journal, store = _open(instance_id)
        try:
            return _snapshot(manifest, journal, store)
        finally:
            journal.close()


def verify_snapshot(value, *, latest=False):
    instance_id = value["manifest"]["binding"]["instance_id"]
    current = snapshot(instance_id)
    count = len(value["transitions"])
    if not count or value["transitions"] != current["transitions"][:count]:
        raise ValueError("lifecycle snapshot is not an authoritative history prefix")
    if latest and current != value:
        raise ValueError("new campaign must freeze complete current lifecycle evidence")
    manifest, journal, store = _open(instance_id)
    try:
        if manifest != value["manifest"] or store.json(digest(canonical(value))) != value:
            raise ValueError("unregistered lifecycle snapshot")
    finally:
        journal.close()


def value_at(value, as_of):
    boundary = timestamp(as_of)
    visible = [e for e in value["transitions"] if timestamp(e["observed_at"]) <= boundary]
    if not visible:
        return dict(value=None, reasons=["event_instance_not_observed"], source_refs=[])
    current = visible[-1]
    refs = sorted({ref for e in visible for ref in e["evidence_refs"]})
    manifest = value["manifest"]
    expired = boundary >= timestamp(manifest["expires_at"])
    if not any(e["state"] == "active_for_registered_research" for e in visible):
        return dict(value=None, reasons=["event_instance_not_admitted"], source_refs=refs)
    if any(e["state"] == "suspended" for e in visible):
        return dict(value=None, reasons=["event_instance_suspended"], source_refs=refs)
    result = 0.0
    if current["state"] == "active_for_registered_research" and not expired:
        result = 1.0
        if manifest["binding"]["template"]["decay"] == "linear":
            ttl = timestamp(manifest["expires_at"]) - timestamp(manifest["created_at"])
            result = (timestamp(manifest["expires_at"]) - boundary) / ttl
    return dict(value=result, reasons=[], source_refs=refs)


def attach_instance(envelope, value):
    """Freeze one observed instance snapshot into an existing ForecastInputs envelope."""
    from quant_rsi.forecast.forecast_inputs import ForecastInputs

    verify_snapshot(value, latest=True)
    manifest, journal, store = _open(value["manifest"]["binding"]["instance_id"])
    try:
        result = decode(canonical(envelope))
        refs = {ref for event in value["transitions"] for ref in event["evidence_refs"]}
        refs.add(store.put_json(value))
        result["evidence_hex"].update({ref: store.get(ref).hex() for ref in refs})
        return ForecastInputs(result).value
    finally:
        journal.close()


def expire(instance_id):
    """Persist due expiration/archival without backdating the transition clock."""
    value = snapshot(instance_id)
    if timestamp(market_contracts.trusted_now()) < timestamp(value["manifest"]["expires_at"]):
        return value
    raw = canonical(value["manifest"])
    evidence = {digest(raw): raw}
    state = value["transitions"][-1]["state"]
    if state in {"proposed", "validated"}:
        value = transition(instance_id, "expired", "rejected", "registered TTL elapsed", evidence)
    elif state in {"experimental", "active_for_registered_research"}:
        value = transition(instance_id, "expired", "inactive", "registered TTL elapsed", evidence)
    if value["transitions"][-1]["state"] in {"inactive", "rejected", "suspended"}:
        value = transition(
            instance_id, "expiry_archived", "archived", "preserve expired history", evidence
        )
    return value
