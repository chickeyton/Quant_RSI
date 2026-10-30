"""Durable prospective RSS observations using the existing authority, Journal and artifacts."""

import re
import xml.etree.ElementTree as ET
from typing import Literal

from pydantic import AwareDatetime, Field

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.core.journal import Journal
from quant_rsi.evaluation.evidence import authority_root
from quant_rsi.forecast.forecast_experiment import commit_event, event_payload
from quant_rsi.forecast.forecast_inputs import SignalSnapshot
from quant_rsi.market import market_contracts, news_feed
from quant_rsi.market.market_contracts import timestamp, utc_text


class CollectionPlan(StrictModel):
    schema_version: Literal[1] = 1
    campaign_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    instrument: str = Field(default="KO", pattern=r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
    start_at: AwareDatetime
    max_requests: int = Field(default=8, ge=1, le=32)
    response_bytes: int = Field(default=262144, ge=1024, le=262144)
    items_per_response: int = Field(default=128, ge=1, le=1024)


def collection_root(campaign_id):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", campaign_id):
        raise ValueError("invalid news collection campaign")
    return authority_root() / "news_collections" / campaign_id


def _open(campaign_id, *, execute=False):
    root = collection_root(campaign_id)
    raw = (root / "manifest.json").read_bytes()
    manifest = decode(raw)
    journal = Journal(root)
    try:
        if manifest["root"] != str(root.resolve()) or dict(
            journal.connection.execute("SELECT key,value FROM metadata")
        ) != {"manifest_digest": digest(raw)}:
            raise ValueError("news collection authority changed")
        store = ArtifactStore(root / "artifacts")
        plan = CollectionPlan.model_validate_json(canonical(store.json(manifest["plan_ref"])))
        source = news_feed.FeedSource.model_validate_json(
            canonical(store.json(manifest["source_ref"]))
        )
        if plan.campaign_id != campaign_id or plan.instrument != source.instrument:
            raise ValueError("collection plan/source binding differs")
        if execute and store.json(manifest["runtime_ref"]) != controller.runtime_identity():
            raise ValueError("new collection requests require the registered source/runtime")
        return manifest, journal, store, plan, source
    except Exception:
        journal.close()
        raise


def initialize(plan):
    root = collection_root(plan.campaign_id)
    root.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(root):
        if (root / "manifest.json").exists():
            manifest, journal, store, saved, source = _open(plan.campaign_id)
            try:
                if saved != plan:
                    raise ValueError("collection identity already binds a different plan")
                return _report(manifest, journal, store, plan)
            finally:
                journal.close()
        source = news_feed.source_for(plan.instrument)
        if plan.start_at >= timestamp(market_contracts.trusted_now()):
            raise ValueError("collection start must precede its observed initialization")
        marker = root / "initializing.json"
        intent = dict(plan=plan.model_dump(mode="json"), source=source.model_dump(mode="json"))
        if marker.exists():
            if decode(marker.read_bytes()) != intent:
                raise ValueError("unfinished collection initialization differs")
        else:
            if any(p.name != ".controller.lock" for p in root.iterdir()):
                raise ValueError("collection manifest missing; cannot renew allowances")
            atomic_write(marker, canonical(intent))
        store = ArtifactStore(root / "artifacts")
        manifest = dict(
            schema_version=1,
            kind="NewsCollectionCampaign",
            root=str(root.resolve()),
            plan_ref=store.put_json(intent["plan"]),
            source_ref=store.put_json(intent["source"]),
            runtime_ref=store.put_json(controller.runtime_identity()),
        )
        journal = Journal(root, digest(canonical(manifest)))
        try:
            atomic_write(root / "manifest.json", canonical(manifest))
            marker.unlink()
            return _report(manifest, journal, store, plan)
        finally:
            journal.close()


def _reservations(journal, plan):
    records = [row["payload"] for row in journal.events() if row["kind"] == "feed_reserved"]
    if (
        len(records) > plan.max_requests
        or [row["ordinal"] for row in records] != list(range(len(records)))
        or len({row["request_id"] for row in records}) != len(records)
    ):
        raise ValueError("collection reservations lost their ordering or allowance")
    return records


def _saved(journal, store, request_id):
    record = event_payload(journal, "feed_result_" + request_id)
    if record is None:
        return None
    value = store.json(record["result_ref"])
    if value["request_id"] != request_id:
        raise ValueError("collection result identity differs")
    for ref in value["evidence_refs"]:
        store.get(ref)
    SignalSnapshot.model_validate_json(canonical(store.json(value["snapshot_ref"])))
    return value


def _seal(manifest, journal, store, plan, source, reservation, receipt):
    request_id = reservation["request_id"]
    prior = _reservations(journal, plan)[: reservation["ordinal"]]
    previous = _saved(journal, store, prior[-1]["request_id"]) if prior else None
    snapshot = (
        store.json(previous["snapshot_ref"])
        if previous
        else dict(schema_version=1, kind="forecast_signal_snapshot", events=[], coverage=[])
    )
    evidence_refs = set(previous["evidence_refs"] if previous else [])
    evidence_refs.update((manifest["source_ref"], receipt["raw_ref"], reservation["query_ref"]))
    receipt_ref = store.put_json(receipt)
    evidence_refs.add(receipt_ref)
    status, reason, parsed = "unknown", receipt.get("error"), []
    if not reason:
        try:
            parsed = news_feed.parse(store.get(receipt["raw_ref"]), source, plan.items_per_response)
            status, reason = (
                "partial",
                "rolling RSS window; absence and historical completeness unknown",
            )
        except (ValueError, ET.ParseError, UnicodeError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
    latest = {}
    for event in snapshot["events"]:
        latest[event["event_id"]] = event
    for item in parsed:
        payload_ref = store.put_json(item["payload"])
        old = latest.get(item["event_id"])
        if old and old["payload_ref"] == payload_ref:
            continue  # Keep the first observation and revision of an unchanged item.
        evidence_refs.add(payload_ref)
        snapshot["events"].append(
            dict(
                schema_version=1,
                kind="news",
                source_id=source.source_id,
                entity=source.instrument,
                event_id=item["event_id"],
                revision=old["revision"] + 1 if old else 0,
                query_ref=reservation["query_ref"],
                published_at=item["published_at"],
                published_date=item["published_date"],
                publication_precision=item["publication_precision"],
                source_available_at=None,
                first_seen_at=receipt["received_at"],
                report_period_end=None,
                payload_ref=payload_ref,
                receipt_ref=receipt_ref,
                extractor_version="issuer_rss_v1",
                fields=dict(title=item["payload"]["title"], content_extent=source.content_extent),
                units=dict(title="text", content_extent="category"),
            )
        )
    snapshot["coverage"].append(
        dict(
            schema_version=1,
            source_id=source.source_id,
            entity=source.instrument,
            query_ref=reservation["query_ref"],
            start_at=utc_text(plan.start_at),
            end_at=receipt["received_at"],
            first_seen_at=receipt["received_at"],
            status=status,
            reason=reason,
            receipt_ref=receipt_ref,
        )
    )
    snapshot = SignalSnapshot.model_validate_json(canonical(snapshot)).model_dump(mode="json")
    result = dict(
        kind="NewsCollectionResult",
        schema_version=1,
        campaign_id=plan.campaign_id,
        request_id=request_id,
        retrieval_mode="rss_polling",
        coverage_status=status,
        reason=reason,
        received_at=receipt["received_at"],
        query_ref=reservation["query_ref"],
        receipt_ref=receipt_ref,
        snapshot_ref=store.put_json(snapshot),
        evidence_refs=sorted(evidence_refs),
        observed_revisions=len(snapshot["events"]),
        response_bytes=len(store.get(receipt["raw_ref"])),
    )
    commit_event(journal, "feed_result_" + request_id, {"result_ref": store.put_json(result)})
    return result


def poll(campaign_id, request_id):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", request_id):
        raise ValueError("invalid collection request ID")
    root = collection_root(campaign_id)
    with controller.controller_lock(root):
        manifest, journal, store, plan, source = _open(campaign_id, execute=True)
        try:
            saved = _saved(journal, store, request_id)
            if saved is not None:
                return saved
            reservations = _reservations(journal, plan)
            reservation = next((r for r in reservations if r["request_id"] == request_id), None)
            fresh = reservation is None
            if fresh:
                if reservations and _saved(journal, store, reservations[-1]["request_id"]) is None:
                    raise ValueError("resolve the pending request before starting another poll")
                if len(reservations) >= plan.max_requests:
                    raise ValueError("news request allowance exhausted")
                now = timestamp(market_contracts.trusted_now())
                if now <= plan.start_at or (
                    reservations
                    and now
                    < timestamp(
                        _saved(journal, store, reservations[-1]["request_id"])["received_at"]
                    )
                ):
                    raise ValueError("collection observation clock moved backwards")
                query_ref = store.put_json(
                    dict(
                        kind="RegisteredIssuerFeedPoll",
                        source_ref=manifest["source_ref"],
                        url=source.url,
                        retrieval_mode="rss_polling",
                        start_at=utc_text(plan.start_at),
                        as_of=utc_text(now),
                        request_id=request_id,
                    )
                )
                reservation = dict(
                    ordinal=len(reservations),
                    request_id=request_id,
                    query_ref=query_ref,
                    requested_at=utc_text(now),
                )
                with journal.connection:
                    journal.event(None, "feed_reserved", reservation)
            receipt_entry = event_payload(journal, "feed_received_" + request_id)
            if receipt_entry is None:
                if fresh:
                    body, metadata = news_feed.fetch(source, plan.response_bytes)
                else:
                    # A remote request may have completed before a controller interruption.
                    # Charge its reservation and retain unknown coverage; never retry implicitly.
                    body, metadata = (
                        b"",
                        dict(
                            error="interrupted_request_outcome_unknown",
                            received_at=market_contracts.trusted_now(),
                        ),
                    )
                if len(body) > plan.response_bytes:
                    raise ValueError("transport exceeded registered response bytes")
                if timestamp(metadata["received_at"]) < timestamp(reservation["requested_at"]):
                    raise ValueError("response receipt precedes registered dispatch")
                receipt = dict(
                    kind="IssuerFeedResponseReceipt",
                    schema_version=1,
                    query_ref=reservation["query_ref"],
                    raw_ref=store.put(body),
                    requested_at=reservation["requested_at"],
                    **metadata,
                )
                commit_event(
                    journal, "feed_received_" + request_id, {"receipt_ref": store.put_json(receipt)}
                )
            else:
                receipt = store.json(receipt_entry["receipt_ref"])
            return _seal(manifest, journal, store, plan, source, reservation, receipt)
        finally:
            journal.close()


def _report(manifest, journal, store, plan):
    reservations = _reservations(journal, plan)
    results = []
    for reservation in reservations:
        result = _saved(journal, store, reservation["request_id"])
        results.append(
            dict(
                request_id=reservation["request_id"],
                status=result["coverage_status"] if result else "pending_unknown",
                result_ref=digest(canonical(result)) if result else None,
            )
        )
    return dict(
        kind="NewsCollectionReport",
        campaign_id=plan.campaign_id,
        artifact_directory=str(store.root),
        source_ref=manifest["source_ref"],
        requests_reserved=len(reservations),
        request_limit=plan.max_requests,
        reserved_bytes=len(reservations) * plan.response_bytes,
        byte_limit=plan.max_requests * plan.response_bytes,
        reserved_item_slots=len(reservations) * plan.items_per_response,
        polls=results,
        provider_search=False,
        fitting_performed=False,
    )


def inspect(campaign_id):
    with controller.controller_lock(collection_root(campaign_id)):
        manifest, journal, store, plan, source = _open(campaign_id)
        try:
            return _report(manifest, journal, store, plan)
        finally:
            journal.close()


def process(campaign_id, request_id, news_campaign_id):
    """Feed a sealed poll into the existing retained-title and lifecycle path; no network/fits."""
    from quant_rsi.factors.event_observer import observe
    from quant_rsi.factors.news import NewsRequest, run_news

    with controller.controller_lock(collection_root(campaign_id)):
        manifest, journal, store, plan, source = _open(campaign_id, execute=True)
        try:
            result = _saved(journal, store, request_id)
            if result is None:
                raise ValueError("process requires a sealed poll")
            request = NewsRequest.model_validate_json(
                canonical(
                    dict(
                        request_id=request_id,
                        campaign_id=news_campaign_id,
                        instrument=source.instrument,
                        source_id=source.source_id,
                        start_at=utc_text(plan.start_at),
                        end_at=result["received_at"],
                        as_of=result["received_at"],
                        max_documents=1024,
                        max_extractions=1024,
                        max_bytes=16777216,
                    )
                )
            )
            # All observations share one extraction/comparison owner; changing a
            # poll ID must not create a fresh downstream research allowance.
            commit_event(journal, "feed_processing", dict(news_campaign_id=news_campaign_id))
            news_result = run_news(
                request,
                SignalSnapshot.model_validate_json(canonical(store.json(result["snapshot_ref"]))),
                {ref: store.get(ref) for ref in result["evidence_refs"]},
                store,
            )
            observation = observe(news_result, source.event_template)
            return dict(
                kind="NewsCollectionProcessing",
                news_result_ref=store.put_json(news_result),
                observation_ref=store.put_json(observation),
                outcomes=observation["outcomes"],
                coverage_status=result["coverage_status"],
                fitting_performed=False,
            )
        finally:
            journal.close()
