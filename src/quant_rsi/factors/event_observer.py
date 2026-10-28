"""Instantiate and retire indicators from registered retained-title evidence.

A title correction deactivates its indicator, not every real-world hypothesis.
This deterministic admission rule grants no predictive or production status.
"""

from pathlib import Path

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, canonical, decode, digest
from quant_rsi.evaluation.evidence import authority_root
from quant_rsi.factors import lifecycle
from quant_rsi.factors.news import Extraction, NewsRequest
from quant_rsi.forecast.forecast_experiment import commit_event
from quant_rsi.forecast.forecast_inputs import SignalSnapshot
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import timestamp


def observe(result, template):
    """Idempotent automatic lifecycle for an already sealed registered search."""
    evidence = {ref: bytes.fromhex(raw) for ref, raw in result["evidence_hex"].items()}
    plan = decode(evidence[result["query_ref"]])
    request = NewsRequest.model_validate_json(
        canonical({key: plan[key] for key in NewsRequest.model_fields})
    )
    root = authority_root() / "news_campaigns" / request.campaign_id
    with controller.controller_lock(root):
        anchor = decode((root / "anchor.json").read_bytes())
        state = decode((root / "news_campaign.json").read_bytes())
        ref = state["runs"][request.request_id].get("result_ref")
        if anchor["root"] != str(root.resolve()) or ref != digest(canonical(result)):
            raise ValueError("observer requires a sealed registered search result")
        if ArtifactStore(Path(anchor["store_root"])).json(ref) != result:
            raise ValueError("registered event result differs")
    now = timestamp(market_contracts.trusted_now())
    snapshot = SignalSnapshot.model_validate_json(canonical(result["snapshot"]))
    lock_root = authority_root() / "factor_event_observer"
    lock_root.mkdir(parents=True, exist_ok=True)
    outcomes = []
    with controller.controller_lock(lock_root):
        for event in sorted(snapshot.events, key=lambda e: (e.event_id, e.revision)):
            identity = dict(
                template=template.model_dump(mode="json"),
                source=event.source_id,
                entity=event.entity,
                event_id=event.event_id,
            )
            instance_id = "news-" + digest(canonical(identity))[:40]
            extraction_ref = event.fields["extraction_ref"]
            extraction = Extraction.model_validate_json(evidence[extraction_ref])
            if event.available_at > now or extraction.completed_at > now:
                raise ValueError("observation cannot precede evidence availability")
            exists = (lifecycle.instance_root(instance_id) / "manifest.json").exists()
            unresolved = extraction.status != "valid" or extraction.value is None
            if unresolved and not exists:
                outcomes.append(
                    dict(event_id=event.event_id, status=extraction.status, instance_id=None)
                )
                continue
            supporting = {
                ref: evidence[ref]
                for ref in (event.payload_ref, event.receipt_ref, event.query_ref, extraction_ref)
            }
            if not exists:
                if not extraction.value:
                    outcomes.append(
                        dict(
                            event_id=event.event_id, status="no_matching_instance", instance_id=None
                        )
                    )
                    continue
                lifecycle.create(instance_id, event.entity, template, supporting)
            manifest, journal, store = lifecycle._open(instance_id)
            try:
                prior = [
                    row["payload"]
                    for row in journal.events()
                    if row["kind"].startswith("news_revision_")
                ]
                seen = next((p for p in prior if p["revision"] == event.revision), None)
                if seen and seen["payload_ref"] != event.payload_ref:
                    raise ValueError("news revision payload changed")
                if prior and event.revision < max(p["revision"] for p in prior):
                    outcomes.append(
                        dict(
                            event_id=event.event_id,
                            status="older_revision_ignored",
                            instance_id=instance_id,
                        )
                    )
                    continue
                for raw in supporting.values():
                    store.put(raw)
                commit_event(
                    journal,
                    f"news_revision_{event.revision}",
                    seen
                    or dict(
                        revision=event.revision,
                        payload_ref=event.payload_ref,
                        extraction_ref=extraction_ref,
                    ),
                )
            finally:
                journal.close()
            value = lifecycle.expire(instance_id)
            current = value["transitions"][-1]["state"]
            if unresolved:
                if "suspended" in lifecycle.STATES[current]:
                    value = lifecycle.transition(
                        instance_id,
                        f"revision_{event.revision}_unresolved",
                        "suspended",
                        "registered revision cannot be reliably extracted",
                        supporting,
                    )
            elif extraction.value:
                for target in ("validated", "experimental", "active_for_registered_research"):
                    if target in lifecycle.STATES[current] and current not in {
                        "inactive",
                        "archived",
                    }:
                        if now >= timestamp(value["manifest"]["expires_at"]):
                            break
                        value = lifecycle.transition(
                            instance_id,
                            "admit_" + target,
                            target,
                            "registered exact-title research indicator",
                            supporting,
                        )
                        current = target
            elif current in {"experimental", "active_for_registered_research"}:
                value = lifecycle.transition(
                    instance_id,
                    f"revision_{event.revision}",
                    "inactive",
                    "registered title indicator no longer matches",
                    supporting,
                )
            outcomes.append(
                dict(
                    event_id=event.event_id,
                    instance_id=instance_id,
                    status=value["transitions"][-1]["state"],
                    snapshot_ref=digest(canonical(value)),
                )
            )
    return dict(
        kind="NewsFactorInstanceObservation",
        outcomes=outcomes,
        predictive_claim=False,
        coverage_scope="registered_retained_corpus_only",
    )
