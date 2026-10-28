"""Bounded, offline search and evidence-backed features over an explicitly retained corpus.

This adapter never claims provider search or complete market-news coverage. Inputs are
existing SignalSnapshots and content-addressed payloads, not instructions to execute.
"""

import re
from datetime import timedelta
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from quant_rsi.core.artifacts import atomic_write, canonical, decode, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.forecast.forecast_inputs import SignalSnapshot
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import Identifier, timestamp, utc_text

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "instruments"
NEWS_TASK = decode((CONFIG_ROOT / "shared" / "issuer_title_task.json").read_bytes())
NEWS_BINDING = decode((CONFIG_ROOT / "US" / "KO" / "news.json").read_bytes())
NEWS_FEATURE_DEFINITION = NEWS_TASK["aggregation"]
SOURCE = NEWS_BINDING["source_id"]
RULE = NEWS_BINDING["rule_version"]
POSITIVE = re.compile(
    r"^(?:"
    + "|".join(re.escape(alias) for alias in NEWS_BINDING["entity_aliases"])
    + r") Reports (?P<quarter>First|Second|Third|Fourth) Quarter "
    + r"(?P<year>20[0-9]{2}) Results[.!]?$"
)


class NewsRequest(StrictModel):
    schema_version: Literal[1] = 1
    request_id: Identifier
    campaign_id: Identifier
    instrument: Literal["KO"] = "KO"
    source_id: Literal["retained_issuer_news"] = SOURCE
    start_at: AwareDatetime
    end_at: AwareDatetime
    as_of: AwareDatetime
    max_documents: int = Field(default=128, ge=1, le=1024)
    max_bytes: int = Field(default=1048576, ge=1, le=16777216)
    max_extractions: int = Field(default=128, ge=1, le=1024)

    @model_validator(mode="after")
    def interval(self) -> Self:
        if not self.start_at < self.end_at <= self.as_of:
            raise ValueError("search interval must end by as_of")
        return self

    @property
    def plan(self):
        return dict(
            **self.model_dump(mode="json"),
            retrieval_mode="retained_corpus_search",
            query=f"entity:{self.instrument} source:{self.source_id}",
            query_version="issuer_all_titles_v1",
            ordering="event_id_revision",
            pagination="one_bounded_page_no_network",
            content_extent="retained_title_only",
            coverage_scope="supplied_retained_corpus_only",
            extractor=RULE,
            task_definition=NEWS_TASK,
            instrument_binding=NEWS_BINDING,
            feature=NEWS_FEATURE_DEFINITION,
        )


class Extraction(StrictModel):
    schema_version: Literal[1]
    kind: Literal["issuer_title_extraction"]
    rule_version: Literal["ko_earnings_title_v1"]
    entity: Identifier
    event_id: Identifier
    revision: int = Field(ge=0)
    payload_ref: str
    status: Literal["valid", "unresolved", "rejected"]
    value: bool | None
    event_key: str | None
    unit: Literal["earnings_release_title_indicator"]
    passage: dict[str, str | int]
    completed_at: AwareDatetime
    reason: str | None


def extract_title(event, raw, completed_at):
    """Classify only a reviewed exact title grammar; never infer reported quantities.

    Payload must be a retained JSON object with an identical title. Nonmatching
    titles are valid negative grammar matches, not claims of no real-world event.
    """
    title = event.fields.get("title", "")
    match = POSITIVE.fullmatch(title)
    status, value, reason = "valid", bool(match), None
    try:
        payload = decode(raw)
        supported = isinstance(payload, dict) and payload.get("title") == title and bool(title)
    except (ValueError, UnicodeError):
        supported = False
    if (
        digest(raw) != event.payload_ref
        or not supported
        or event.entity != NEWS_BINDING["instrument"]
    ):
        status, value, reason = "rejected", None, "title_entity_or_payload_not_supported"
    elif len(title) > 4096:
        status, value, reason = "unresolved", None, "title_exceeds_rule_limit"
    return Extraction.model_validate_json(
        canonical(
            dict(
                schema_version=1,
                kind="issuer_title_extraction",
                rule_version=RULE,
                entity=NEWS_BINDING["instrument"],
                event_id=event.event_id,
                revision=event.revision,
                payload_ref=event.payload_ref,
                status=status,
                value=value,
                event_key=(
                    f"{NEWS_BINDING['instrument']}:earnings_release:{match['year']}:{match['quarter']}"
                    if match and value is True
                    else None
                ),
                unit="earnings_release_title_indicator",
                passage=dict(field="title", start=0, end=len(title), text=title),
                completed_at=utc_text(completed_at),
                reason=reason,
            )
        )
    ).model_dump(mode="json")


def run_news(request, snapshot, evidence, store, root: Path | None = None, *, completed_at=None):
    """Search a registered retained corpus with cumulative campaign reservations.

    Intent/reservation and each extraction are checkpointed using existing atomic
    artifacts and controller locking. Terminal repeats read saved outputs. An
    interrupted local rule may be recomputed only if its output was not committed;
    no remote operation or charge is possible in this adapter.
    """
    from quant_rsi.core.controller import controller_lock
    from quant_rsi.evaluation.evidence import authority_root

    now = timestamp(market_contracts.trusted_now())
    if completed_at is not None and timestamp(completed_at) != now:
        raise ValueError("completion time must come from the trusted clock")
    completed_at = now
    if completed_at < request.as_of:
        raise ValueError("extraction completion precedes search boundary")
    for ref, raw in evidence.items():
        if digest(raw) != ref:
            raise ValueError("corpus evidence hash mismatch")
    for event in snapshot.events:
        for ref in (event.payload_ref, event.receipt_ref, event.query_ref):
            if ref not in evidence:
                raise ValueError("missing corpus evidence")
    for coverage in snapshot.coverage:
        if coverage.receipt_ref not in evidence or coverage.query_ref not in evidence:
            raise ValueError("missing coverage evidence")
    registered = authority_root() / "news_campaigns"
    if root is not None and Path(root).resolve() != registered.resolve():
        raise ValueError("news campaign must use its registered authority path")
    root = registered / request.campaign_id
    root.mkdir(parents=True, exist_ok=True)
    binding = digest(canonical(dict(plan=request.plan, snapshot=snapshot.model_dump(mode="json"))))
    limits = dict(
        documents=request.max_documents,
        bytes=request.max_bytes,
        extractions=request.max_extractions,
    )
    with controller_lock(root):
        path = root / "news_campaign.json"
        anchor = root / "anchor.json"
        identity = dict(
            campaign_id=request.campaign_id,
            root=str(root.resolve()),
            store_root=str(store.root.resolve()),
            limits=limits,
        )
        if anchor.exists():
            if decode(anchor.read_bytes()) != identity or not path.exists():
                raise ValueError("news campaign identity or ledger is missing or changed")
        elif path.exists():
            raise ValueError("news campaign anchor missing")
        else:
            if any(p.name != ".controller.lock" for p in root.iterdir()):
                raise ValueError("news campaign has unbound prior artifacts")
            atomic_write(anchor, canonical(identity))
            atomic_write(
                path,
                canonical(
                    dict(
                        schema_version=1,
                        limits=limits,
                        used=dict(documents=0, bytes=0, extractions=0),
                        runs={},
                    )
                ),
            )
        state = (
            decode(path.read_bytes())
            if path.exists()
            else dict(
                schema_version=1,
                limits=limits,
                used=dict(documents=0, bytes=0, extractions=0),
                runs={},
            )
        )
        if state["limits"] != limits:
            raise ValueError("campaign allowance cannot be renewed or changed")
        old = state["runs"].get(request.request_id)
        if old and old["binding"] != binding:
            raise ValueError("request identity already binds different inputs")
        if old and old.get("result_ref"):
            return store.json(old["result_ref"])
        if old is None:
            candidates = [
                e
                for e in snapshot.events
                if e.entity == request.instrument
                and e.source_id == request.source_id
                and e.kind == "news"
                and e.available_at <= request.as_of
            ]
            requested_ids = {
                e.event_id
                for e in candidates
                if e.published_at is None or request.start_at <= e.published_at < request.end_at
            }
            # Keep the full visible lineage for any intersecting document. Filtering
            # individual revisions first could resurrect a corrected older story.
            candidates = [e for e in candidates if e.event_id in requested_ids]
            candidates.sort(key=lambda e: (e.event_id, e.revision))
            chosen, size = [], 0
            remaining = {k: limits[k] - state["used"][k] for k in limits}
            available_coverage = [
                c
                for c in snapshot.coverage
                if c.source_id == SOURCE
                and c.entity == NEWS_BINDING["instrument"]
                and c.first_seen_at <= request.as_of
            ]
            retained_refs = {
                ref for c in available_coverage for ref in (c.query_ref, c.receipt_ref)
            }
            size = sum(len(evidence[ref]) for ref in retained_refs)
            coverage_truncated = size > remaining["bytes"]
            if coverage_truncated:
                available_coverage, retained_refs, size = [], set(), 0
            for event in candidates:
                new_refs = {event.payload_ref, event.receipt_ref, event.query_ref} - retained_refs
                nbytes = sum(len(evidence[ref]) for ref in new_refs)
                if (
                    len(chosen) >= min(remaining["documents"], remaining["extractions"])
                    or size + nbytes > remaining["bytes"]
                ):
                    break
                chosen.append(event.model_dump(mode="json"))
                size += nbytes
                retained_refs.update(new_refs)
            usage = dict(documents=len(chosen), bytes=size, extractions=len(chosen))
            old = dict(
                binding=binding,
                query_ref=digest(canonical(request.plan)),
                selected=chosen,
                coverage=[c.model_dump(mode="json") for c in available_coverage],
                retained_refs=sorted(retained_refs),
                usage=usage,
                outputs={},
                completed_at=utc_text(completed_at),
                truncated=coverage_truncated or len(chosen) != len(candidates),
            )
            state["runs"][request.request_id] = old
            for key in usage:
                state["used"][key] += usage[key]
            atomic_write(path, canonical(state))
        query_ref = store.put_json(request.plan)
        retained = {ref: evidence[ref] for ref in old["retained_refs"]}
        retained[query_ref] = store.get(query_ref)
        events = []
        from quant_rsi.forecast.forecast_inputs import SignalEvent

        for item in old["selected"]:
            event = SignalEvent.model_validate_json(canonical(item))
            key = f"{event.event_id}:{event.revision}"
            ref = old["outputs"].get(key)
            if ref is None:
                extracted = extract_title(
                    event, evidence[event.payload_ref], market_contracts.trusted_now()
                )
                extracted["completed_at"] = utc_text(market_contracts.trusted_now())
                ref = store.put_json(extracted)
                old["outputs"][key] = ref
                atomic_write(path, canonical(state))
            retained[ref] = store.get(ref)
            events.append(
                item
                | dict(
                    query_ref=query_ref,
                    fields=event.fields
                    | {"extraction_ref": ref, "content_extent": "retained_title_only"},
                    units=event.units
                    | {"extraction_ref": "artifact_sha256", "content_extent": "category"},
                )
            )
        coverage = []
        for original in old["coverage"]:
            value = dict(original)
            # Return only the interval actually searched. The original coverage
            # and receipts remain unchanged in the campaign ledger/evidence.
            start = max(timestamp(value["start_at"]), request.start_at)
            end = min(timestamp(value["end_at"]), request.end_at)
            if start >= end:
                continue
            value.update(start_at=utc_text(start), end_at=utc_text(end))
            value["first_seen_at"] = utc_text(
                max(timestamp(value["first_seen_at"]), timestamp(market_contracts.trusted_now()))
            )
            if old["truncated"]:
                value.update(status="partial", reason="campaign_allowance_truncated_corpus")
            coverage.append(value | dict(query_ref=query_ref))
        result = dict(
            kind="RetainedNewsSearchResult",
            schema_version=1,
            query_ref=query_ref,
            snapshot=dict(
                schema_version=1, kind="forecast_signal_snapshot", events=events, coverage=coverage
            ),
            evidence_hex={ref: raw.hex() for ref, raw in retained.items()},
            usage=old["usage"] | dict(network_requests=0, tokens=None, monetary_cost=None),
            cumulative_usage=state["used"].copy(),
            truncated=old["truncated"],
            qualifications=[
                "retained_corpus_only_not_complete_market_news",
                "historical_features_require_observed_extraction_completion",
                "exact_title_grammar_not_semantic_event_confirmation",
            ],
        )
        # Validate before sealing, including revision monotonicity and duplicate coverage.
        result["snapshot"] = SignalSnapshot.model_validate_json(
            canonical(result["snapshot"])
        ).model_dump(mode="json")
        old["result_ref"] = store.put_json(result)
        atomic_write(path, canonical(state))
        return result


def news_scalar(inputs, origin, as_of):
    """Corpus-relative seven-calendar-day count, never missing-to-zero imputation."""
    boundary = timestamp(as_of)
    reasons, refs = [], set()
    if inputs.target[0].instruments[0] != NEWS_BINDING["instrument"]:
        return dict(value=None, reasons=["news_instrument_not_supported"], source_refs=[])
    snapshot = inputs.signals
    if snapshot is None:
        return dict(value=None, reasons=["news_coverage_missing"], source_refs=[])
    coverage = [
        c
        for c in snapshot.coverage
        if c.source_id == SOURCE
        and c.entity == NEWS_BINDING["instrument"]
        and c.start_at <= c.end_at - timedelta(days=NEWS_FEATURE_DEFINITION["window_days"])
        and boundary - timedelta(days=1) <= c.end_at <= boundary
        and c.first_seen_at <= boundary
    ]
    if not coverage:
        return dict(value=None, reasons=["news_coverage_missing"], source_refs=[])
    latest_coverage = max(coverage, key=lambda c: (c.end_at, c.first_seen_at))
    end = latest_coverage.end_at
    start = end - timedelta(days=NEWS_FEATURE_DEFINITION["window_days"])
    refs.update((latest_coverage.receipt_ref, latest_coverage.query_ref))
    if latest_coverage.status != "complete":
        reasons.append("news_coverage_" + latest_coverage.status)
    query = decode(inputs.evidence[latest_coverage.query_ref])
    if (
        query.get("retrieval_mode") != "retained_corpus_search"
        or query.get("extractor") != RULE
        or query.get("feature") != NEWS_FEATURE_DEFINITION
        or query.get("task_definition") != NEWS_TASK
        or query.get("instrument_binding") != NEWS_BINDING
        or timestamp(query["start_at"]) > start
        or timestamp(query["end_at"]) < end
    ):
        reasons.append("news_query_binding_invalid")
    latest = {}
    for event in snapshot.events:
        if (
            event.source_id != SOURCE
            or event.entity != NEWS_BINDING["instrument"]
            or event.kind != "news"
            or event.available_at > boundary
        ):
            continue
        if event.event_id not in latest or event.revision > latest[event.event_id].revision:
            latest[event.event_id] = event
    count = 0
    counted_payloads = set()
    for event in latest.values():
        if event.published_at is None:
            reasons.append("news_publication_time_unknown")
            continue
        if not start <= event.published_at < end:
            continue
        refs.update((event.payload_ref, event.receipt_ref, event.query_ref))
        event_query = decode(inputs.evidence[event.query_ref])
        if (
            event_query.get("extractor") != RULE
            or event_query.get("retrieval_mode") != "retained_corpus_search"
        ):
            reasons.append("news_event_query_invalid")
            continue
        ref = event.fields.get("extraction_ref")
        if ref is None or ref not in inputs.evidence:
            reasons.append("news_extraction_missing")
            continue
        refs.add(ref)
        try:
            result = Extraction.model_validate_json(inputs.evidence[ref])
        except ValueError:
            reasons.append("news_extraction_malformed")
            continue
        if (
            result.event_id != event.event_id
            or result.revision != event.revision
            or result.payload_ref != event.payload_ref
        ):
            reasons.append("news_extraction_binding_invalid")
        elif result.completed_at > boundary:
            reasons.append("news_extraction_not_available")
        elif result.status != "valid" or result.value is None:
            reasons.append("news_extraction_" + result.status)
        else:
            # Revalidate exact passage and typed rule output, not just a claimed artifact ID.
            expected = extract_title(event, inputs.evidence[event.payload_ref], result.completed_at)
            if expected != result.model_dump(mode="json"):
                reasons.append("news_extraction_evidence_invalid")
            else:
                if result.value and result.event_key not in counted_payloads:
                    count += 1
                    counted_payloads.add(result.event_key)
    return dict(
        value=float(count) if not reasons else None,
        reasons=sorted(set(reasons)),
        source_refs=sorted(refs),
    )


def attach_news_result(envelope, result):
    """Accumulate prospective observations without replacing prior extraction evidence.

    Identical revisions keep their first committed extraction; changed payloads
    require new revision IDs. The returned ForecastInputs envelope remains v1,
    with all new definitions bound by its existing evidence and signal references.
    """
    from quant_rsi.forecast.forecast_inputs import ForecastInputs

    value = decode(canonical(envelope))
    previous = value["signals"] or dict(events=[], coverage=[])
    events = {}
    for event in [*previous["events"], *result["snapshot"]["events"]]:
        key = (event["source_id"], event["entity"], event["event_id"], event["revision"])
        if key in events and events[key]["payload_ref"] != event["payload_ref"]:
            raise ValueError("changed document content requires a new revision")
        events.setdefault(key, event)
    coverage = {
        digest(canonical(c)): c for c in [*previous["coverage"], *result["snapshot"]["coverage"]]
    }
    value["signals"] = dict(
        schema_version=1,
        kind="forecast_signal_snapshot",
        events=list(events.values()),
        coverage=list(coverage.values()),
    )
    value["evidence_hex"].update(result["evidence_hex"])
    return ForecastInputs(value).value


def authorize_news_comparison(
    inputs, experiment_id, fold_origins, *, verify_only=False, comparison_owner=None
):
    """Bind exactly one paired fit allowance to the existing news campaign ledger.

    Query changes consume the same search allowance but never mint another fit
    allowance. A completed or failed comparison cannot be renamed to fit again.
    """
    from quant_rsi.core.artifacts import ArtifactStore
    from quant_rsi.core.controller import controller_lock
    from quant_rsi.evaluation.evidence import authority_root

    snapshot = inputs.signals
    if snapshot is None or not snapshot.coverage:
        raise ValueError("news comparison requires registered campaign evidence")
    coverage = [
        c
        for c in snapshot.coverage
        if c.source_id == SOURCE and c.entity == NEWS_BINDING["instrument"]
    ]
    events = [
        e
        for e in snapshot.events
        if e.source_id == SOURCE and e.entity == NEWS_BINDING["instrument"] and e.kind == "news"
    ]
    query_refs = {c.query_ref for c in coverage} | {e.query_ref for e in events}
    if not query_refs:
        raise ValueError("news comparison requires registered campaign evidence")
    plans = [decode(inputs.evidence[ref]) for ref in query_refs]
    campaigns = {p.get("campaign_id") for p in plans}
    if len(campaigns) != 1 or None in campaigns:
        raise ValueError("news comparison requires exactly one campaign")
    campaign = next(iter(campaigns))
    if not isinstance(campaign, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", campaign
    ):
        raise ValueError("invalid campaign identity")
    root = authority_root() / "news_campaigns" / campaign
    if not root.is_dir():
        raise ValueError("news campaign is not registered")
    with controller_lock(root):
        anchor = decode((root / "anchor.json").read_bytes())
        path = root / "news_campaign.json"
        state = decode(path.read_bytes())
        if (
            anchor["root"] != str(root.resolve())
            or anchor["limits"] != state["limits"]
            or anchor["campaign_id"] != campaign
        ):
            raise ValueError("news campaign authority differs")
        if verify_only and "comparison" not in state:
            raise ValueError("news comparison allowance has not been registered")
        store = ArtifactStore(Path(anchor["store_root"]))
        finished = {
            r["query_ref"]: r["result_ref"] for r in state["runs"].values() if r.get("result_ref")
        }
        # First authorization must include all completed searches, not a subset
        # chosen by the input envelope. Later searches must not change replay of
        # the already-frozen comparison or grant another fit allowance.
        if "comparison" in state:
            searches = state.get("comparison_searches")
            if not isinstance(searches, dict) or not searches:
                raise ValueError("news comparison frozen searches are missing")
            if any(finished.get(query) != result for query, result in searches.items()):
                raise ValueError("news comparison frozen searches differ from registered output")
            receipt = state.get("comparison_receipt", {})
            if (
                receipt.get("searches") != searches
                or receipt.get("comparison") != state["comparison"]
                or receipt.get("campaign_id") != campaign
                or receipt.get("limits") != state["limits"]
            ):
                raise ValueError("news comparison receipt differs from frozen campaign state")
        else:
            searches = finished
        if not query_refs <= searches.keys():
            raise ValueError("query is not a completed registered search")
        snapshots = [store.json(ref)["snapshot"] for ref in searches.values()]
        expected_coverage = {digest(canonical(c)) for s in snapshots for c in s["coverage"]}
        actual_coverage = {digest(canonical(c.model_dump(mode="json"))) for c in coverage}
        if actual_coverage != expected_coverage:
            raise ValueError("comparison coverage differs from registered search output")
        expected_events = {}
        for s in snapshots:
            for event in s["events"]:
                key = (event["source_id"], event["entity"], event["event_id"], event["revision"])
                expected_events.setdefault(key, set()).add(digest(canonical(event)))
        actual_events = {
            (e.source_id, e.entity, e.event_id, e.revision): digest(
                canonical(e.model_dump(mode="json"))
            )
            for e in events
        }
        if set(actual_events) != set(expected_events) or any(
            ref not in expected_events[key] for key, ref in actual_events.items()
        ):
            raise ValueError("comparison evidence differs from registered search output")
        binding = dict(
            experiment_id=experiment_id,
            input_ref=inputs.ref,
            fold_origins=[list(fold) for fold in fold_origins],
        )
        if comparison_owner is not None:
            binding["owner"] = comparison_owner
        if "comparison" in state and state["comparison"] != binding:
            raise ValueError("news campaign already bound its single comparison allowance")
        if "comparison" not in state:
            if verify_only:
                raise ValueError("news comparison allowance has not been registered")
            state["comparison"] = binding
            state["comparison_searches"] = searches
            state["comparison_receipt"] = dict(
                campaign_id=campaign,
                comparison=binding,
                searches=searches,
                limits=state["limits"],
                cumulative_usage=state["used"].copy(),
                fit_allowance_policy=(
                    "bounded_factor_discovery"
                    if comparison_owner is not None
                    else "single_paired_comparison"
                ),
            )
            atomic_write(path, canonical(state))
        return state["comparison_receipt"]
