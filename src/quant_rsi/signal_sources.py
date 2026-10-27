"""Bounded public signals; no registry writes, fitting, or historical PIT certification.

collect_signals accepts schema_version=1, kind="forecast_signal_collection",
start_at/end_at (aware, half-open), and sources:
  {"source_id": "nvidia_newsroom_rss", "entity": "NVDA"}
  {"source_id": "sec_companyfacts", "entity": "NVDA", "cik": "0001045810",
   "concepts": [{"taxonomy": "us-gaap", "concept": "Revenues", "unit": "USD"}]}
RSS uses its explicit publication precision; undated items are retained as unknown.
SEC queries use accession acceptance time, or intersecting filed dates when absent.
SEC acceptance/filing are metadata, not claimed publication/availability clocks.
Every first_seen_at is the actual response receipt time, never a historical date.
"""

import json
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from quant_rsi.artifacts import canonical, digest, reject_constant
from quant_rsi.forecast_inputs import SignalEvent, SignalSnapshot
from quant_rsi.market_contracts import timestamp

RSS_URL = "https://nvidianews.nvidia.com/rss.xml"
SEC_BASE = "https://data.sec.gov"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 15
EXTRACTOR_VERSION = "public_signals_v1"
QUALIFICATIONS = [
    "Historical retrieval is not observed historical PIT availability.",
    "SEC acceptance and filed dates do not prove the first public dissemination instant.",
    "RSS summaries only; article bodies and complete historical news are not collected.",
    "No cross-collection revision reconciliation; immutable accession/item variants are retained.",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _spec(value):
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "kind",
        "start_at",
        "end_at",
        "sources",
    }:
        raise ValueError("invalid signal collection spec fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported signal collection schema")
    if value["kind"] != "forecast_signal_collection":
        raise ValueError("unsupported signal collection kind")
    start, end = timestamp(value["start_at"]), timestamp(value["end_at"])
    if start >= end:
        raise ValueError("query interval must be nonempty")
    sources = value["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 2:
        raise ValueError("one or two supported sources are required")
    seen, normalized = set(), []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source must be an object")
        sid, entity = source.get("source_id"), source.get("entity")
        if sid in seen:
            raise ValueError("duplicate source")
        if not isinstance(entity, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", entity
        ):
            raise ValueError("invalid entity")
        seen.add(sid)
        if sid == "nvidia_newsroom_rss":
            if set(source) != {"source_id", "entity"} or entity != "NVDA":
                raise ValueError("NVIDIA RSS requires entity NVDA")
            normalized.append(dict(source))
        elif sid == "sec_companyfacts":
            if set(source) != {"source_id", "entity", "cik", "concepts"}:
                raise ValueError("invalid SEC source fields")
            cik = source["cik"]
            if not isinstance(cik, str) or not re.fullmatch(r"[0-9]{1,10}", cik) or int(cik) == 0:
                raise ValueError("CIK must be one to ten digits, nonzero")
            concepts = source["concepts"]
            if not isinstance(concepts, list) or not 1 <= len(concepts) <= 8:
                raise ValueError("one to eight explicit concepts are required")
            concept_keys = set()
            for concept in concepts:
                if not isinstance(concept, dict) or set(concept) != {"taxonomy", "concept", "unit"}:
                    raise ValueError("invalid concept fields")
                for key in ("taxonomy", "concept"):
                    if not isinstance(concept[key], str) or not re.fullmatch(
                        r"[A-Za-z][A-Za-z0-9_-]{0,127}", concept[key]
                    ):
                        raise ValueError("invalid taxonomy/concept")
                if not isinstance(concept["unit"], str) or not 1 <= len(concept["unit"]) <= 80:
                    raise ValueError("invalid concept unit")
                key = (concept["taxonomy"], concept["concept"], concept["unit"])
                if key in concept_keys:
                    raise ValueError("duplicate concept/unit")
                concept_keys.add(key)
            normalized.append({**source, "cik": cik.zfill(10)})
        else:
            raise ValueError("unsupported signal source")
    return {
        **value,
        "start_at": start.isoformat(),
        "end_at": end.isoformat(),
        "sources": normalized,
    }


def _put(out, name, body, evidence):
    path = out / name
    with path.open("xb") as stream:
        stream.write(body)
    ref = digest(body)
    evidence.setdefault(ref, str(path.absolute()))
    return ref


def _json(out, name, value, evidence):
    return _put(out, name, canonical(value), evidence)


def _fetch(url, out, name, records, evidence):
    record = {"url": url, "requested_at": _now()}
    response, body, error = None, b"", None
    try:
        try:
            response = urlopen(
                Request(url, headers={"User-Agent": "Mozilla/5.0 QuantRSI public-signal research"}),
                timeout=TIMEOUT_SECONDS,
            )
        except HTTPError as exc:
            response = exc
        with response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            record.update(
                response_url=response.geturl(),
                http_status=response.status,
                content_type=response.headers.get("Content-Type"),
                truncated=len(body) > MAX_RESPONSE_BYTES,
            )
        record["raw_ref"] = _put(out, name, body, evidence)
        if record["truncated"]:
            raise ValueError("response exceeds 8 MiB; bounded prefix retained")
        if record["http_status"] != 200:
            raise ValueError(f"HTTP {record['http_status']}; response retained")
        if urlparse(record["response_url"]).hostname != urlparse(url).hostname:
            raise ValueError("unexpected response host")
    except Exception as exc:
        error = exc
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["received_at"] = _now()
        record["bytes"] = len(body)
        records.append(record)
    if error is not None:
        raise error
    return body


def _day_overlaps(day, start, end):
    boundary = datetime.combine(date.fromisoformat(day), time.min, timezone.utc)
    # A date without a timezone cannot be treated as midnight UTC. Retain any
    # overlap under civil offsets UTC-12..UTC+14; do not invent publication time.
    return boundary - timedelta(hours=14) < end and boundary + timedelta(days=1, hours=12) > start


def _publication(raw):
    raw = raw.strip()
    if not raw:
        return None, None, "unknown"
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", raw):
        return None, date.fromisoformat(raw).isoformat(), "day"
    try:
        value = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None, None, "unknown"
    if value.tzinfo is None:
        return None, None, "unknown"
    return value.astimezone(timezone.utc).isoformat(), None, "second"


def _event(source, query_ref, receipt_ref, seen, event_id, payload_ref, **values):
    event = dict(
        schema_version=1,
        source_id=source["source_id"],
        entity=source["entity"],
        event_id=event_id,
        revision=0,
        query_ref=query_ref,
        first_seen_at=seen,
        receipt_ref=receipt_ref,
        payload_ref=payload_ref,
        extractor_version=EXTRACTOR_VERSION,
        source_available_at=None,
        **values,
    )
    return SignalEvent.model_validate_json(canonical(event)).model_dump(mode="json")


def _rss(body, source, query, query_ref, receipt_ref, seen, raw_refs, out, evidence):
    if re.search(rb"<!DOCTYPE|<!ENTITY", body, re.IGNORECASE):
        raise ValueError("RSS document declarations/entities are unsupported")
    channel = ET.fromstring(body).find("channel")
    if channel is None:
        raise ValueError("expected RSS channel")
    items = channel.findall("item")
    if len(items) > 1000:
        raise ValueError("RSS item bound exceeded")
    start, end = timestamp(query["start_at"]), timestamp(query["end_at"])
    events, identities = [], {}
    for item in items:
        fields = {
            key: item.findtext(key, "").strip()
            for key in ("guid", "link", "title", "description", "pubDate")
        }
        identity = fields["guid"] or fields["link"]
        if not identity:
            raise ValueError("RSS item has no GUID or link")
        if fields["link"] and (
            urlparse(fields["link"]).scheme != "https"
            or urlparse(fields["link"]).hostname
            not in {"nvidianews.nvidia.com", "blogs.nvidia.com"}
        ):
            raise ValueError("RSS item links outside supported official NVIDIA hosts")
        published, day, precision = _publication(fields["pubDate"])
        if published and not start <= timestamp(published) < end:
            continue
        if day and not _day_overlaps(day, start, end):
            continue
        content = canonical({"item": fields, "raw_refs": raw_refs})
        # A source GUID repeated with conflicting content in one response is ambiguous.
        item_ref = digest(canonical(fields))
        if identity in identities:
            if identities[identity] != item_ref:
                raise ValueError("conflicting duplicate RSS item")
            continue
        identities[identity] = item_ref
        payload_ref = _put(out, f"news-{item_ref}.json", content, evidence)
        events.append(
            _event(
                source,
                query_ref,
                receipt_ref,
                seen,
                "news-" + item_ref,
                payload_ref,
                kind="news",
                published_at=published,
                published_date=day,
                publication_precision=precision,
                report_period_end=None,
                fields=fields,
                units={
                    key: {"guid": "identifier", "link": "url"}.get(key, "text") for key in fields
                },
            )
        )
    return (
        events,
        "partial",
        "rolling RSS window; historical completeness and undated-item times unknown",
    )


def _sec(bodies, source, query, query_ref, receipt_ref, seen, raw_refs, out, evidence):
    submissions, facts = [
        json.loads(b, parse_float=Decimal, parse_constant=reject_constant) for b in bodies
    ]
    if any(str(obj.get("cik", "")).zfill(10) != source["cik"] for obj in (submissions, facts)):
        raise ValueError("SEC CIK mismatch")
    if source["entity"] not in submissions.get("tickers", []):
        raise ValueError("SEC requested entity is not a reported ticker")
    recent = submissions["filings"]["recent"]
    columns = ["accessionNumber", "acceptanceDateTime", "filingDate", "form", "primaryDocument"]
    if any(not isinstance(recent.get(c), list) for c in columns):
        raise ValueError("invalid submissions columns")
    if len({len(recent[c]) for c in columns}) != 1:
        raise ValueError("submissions column length mismatch")
    by_accession = {}
    for row in zip(*(recent[c] for c in columns), strict=True):
        record = dict(zip(columns, row, strict=True))
        accession = record["accessionNumber"]
        if accession in by_accession and by_accession[accession] != record:
            raise ValueError("conflicting accession records")
        by_accession[accession] = record
    start, end = timestamp(query["start_at"]), timestamp(query["end_at"])
    events, dedup, missing, unmatched = [], set(), [], set()
    for requested in source["concepts"]:
        taxonomy, concept, unit = (requested[k] for k in ("taxonomy", "concept", "unit"))
        values = (
            facts.get("facts", {}).get(taxonomy, {}).get(concept, {}).get("units", {}).get(unit)
        )
        if values is None:
            missing.append(f"{taxonomy}:{concept}/{unit}")
            continue
        if not isinstance(values, list):
            raise ValueError("invalid companyfacts records")
        for fact in values:
            accession, filed = fact["accn"], date.fromisoformat(fact["filed"]).isoformat()
            period_end = date.fromisoformat(fact["end"]).isoformat()
            if fact.get("start") is not None:
                if date.fromisoformat(fact["start"]) > date.fromisoformat(period_end):
                    raise ValueError("inverted fact report period")
            if not re.fullmatch(r"[0-9]{10}-[0-9]{2}-[0-9]{6}", accession):
                raise ValueError("invalid accession")
            submission = by_accession.get(accession)
            accepted = submission.get("acceptanceDateTime") if submission else None
            if accepted:
                accepted = timestamp(accepted).isoformat()
                if not start <= timestamp(accepted) < end:
                    continue
            elif not _day_overlaps(filed, start, end):
                continue
            if submission and (
                submission["filingDate"] != filed or submission["form"] != fact["form"]
            ):
                raise ValueError("fact and submission metadata disagree")
            if not accepted:
                unmatched.add(accession)
            value = fact["val"]
            if type(value) is int:
                value = str(value)
            elif isinstance(value, Decimal) and value.is_finite():
                value = format(value, "f")
            else:
                raise ValueError("fundamental value must be a finite JSON number")
            normalized = {**fact, "val": value}
            item = {"requested": requested, "fact": normalized}
            item_ref = digest(canonical(item))
            if item_ref in dedup:
                continue
            dedup.add(item_ref)
            payload_ref = _json(
                out,
                f"fact-{item_ref}.json",
                {**item, "submission": submission, "raw_refs": raw_refs},
                evidence,
            )
            fields = {
                "value": value,
                "taxonomy": taxonomy,
                "concept": concept,
                "accession": accession,
                "filed_date": filed,
                "form": fact["form"],
            }
            units = {key: "identifier" for key in fields}
            units.update(value=unit, filed_date="date")
            if accepted:
                fields.update(acceptance_at=accepted, acceptance_clock_kind="edgar_acceptance")
                units.update(acceptance_at="timestamp", acceptance_clock_kind="identifier")
            for source_key, field, field_unit in [
                ("start", "report_period_start", "date"),
                ("fy", "fiscal_year", "year"),
                ("fp", "fiscal_period", "identifier"),
                ("frame", "frame", "identifier"),
            ]:
                optional = fact.get(source_key)
                if optional is None or optional == "":
                    continue
                expected = int if source_key == "fy" else str
                if type(optional) is not expected:
                    raise ValueError(f"invalid optional SEC {source_key}")
                fields[field] = str(optional)
                units[field] = field_unit
            events.append(
                _event(
                    source,
                    query_ref,
                    receipt_ref,
                    seen,
                    "sec-" + item_ref,
                    payload_ref,
                    kind="fundamental",
                    published_at=None,
                    published_date=None,
                    publication_precision="unknown",
                    report_period_end=period_end,
                    fields=fields,
                    units=units,
                )
            )
    reason = (
        "selected SEC concepts/backfill only; archive pagination and dissemination not certified"
    )
    if missing:
        reason += "; missing concept/units: " + ", ".join(missing)
    if unmatched:
        reason += f"; {len(unmatched)} accessions lack a joined acceptance timestamp"
    return events, "partial", reason


def collect_signals(spec, output_dir):
    """Collect one immutable response batch; source failures yield explicit unknown coverage."""
    query = _spec(spec)
    out = Path(output_dir).absolute()
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise FileExistsError("signal collection requires a fresh or empty directory")
    out.mkdir(parents=True, exist_ok=True)
    evidence, events, coverage, qualifications, source_receipts = (
        {},
        [],
        [],
        list(QUALIFICATIONS),
        [],
    )
    query_ref = _json(out, "query.json", query, evidence)
    for source in query["sources"]:
        sid, records, bodies, error = source["source_id"], [], [], None
        urls = (
            [RSS_URL]
            if sid == "nvidia_newsroom_rss"
            else [
                f"{SEC_BASE}/submissions/CIK{source['cik']}.json",
                f"{SEC_BASE}/api/xbrl/companyfacts/CIK{source['cik']}.json",
            ]
        )
        try:
            for i, url in enumerate(urls):
                bodies.append(_fetch(url, out, f"{sid}-{i}.raw", records, evidence))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        # Freeze response receipt before deriving any event; no self-reference or event hash.
        seen = max(record["received_at"] for record in records)
        receipt_ref = _json(
            out,
            f"{sid}.receipt.json",
            {
                "schema_version": 1,
                "kind": "signal_response_receipt",
                "query_ref": query_ref,
                "source": source,
                "responses": records,
                "first_seen_at": seen,
                "error": error,
            },
            evidence,
        )
        source_receipts.append(receipt_ref)
        status, reason, parsed = "unknown", error, []
        if error is None:
            raw_refs = [record["raw_ref"] for record in records]
            try:
                parser = _rss if sid == "nvidia_newsroom_rss" else _sec
                parsed, status, reason = parser(
                    bodies[0] if sid == "nvidia_newsroom_rss" else bodies,
                    source,
                    query,
                    query_ref,
                    receipt_ref,
                    seen,
                    raw_refs,
                    out,
                    evidence,
                )
            except Exception as exc:
                reason = f"parse rejected: {type(exc).__name__}: {exc}"
        if status == "unknown":
            qualifications.append(f"{sid}: {reason}")
        events.extend(parsed)
        coverage.append(
            dict(
                schema_version=1,
                source_id=sid,
                entity=source["entity"],
                query_ref=query_ref,
                start_at=query["start_at"],
                end_at=query["end_at"],
                first_seen_at=seen,
                status=status,
                reason=reason,
                receipt_ref=receipt_ref,
            )
        )
    snapshot = dict(
        schema_version=1, kind="forecast_signal_snapshot", events=events, coverage=coverage
    )
    snapshot = SignalSnapshot.model_validate_json(canonical(snapshot)).model_dump(mode="json")
    snapshot_ref = _json(out, "snapshot.json", snapshot, evidence)
    _json(
        out,
        "collection.receipt.json",
        {
            "schema_version": 1,
            "kind": "signal_collection_receipt",
            "query_ref": query_ref,
            "snapshot_ref": snapshot_ref,
            "response_receipts": source_receipts,
            "completed_at": _now(),
            "qualifications": qualifications,
        },
        evidence,
    )
    return dict(
        snapshot=snapshot,
        evidence_paths=evidence,
        receipt_path=str(out / "collection.receipt.json"),
        qualifications=qualifications,
    )
