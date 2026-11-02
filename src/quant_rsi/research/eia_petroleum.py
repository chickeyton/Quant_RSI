"""Retain dated EIA weekly inventory releases for external business hypotheses."""

import csv
import fcntl
import html
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, build_opener

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.market.news_feed import NoRedirect
from quant_rsi.research.eia_price_history import collect_price_publications
from quant_rsi.research.eia_prices import PRICE_FIELDS, parse_prices

ARCHIVE_URL = "https://www.eia.gov/petroleum/supply/weekly/archive/"
MAX_BYTES = 2 * 1024 * 1024
MAX_RELEASES = 104
FIELDS = {
    "commercial_crude": "Commercial (Excluding SPR)",
    "gasoline": "Total Motor Gasoline",
    "distillate": "Distillate Fuel Oil",
}
NOTICE = (
    "EIA archived weekly petroleum inventories, reconstructed from documents retrieved now. "
    "Inventory changes describe US industry stocks, not the target company's inventory. "
    "When linked by a dated archive, Table 11 adds WTI/Brent five-observation daily-price "
    "returns; weekly-average columns are excluded. Current EIA daily-history pages add "
    "their latest six quotes together at that page's publication date, never at each quote's "
    "observation date. Shared study history retains successive publications. "
    "Earlier gaps remain missing. "
    "Date-only releases become usable at 05:00 UTC the following day; the observation week "
    "is not the publication date. Retained documents do not prove their exact historical "
    "version. Values expire 14 days after availability; missing data are not filled with zero."
)


def release_links(raw, since, through):
    """Use the dated official archive links, including holiday-delayed releases."""
    found = {}
    for href in re.findall(r'href=["\']([^"\']+)', raw.decode("utf-8")):
        url = urljoin(ARCHIVE_URL, href)
        match = re.fullmatch(
            re.escape(ARCHIVE_URL) + r"(\d{4})/(\d{4}_\d{2}_\d{2})/wpsr_\2\.php", url
        )
        if not match or match[1] != match[2][:4]:
            continue
        released = date.fromisoformat(match[2].replace("_", "-"))
        if since - timedelta(days=21) <= released <= through:
            found[released] = url
    if not found:
        raise ValueError("No EIA archive releases cover the requested dates")
    if len(found) > MAX_RELEASES:
        raise ValueError("Requested EIA history exceeds the bounded release count")
    return sorted(found.items())


def parse_table(raw, released):
    """Read current/prior week values from Table 4, whose units are million barrels."""
    rows = list(csv.reader(io.StringIO(raw.decode("cp1252"))))
    if not rows or len(rows) > 100 or len(rows[0]) != 8 or rows[0][0] != "STUB_1":
        raise ValueError("Unsupported EIA Table 4 layout")
    current, previous = (datetime.strptime(value, "%m/%d/%y").date() for value in rows[0][1:3])
    if (current - previous).days != 7 or not 0 < (released - current).days <= 14:
        raise ValueError("EIA observation weeks do not match the dated release")
    records = []
    available = datetime.combine(released + timedelta(days=1), datetime.min.time(), timezone.utc)
    available += timedelta(hours=5)
    for field, label in FIELDS.items():
        selected = [row for row in rows[1:] if row and row[0].strip() == label]
        if len(selected) != 1 or len(selected[0]) != 8:
            raise ValueError(f"Missing or duplicate EIA row: {label}")
        current_value, prior_value = (float(v.replace(",", "")) for v in selected[0][1:3])
        if not all(math.isfinite(v) and v > 0 for v in (current_value, prior_value)):
            raise ValueError(f"Invalid EIA stock value: {label}")
        records.append(
            {
                "field": field,
                "observation_date": current.isoformat(),
                "prior_observation_date": previous.isoformat(),
                "released_on": released.isoformat(),
                "available_at": available.isoformat(),
                "value": current_value,
                "prior_value": prior_value,
                "unit": "million_barrels",
                "row_label": label,
                "cells": selected[0][1:3],
            }
        )
    return records


def _page_dates(raw, released):
    page = raw.decode("utf-8")
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", page)).split())
    dates = []
    for label in ("Release Date:", "Data for week ending"):
        match = re.search(re.escape(label) + r"\s+([A-Z][a-z]+ \d{1,2}, \d{4})", text)
        if not match:
            raise ValueError("EIA archive page lacks explicit release/observation dates")
        dates.append(datetime.strptime(match[1], "%B %d, %Y").date())
    if dates[0] != released:
        raise ValueError("EIA release date differs from the archive link")
    return dates


def parse_release_page(raw, page_url, released, table=4):
    """Follow the archived page's actual CSV link; older archives use a different path."""
    dates = _page_dates(raw, released)
    page = raw.decode("utf-8")
    allowed = {
        urljoin(page_url, path) for path in (f"download/table{table}.csv", f"csv/table{table}.csv")
    }
    links = {
        urljoin(page_url, href)
        for href in re.findall(r'href=["\']([^"\']+)', page)
        if urljoin(page_url, href) in allowed
    }
    if not links and table == 11:
        return None, dates[1]
    if len(links) != 1:
        raise ValueError(f"EIA archive page lacks one unambiguous Table {table} CSV link")
    return links.pop(), dates[1]


def _download(url):
    request = Request(
        url, headers={"User-Agent": "QuantRSI research (github.com/natureofnature/Quant_RSI)"}
    )
    with build_opener(NoRedirect).open(request, timeout=30) as response:
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("EIA resource exceeds the collection bound")
    return raw


def collect_petroleum(directory, since, through, *, cache_directory=None, history_directory=None):
    """Reuse retained inputs; fresh collection never overwrites an existing snapshot."""
    directory = Path(directory)
    since, through = date.fromisoformat(since), date.fromisoformat(through)
    if since > through or through > datetime.now(timezone.utc).date():
        raise ValueError("Invalid EIA collection date interval")
    bounds = {"since": since.isoformat(), "through": through.isoformat()}
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        if (
            snapshot["provider"] != "eia_weekly_petroleum"
            or snapshot["bounds"] != bounds
            or digest(canonical(snapshot["records"])) != snapshot["sha256"]
        ):
            raise ValueError("Retained EIA snapshot identity changed")
        for resource in snapshot["resources"]:
            if not re.fullmatch(r"[0-9a-f]{64}\.body", resource["file"]):
                raise ValueError("Invalid retained EIA resource path")
            if digest((directory / resource["file"]).read_bytes()) != resource["sha256"]:
                raise ValueError("Retained EIA resource changed")
        return snapshot
    resources, failures = [], []
    shared = (
        Path(cache_directory) / f"{since.isoformat()}_{through.isoformat()}"
        if cache_directory is not None
        else directory
    )
    shared.mkdir(parents=True, exist_ok=True)

    def fetch(url, retained_resources):
        identity = digest(url.encode())
        # Keep the bytes/receipt pair consistent for concurrent cache readers and writers.
        with (shared / f".{identity}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            local = directory / f"{identity}.json"
            storage = directory if local.exists() else shared
            metadata = storage / f"{identity}.json"
            body = storage / f"{identity}.body"
            if metadata.exists():
                item = json.loads(metadata.read_bytes())
                raw = body.read_bytes()
                if item["url"] != url or item["file"] != body.name or item["sha256"] != digest(raw):
                    raise ValueError("Cached EIA resource changed")
            else:
                raw = _download(url)
                item = {
                    "url": url,
                    "file": body.name,
                    "sha256": digest(raw),
                    "received_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_write(body, raw)
                atomic_write(metadata, canonical(item))
            if storage != directory:
                # Every task stays independently replayable if the batch cache is removed.
                atomic_write(directory / body.name, raw)
                atomic_write(local, canonical(item))
        retained_resources.append(item)
        return raw, item

    links = []
    try:
        index, _ = fetch(ARCHIVE_URL, resources)
    except OSError as exc:
        failures.append({"url": ARCHIVE_URL, "error_type": type(exc).__name__, "reason": str(exc)})
    else:
        try:
            links = release_links(index, since, through)
        except ValueError as exc:
            failures.append(
                {"url": ARCHIVE_URL, "error_type": type(exc).__name__, "reason": str(exc)}
            )

    def collect_release(link):
        released, page_url = link
        records, retained_resources, release_failures = [], [], []
        url = page_url
        try:
            page, _ = fetch(page_url, retained_resources)
            _page_dates(page, released)
        except (ValueError, OSError) as exc:
            release_failures.append(
                {"url": url, "error_type": type(exc).__name__, "reason": str(exc)}
            )
            return records, retained_resources, release_failures
        for table in (4, 11):
            url = page_url
            try:
                url, observation = parse_release_page(page, page_url, released, table)
                if url is None:
                    continue
                raw, resource = fetch(url, retained_resources)
                parsed = (
                    parse_table(raw, released)
                    if table == 4
                    else parse_prices(raw, released, observation)
                )
                if table == 4 and any(
                    record["observation_date"] != observation.isoformat() for record in parsed
                ):
                    raise ValueError("EIA table observation differs from its archive page")
                records.extend(
                    record | {"source_url": url, "resource_sha256": resource["sha256"]}
                    for record in parsed
                )
            except (ValueError, OSError) as exc:
                release_failures.append(
                    {"url": url, "error_type": type(exc).__name__, "reason": str(exc)}
                )
        return records, retained_resources, release_failures

    records = []
    # map preserves archive order even when later releases finish first.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for rows, items, errors in executor.map(collect_release, links):
            records.extend(rows)
            resources.extend(items)
            failures.extend(errors)
    prices, publications, errors = collect_price_publications(
        lambda url: fetch(url, resources), directory, since, through, history_directory
    )
    records.extend(prices)
    resources.extend(publications)
    failures.extend(errors)
    snapshot = {
        "provider": "eia_weekly_petroleum",
        "bounds": bounds,
        "records": records,
        "sha256": digest(canonical(records)),
        "resources": resources,
        "failures": failures,
        "notice": NOTICE,
        "price_tables": "archived_and_retained_daily_price_publications",
    }
    atomic_write(retained, canonical(snapshot))
    return snapshot


def prepare_petroleum(snapshot, cutoffs, required):
    """Expose only published, non-stale weekly changes on the existing fixed rows."""
    columns, definitions, evidence = {}, [], []
    records = snapshot["records"]
    for cutoff in cutoffs:
        cutoff = datetime.fromisoformat(cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("External factor cutoffs require timezones")
        selected = {}
        for record in records:
            available = datetime.fromisoformat(record["available_at"])
            if timedelta(0) <= cutoff - available <= timedelta(days=14):
                field = record["field"]
                if field not in selected or (
                    record["available_at"],
                    record.get("observation_date", ""),
                ) > (
                    selected[field]["available_at"],
                    selected[field].get("observation_date", ""),
                ):
                    selected[field] = record
        evidence.append(selected)
    fields = dict(FIELDS)
    if snapshot.get("price_tables"):
        fields.update(PRICE_FIELDS)
    for field, label in fields.items():
        price = field in PRICE_FIELDS
        metric = f"eia_{field}_5obs_return" if price else f"eia_{field}_wow"
        identity = f"external.{metric}"
        values = [
            row[field]["value"] / row[field]["prior_value"] - 1 if field in row else None
            for row in evidence
        ]
        missing = sum(values[int(i)] is None for i in required)
        columns[identity] = values
        definitions.append(
            {
                "factor_id": identity,
                "kind": "numeric",
                "input_kind": (
                    "publication_dated_oil_prices" if price else "publication_dated_industry_stocks"
                ),
                "description": (
                    f"EIA {label}: latest daily quote / fifth preceding daily observation - 1; "
                    "available after publication, not on the quoted date."
                    if price
                    else f"EIA US {label}: current / prior week inventory - 1."
                ),
                "recipe": {"op": "external", "metric": metric},
                "runnable": missing == 0,
                "missing_required_rows": missing,
                "source": {"provider": snapshot["provider"], "snapshot_sha256": snapshot["sha256"]},
                "reason": "available" if missing == 0 else "missing_or_stale_publication",
            }
        )
    return {
        "provider": snapshot["provider"],
        "definitions": definitions,
        "columns": columns,
        "evidence": evidence,
        "snapshot_sha256": snapshot["sha256"],
        "failures": snapshot["failures"],
        "notice": snapshot["notice"],
    }


def simulated_petroleum(cutoffs, required):
    """Explicit offline fixture for the same connected workflow, never market evidence."""
    release = datetime.fromisoformat(cutoffs[0]) - timedelta(days=14)
    end = datetime.fromisoformat(cutoffs[-1])
    records, week = [], 0
    while release < end:
        for index, field in enumerate(FIELDS | PRICE_FIELDS):
            records.append(
                {
                    "field": field,
                    "available_at": release.isoformat(),
                    "value": 100 + (week * (index + 1)) % 17,
                    "prior_value": 100,
                }
            )
        release += timedelta(days=7)
        week += 1
    return prepare_petroleum(
        {
            "records": records,
            "provider": "simulated_petroleum",
            "sha256": digest(canonical(records)),
            "failures": [],
            "notice": (
                "SIMULATED industry inventories/prices; engineering fixture, not market evidence."
            ),
            "price_tables": "simulated_daily_prices",
        },
        cutoffs,
        required,
    )
