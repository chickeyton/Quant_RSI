"""The latest publication of EIA daily price history, never a historical backfill."""

import fcntl
import html
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical, digest

PRICE_HISTORY_URLS = {
    "wti": "https://www.eia.gov/dnav/pet/hist/RWTCD.htm",
    "brent": "https://www.eia.gov/dnav/pet/hist/rbrted.htm",
}
TITLES = {
    "wti": "Cushing, OK WTI Spot Price FOB (Dollars per Barrel)",
    "brent": "Europe Brent Spot Price FOB (Dollars per Barrel)",
}


def collect_price_publications(fetch, directory, since, through, history_directory=None):
    """Keep dated publications across batches; each task retains its own raw evidence."""
    records, resources, failures = [], [], []
    history = Path(history_directory) / "eia_price_publications" if history_directory else None
    if history:
        history.mkdir(parents=True, exist_ok=True)

    def retain(raw, resource, field):
        record = parse_price_history(raw, field, resource["received_at"])
        released = datetime.fromisoformat(record["released_on"]).date()
        if since - timedelta(days=21) <= released <= through:
            item = resource | {"file": digest(raw) + ".body"}
            atomic_write(Path(directory) / item["file"], raw)
            resources.append(item)
            records.append(record | {"source_url": item["url"], "resource_sha256": item["sha256"]})

    for field, url in PRICE_HISTORY_URLS.items():
        try:
            raw, resource = fetch(url)
            record = parse_price_history(raw, field, resource["received_at"])
            if history:
                with (history / ".lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    receipt = history / f"{field}_{record['released_on']}.json"
                    if receipt.exists():
                        previous = json.loads(receipt.read_bytes())
                        if previous["sha256"] != resource["sha256"]:
                            raise ValueError(
                                "EIA price publication changed; retained original version"
                            )
                    else:
                        item = resource | {"file": resource["sha256"] + ".body"}
                        atomic_write(history / item["file"], raw)
                        atomic_write(receipt, canonical(item))
            else:
                retain(raw, resource, field)
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"url": url, "error_type": type(exc).__name__, "reason": str(exc)})
        if history:
            for receipt in sorted(history.glob(f"{field}_????-??-??.json")):
                try:
                    released = datetime.strptime(
                        receipt.stem.removeprefix(field + "_"), "%Y-%m-%d"
                    ).date()
                    if not since - timedelta(days=21) <= released <= through:
                        continue
                    item = json.loads(receipt.read_bytes())
                    if item["url"] != url or not re.fullmatch(r"[0-9a-f]{64}\.body", item["file"]):
                        raise ValueError("Invalid EIA price publication receipt")
                    raw = (history / item["file"]).read_bytes()
                    if digest(raw) != item["sha256"]:
                        raise ValueError("Retained EIA price publication changed")
                    record = parse_price_history(raw, field, item["received_at"])
                    if record["released_on"] != released.isoformat():
                        raise ValueError("EIA price publication date differs from receipt")
                    retain(raw, item, field)
                except (OSError, ValueError, KeyError) as exc:
                    failures.append(
                        {"url": url, "error_type": type(exc).__name__, "reason": str(exc)}
                    )
    return records, resources, failures


def parse_price_history(raw, field, received_at):
    """All six quotes become visible together after the page's explicit release date."""
    received = datetime.fromisoformat(received_at)
    if received.tzinfo is None:
        raise ValueError("EIA price receipts require a timezone")
    page = raw.decode("utf-8")

    def text(value):
        return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())

    title = re.findall(r"<title>(.*?)</title>", page, re.I | re.S)
    if len(title) != 1 or text(title[0]) != TITLES[field]:
        raise ValueError("EIA price-history series or unit differs from requested input")
    releases = re.findall(
        r"<td\b[^>]*>\s*Release Date:\s*(\d{1,2}/\d{1,2}/\d{4})\s*</td>", page, re.I
    )
    if len(releases) != 1:
        raise ValueError("EIA price history lacks one explicit release date")
    released = datetime.strptime(releases[0], "%m/%d/%Y").date()
    if released > received.astimezone(timezone.utc).date():
        raise ValueError("EIA price-history release is after its receipt")
    observations = {}
    rows = re.findall(
        r"<td\b[^>]*\bclass=['\"]B6['\"][^>]*>(.*?)</td>(.*?)</tr>", page, re.I | re.S
    )
    for label, cells in rows:
        match = re.fullmatch(
            r"(\d{4}) ([A-Z][a-z]{2})-\s*(\d+) to ([A-Z][a-z]{2})-\s*(\d+)", text(label)
        )
        if not match:
            raise ValueError("Unsupported EIA price-history week label")
        start = datetime.strptime(f"{match[1]} {match[2]} {match[3]}", "%Y %b %d").date()
        if start < released - timedelta(days=40):
            continue
        end = start + timedelta(days=4)
        if start.weekday() != 0 or end.strftime("%b") != match[4] or end.day != int(match[5]):
            raise ValueError("EIA price-history week is not Monday through Friday")
        values = re.findall(r"<td\b[^>]*>(.*?)</td>", cells, re.I | re.S)
        if len(values) != 5:
            raise ValueError("EIA price-history week lacks five daily cells")
        for index, cell in enumerate(values):
            value = text(cell)
            if value in {"", "-", "--", "NA", "W"}:
                continue
            observed = start + timedelta(days=index)
            if observed > released or observed in observations:
                raise ValueError("EIA price history contains future or duplicate observations")
            price = float(value.replace(",", ""))
            if not math.isfinite(price) or price <= 0:
                raise ValueError("Invalid EIA daily price")
            observations[observed] = price
    ordered = sorted(observations.items())
    if len(ordered) < 6 or (released - ordered[-1][0]).days > 14:
        raise ValueError("EIA price history lacks six recent daily observations")
    prior, current = ordered[-6], ordered[-1]
    available = datetime.combine(released + timedelta(days=1), datetime.min.time(), timezone.utc)
    available += timedelta(hours=5)
    return {
        "field": field,
        "observation_date": current[0].isoformat(),
        "prior_observation_date": prior[0].isoformat(),
        "released_on": released.isoformat(),
        "available_at": available.isoformat(),
        "value": current[1],
        "prior_value": prior[1],
        "unit": "usd_per_barrel",
        "row_label": TITLES[field],
        "daily_observation_dates": [d.isoformat() for d, _ in ordered[-6:]],
        "source_kind": "current_price_history_publication",
    }
