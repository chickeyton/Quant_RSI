"""Shared quarterly business features from retained, publication-dated public facts."""

import json
import math
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from quant_rsi.core.artifacts import atomic_write, canonical, digest

NOTICE = (
    "Retrospective reconstruction from currently retained SEC facts, not a complete historical "
    "vendor archive. A filing date without an acceptance time becomes usable at 05:00 UTC on "
    "the following day. Only directly reported 70–110 day quarters in USD are used; no annual "
    "or year-to-date values are silently converted. Revisions are selected as of each cutoff."
)
METRICS = {
    "revenue_yoy": "Quarterly revenue / comparable prior-year revenue - 1",
    "operating_margin": "Quarterly operating income / revenue",
    "operating_margin_yoy": "Quarterly operating margin minus comparable prior-year margin",
}


def collect_financials(symbol, directory, factor_root, source="auto"):
    if source not in {"auto", "sec", "issuer"}:
        raise ValueError("Unknown financial data source")
    directory = Path(directory)
    profile_path = Path(factor_root) / "US" / symbol / "financials.json"
    if not profile_path.exists():
        raise ValueError(f"No financial data profile for {symbol}")
    profile = json.loads(profile_path.read_bytes())
    if profile["symbol"] != symbol or not str(profile["cik"]).isdigit():
        raise ValueError("Financial profile identity mismatch")
    retained = directory / "snapshot.json"
    if retained.exists():
        snapshot = json.loads(retained.read_bytes())
        payload = (
            snapshot.get("records")
            if snapshot["provider"] == "issuer_earnings"
            else snapshot["companyfacts"]
        )
        if (
            snapshot["symbol"] != symbol
            or snapshot["profile"] != profile
            or (
                snapshot["provider"] != "issuer_earnings"
                and int(snapshot["companyfacts"]["cik"]) != int(profile["cik"])
            )
            or digest(canonical(payload)) != snapshot["sha256"]
        ):
            raise ValueError("Retained financial snapshot identity changed")
        if (
            source != "auto"
            and snapshot["provider"]
            != {"issuer": "issuer_earnings", "sec": "sec_companyfacts"}[source]
        ):
            raise ValueError("Retained financial provider differs from requested source")
        for resource in snapshot.get("resources", []):
            if (
                Path(resource["file"]).name != resource["file"]
                or digest((directory / resource["file"]).read_bytes()) != resource["sha256"]
            ):
                raise ValueError("Retained financial document identity changed")
        return snapshot
    attempts = []
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = None
    if source in {"auto", "sec"}:
        try:
            snapshot = collect_sec(symbol, profile)
            attempts.append({"provider": "sec_companyfacts", "status": "available"})
        except (OSError, ValueError, KeyError) as exc:
            attempts.append(
                {"provider": "sec_companyfacts", "status": "unavailable", "reason": str(exc)}
            )
            atomic_write(directory / "source_attempts.json", canonical(attempts))
            if source == "sec":
                raise
    if snapshot is None:
        from quant_rsi.research.issuer_financials import collect_issuer

        if "issuer" not in profile:
            raise ValueError("Financial profile has no official issuer source")
        try:
            snapshot = collect_issuer(symbol, directory, profile)
        except (OSError, ValueError, KeyError) as exc:
            attempts.append(
                {"provider": "issuer_earnings", "status": "unavailable", "reason": str(exc)}
            )
            atomic_write(directory / "source_attempts.json", canonical(attempts))
            raise
        attempts.append({"provider": "issuer_earnings", "status": "available"})
    snapshot["source_attempts"] = attempts
    atomic_write(directory / "source_attempts.json", canonical(attempts))
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write(retained, canonical(snapshot))
    return snapshot


def collect_sec(symbol, profile):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(profile['cik']):010d}.json"
    request = Request(
        url,
        headers={
            "User-Agent": os.environ.get(
                "SEC_USER_AGENT", "QuantRSI/0.1 https://github.com/natureofnature/Quant_RSI"
            ),
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=25) as response:
        raw = response.read(20 * 1024 * 1024 + 1)
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError("SEC company facts exceeded the 20 MiB limit")
    facts = json.loads(raw)
    if int(facts["cik"]) != int(profile["cik"]):
        raise ValueError("SEC company identity mismatch")
    return {
        "symbol": symbol,
        "provider": "sec_companyfacts",
        "url": url,
        "profile": profile,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "companyfacts": facts,
        "sha256": digest(canonical(facts)),
        "notice": NOTICE,
    }


def quarterly_records(snapshot):
    if snapshot["provider"] == "issuer_earnings":
        return snapshot["records"]
    records = []
    for field, tags in snapshot["profile"]["fields"].items():
        for priority, tag in enumerate(tags):
            for fact in (
                snapshot["companyfacts"]
                .get("facts", {})
                .get("us-gaap", {})
                .get(tag, {})
                .get("units", {})
                .get("USD", [])
            ):
                if (
                    fact.get("form") not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}
                    or "start" not in fact
                ):
                    continue
                start, end, filed = (date.fromisoformat(fact[k]) for k in ("start", "end", "filed"))
                value = fact.get("val")
                if (
                    not 70 <= (end - start).days <= 110
                    or end > filed
                    or type(value) not in {int, float}
                    or not math.isfinite(value)
                ):
                    continue
                available = datetime.combine(
                    filed + timedelta(days=1), datetime.min.time(), timezone.utc
                ) + timedelta(hours=5)
                records.append(
                    {
                        "field": field,
                        "tag": tag,
                        "priority": priority,
                        "start": str(start),
                        "end": str(end),
                        "filed": str(filed),
                        "available_at": available.isoformat(),
                        "accession": fact["accn"],
                        "value": float(value),
                        "unit": "USD",
                    }
                )
    return records


def at_cutoff(records, cutoff):
    instant = datetime.fromisoformat(cutoff)
    if instant.tzinfo is None:
        raise ValueError("Financial cutoffs require a timezone")
    visible = [r for r in records if datetime.fromisoformat(r["available_at"]) <= instant]
    periods = {}
    for row in visible:
        periods.setdefault((row["start"], row["end"]), {}).setdefault(row["field"], []).append(row)
    quarters = []
    for (start, end), fields in periods.items():
        selected = {}
        for field, versions in fields.items():
            rank = max((r["available_at"], -r["priority"]) for r in versions)
            choices = [r for r in versions if (r["available_at"], -r["priority"]) == rank]
            if len({r["value"] for r in choices}) != 1:
                continue
            selected[field] = sorted(
                choices, key=lambda r: r.get("statement_id", r.get("accession"))
            )[-1]
        if "revenue" in selected and selected["revenue"]["value"] > 0:
            quarters.append({"start": start, "end": end, "facts": selected})
    if not quarters:
        return {}, {"reason": "no_visible_reported_quarter", "cutoff": cutoff}
    latest = max(quarters, key=lambda q: q["end"])
    if (instant.date() - date.fromisoformat(latest["end"])).days > 270:
        return {}, {"reason": "latest_reported_quarter_stale", "cutoff": cutoff}
    previous = [
        q
        for q in quarters
        if 350 <= (date.fromisoformat(latest["end"]) - date.fromisoformat(q["end"])).days <= 380
        and (
            latest["start"] is None
            or q["start"] is None
            or abs(
                (date.fromisoformat(latest["start"]) - date.fromisoformat(q["start"])).days - 365
            )
            <= 15
        )
    ]
    prior = max(previous, key=lambda q: q["end"]) if previous else None

    def margin(quarter):
        facts = quarter["facts"]
        income, revenue = facts.get("operating_income"), facts["revenue"]
        if income and income.get("statement_id", income.get("accession")) == revenue.get(
            "statement_id", revenue.get("accession")
        ):
            return income["value"] / revenue["value"]
        return None

    values = {}
    current_margin = margin(latest)
    if current_margin is not None:
        values["operating_margin"] = current_margin
    if prior:
        values["revenue_yoy"] = (
            latest["facts"]["revenue"]["value"] / prior["facts"]["revenue"]["value"] - 1
        )
        previous_margin = margin(prior)
        if current_margin is not None and previous_margin is not None:
            values["operating_margin_yoy"] = current_margin - previous_margin
    return values, {"cutoff": cutoff, "current": latest, "prior_year": prior, "values": values}


def prepare_financials(snapshot, cutoffs, required):
    records = quarterly_records(snapshot)
    rows = [at_cutoff(records, cutoff) for cutoff in cutoffs]
    columns, definitions = {}, []
    for metric, description in METRICS.items():
        identity = "financial." + metric
        values = [value.get(metric) for value, _ in rows]
        missing = sum(values[int(i)] is None for i in required)
        columns[identity] = values
        definitions.append(
            {
                "factor_id": identity,
                "kind": "numeric",
                "input_kind": "publication_dated_financials",
                "description": description,
                "recipe": {"op": "financial", "metric": metric},
                "runnable": missing == 0,
                "missing_required_rows": missing,
                "source": {"symbol": snapshot["symbol"], "snapshot_sha256": snapshot["sha256"]},
                "reason": "available"
                if missing == 0
                else "missing_or_incompatible_reported_quarters",
            }
        )
    return {
        "definitions": definitions,
        "columns": columns,
        "evidence": [e for _, e in rows],
        "notice": snapshot["notice"],
        "snapshot_sha256": snapshot["sha256"],
        "provider": snapshot["provider"],
        "source_attempts": snapshot.get("source_attempts", []),
        "source_failures": snapshot.get("failures", []),
        "coverage": {
            "reported_periods": sorted({r["end"] for r in records}),
            "first_available_at": min((r["available_at"] for r in records), default=None),
            "last_available_at": max((r["available_at"] for r in records), default=None),
            "required_rows": len(required),
            "runnable_factors": sum(d["runnable"] for d in definitions),
        },
    }


def simulated_financials(symbol, cutoffs, required):
    # Explicit fixture mode exercises the same filing-date parser and feature computation.
    last = datetime.fromisoformat(cutoffs[-1]).date()
    facts = {"cik": 0, "facts": {"us-gaap": {}}}
    for tag, scale in (("Revenues", 1000), ("OperatingIncomeLoss", 180)):
        rows = []
        for year in range(last.year - 3, last.year + 1):
            for month in (3, 6, 9, 12):
                end = date(year, month, 28)
                start = date(year, month - 2, 1)
                rows.append(
                    {
                        "start": str(start),
                        "end": str(end),
                        "filed": str(end + timedelta(days=25)),
                        "form": "10-Q",
                        "accn": f"fixture-{year}-{month}",
                        "val": scale * (1 + 0.08 * (year - last.year + 3) + 0.01 * month),
                    }
                )
        facts["facts"]["us-gaap"][tag] = {"units": {"USD": rows}}
    snapshot = {
        "symbol": symbol,
        "provider": "simulated_financials",
        "profile": {
            "fields": {"revenue": ["Revenues"], "operating_income": ["OperatingIncomeLoss"]}
        },
        "companyfacts": facts,
        "sha256": digest(canonical(facts)),
        "notice": "SIMULATED financial statements; engineering fixture only. " + NOTICE,
    }
    return prepare_financials(snapshot, cutoffs, required)
