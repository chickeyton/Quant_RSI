"""Shared quarterly business features from retained, publication-dated public facts."""

import json
import math
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from quant_rsi.core.artifacts import atomic_write, canonical, digest

NOTICE = (
    "Retrospective reconstruction from currently retained SEC facts, not a complete historical "
    "vendor archive. A filing date without an acceptance time becomes usable at 05:00 UTC on "
    "the following day. Only directly reported 70–110 day quarters in USD are used; no annual "
    "income values are silently converted. Cash-flow facts retain their reported 3/6/9/12-month "
    "durations and compare only matching year-over-year periods. Revisions are selected "
    "as of each cutoff."
    " Inventory uses reported balance-sheet instants and comparable prior-year dates, "
    "not cash-flow inventory changes or the previous fiscal year-end by default."
)
METRICS = {
    "revenue_yoy": "Quarterly revenue / comparable prior-year revenue - 1",
    "operating_margin": "Quarterly operating income / revenue",
    "operating_margin_yoy": "Quarterly operating margin minus comparable prior-year margin",
    "inventory_yoy": "Reported inventory / comparable prior-year inventory - 1",
    "operating_cashflow_yoy": (
        "Reported cumulative operating cash flow change / absolute prior-year "
        "value; matching 3/6/9/12-month periods"
    ),
    "free_cashflow_yoy": (
        "Reported cumulative (operating cash flow - capital expenditure) change / "
        "absolute prior-year value; matching periods, not a single-quarter estimate"
    ),
}
CASHFLOW_FIELDS = {"operating_cashflow", "capital_expenditure"}
DURATION_BOUNDS = {
    3: (70, 110),
    6: (150, 200),
    9: (240, 290),
    12: (330, 380),
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


def reported_records(snapshot):
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
                balance = field == "inventory"
                if fact.get("form") not in {"10-Q", "10-Q/A", "10-K", "10-K/A"}:
                    continue
                if (balance and "start" in fact) or (not balance and "start" not in fact):
                    continue
                end, filed = (date.fromisoformat(fact[k]) for k in ("end", "filed"))
                start = None if balance else date.fromisoformat(fact["start"])
                value = fact.get("val")
                months = next(
                    (
                        m
                        for m, (low, high) in DURATION_BOUNDS.items()
                        if start is not None and low <= (end - start).days <= high
                    ),
                    None,
                )
                if (
                    (not balance and months is None)
                    or (not balance and field not in CASHFLOW_FIELDS and months != 3)
                    or end > filed
                    or type(value) not in {int, float}
                    or not math.isfinite(value)
                    or (balance and value < 0)
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
                        "start": str(start) if start else None,
                        "end": str(end),
                        "filed": str(filed),
                        "available_at": available.isoformat(),
                        "accession": fact["accn"],
                        "value": float(value),
                        "unit": "USD",
                    }
                    | (
                        {"period_kind": "reported_balance_instant"}
                        if balance
                        else {"period_months": months, "period_kind": "reported_cashflow_duration"}
                        if field in CASHFLOW_FIELDS
                        else {}
                    )
                )
    return records


def select_facts(fields):
    selected = {}
    for field, versions in fields.items():
        rank = max((r["available_at"], -r["priority"]) for r in versions)
        choices = [r for r in versions if (r["available_at"], -r["priority"]) == rank]
        if len({r["value"] for r in choices}) != 1:
            continue
        selected[field] = sorted(choices, key=lambda r: r.get("statement_id", r.get("accession")))[
            -1
        ]
    return selected


def at_cutoff(records, cutoff):
    instant = datetime.fromisoformat(cutoff)
    if instant.tzinfo is None:
        raise ValueError("Financial cutoffs require a timezone")
    visible = [
        r
        for r in records
        if r["field"] not in CASHFLOW_FIELDS | {"inventory"}
        and datetime.fromisoformat(r["available_at"]) <= instant
    ]
    periods = {}
    for row in visible:
        periods.setdefault((row["start"], row["end"]), {}).setdefault(row["field"], []).append(row)
    quarters = []
    for (start, end), fields in periods.items():
        selected = select_facts(fields)
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


def cashflow_at_cutoff(records, cutoff):
    """Compare reported durations; never derive quarters by subtracting YTD filings."""
    instant = datetime.fromisoformat(cutoff)
    if instant.tzinfo is None:
        raise ValueError("Financial cutoffs require a timezone")
    periods = {}
    for row in records:
        if (
            row["field"] not in CASHFLOW_FIELDS
            or datetime.fromisoformat(row["available_at"]) > instant
        ):
            continue
        months = row.get("period_months")
        if months not in DURATION_BOUNDS:
            continue
        periods.setdefault((row["start"], row["end"], months), {}).setdefault(
            row["field"], []
        ).append(row)
    periods = [
        dict(start=start, end=end, months=months, facts=select_facts(fields))
        for (start, end, months), fields in periods.items()
    ]
    if not periods:
        return {}, {"reason": "no_visible_cashflow_period"}
    # If a newer period is unusable, do not silently fall back to an older ratio.
    end = max(p["end"] for p in periods)
    months = max(p["months"] for p in periods if p["end"] == end)
    latest = [p for p in periods if p["end"] == end and p["months"] == months]
    if len(latest) != 1 or (instant.date() - date.fromisoformat(end)).days > 270:
        return {}, {"reason": "ambiguous_or_stale_cashflow_period"}
    current = latest[0]
    matches = [
        p
        for p in periods
        if p["months"] == current["months"]
        and 350 <= (date.fromisoformat(end) - date.fromisoformat(p["end"])).days <= 380
        and (
            current["start"] is None
            or p["start"] is None
            or 350
            <= (date.fromisoformat(current["start"]) - date.fromisoformat(p["start"])).days
            <= 380
        )
    ]
    prior = matches[0] if len(matches) == 1 else None
    evidence = {"current": current, "prior_year": prior}
    if prior is None:
        return {}, evidence | {"reason": "no_unique_comparable_cashflow_period"}

    def amounts(period):
        facts = period["facts"]
        operating, capex = facts.get("operating_cashflow"), facts.get("capital_expenditure")
        result = {}
        if operating:
            result["operating_cashflow_yoy"] = operating["value"]
            if (
                capex
                and capex["value"] >= 0
                and capex.get("statement_id", capex.get("accession"))
                == operating.get("statement_id", operating.get("accession"))
            ):
                result["free_cashflow_yoy"] = operating["value"] - capex["value"]
        return result

    now, before = amounts(current), amounts(prior)
    values = {
        metric: (value - before[metric]) / abs(before[metric])
        for metric, value in now.items()
        if metric in before and before[metric] != 0
    }
    return values, evidence | {"values": values}


def inventory_at_cutoff(records, cutoff):
    instant = datetime.fromisoformat(cutoff)
    if instant.tzinfo is None:
        raise ValueError("Financial cutoffs require a timezone")
    periods = {}
    for row in records:
        if (
            row["field"] == "inventory"
            and row.get("period_kind") == "reported_balance_instant"
            and datetime.fromisoformat(row["available_at"]) <= instant
        ):
            periods.setdefault(row["end"], []).append(row)
    if not periods:
        return {}, {"reason": "no_visible_inventory_balance"}
    end = max(periods)
    current = select_facts({"inventory": periods[end]}).get("inventory")
    if current is None or (instant.date() - date.fromisoformat(end)).days > 270:
        return {}, {"reason": "ambiguous_or_stale_inventory_balance"}
    comparable = [
        prior
        for prior in periods
        if 350 <= (date.fromisoformat(end) - date.fromisoformat(prior)).days <= 380
    ]
    evidence = {"current": current}
    if len(comparable) != 1:
        return {}, evidence | {"reason": "no_unique_comparable_inventory_balance"}
    prior = select_facts({"inventory": periods[comparable[0]]}).get("inventory")
    evidence["prior_year"] = prior
    if prior is None or prior["value"] <= 0:
        return {}, evidence | {"reason": "ambiguous_or_nonpositive_prior_inventory"}
    values = {"inventory_yoy": current["value"] / prior["value"] - 1}
    return values, evidence | {"values": values}


GAP_GUIDANCE = {
    "no_visible_reported_quarter": (
        "No usable revenue quarter was visible at the required cutoff.",
        "Check supported income tags, source failures and earlier publication-dated statements.",
    ),
    "latest_reported_quarter_stale": (
        "The latest usable income quarter was more than 270 days old.",
        "Collect more recent supported income statements with their original publication dates.",
    ),
    "no_comparable_income_quarter": (
        "The visible income quarter lacks a comparable prior-year quarter.",
        "Collect the comparable prior-year quarterly revenue and income, not annual totals.",
    ),
    "missing_or_incompatible_margin_facts": (
        "Required operating income is missing, ambiguous or incompatible with revenue.",
        "Check revenue and operating income from the same filing for each required quarter.",
    ),
    "no_visible_cashflow_period": (
        "No supported cash-flow period was visible at the required cutoff.",
        "Check supported cash-flow tags, source failures and earlier reported duration facts.",
    ),
    "ambiguous_or_stale_cashflow_period": (
        "The latest cash-flow period is ambiguous or more than 270 days old.",
        "Inspect conflicting periods or collect newer statements; do not fall back silently.",
    ),
    "no_unique_comparable_cashflow_period": (
        "There is no unique prior-year cash-flow period with the same reported duration.",
        "Check comparable 3/6/9/12-month statements; do not mix durations or subtract YTD values.",
    ),
    "missing_or_incompatible_cashflow_values": (
        "Required cash-flow values are missing, conflicting, incompatible or have a zero base.",
        "Inspect operating cash flow and, for free cash flow, same-filing capital expenditure "
        "in both periods. A zero comparison base remains unavailable.",
    ),
    "no_visible_inventory_balance": (
        "No reported inventory balance was visible at the required cutoff.",
        "Check supported inventory tags and earlier balance sheets, not cash-flow adjustments.",
    ),
    "ambiguous_or_stale_inventory_balance": (
        "The latest inventory balance is ambiguous or more than 270 days old.",
        "Inspect conflicting balance facts or collect newer publication-dated balance sheets.",
    ),
    "no_unique_comparable_inventory_balance": (
        "There is no unique comparable prior-year inventory balance.",
        "Collect a balance 350–380 days before the current balance; the prior fiscal year-end "
        "is not automatically comparable.",
    ),
    "ambiguous_or_nonpositive_prior_inventory": (
        "The comparable prior inventory is ambiguous, zero or negative.",
        "Inspect the prior balance and conflicting filings. A nonpositive base remains "
        "unavailable; collecting more history cannot guarantee a usable ratio.",
    ),
}


def _gap_reason(metric, evidence):
    if metric in {"operating_cashflow_yoy", "free_cashflow_yoy"}:
        return evidence["cashflow"].get("reason", "missing_or_incompatible_cashflow_values")
    if metric == "inventory_yoy":
        return evidence["inventory"]["reason"]
    if "reason" in evidence:
        return evidence["reason"]
    if metric.endswith("_yoy") and evidence["prior_year"] is None:
        return "no_comparable_income_quarter"
    return "missing_or_incompatible_margin_facts"


def prepare_financials(snapshot, cutoffs, required):
    records = reported_records(snapshot)
    rows = []
    for cutoff in cutoffs:
        values, evidence = at_cutoff(records, cutoff)
        cash_values, cash_evidence = cashflow_at_cutoff(records, cutoff)
        inventory_values, inventory_evidence = inventory_at_cutoff(records, cutoff)
        rows.append(
            (
                values | cash_values | inventory_values,
                evidence | {"cashflow": cash_evidence, "inventory": inventory_evidence},
            )
        )
    columns, definitions, gaps = {}, [], []
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
                else "missing_or_incompatible_reported_periods",
            }
        )
        missing_rows = [int(i) for i in required if values[int(i)] is None]
        reasons = [_gap_reason(metric, rows[i][1]) for i in missing_rows]
        for reason, count in Counter(reasons).items():
            description, next_step = GAP_GUIDANCE[reason]
            gaps.append(
                {
                    "factor_id": identity,
                    "reason": reason,
                    "missing_required_rows": count,
                    "example_cutoff": cutoffs[missing_rows[reasons.index(reason)]],
                    "description": description,
                    "next_step": next_step,
                }
            )
    return {
        "definitions": definitions,
        "coverage_gaps": gaps,
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
    for tag, scale in (
        ("Revenues", 1000),
        ("OperatingIncomeLoss", 180),
        ("NetCashProvidedByUsedInOperatingActivities", 220),
        ("PaymentsToAcquirePropertyPlantAndEquipment", 50),
        ("InventoryNet", 140),
    ):
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
                if tag == "InventoryNet":
                    rows[-1].pop("start")
        facts["facts"]["us-gaap"][tag] = {"units": {"USD": rows}}
    snapshot = {
        "symbol": symbol,
        "provider": "simulated_financials",
        "profile": {
            "fields": {
                "revenue": ["Revenues"],
                "operating_income": ["OperatingIncomeLoss"],
                "operating_cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
                "capital_expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment"],
                "inventory": ["InventoryNet"],
            }
        },
        "companyfacts": facts,
        "sha256": digest(canonical(facts)),
        "notice": "SIMULATED financial statements; engineering fixture only. " + NOTICE,
    }
    return prepare_financials(snapshot, cutoffs, required)
