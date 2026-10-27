"""Prepare frozen target/industry/market context data from a supplied audit package.

No HTTP, fits, or old-registry access. Required package layout is raw/<response>
and <response>.receipt.json, covered by manifest.json, as in the source audit.
The public function uses an isolated child only for the existing ingest/export API.
Its LocalDataSpec cutoff is a legal storage placeholder, not a fold/publication rule.
"""

import csv
import io
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from quant_rsi import daily_data
from quant_rsi.artifacts import atomic_write, canonical, decode, digest
from quant_rsi.forecast_contracts import ForecastNotice, ForecastObjective
from quant_rsi.forecast_inputs import ForecastInputs, SignalSnapshot
from quant_rsi.forecast_kernel import cash_interpretation_matches
from quant_rsi.local_research import export_local_data
from quant_rsi.market_contracts import CSV_FIELDS, DataContract, timestamp
from quant_rsi.market_data import DataRegistry, ingest, market_context
from quant_rsi.schemas import LocalDataSpec

ROLES = {"target": "SOXL", "industry": "XSD", "market": "SPY"}
CALENDAR_SOURCES = {
    "NYSE_calendar_current.html": "https://www.nyse.com/markets/hours-calendars",
    "NYSE_holidays_2023_2025.html": (
        "https://ir.theice.com/press/news-details/2022/"
        "NYSE-Group-Announces-2023-2024-and-2025-Holiday-and-Early-Closings-Calendar/default.aspx"
    ),
    "NYSE_holidays_2024_2026.html": (
        "https://ir.theice.com/press/news-details/2023/"
        "NYSE-Group-Announces-2024-2025-and-2026-Holiday-and-Early-Closings-Calendar/default.aspx"
    ),
    "NYSE_closure_20250109.html": (
        "https://ir.theice.com/press/news-details/2024/"
        "The-New-York-Stock-Exchange-Will-Close-Markets-on-January-9-to-Honor-the-Passing-of-"
        "Former-President-Jimmy-Carter-on-National-Day-of-Mourning/default.aspx"
    ),
}
QUALIFICATIONS = [
    "Historical price availability at event time is an explicit research assumption, not PIT.",
    "Original HTTP receipt clocks and new registry first-seen clocks are distinct and preserved.",
    "XSD is a semiconductor context proxy, not the exact SOXL underlying or a leverage replica.",
    "USD per share/raw provider quote interpretation is an explicit assumption.",
    "All returned actions are retained; completeness and unknown corporate actions are not waived.",
    "Close plus 15 minutes is a completion policy, not a guarantee against provider revisions.",
]


def _json_file(path, value):
    body = canonical(value)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"preparation content conflict: {path.name}")
    else:
        atomic_write(path, body)


def _evidence(out, body, paths):
    ref = digest(body)
    path = out / "evidence" / ref
    if not path.exists():
        atomic_write(path, body)
    elif path.read_bytes() != body:
        raise ValueError("evidence content conflict")
    paths[ref] = str(path)
    return ref


def _read_member(audit, members, name):
    if name not in members:
        raise ValueError(f"audit manifest missing {name}")
    path = audit / name
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(audit):
        raise ValueError("audit member leaves package")
    body = path.read_bytes()
    row = members[name]
    if row["bytes"] != len(body) or row["sha256"] != digest(body):
        raise ValueError(f"audit member mismatch: {name}")
    return body


def _response(audit, members, name, url, out, paths):
    body = _read_member(audit, members, "raw/" + name)
    receipt_body = _read_member(audit, members, name + ".receipt.json")
    receipt = decode(receipt_body)
    if (
        receipt["url"] != url
        or receipt["status"] != 200
        or receipt["error"] is not None
        or receipt["bytes"] != len(body)
        or receipt["sha256"] != digest(body)
        or timestamp(receipt["requested_at"]) > timestamp(receipt["received_at"])
        or timestamp(receipt["received_at"]) > datetime.now(timezone.utc)
    ):
        raise ValueError(f"invalid original response receipt: {name}")
    return body, receipt, _evidence(out, body, paths), _evidence(out, receipt_body, paths)


def _historical_calendar(body):
    page = daily_data._CalendarHTML()
    page.feed(body.decode("utf-8"))
    # The historical publisher includes all-empty spacer rows around the table.
    tables = [[row for row in table if any(cell.strip() for cell in row)] for table in page.tables]
    tables = [t for t in tables if t and t[0] and t[0][0].lower() == "holiday"]
    if len(tables) != 1:
        raise ValueError("official historical holiday table absent/ambiguous")
    table = tables[0]
    if len(table) != 11 or any(not re.fullmatch(r"20\d{2}", x) for x in table[0][1:]):
        raise ValueError("historical calendar requires ten published holidays per year")
    years = [int(x) for x in table[0][1:]]
    if not years or len(set(years)) != len(years):
        raise ValueError("invalid historical year headers")
    holidays = {year: set() for year in years}
    for row in table[1:]:
        if len(row) != len(years) + 1:
            raise ValueError("incomplete historical holiday row")
        for year, cell in zip(years, row[1:], strict=True):
            match = re.search(daily_data.DATE_PATTERN, cell)
            if match is None:
                raise ValueError("unrecognized historical holiday")
            day = date(year, daily_data.MONTHS[match[1]], int(match[2]))
            if day.strftime("%A") not in cell:
                raise ValueError("historical holiday weekday mismatch")
            holidays[year].add(day)
    if any(len(values) != 10 for values in holidays.values()):
        raise ValueError("duplicate historical holidays")
    early = set()
    paragraphs = [p for p in page.paragraphs if "Each market will close early" in p]
    if not paragraphs:
        raise ValueError("historical early-close evidence missing")
    for paragraph in paragraphs:
        if "close early at 1:00 p.m." not in paragraph:
            raise ValueError("unsupported historical early-close time")
        values = re.findall(daily_data.DATE_PATTERN + r",\s*(20\d{2})", paragraph)
        if not values:
            raise ValueError("unrecognized historical early-close dates")
        early.update(date(int(y), daily_data.MONTHS[m], int(d)) for m, d, y in values)
    return holidays, early


def _calendar(responses):
    holidays, early = daily_data._calendar(responses["NYSE_calendar_current.html"][0])
    for name in ("NYSE_holidays_2023_2025.html", "NYSE_holidays_2024_2026.html"):
        old, closing = _historical_calendar(responses[name][0])
        for year, days in old.items():
            if year in holidays and holidays[year] != days:
                raise ValueError("official holiday sources disagree")
            holidays[year] = days
        early.update(closing)
    page = daily_data._CalendarHTML()
    page.feed(responses["NYSE_closure_20250109.html"][0].decode("utf-8"))
    text = " ".join(" ".join(page.text).split())
    if not all(
        fragment in text
        for fragment in (
            "close all NYSE Group equity and options markets",
            "Thursday, January 9, 2025",
            "NYSE Arca Equities",
        )
    ):
        raise ValueError("specific 2025-01-09 closure evidence absent")
    holidays[2025].add(date(2025, 1, 9))
    return holidays, early


def _last_complete(received, holidays, early):
    day = received.astimezone(daily_data.EASTERN).date()
    for _ in range(15):
        session = daily_data._session(day, holidays, early)
        if session and timestamp(session["close_at"]) + timedelta(minutes=15) <= received:
            return day
        day -= timedelta(days=1)
    raise ValueError("no recent completed official session")


def _grid(origin, count, holidays, early):
    historical, day = [], origin
    while len(historical) < count:
        session = daily_data._session(day, holidays, early)
        if session:
            historical.append(session)
        day -= timedelta(days=1)
    historical.reverse()
    future, day = [], origin
    while len(future) < 5:
        day += timedelta(days=1)
        session = daily_data._session(day, holidays, early)
        if session:
            future.append(session)
    return historical, future


def _quotes(data, stamps, quotes, history, received, holidays, early):
    start, end = history[0]["session_id"], history[-1]["session_id"]
    selected = {}
    for index, stamp in enumerate(stamps):
        at = datetime.fromtimestamp(stamp, daily_data.EASTERN)
        day = at.date().isoformat()
        if not start <= day <= end:
            continue
        session = daily_data._session(at.date(), holidays, early)
        if session is None or timestamp(session["open_at"]) != at:
            raise ValueError("provider bar on closed day or off the official open")
        if day in selected:
            raise ValueError("multiple daily bars for one official session")
        if timestamp(session["close_at"]) + timedelta(minutes=15) > received:
            raise ValueError("selected bar has not passed close plus 15 minutes")
        selected[day] = (
            daily_data._price(quotes["open"][index]),
            daily_data._price(quotes["close"][index]),
        )
    if set(selected) != {s["session_id"] for s in history}:
        raise ValueError(
            f"{data['meta']['symbol']}: missing registered session; no fill/holiday inference"
        )
    return selected


def _collection(ticker, response, history, future, calendar_ref, out, paths):
    raw, receipt, raw_ref, http_receipt_ref, data, prices = response
    sessions = history + future
    unit_ref = _evidence(
        out,
        canonical(
            {
                "kind": "ExplicitProviderFieldInterpretation",
                "source_ref": raw_ref,
                "http_receipt_ref": http_receipt_ref,
                "symbol": ticker,
                "currency": "USD",
                "quote_fields": ["indicators.quote[0].open", "indicators.quote[0].close"],
                "price_unit": "raw_currency_per_share",
                "price_unit_claim": "explicit_assumption",
                "quote_decimal_normalization": {
                    "max_fractional_places": 12,
                    "rounding": "ROUND_HALF_EVEN",
                },
                "cash_unit_claim": "Yahoo dividends.amount interpreted as USD/share; not certified",
            }
        ),
        paths,
    )
    availability_ref = _evidence(
        out,
        canonical(
            {
                "kind": "HistoricalAvailabilityAssumption",
                "knowledge_basis": "assumed_historical_availability",
                "assumption": "available_at_event_time",
                "source_ref": raw_ref,
                "http_receipt_ref": http_receipt_ref,
                "http_received_at": receipt["received_at"],
                "qualification": "Assumption only; registry receipt is separate from HTTP receipt.",
            }
        ),
        paths,
    )
    actions, cash, unsupported = daily_data._action_events(data, ticker, raw_ref)
    exact = json.loads(raw, parse_float=Decimal)["chart"]["result"][0].get("events", {})
    for kind, records in exact.items():
        for key, action in records.items():
            identity = digest(json.dumps([kind, key]).encode())[:24]
            event_id = f"{ticker}.action.{identity}"
            if event_id in cash and Decimal(cash[event_id]["amount"]) != Decimal(
                str(action["amount"])
            ):
                del cash[event_id]
                unsupported.append(event_id)
    contract = dict(
        schema_version=1,
        source_id=f"yahoo_context_{ticker}_{history[0]['session_id']}_{history[-1]['session_id']}",
        provider_statement=(
            "Yahoo daily quote fields; historical availability and USD/share "
            "are explicit research assumptions."
        ),
        currency="USD",
        price_basis="raw",
        frequency="daily_equities",
        calendar=dict(
            calendar_id="nyse_context_" + digest(canonical(sessions))[:20],
            timezone="America/New_York",
            sessions=sessions,
        ),
        instruments=[ticker],
        knowledge_basis="assumed_historical_availability",
        availability_evidence_ref=availability_ref,
        availability_claim="explicit_assumption",
        historical_assumption="available_at_event_time",
        universe_provenance="unknown_exploration",
        corporate_action_coverage="unknown",
        columns={key: key for key in sorted(CSV_FIELDS)},
    )
    DataContract.model_validate_json(canonical(contract))
    events = [
        dict(
            event_id=f"{ticker}.research_membership",
            revision=0,
            kind="UNIVERSE_STATUS",
            instrument=ticker,
            event_at=history[0]["open_at"],
            status="active",
            quality="valid",
        )
    ]
    for session in history:
        for kind, phase, value in [
            ("OPEN_QUOTE", "open_at", prices[session["session_id"]][0]),
            ("CLOSE_QUOTE", "close_at", prices[session["session_id"]][1]),
        ]:
            events.append(
                dict(
                    event_id=f"{ticker}.{kind}.{session['session_id']}",
                    revision=0,
                    kind=kind,
                    instrument=ticker,
                    event_at=session[phase],
                    session_id=session["session_id"],
                    quality="valid",
                    price=value,
                )
            )
    events.extend(actions)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=sorted(CSV_FIELDS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)
    csv_ref = _evidence(out, text.getvalue().encode(), paths)
    return dict(
        contract=contract,
        csv_ref=csv_ref,
        cash_actions=cash,
        unsupported_actions=unsupported,
        price_unit_evidence_ref=unit_ref,
        calendar_evidence_ref=calendar_ref,
        availability_evidence_ref=availability_ref,
        http_received_at=receipt["received_at"],
        http_receipt_ref=http_receipt_ref,
        raw_ref=raw_ref,
    )


def _ingest_prepared(prepared_path):
    """Private child entry: existing registry APIs only, never a worker or fit."""
    prepared_path = Path(prepared_path).absolute()
    prepared = decode(prepared_path.read_bytes())
    out = prepared_path.parent
    workspace = out / "workspace"
    if os.environ.get("QUANT_RSI_WORKSPACE") != str(workspace):
        raise ValueError("preparation child requires its exact isolated workspace")
    paths = prepared["evidence_paths"]
    anchor = out / "registry.identity.json"
    registry_root = workspace / ".quant_rsi" / "market"
    if anchor.exists():
        if (registry_root / "identity.json").read_bytes() != anchor.read_bytes():
            raise ValueError("preparation registry identity changed")
        with DataRegistry(registry_root):
            pass
    else:
        # Persist outside the registry before any receipt can be imported.
        with market_context(initialize=True) as (registry, _):
            _json_file(anchor, registry.anchor)
    markets, imports = {}, {}
    for role, item in prepared["collections"].items():
        contract = DataContract.model_validate_json(canonical(item["contract"]))
        imported = ingest(
            contract,
            Path(paths[item["csv_ref"]]),
            request_id=f"context.{role}.v1",
            evidence_file=Path(paths[item["availability_evidence_ref"]]),
        )
        spec = LocalDataSpec.model_validate_json(
            canonical(
                dict(
                    source="local_event_snapshot",
                    sessions=prepared["history_sessions"],
                    instruments=1,
                    train_cutoff=prepared["history_sessions"] - 5,
                    seed=7,
                    snapshot_ref=imported["snapshot_ref"],
                    decision_delay_seconds=60,
                )
            )
        )
        data = export_local_data(spec)
        notices, unsupported = [], list(item["unsupported_actions"])
        for event in data["snapshot"]["events"]:
            if event["event"]["kind"] != "CORPORATE_ACTION_NOTICE":
                continue
            event_id = event["event"]["event_id"]
            interpretation = item["cash_actions"].get(event_id)
            if interpretation is None:
                continue
            try:
                notice = ForecastNotice.model_validate_json(
                    canonical(
                        dict(
                            schema_version=1,
                            notice_ref=event["payload_ref"],
                            action_kind="cash_dividend",
                            **interpretation,
                        )
                    )
                )
            except ValueError:
                unsupported.append(event_id)
                continue
            if not cash_interpretation_matches(event, notice):
                raise ValueError("cash interpretation differs from source event")
            notices.append(notice.model_dump(mode="json"))
        objective = ForecastObjective.model_validate_json(
            canonical(
                dict(
                    schema_version=1,
                    kind="five_session_raw_close",
                    horizon_sessions=5,
                    target="log_close_ratio",
                    currency="USD",
                    price_unit="raw_currency_per_share",
                    price_unit_claim="explicit_assumption",
                    price_unit_evidence_ref=item["price_unit_evidence_ref"],
                    calendar_evidence_ref=item["calendar_evidence_ref"],
                    notices=notices,
                    unknown_action_policy="affected_or_unknown_interval_abstain",
                    prior_exposure="previously_exposed_historical_exploration",
                    source_qualifications=prepared.get("qualifications", QUALIFICATIONS)
                    + (
                        [f"{len(unsupported)} unknown notices retained for kernel abstention"]
                        if unsupported
                        else []
                    ),
                )
            )
        )
        markets[role] = {"data": data, "objective": objective.model_dump(mode="json")}
        imports[role] = {
            "receipt": imported,
            "http_received_at": item["http_received_at"],
            "unsupported_actions": sorted(set(unsupported)),
        }
    evidence_hex = {ref: Path(path).read_bytes().hex() for ref, path in paths.items()}
    inputs = dict(
        kind="ForecastInputs",
        schema_version=1,
        target=markets["target"],
        contexts={role: markets[role] for role in ("industry", "market")},
        signals=prepared["signals"],
        evidence_hex=evidence_hex,
    )
    ForecastInputs(inputs)
    _json_file(out / "inputs.json", inputs)
    _json_file(out / "imports.json", imports)


def prepare_context_data(
    audit_dir,
    output_dir,
    *,
    history_sessions=756,
    origin_session_id=None,
    signals=None,
    roles=None,
    resume=False,
    parent_lock_fd=None,
):
    """Return inputs/evidence_paths/receipt_path/qualifications/meta; no implicit source paths."""
    if type(history_sessions) is not int or not 24 <= history_sessions <= 1000:
        raise ValueError("history_sessions must be an integer in 24..1000")
    roles = validate_roles(ROLES if roles is None else roles)
    qualifications = context_qualifications(roles)
    audit = Path(audit_dir).resolve(strict=True)
    out = Path(output_dir).absolute()
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise ValueError("preparation output must be a directory, not a symlink")
    nonempty = out.exists() and any(out.iterdir())
    if nonempty and not resume:
        raise FileExistsError("context preparation requires a fresh/empty output directory")
    out = out.resolve()
    if out.is_relative_to(audit) or audit.is_relative_to(out):
        raise ValueError("output and audit package must be disjoint")
    binding = dict(
        audit_manifest_ref=digest((audit / "manifest.json").read_bytes()),
        history_sessions=history_sessions,
        origin_session_id=origin_session_id,
        roles=roles,
        signals=signals,
    )
    intent = out / "intent.json"
    if nonempty:
        pending_only = all(
            p.name.startswith(".pending-")
            and p.is_file()
            and not p.is_symlink()
            and p.stat().st_size <= daily_data.MAX_RESPONSE_BYTES + 1
            for p in out.iterdir()
        )
        if intent.exists():
            if decode(intent.read_bytes()) != binding:
                raise ValueError("preparation identity is missing or binds different inputs")
        elif not pending_only:
            raise ValueError("preparation identity is missing or binds different inputs")
    out.mkdir(parents=True, exist_ok=True)
    _json_file(intent, binding)
    (out / "evidence").mkdir(exist_ok=True)
    if (out / "prepared.binding.json").exists():
        _verify_prepared_binding(out)
        return _finish_preparation(out, parent_lock_fd=parent_lock_fd)
    if (out / "registry.identity.json").exists():
        raise ValueError("prepared binding missing from initialized registry")
    paths = {}
    try:
        manifest_body = (audit / "manifest.json").read_bytes()
        manifest = decode(manifest_body)
        members = {item["path"]: item for item in manifest["files"]}
        if len(members) != len(manifest["files"]):
            raise ValueError("duplicate audit manifest members")
        audit_ref = _evidence(out, manifest_body, paths)
        calendars = {
            name: _response(audit, members, name, url, out, paths)
            for name, url in CALENDAR_SOURCES.items()
        }
        holidays, early = _calendar(calendars)
        responses, last = {}, []
        for ticker in roles.values():
            name = f"{ticker}_chart_5y.json"
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5y&interval=1d&events=div%2Csplits"
            response = _response(audit, members, name, url, out, paths)
            data, stamps, quotes = daily_data._chart(response[0], ticker)
            received = timestamp(response[1]["received_at"])
            last.append(_last_complete(received, holidays, early))
            responses[ticker] = (*response, data, stamps, quotes)
        origin = min(last) if origin_session_id is None else date.fromisoformat(origin_session_id)
        if origin > min(last) or daily_data._session(origin, holidays, early) is None:
            raise ValueError("origin is not complete for every source")
        history, future = _grid(origin, history_sessions, holidays, early)
        calendar_ref = _evidence(
            out,
            canonical(
                dict(
                    kind="ParsedPublishedNYSEContextCalendar",
                    timezone="America/New_York",
                    source_refs=[r[2] for r in calendars.values()],
                    receipt_refs=[r[3] for r in calendars.values()],
                    holidays={
                        str(year): sorted(day.isoformat() for day in days)
                        for year, days in holidays.items()
                    },
                    early_closes_1300=sorted(day.isoformat() for day in early),
                    explicit_exceptional_closures=["2025-01-09"],
                    sessions=history + future,
                    qualification="Official schedule plus explicit closure; no inferred holidays.",
                )
            ),
            paths,
        )
        collections = {}
        for role, ticker in roles.items():
            raw, receipt, raw_ref, receipt_ref, data, stamps, quotes = responses[ticker]
            prices = _quotes(
                data, stamps, quotes, history, timestamp(receipt["received_at"]), holidays, early
            )
            collections[role] = _collection(
                ticker,
                (raw, receipt, raw_ref, receipt_ref, data, prices),
                history,
                future,
                calendar_ref,
                out,
                paths,
            )
        signal_snapshot = None
        if signals is not None:
            signal_snapshot = SignalSnapshot.model_validate_json(
                canonical(signals["snapshot"])
            ).model_dump(mode="json")
            for ref, path in signals["evidence_paths"].items():
                body = Path(path).read_bytes()
                if digest(body) != ref:
                    raise ValueError("signal evidence hash mismatch")
                _evidence(out, body, paths)
        workspace = out / "workspace"
        workspace.mkdir(exist_ok=True)
        prepared = dict(
            history_sessions=history_sessions,
            origin_session_id=origin.isoformat(),
            audit_manifest_ref=audit_ref,
            collections=collections,
            evidence_paths=paths,
            signals=signal_snapshot,
            roles=roles,
            qualifications=qualifications,
            future_sessions=[item["session_id"] for item in future],
        )
        prepared_path = out / "prepared.json"
        _json_file(prepared_path, prepared)
        _json_file(
            out / "prepared.binding.json",
            dict(
                intent_ref=digest((out / "intent.json").read_bytes()),
                prepared_ref=digest(prepared_path.read_bytes()),
            ),
        )
        return _finish_preparation(out, parent_lock_fd=parent_lock_fd)
    except Exception as exc:
        # Keep the first failure as evidence, and never replace a successful receipt.
        failure = out / "failure.receipt.json"
        if not failure.exists():
            _json_file(
                failure,
                dict(
                    kind="forecast_context_preparation_failure",
                    error=f"{type(exc).__name__}: {exc}",
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                    qualification="Partial output retained; no fit was requested.",
                ),
            )
        raise


def validate_roles(roles):
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        raise ValueError("exact target, industry and market tickers are required")
    if (
        any(
            not isinstance(t, str) or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,11}", t)
            for t in roles.values()
        )
        or len(set(roles.values())) != 3
    ):
        raise ValueError("three distinct explicit uppercase tickers are required")
    return {role: roles[role] for role in ROLES}


def context_qualifications(roles):
    if roles == ROLES:
        return list(QUALIFICATIONS)
    return [
        (
            f"{roles['industry']} and {roles['market']} are explicitly selected context proxies "
            f"for {roles['target']}; no benchmark replication or economic equivalence is implied."
        )
        if text == QUALIFICATIONS[2]
        else text
        for text in QUALIFICATIONS
    ]


def _verify_prepared_binding(out):
    binding = decode((out / "prepared.binding.json").read_bytes())
    if binding != dict(
        intent_ref=digest((out / "intent.json").read_bytes()),
        prepared_ref=digest((out / "prepared.json").read_bytes()),
    ):
        raise ValueError("prepared input binding changed")


def _preparation_result(out):
    prepared = decode((out / "prepared.json").read_bytes())
    inputs = decode((out / "inputs.json").read_bytes())
    receipt = decode((out / "preparation.receipt.json").read_bytes())
    if ForecastInputs(inputs).ref != receipt["inputs_ref"]:
        raise ValueError("prepared inputs differ from receipt")
    if decode((out / "imports.json").read_bytes()) != receipt["imports"]:
        raise ValueError("prepared imports differ from receipt")
    return dict(
        inputs=inputs,
        evidence_paths=prepared["evidence_paths"],
        receipt_path=str(out / "preparation.receipt.json"),
        qualifications=receipt["qualifications"],
        meta={
            key: receipt[key]
            for key in (
                "workspace",
                "registry_path",
                "origin_session_id",
                "historical_sessions",
                "future_sessions",
                "inputs_ref",
            )
        },
    )


def verify_preparation(output_dir):
    """Read-only integrity check, including the original first-seen authority."""
    out = Path(output_dir)
    _verify_prepared_binding(out)
    prepared = decode((out / "prepared.json").read_bytes())
    for ref, path in prepared["evidence_paths"].items():
        member = Path(path)
        if member.is_symlink() or not member.resolve().is_relative_to(out.resolve()):
            raise ValueError("preparation evidence leaves its registered directory")
        if digest(member.read_bytes()) != ref:
            raise ValueError("prepared evidence changed or is missing")
    registry_root = out / "workspace" / ".quant_rsi" / "market"
    if (registry_root / "identity.json").read_bytes() != (
        out / "registry.identity.json"
    ).read_bytes():
        raise ValueError("preparation registry identity changed")
    with DataRegistry(registry_root) as registry:
        for item in decode((out / "imports.json").read_bytes()).values():
            saved = item["receipt"]
            row = registry.connection.execute(
                "SELECT snapshot_ref, first_seen_at FROM receipts WHERE request_id=?",
                (saved["request_id"],),
            ).fetchone()
            if row is None or tuple(row) != (saved["snapshot_ref"], saved["first_seen_at"]):
                raise ValueError("preparation receipt history changed")
            registry.snapshot(saved["snapshot_ref"])
    return _preparation_result(out)


def _finish_preparation(out, *, parent_lock_fd=None):
    _verify_prepared_binding(out)
    prepared_path = out / "prepared.json"
    prepared = decode(prepared_path.read_bytes())
    for ref, path in prepared["evidence_paths"].items():
        if digest(Path(path).read_bytes()) != ref:
            raise ValueError("prepared evidence changed")
    if (out / "preparation.receipt.json").exists():
        return verify_preparation(out)
    workspace = out / "workspace"
    env = dict(
        os.environ,
        QUANT_RSI_WORKSPACE=str(workspace),
        PYTHONPATH=str(Path(__file__).absolute().parent.parent),
        PYTHONDONTWRITEBYTECODE="1",
    )
    with (out / "ingest.log").open("ab") as log:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from quant_rsi.forecast_context_data import _ingest_prepared; "
                "_ingest_prepared(sys.argv[1])",
                str(prepared_path),
            ],
            env=env,
            cwd=out,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
            pass_fds=tuple(
                dict.fromkeys(
                    ()
                    if parent_lock_fd is None
                    else (parent_lock_fd,)
                    if isinstance(parent_lock_fd, int)
                    else parent_lock_fd
                )
            ),
        )
    if completed.returncode:
        raise ValueError(f"isolated registry preparation failed; see {out / 'ingest.log'}")
    inputs = ForecastInputs(decode((out / "inputs.json").read_bytes()))
    receipt = dict(
        schema_version=1,
        kind="forecast_context_preparation_receipt",
        status="complete",
        audit_manifest_ref=prepared["audit_manifest_ref"],
        inputs_ref=inputs.ref,
        origin_session_id=prepared["origin_session_id"],
        historical_sessions=prepared["history_sessions"],
        future_sessions=prepared["future_sessions"],
        workspace=str(workspace),
        registry_path=str(workspace / ".quant_rsi" / "market"),
        imports=decode((out / "imports.json").read_bytes()),
        qualifications=prepared["qualifications"],
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _json_file(out / "preparation.receipt.json", receipt)
    return verify_preparation(out)
