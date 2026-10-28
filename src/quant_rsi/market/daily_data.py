"""Bounded public daily-bar collection for explicitly supported NYSE-family USD listings.

This is an exploratory source adapter, not a PIT certification or an exchange feed.
It writes a fresh collection directory and never fits a model or submits an order.
"""

import csv
import hashlib
import json
import math
import re
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from quant_rsi.core.artifacts import atomic_write
from quant_rsi.market.market_contracts import CSV_FIELDS, DataContract

CALENDAR_URL = "https://www.nyse.com/markets/hours-calendars"
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
SUPPORTED_EXCHANGES = {"PCX": "NYSE Arca", "NYQ": "NYSE", "ASE": "NYSE American"}
HISTORY_SESSIONS = 120
CLOSE_BUFFER_SECONDS = 900
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
EASTERN = ZoneInfo("America/New_York")
MONTHS = {
    name: i
    for i, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        1,
    )
}
DATE_PATTERN = r"(" + "|".join(MONTHS) + r")\s+(\d{1,2})"


def _now():
    return datetime.now(timezone.utc)


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _json_file(path, value):
    body = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        raise FileExistsError(path)
    atomic_write(path, body)
    return _sha(body)


def _download(url, path, records):
    started = _now().isoformat()
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 QuantRSI daily research"})
    error = None
    try:
        response = urlopen(request, timeout=15)
    except HTTPError as exc:
        response, error = exc, exc
    with response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        record = {
            "url": url,
            "response_url": response.geturl(),
            "requested_at": started,
            "received_at": _now().isoformat(),
            "http_status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "path": str(path),
            "bytes": len(body),
            "sha256": _sha(body),
            "truncated": len(body) > MAX_RESPONSE_BYTES,
        }
    with path.open("xb") as handle:
        handle.write(body)
    records.append(record)
    if record["truncated"]:
        raise ValueError("Public response exceeds the 8 MiB collection limit; prefix preserved")
    if error is not None:
        raise ValueError(f"Public source HTTP {error.code}; response preserved") from error
    return body


class _CalendarHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.paragraphs, self.text = [], [], []
        self.table = self.row = self.cell = self.paragraph = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        if tag == "p":
            self.paragraph = []

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)
        if self.paragraph is not None:
            self.paragraph.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None
        if tag == "p" and self.paragraph is not None:
            self.paragraphs.append(" ".join(" ".join(self.paragraph).split()))
            self.paragraph = None


def _calendar(body):
    page = _CalendarHTML()
    page.feed(body.decode("utf-8"))
    text = " ".join(" ".join(page.text).split())
    if "All NYSE markets observe" not in text or not re.search(
        r"Core Trading Session:\s*9:30 a\.m\.\s+to\s+4:00 p\.m\.\s+ET", text
    ):
        raise ValueError("Official NYSE core-hours evidence is absent or has changed")
    tables = [
        table for table in page.tables if table and table[0] and table[0][0].lower() == "holiday"
    ]
    if len(tables) != 1:
        raise ValueError("Official NYSE holiday table is missing or ambiguous")
    table = tables[0]
    if len(table[0]) < 2 or any(not re.fullmatch(r"20\d{2}", x) for x in table[0][1:]):
        raise ValueError("Unsupported official holiday-year headers")
    years = [int(x) for x in table[0][1:]]
    if len(set(years)) != len(years) or len(table) != 11:
        raise ValueError("Expected ten explicitly published NYSE holidays per year")
    holidays = {year: set() for year in years}
    for row in table[1:]:
        if len(row) != len(years) + 1:
            raise ValueError("Incomplete official holiday row")
        for year, value in zip(years, row[1:], strict=True):
            match = re.search(DATE_PATTERN, value)
            if match is None:
                raise ValueError("Unrecognized official holiday date")
            day = date(year, MONTHS[match[1]], int(match[2]))
            if day.strftime("%A") not in value:
                raise ValueError("Official holiday weekday/date mismatch")
            holidays[year].add(day)
    if any(len(days) != 10 for days in holidays.values()):
        raise ValueError("Duplicate or missing official holidays")
    early = set()
    paragraphs = [x for x in page.paragraphs if "Each market will close early" in x]
    if not paragraphs:
        raise ValueError("Official early-close evidence is absent")
    for paragraph in paragraphs:
        if "close early at 1:00 p.m." not in paragraph:
            raise ValueError("Unsupported official early-close time")
        found = re.findall(DATE_PATTERN + r",\s*(20\d{2})", paragraph)
        if not found:
            raise ValueError("Unrecognized official early-close dates")
        for month, day, year in found:
            early.add(date(int(year), MONTHS[month], int(day)))
    return holidays, early


def _session(day, holidays, early):
    if day.year not in holidays:
        raise ValueError(f"Official calendar evidence does not cover {day.year}")
    if day.weekday() >= 5 or day in holidays[day.year]:
        return None
    return {
        "session_id": day.isoformat(),
        "open_at": datetime.combine(day, time(9, 30), EASTERN).isoformat(),
        "close_at": datetime.combine(day, time(13 if day in early else 16), EASTERN).isoformat(),
    }


def _price(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("A completed daily bar has a missing/invalid positive price")
    # Preserve the public response separately; the registry accepts at most 12
    # fractional decimal places. This is representation normalization, not cents rounding.
    number = Decimal(str(value))
    if number.adjusted() > 15:
        raise ValueError("Daily quote exceeds the registry's supported decimal magnitude")
    if number.as_tuple().exponent < -12:
        with localcontext() as context:
            context.prec = 40
            number = number.quantize(Decimal("0.000000000001"), rounding=ROUND_HALF_EVEN)
    if number <= 0:
        raise ValueError("Daily quote is too small for the registry's decimal precision")
    return format(number.normalize(), "f")


def _chart(body, ticker):
    payload = json.loads(body)
    chart = payload.get("chart", {})
    if chart.get("error") is not None:
        raise ValueError(f"Unknown ticker or public chart error: {chart['error']}")
    result = chart.get("result")
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], dict):
        raise ValueError("Expected exactly one public daily chart")
    data = result[0]
    meta = data.get("meta", {})
    if meta.get("symbol") != ticker:
        raise ValueError("Public response symbol differs from requested ticker")
    if (
        meta.get("currency") != "USD"
        or meta.get("instrumentType") not in {"EQUITY", "ETF"}
        or meta.get("exchangeName") not in SUPPORTED_EXCHANGES
        or meta.get("exchangeTimezoneName") != "America/New_York"
        or meta.get("dataGranularity") != "1d"
    ):
        raise ValueError(
            "Supported scope is USD NYSE/NYSE Arca/NYSE American EQUITY or ETF daily bars; "
            "other exchanges, including NASDAQ, require their own verified calendar adapter"
        )
    timestamps = data.get("timestamp")
    quotes = data.get("indicators", {}).get("quote")
    if not isinstance(timestamps, list) or not timestamps or not isinstance(quotes, list):
        raise ValueError("Missing daily timestamp/quote arrays")
    if len(quotes) != 1 or not isinstance(quotes[0], dict):
        raise ValueError("Ambiguous daily quote arrays")
    quotes = quotes[0]
    if any(type(t) is not int for t in timestamps) or timestamps != sorted(set(timestamps)):
        raise ValueError("Daily timestamps must be unique ordered epoch seconds")
    if not {"open", "close"} <= quotes.keys() or any(
        not isinstance(values, list) or len(values) != len(timestamps) for values in quotes.values()
    ):
        raise ValueError("Daily quote arrays do not match the timestamp array")
    return data, timestamps, quotes


def _complete_bars(timestamps, quotes, holidays, early, received_at):
    complete, previous = [], None
    for i, epoch in enumerate(timestamps):
        opening = datetime.fromtimestamp(epoch, EASTERN)
        if previous is not None and opening.date() <= previous:
            raise ValueError("Duplicate or out-of-order daily session")
        previous = opening.date()
        # The source can contain a current partial bar, but cannot define its close.
        session = _session(opening.date(), holidays, early)
        if session is None or opening != datetime.fromisoformat(session["open_at"]):
            raise ValueError("Daily timestamp differs from the official registered opening")
        close_at = datetime.fromisoformat(session["close_at"])
        if received_at < close_at + timedelta(seconds=CLOSE_BUFFER_SECONDS):
            if opening > received_at:
                raise ValueError("Public chart contains a future bar")
            continue
        complete.append((session, _price(quotes["open"][i]), _price(quotes["close"][i])))
    if len(complete) < HISTORY_SESSIONS:
        raise ValueError(f"Need {HISTORY_SESSIONS} complete daily bars; received {len(complete)}")
    complete = complete[-HISTORY_SESSIONS:]
    start = date.fromisoformat(complete[0][0]["session_id"])
    end = date.fromisoformat(complete[-1][0]["session_id"])
    expected = []
    while start <= end:
        session = _session(start, holidays, early)
        if session is not None:
            expected.append(session["session_id"])
        start += timedelta(days=1)
    if expected != [bar[0]["session_id"] for bar in complete]:
        raise ValueError("Missing completed exchange sessions; prices will not be imputed")
    # Do not silently forecast from an old quote when a later session is already complete.
    day = end + timedelta(days=1)
    while day <= received_at.astimezone(EASTERN).date():
        session = _session(day, holidays, early)
        if session and received_at >= datetime.fromisoformat(session["close_at"]) + timedelta(
            seconds=CLOSE_BUFFER_SECONDS
        ):
            raise ValueError("Latest completed exchange session is absent from the public chart")
        day += timedelta(days=1)
    future = []
    day = end
    while len(future) < 5:
        day += timedelta(days=1)
        session = _session(day, holidays, early)
        if session:
            future.append(session)
    return complete, future


def _action_events(data, ticker, raw_ref):
    events, cash, unsupported = [], {}, []
    groups = data.get("events", {})
    if not isinstance(groups, dict):
        raise ValueError("Malformed public corporate-action groups")
    for kind, records in groups.items():
        if not isinstance(records, dict):
            raise ValueError("Malformed corporate-action records")
        for key, action in sorted(records.items()):
            if not isinstance(action, dict) or type(action.get("date")) is not int:
                raise ValueError("Corporate action has no explicit source event time")
            if {"provider", "event_kind"} & action.keys():
                raise ValueError("Corporate action conflicts with normalized provenance fields")
            at = datetime.fromtimestamp(action["date"], EASTERN).isoformat()
            identity = _sha(json.dumps([kind, key]).encode())[:24]
            event_id = f"{ticker}.action.{identity}"
            events.append(
                {
                    "event_id": event_id,
                    "revision": 0,
                    "kind": "CORPORATE_ACTION_NOTICE",
                    "instrument": ticker,
                    "event_at": at,
                    "quality": "valid",
                    "action": json.dumps(
                        {"provider": "Yahoo", "event_kind": kind, **action},
                        sort_keys=True,
                        allow_nan=False,
                    ),
                }
            )
            try:
                amount = Decimal(str(action.get("amount")))
                valid = amount.is_finite() and amount > 0
            except InvalidOperation:
                valid = False
            if kind == "dividends" and set(action) == {"amount", "date"} and valid:
                cash[event_id] = {
                    "effective_at": at,
                    "amount": str(amount),
                    "currency": "USD",
                    "unit": "currency_per_share",
                    "source_format": "yahoo_chart_dividends_v1",
                    "evidence_ref": raw_ref,
                }
            else:
                unsupported.append(event_id)
    return events, cash, unsupported


def collect_daily(ticker, output_dir) -> dict:
    """Collect into a fresh directory; errors preserve raw evidence and a failure receipt."""
    output = Path(output_dir).absolute()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(
            "Collection output directory must be empty; existing evidence is preserved"
        )
    downloads = []
    receipt = {
        "kind": "PublicDailyCollection",
        "schema_version": 1,
        "requested_ticker": ticker,
        "started_at": _now().isoformat(),
        "downloads": downloads,
        "status": "collecting",
        "complete_bar_policy": {
            "scheduled_close_buffer_seconds": CLOSE_BUFFER_SECONDS,
            "claim": "conservative_collection_assumption_not_finality",
        },
    }
    receipt_path = output / "collection_receipt.json"
    try:
        if not isinstance(ticker, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,14}", ticker):
            raise ValueError("Ticker must be a supported plain US equity/ETF symbol")
        ticker = ticker.upper()
        url = CHART_URL + quote(ticker, safe="") + "?range=1y&interval=1d&events=div%2Csplits"
        raw_path = output / "yahoo_chart_raw.json"
        raw = _download(url, raw_path, downloads)
        data, timestamps, quotes = _chart(raw, ticker)
        bars_received_at = datetime.fromisoformat(downloads[-1]["received_at"])
        calendar_path = output / "nyse_hours_calendar.html"
        calendar_body = _download(CALENDAR_URL, calendar_path, downloads)
        holidays, early = _calendar(calendar_body)
        bars, future = _complete_bars(timestamps, quotes, holidays, early, bars_received_at)
        sessions = [bar[0] for bar in bars] + future
        calendar_evidence = {
            "kind": "ParsedPublishedNYSECalendar",
            "source_url": CALENDAR_URL,
            "source_ref": _sha(calendar_body),
            "timezone": "America/New_York",
            "holidays": {
                str(y): sorted(d.isoformat() for d in days) for y, days in holidays.items()
            },
            "early_closes_1300": sorted(d.isoformat() for d in early),
            "sessions": sessions,
            "qualification": "published scheduled core sessions; emergency closures not certified",
        }
        calendar_json = output / "calendar_evidence.json"
        calendar_ref = _json_file(calendar_json, calendar_evidence)
        unit_json = output / "price_unit_evidence.json"
        unit_ref = _json_file(
            unit_json,
            {
                "kind": "ExplicitProviderFieldInterpretation",
                "source_ref": _sha(raw),
                "symbol": ticker,
                "currency": data["meta"]["currency"],
                "quote_fields": ["indicators.quote[0].open", "indicators.quote[0].close"],
                "price_unit": "raw_currency_per_share",
                "price_unit_claim": "explicit_assumption",
                "quote_decimal_normalization": {
                    "schema_version": 1,
                    "max_fractional_places": 12,
                    "rounding": "ROUND_HALF_EVEN",
                    "raw_response_preserved": True,
                },
                "cash_unit": (
                    "dividends.amount interpreted as USD per share; not independently certified"
                ),
                "qualification": (
                    "provider raw fields and units, not adjusted/total-return or PIT-certified"
                ),
            },
        )
        action_events, cash, unsupported = _action_events(data, ticker, _sha(raw))
        contract = {
            "schema_version": 1,
            "source_id": f"yahoo_{ticker}_{sessions[0]['session_id']}_{bars[-1][0]['session_id']}",
            "provider_statement": (
                "Yahoo daily chart raw quote fields; selected NYSE-family USD listing. "
                "Historical event-time availability and USD/share units are explicit exploration "
                "assumptions, not provider publication times or PIT certification."
            ),
            "currency": "USD",
            "price_basis": "raw",
            "frequency": "daily_equities",
            "calendar": {
                "calendar_id": "nyse_core_" + _sha(json.dumps(sessions).encode())[:20],
                "timezone": "America/New_York",
                "sessions": sessions,
            },
            "instruments": [ticker],
            "knowledge_basis": "assumed_historical_availability",
            "availability_evidence_ref": None,
            "availability_claim": "explicit_assumption",
            "historical_assumption": "available_at_event_time",
            "universe_provenance": "unknown_exploration",
            "corporate_action_coverage": "unknown",
            "columns": {key: key for key in sorted(CSV_FIELDS)},
        }
        DataContract.model_validate_json(json.dumps(contract))
        events = [
            {
                "event_id": f"{ticker}.research_membership",
                "revision": 0,
                "kind": "UNIVERSE_STATUS",
                "instrument": ticker,
                "event_at": sessions[0]["open_at"],
                "status": "active",
                "quality": "valid",
            }
        ]
        for session, opening, closing in bars:
            for kind, phase, value in [
                ("OPEN_QUOTE", "open_at", opening),
                ("CLOSE_QUOTE", "close_at", closing),
            ]:
                events.append(
                    {
                        "event_id": f"{ticker}.{kind}.{session['session_id']}",
                        "revision": 0,
                        "kind": kind,
                        "instrument": ticker,
                        "event_at": session[phase],
                        "session_id": session["session_id"],
                        "quality": "valid",
                        "price": value,
                    }
                )
        events.extend(action_events)
        events_path = output / "events.csv"
        with events_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(CSV_FIELDS))
            writer.writeheader()
            writer.writerows(events)
        contract_file_ref = _json_file(output / "data_contract.json", contract)
        qualifications = [
            "public_source_not_PIT_or_corporate_action_complete",
            "historical_available_at_event_time_is_explicit_assumption",
            "cash_and_quote_USD_per_share_are_explicit_provider_field_interpretations",
            "publication_must_use_observed_receipts_and_actual_model_effective_time",
            "user_selected_single_instrument_universe_not_historical_membership_certification",
            "15_minute_close_buffer_does_not_prevent_later_provider_revisions",
            "quote_decimals_normalized_to_12_fractional_places_half_even_raw_preserved",
            "unversioned_provider_events_use_revision_zero_conflicts_must_not_be_overwritten",
        ]
        if unsupported:
            qualifications.append("unsupported_action_notices_retained_for_forecast_abstention")
        evidence = {
            _sha(raw): str(raw_path),
            _sha(calendar_body): str(calendar_path),
            calendar_ref: str(calendar_json),
            unit_ref: str(unit_json),
        }
        receipt.update(
            status="collected",
            ticker=ticker,
            completed_at=_now().isoformat(),
            origin_session_id=bars[-1][0]["session_id"],
            historical_sessions=len(bars),
            future_sessions=future,
            events=len(events),
            cash_action_event_ids=sorted(cash),
            unsupported_action_event_ids=unsupported,
            qualifications=qualifications,
            evidence_paths=evidence,
            normalized_inputs={
                "contract_file_sha256": contract_file_ref,
                "events_file_sha256": _sha(events_path.read_bytes()),
            },
        )
        result = {
            "contract": contract,
            "events_path": str(events_path),
            "raw_path": str(raw_path),
            "receipt_path": str(receipt_path),
            "origin_session_id": bars[-1][0]["session_id"],
            "evidence_paths": evidence,
            "cash_actions": cash,
            "qualifications": qualifications,
            "calendar_evidence_ref": calendar_ref,
            "price_unit_evidence_ref": unit_ref,
        }
    except Exception as exc:
        receipt.update(
            status="failed",
            completed_at=_now().isoformat(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        _json_file(receipt_path, receipt)
        raise ValueError(f"Daily collection failed: {exc}; receipt: {receipt_path}") from exc
    # The atomic success receipt is also the full collection manifest.
    receipt["result"] = result
    _json_file(receipt_path, receipt)
    return result
