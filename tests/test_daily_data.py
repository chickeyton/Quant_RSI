"""Offline adapter tests: no public requests, model training, or registry mutation."""

import csv
import hashlib
import io
import json
from datetime import date, datetime, time, timedelta
from urllib.error import HTTPError
from zoneinfo import ZoneInfo

import pytest

from quant_rsi.market import daily_data
from quant_rsi.market.market_contracts import DataContract, SourceEvent

HOLIDAYS = {
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
}
EASTERN = ZoneInfo("America/New_York")
CALENDAR = (
    "<h4>All NYSE markets observe the following holidays.</h4>"
    "<p>NYSE Arca Core Trading Session: 9:30 a.m. to 4:00 p.m. ET</p>"
    "<table><tr><th>Holiday</th><th>2026</th></tr>"
    + "".join(
        f"<tr><td>Holiday {i}</td><td>{date.fromisoformat(day).strftime('%A, %B %d')}</td></tr>"
        for i, day in enumerate(sorted(HOLIDAYS))
    )
    + "</table><p>Each market will close early at 1:00 p.m. on "
    "Friday, November 27, 2026 and Thursday, December 24, 2026.</p>"
).encode()


def chart(last="2026-10-23"):
    dates = []
    day = date(2026, 1, 2)
    while day <= date.fromisoformat(last):
        if day.weekday() < 5 and day.isoformat() not in HOLIDAYS:
            dates.append(day)
        day += timedelta(days=1)
    stamps = [int(datetime.combine(day, time(9, 30), EASTERN).timestamp()) for day in dates]
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {
                        "symbol": "SOXL",
                        "currency": "USD",
                        "instrumentType": "ETF",
                        "exchangeName": "PCX",
                        "exchangeTimezoneName": "America/New_York",
                        "dataGranularity": "1d",
                    },
                    "timestamp": stamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [30.0 + i / 100 for i in range(len(dates))],
                                "close": [31.0 + i / 100 for i in range(len(dates))],
                            }
                        ]
                    },
                    "events": {
                        "dividends": {
                            "old-cash": {
                                "amount": 0.25,
                                "date": int(
                                    datetime(2026, 2, 2, 9, 30, tzinfo=EASTERN).timestamp()
                                ),
                            }
                        },
                        "splits": {
                            "split": {
                                "date": int(
                                    datetime(2026, 6, 1, 9, 30, tzinfo=EASTERN).timestamp()
                                ),
                                "numerator": 2,
                                "denominator": 1,
                                "splitRatio": "2:1",
                            }
                        },
                    },
                }
            ],
        },
    }


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, body, url):
        super().__init__(body)
        self.url = url

    def geturl(self):
        return self.url


def install(monkeypatch, data=None, *, now="2026-10-26T12:00:00+00:00", calendar=CALENDAR):
    data = chart() if data is None else data
    raw = json.dumps(data).encode()
    calls = []

    def get(request, timeout):
        assert timeout == 15
        calls.append(request.full_url)
        body = calendar if request.full_url == daily_data.CALENDAR_URL else raw
        return Response(body, request.full_url)

    monkeypatch.setattr(daily_data, "urlopen", get)
    monkeypatch.setattr(daily_data, "_now", lambda: datetime.fromisoformat(now))
    return raw, calls


def read_events(result):
    with open(result["events_path"], newline="") as handle:
        return list(csv.DictReader(handle))


def test_collects_complete_window_provenance_and_all_actions(tmp_path, monkeypatch):
    raw, calls = install(monkeypatch)
    result = daily_data.collect_daily("soxl", tmp_path / "collected")
    contract = DataContract.model_validate_json(json.dumps(result["contract"]))
    assert contract.knowledge_basis == "assumed_historical_availability"
    assert contract.historical_assumption == "available_at_event_time"
    assert contract.currency == "USD" and contract.price_basis == "raw"
    assert len(contract.calendar.sessions) == 125
    assert result["origin_session_id"] == "2026-10-23"
    assert [s.session_id for s in contract.calendar.sessions[-5:]] == [
        "2026-10-26",
        "2026-10-27",
        "2026-10-28",
        "2026-10-29",
        "2026-10-30",
    ]
    assert (tmp_path / "collected/yahoo_chart_raw.json").read_bytes() == raw
    events = read_events(result)
    assert len(events) == 243
    notices = [row for row in events if row["kind"] == "CORPORATE_ACTION_NOTICE"]
    assert len(notices) == 2
    assert all(row["source_available_at"] == "" for row in events)
    assert min(row["event_at"] for row in notices) < events[0]["event_at"]
    assert len(result["cash_actions"]) == 1
    cash = next(iter(result["cash_actions"].values()))
    assert cash["unit"] == "currency_per_share"
    assert cash["amount"] == "0.25" and cash["evidence_ref"] == hashlib.sha256(raw).hexdigest()
    for reference, path in result["evidence_paths"].items():
        with open(path, "rb") as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == reference
    for row in events:
        clean = {key: (value if value else None) for key, value in row.items()}
        clean["revision"] = int(clean["revision"])
        SourceEvent.model_validate_json(json.dumps(clean))
    receipt = json.loads((tmp_path / "collected/collection_receipt.json").read_text())
    assert receipt["status"] == "collected"
    assert len(receipt["downloads"]) == len(calls) == 2
    assert receipt["downloads"][0]["received_at"] == "2026-10-26T12:00:00+00:00"
    assert "PIT" in " ".join(result["qualifications"])
    assert receipt["unsupported_action_event_ids"]


def test_current_partial_bar_is_excluded(tmp_path, monkeypatch):
    data = chart("2026-10-26")
    data["chart"]["result"][0]["indicators"]["quote"][0]["close"][-1] = None
    install(monkeypatch, data, now="2026-10-26T19:30:00+00:00")
    result = daily_data.collect_daily("SOXL", tmp_path)
    assert result["origin_session_id"] == "2026-10-23"
    assert not any(
        row["session_id"] == "2026-10-26" and row["kind"] == "CLOSE_QUOTE"
        for row in read_events(result)
    )


def test_holiday_and_half_day_future_sessions(tmp_path, monkeypatch):
    install(monkeypatch, chart("2026-11-20"), now="2026-11-21T12:00:00+00:00")
    result = daily_data.collect_daily("SOXL", tmp_path)
    future = result["contract"]["calendar"]["sessions"][-5:]
    assert [s["session_id"] for s in future] == [
        "2026-11-23",
        "2026-11-24",
        "2026-11-25",
        "2026-11-27",
        "2026-11-30",
    ]
    assert future[3]["close_at"] == "2026-11-27T13:00:00-05:00"
    assert future[0]["open_at"] == "2026-11-23T09:30:00-05:00"


def saturday_new_year_calendar():
    # Published NYSE 2027/2028 rows; only New Year's Day 2028 is not observed.
    dates = [
        ("New Year’s Day", "2027-01-01", None),
        ("Martin Luther King, Jr. Day", "2027-01-18", "2028-01-17"),
        ("Washington's Birthday", "2027-02-15", "2028-02-21"),
        ("Good Friday", "2027-03-26", "2028-04-14"),
        ("Memorial Day", "2027-05-31", "2028-05-29"),
        ("Juneteenth", "2027-06-18", "2028-06-19"),
        ("Independence Day", "2027-07-05", "2028-07-04"),
        ("Labor Day", "2027-09-06", "2028-09-04"),
        ("Thanksgiving Day", "2027-11-25", "2028-11-23"),
        ("Christmas Day", "2027-12-24", "2028-12-25"),
    ]
    rows = "".join(
        f"<tr><td>{name}</td>"
        + "".join(
            f"<td>{date.fromisoformat(day).strftime('%A, %B %d') if day else '—*'}</td>"
            for day in (first, second)
        )
        + "</tr>"
        for name, first, second in dates
    )
    return (
        "All NYSE markets observe the following holidays. "
        "Core Trading Session: 9:30 a.m. to 4:00 p.m. ET"
        "<table><tr><th>Holiday</th><th>2027</th><th>2028</th></tr>"
        + rows
        + "</table><p>* Because the holiday falls on Saturday, January 1, 2028, "
        "no New Year’s Day holiday is observed.</p>"
        "<p>Each market will close early at 1:00 p.m. on Monday, July 3, 2028, "
        "Friday, November 26, 2027, and Friday, November 24, 2028.</p>"
    )


def test_explicit_nonobserved_new_year_preserves_calendar_coverage():
    holidays, early = daily_data._calendar(saturday_new_year_calendar().encode())
    assert len(holidays[2027]) == 10
    assert len(holidays[2028]) == 9
    assert daily_data._session(date(2027, 12, 31), holidays, early) is not None
    assert daily_data._session(date(2028, 1, 1), holidays, early) is None
    assert daily_data._session(date(2028, 1, 3), holidays, early) is not None
    assert daily_data._session(date(2028, 7, 3), holidays, early)["close_at"].endswith(
        "13:00:00-04:00"
    )


@pytest.mark.parametrize(
    "old,new",
    [
        ("no New Year’s Day holiday is observed.", ""),
        ("Saturday, January 1, 2028", "Saturday, January 1, 2027"),
        ("<td>New Year’s Day</td>", "<td>Other holiday</td>"),
        ("—*", "—"),
        ("Monday, January 17", "Monday, February 21"),
    ],
)
def test_nonobserved_holiday_requires_exact_evidence_and_other_dates(old, new):
    body = saturday_new_year_calendar().replace(old, new).encode()
    with pytest.raises(ValueError):
        daily_data._calendar(body)


@pytest.mark.parametrize(
    "last,received,problem,error",
    [
        ("2026-10-23", "2026-10-24T12:00:00+00:00", None, None),
        ("2026-05-01", "2026-05-02T12:00:00+00:00", None, "Need 120 complete"),
        ("2026-10-23", "2026-10-24T12:00:00+00:00", "gap", "Missing completed"),
        ("2026-12-31", "2026-12-31T22:00:00+00:00", None, "does not cover 2027"),
    ],
)
def test_leading_uncovered_bars_do_not_replace_required_covered_window(
    last, received, problem, error
):
    data = chart(last)["chart"]["result"][0]
    stamps = data["timestamp"]
    quotes = data["indicators"]["quote"][0]
    # The source's older prefix is not assigned guessed exchange-session metadata.
    stamps.insert(0, int(datetime(2025, 12, 31, 9, 30, tzinfo=EASTERN).timestamp()))
    for values in quotes.values():
        values.insert(0, None)
    if problem == "gap":
        stamps.pop(-10)
        for values in quotes.values():
            values.pop(-10)
    holidays, early = daily_data._calendar(CALENDAR)
    if error:
        with pytest.raises(ValueError, match=error):
            daily_data._complete_bars(
                stamps, quotes, holidays, early, datetime.fromisoformat(received)
            )
    else:
        bars, future = daily_data._complete_bars(
            stamps, quotes, holidays, early, datetime.fromisoformat(received)
        )
        assert len(bars) == 120 and len(future) == 5
        assert bars[-1][0]["session_id"] == last
        assert all(bar[0]["session_id"].startswith("2026-") for bar in bars)


@pytest.mark.parametrize("minute,origin", [(10, "2026-11-25"), (16, "2026-11-27")])
def test_half_day_completion_buffer(tmp_path, monkeypatch, minute, origin):
    install(monkeypatch, chart("2026-11-27"), now=f"2026-11-27T18:{minute}:00+00:00")
    assert daily_data.collect_daily("SOXL", tmp_path)["origin_session_id"] == origin


@pytest.mark.parametrize(
    "problem",
    [
        "symbol",
        "currency",
        "exchange",
        "type",
        "array",
        "duplicate",
        "bad_price",
        "missing_session",
        "stale",
        "short",
        "action_date",
        "provider_error",
    ],
)
def test_invalid_inputs_leave_failure_receipt_and_original_response(tmp_path, monkeypatch, problem):
    data = chart("2026-10-22" if problem == "stale" else "2026-10-23")
    result = data["chart"]["result"][0]
    quotes = result["indicators"]["quote"][0]
    if problem in {"symbol", "currency", "exchange", "type"}:
        key = {
            "symbol": "symbol",
            "currency": "currency",
            "exchange": "exchangeName",
            "type": "instrumentType",
        }[problem]
        result["meta"][key] = {
            "symbol": "OTHER",
            "currency": "EUR",
            "exchange": "NMS",
            "type": "CRYPTOCURRENCY",
        }[problem]
    elif problem == "array":
        quotes["close"].pop()
    elif problem == "duplicate":
        result["timestamp"][-1] = result["timestamp"][-2]
    elif problem == "bad_price":
        quotes["close"][-1] = None
    elif problem in {"missing_session", "short"}:
        indices = (
            [len(result["timestamp"]) - 10] if problem == "missing_session" else list(range(100))
        )
        for i in sorted(indices, reverse=True):
            result["timestamp"].pop(i)
            for values in quotes.values():
                values.pop(i)
    elif problem == "action_date":
        del result["events"]["dividends"]["old-cash"]["date"]
    elif problem == "provider_error":
        data["chart"]["error"] = {"code": "Not Found", "description": "No data"}
    raw, _ = install(monkeypatch, data)
    with pytest.raises(ValueError, match="Daily collection failed:"):
        daily_data.collect_daily("SOXL", tmp_path)
    receipt = json.loads((tmp_path / "collection_receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["error"]["message"]
    assert (tmp_path / "yahoo_chart_raw.json").read_bytes() == raw


def test_bad_ticker_and_existing_evidence_not_overwritten(tmp_path, monkeypatch):
    _, calls = install(monkeypatch)
    with pytest.raises(ValueError, match="Ticker"):
        daily_data.collect_daily("../SOXL", tmp_path)
    saved = (tmp_path / "collection_receipt.json").read_bytes()
    with pytest.raises(ValueError, match="must be empty"):
        daily_data.collect_daily("SOXL", tmp_path)
    assert (tmp_path / "collection_receipt.json").read_bytes() == saved
    assert not calls


@pytest.mark.parametrize(
    "calendar",
    [
        CALENDAR.replace(b"close early at 1:00", b"close early at 2:00"),
        CALENDAR.replace(b"2026</th>", b"2027</th>"),
        CALENDAR.replace(b"Core Trading Session: 9:30", b"Core Trading Session: 10:30"),
    ],
)
def test_calendar_evidence_changes_fail_closed(tmp_path, monkeypatch, calendar):
    install(monkeypatch, calendar=calendar)
    with pytest.raises(ValueError, match="Daily collection failed:"):
        daily_data.collect_daily("SOXL", tmp_path)
    assert json.loads((tmp_path / "collection_receipt.json").read_text())["status"] == "failed"


def test_http_error_is_preserved_without_retry(tmp_path, monkeypatch):
    calls = []

    def fail(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(
            request.full_url,
            429,
            "Too Many Requests",
            {"Content-Type": "text/plain"},
            io.BytesIO(b"rate limited"),
        )

    monkeypatch.setattr(daily_data, "urlopen", fail)
    with pytest.raises(ValueError, match="HTTP 429"):
        daily_data.collect_daily("SOXL", tmp_path)
    assert len(calls) == 1
    assert (tmp_path / "yahoo_chart_raw.json").read_bytes() == b"rate limited"
    receipt = json.loads((tmp_path / "collection_receipt.json").read_text())
    assert receipt["downloads"][0]["http_status"] == 429


def test_unrecognized_cash_schema_is_retained_not_interpreted(tmp_path, monkeypatch):
    data = chart()
    data["chart"]["result"][0]["events"]["dividends"]["old-cash"]["unknown_unit"] = "per_lot"
    install(monkeypatch, data)
    result = daily_data.collect_daily("SOXL", tmp_path)
    assert result["cash_actions"] == {}
    assert (
        len([row for row in read_events(result) if row["kind"] == "CORPORATE_ACTION_NOTICE"]) == 2
    )


def test_response_size_is_bounded_and_truncation_is_recorded(tmp_path, monkeypatch):
    install(monkeypatch)
    monkeypatch.setattr(daily_data, "MAX_RESPONSE_BYTES", 16)
    with pytest.raises(ValueError, match="collection limit"):
        daily_data.collect_daily("SOXL", tmp_path)
    assert (tmp_path / "yahoo_chart_raw.json").stat().st_size == 17
    receipt = json.loads((tmp_path / "collection_receipt.json").read_text())
    assert receipt["downloads"][0]["truncated"] is True
    assert receipt["status"] == "failed"


def test_future_calendar_year_requires_published_evidence(tmp_path, monkeypatch):
    install(monkeypatch, chart("2026-12-28"), now="2026-12-29T12:00:00+00:00")
    with pytest.raises(ValueError, match="does not cover 2027"):
        daily_data.collect_daily("SOXL", tmp_path)
    assert json.loads((tmp_path / "collection_receipt.json").read_text())["status"] == "failed"


def test_public_binary_float_prices_fit_registered_decimal_precision():
    from quant_rsi.market.market_contracts import decimal_text

    normalized = daily_data._price(144.16000366210938)
    assert normalized == "144.160003662109"
    assert decimal_text(normalized, positive=True) == normalized
    assert daily_data._price(83.55999755859375) == "83.559997558594"
    with pytest.raises(ValueError, match="too small"):
        daily_data._price(1e-13)
    with pytest.raises(ValueError, match="magnitude"):
        daily_data._price(1e16)
