"""Offline audit-package fixtures; isolated registry only, never a fit."""

import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant_rsi import forecast_context_data as context
from quant_rsi.artifacts import canonical, digest
from quant_rsi.forecast_inputs import ForecastInputs
from quant_rsi.forecast_kernel import action_assessment

ET = ZoneInfo("America/New_York")
HOLIDAYS = {
    2023: "01-02 01-16 02-20 04-07 05-29 06-19 07-04 09-04 11-23 12-25",
    2024: "01-01 01-15 02-19 03-29 05-27 06-19 07-04 09-02 11-28 12-25",
    2025: "01-01 01-20 02-17 04-18 05-26 06-19 07-04 09-01 11-27 12-25",
    2026: "01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25",
    2027: "01-01 01-18 02-15 03-26 05-31 06-18 07-05 09-06 11-25 12-24",
}
CLOSED = {f"{year}-{value}" for year, days in HOLIDAYS.items() for value in days.split()}
CLOSED.add("2025-01-09")
EARLY = {
    "2023-07-03",
    "2023-11-24",
    "2024-07-03",
    "2024-11-29",
    "2024-12-24",
    "2025-07-03",
    "2025-11-28",
    "2025-12-24",
    "2026-11-27",
    "2026-12-24",
}


def calendar_html(years, *, current=False):
    text = (
        "<p>All NYSE markets observe these holidays.</p>"
        "<p>Core Trading Session: 9:30 a.m. to 4:00 p.m. ET</p>"
        if current
        else ""
    )
    text += "<table>"
    if not current:
        text += "<tr><td></td><td> </td><td>&nbsp;</td><td></td></tr>"
    text += "<tr><th>HOLIDAY</th>"
    text += "".join(f"<th>{year}</th>" for year in years) + "</tr>"
    for i in range(10):
        text += f"<tr><td>Holiday {i}</td>"
        for year in years:
            day = date.fromisoformat(f"{year}-{HOLIDAYS[year].split()[i]}")
            text += f"<td>{day.strftime('%A, %B %d')}</td>"
        text += "</tr>"
    text += "</table>"
    early = [date.fromisoformat(d) for d in sorted(EARLY) if int(d[:4]) in years]
    text += "<p>Each market will close early at 1:00 p.m. on "
    text += "; ".join(d.strftime("%A, %B %d, %Y") for d in early) + ".</p>"
    return text.encode()


def session(day):
    return day.weekday() < 5 and day.isoformat() not in CLOSED


def write_manifest(audit):
    rows = []
    for p in sorted(audit.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            body = p.read_bytes()
            rows.append(
                {"path": str(p.relative_to(audit)), "bytes": len(body), "sha256": digest(body)}
            )
    (audit / "manifest.json").write_bytes(canonical({"files": rows}))


def audit_package(tmp_path, origin="2026-10-26", *, cash=True, split=False, received=None):
    audit = tmp_path / "audit"
    (audit / "raw").mkdir(parents=True)
    end = date.fromisoformat(origin)
    days, day = [], end
    while len(days) < 42:
        if session(day):
            days.append(day)
        day -= timedelta(days=1)
    days.reverse()
    close = datetime.combine(end, time(13 if origin in EARLY else 16), ET)
    received = received or (close + timedelta(minutes=30)).isoformat()
    urls = dict(context.CALENDAR_SOURCES)
    bodies = {
        "NYSE_calendar_current.html": calendar_html([2025, 2026, 2027], current=True),
        "NYSE_holidays_2023_2025.html": calendar_html([2023, 2024, 2025]),
        "NYSE_holidays_2024_2026.html": calendar_html([2024, 2025, 2026]),
        "NYSE_closure_20250109.html": (
            b"<p>close all NYSE Group equity and options markets on "
            b"Thursday, January 9, 2025, including NYSE Arca Equities.</p>"
        ),
    }
    for ticker in context.ROLES.values():
        actions = {}
        stamp = int(datetime.combine(days[-4], time(9, 30), ET).timestamp())
        if cash:
            actions["dividends"] = {str(stamp): {"amount": 0.093, "date": stamp}}
        if split:
            actions["splits"] = {
                str(stamp - 86400): {
                    "date": stamp - 86400,
                    "numerator": 2,
                    "denominator": 1,
                    "splitRatio": "2:1",
                }
            }
        quote = {
            "open": [100.0 + i / 10 for i in range(len(days))],
            "close": [100.1 + i / 10 for i in range(len(days))],
        }
        chart = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "symbol": ticker,
                            "currency": "USD",
                            "instrumentType": "ETF",
                            "exchangeName": "PCX",
                            "exchangeTimezoneName": "America/New_York",
                            "dataGranularity": "1d",
                        },
                        "timestamp": [
                            int(datetime.combine(d, time(9, 30), ET).timestamp()) for d in days
                        ],
                        "indicators": {"quote": [quote]},
                        "events": actions,
                    }
                ],
            }
        }
        name = f"{ticker}_chart_5y.json"
        bodies[name] = canonical(chart)
        urls[name] = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            "?range=5y&interval=1d&events=div%2Csplits"
        )
    for name, body in bodies.items():
        (audit / "raw" / name).write_bytes(body)
        receipt = dict(
            url=urls[name],
            status=200,
            error=None,
            bytes=len(body),
            sha256=digest(body),
            requested_at=received,
            received_at=received,
        )
        (audit / (name + ".receipt.json")).write_bytes(canonical(receipt))
    write_manifest(audit)
    return audit


def replace_response(audit, name, body):
    (audit / "raw" / name).write_bytes(body)
    p = audit / (name + ".receipt.json")
    receipt = json.loads(p.read_text())
    receipt.update(bytes=len(body), sha256=digest(body))
    p.write_bytes(canonical(receipt))
    write_manifest(audit)


def test_isolated_inputs_roles_future_dst_and_clock(tmp_path, monkeypatch):
    audit = audit_package(tmp_path)
    prior = tmp_path / "must-not-touch"
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(prior))
    result = context.prepare_context_data(audit, tmp_path / "prepared", history_sessions=24)
    inputs = ForecastInputs(result["inputs"])
    assert os.environ["QUANT_RSI_WORKSPACE"] == str(prior) and not prior.exists()
    assert inputs.target[0].instruments == ("SOXL",)
    assert inputs.contexts["industry"][0].instruments == ("XSD",)
    assert inputs.contexts["market"][0].instruments == ("SPY",)
    receipt = json.loads(Path(result["receipt_path"]).read_text())
    for role in ["target", "industry", "market"]:
        wrapped = (
            result["inputs"]["target"] if role == "target" else result["inputs"]["contexts"][role]
        )
        data = wrapped["data"]
        assert data["specification"]["sessions"] == 24
        assert data["specification"]["train_cutoff"] == 19
        assert data["specification"]["seed"] == 7
        assert len(data["contract"]["calendar"]["sessions"]) == 29
        assert data["contract"]["knowledge_basis"] == "assumed_historical_availability"
        assert data["availability_evidence_hex"]
        assert len(wrapped["objective"]["notices"]) == 1
        http = datetime.fromisoformat(receipt["imports"][role]["http_received_at"])
        for row in data["snapshot"]["receipts"]:
            assert datetime.fromisoformat(row["first_seen_at"]) >= http
    sessions = inputs.target[0].contract.calendar.sessions
    assert sessions[-1].session_id == "2026-11-02"
    assert sessions[-1].open_at.astimezone(timezone.utc).hour == 14
    assert sessions[-6].open_at.astimezone(timezone.utc).hour == 13
    assert Path(result["meta"]["registry_path"]).is_dir()
    assert all(digest(Path(p).read_bytes()) == h for h, p in result["evidence_paths"].items())


def test_exceptional_closure_and_early_close_from_official_evidence(tmp_path):
    audit = audit_package(tmp_path, origin="2025-01-10")
    result = context.prepare_context_data(audit, tmp_path / "prepared", history_sessions=24)
    data = ForecastInputs(result["inputs"]).target[0]
    sessions = {s.session_id: s for s in data.contract.calendar.sessions}
    assert "2025-01-09" not in sessions
    assert sessions["2024-12-24"].close_at.hour == 13
    assert sessions["2025-01-10"].close_at.hour == 16


@pytest.mark.parametrize("defect", ["common_missing", "closed_day", "wrong_open", "wrong_symbol"])
def test_price_calendar_mismatch_rejected_before_registry(tmp_path, defect):
    audit = audit_package(tmp_path, origin="2025-01-10")
    for ticker in context.ROLES.values():
        name = f"{ticker}_chart_5y.json"
        payload = json.loads((audit / "raw" / name).read_text())
        value = payload["chart"]["result"][0]
        if defect == "common_missing":
            value["timestamp"].pop(-2)
            for col in value["indicators"]["quote"][0].values():
                col.pop(-2)
        elif defect == "closed_day":
            stamp = int(datetime(2025, 1, 9, 9, 30, tzinfo=ET).timestamp())
            value["timestamp"].insert(-1, stamp)
            for col in value["indicators"]["quote"][0].values():
                col.insert(-1, 100.0)
        elif defect == "wrong_open":
            value["timestamp"][-1] += 60
        else:
            value["meta"]["symbol"] = "WRONG"
        replace_response(audit, name, canonical(payload))
    out = tmp_path / "prepared"
    with pytest.raises(ValueError):
        context.prepare_context_data(audit, out, history_sessions=24)
    assert not (out / "workspace").exists()
    assert (out / "failure.receipt.json").exists()


def test_incomplete_early_close_bar_is_not_complete(tmp_path):
    audit = audit_package(tmp_path, origin="2024-11-29", received="2024-11-29T18:14:00+00:00")
    with pytest.raises(ValueError, match="origin is not complete"):
        context.prepare_context_data(
            audit, tmp_path / "prepared", history_sessions=24, origin_session_id="2024-11-29"
        )


def test_manifest_and_special_closure_fail_closed(tmp_path):
    audit = audit_package(tmp_path)
    name = "NYSE_closure_20250109.html"
    replace_response(audit, name, b"<p>Unknown schedule</p>")
    with pytest.raises(ValueError, match="closure evidence"):
        context.prepare_context_data(audit, tmp_path / "bad-calendar", history_sessions=24)
    (audit / "raw" / "SOXL_chart_5y.json").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        context.prepare_context_data(audit, tmp_path / "bad-hash", history_sessions=24)


def test_unknown_split_retained_and_kernel_abstains(tmp_path):
    audit = audit_package(tmp_path, split=True)
    result = context.prepare_context_data(audit, tmp_path / "prepared", history_sessions=24)
    data, objective = ForecastInputs(result["inputs"]).target
    actions = [
        item
        for item in data.snapshot["events"]
        if item["event"]["kind"] == "CORPORATE_ACTION_NOTICE"
    ]
    assert len(actions) == 2 and len(objective.notices) == 1
    assessment = action_assessment(
        actions,
        "SOXL",
        data.contract.calendar.sessions[0].open_at,
        data.contract.calendar.sessions[-1].close_at,
        objective,
    )
    assert "forecast_action_effect_or_type_unknown" in str(assessment)


def test_signal_evidence_is_copied_and_bound(tmp_path):
    audit = audit_package(tmp_path)
    evidence = tmp_path / "signal-evidence"
    evidence.write_bytes(b"signal source evidence")
    signals = {
        "snapshot": {
            "schema_version": 1,
            "kind": "forecast_signal_snapshot",
            "events": [],
            "coverage": [],
        },
        "evidence_paths": {digest(evidence.read_bytes()): str(evidence)},
    }
    result = context.prepare_context_data(
        audit, tmp_path / "prepared", history_sessions=24, signals=signals
    )
    evidence.unlink()
    parsed = ForecastInputs(result["inputs"])
    assert parsed.signals is not None
    assert bytes.fromhex(result["inputs"]["evidence_hex"][digest(b"signal source evidence")]) == (
        b"signal source evidence"
    )


def test_no_fixed_path_no_output_overwrite(tmp_path):
    audit = audit_package(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        context.prepare_context_data(audit, audit / "new-output", history_sessions=24)
    assert not (audit / "new-output").exists()
    out = tmp_path / "prepared"
    out.mkdir()
    (out / "keep").write_text("untouched")
    with pytest.raises(FileExistsError):
        context.prepare_context_data(audit, out, history_sessions=24)
    assert (out / "keep").read_text() == "untouched"


def test_cash_precision_is_not_rounded_into_a_supported_notice(tmp_path):
    audit = audit_package(tmp_path)
    for ticker in context.ROLES.values():
        name = f"{ticker}_chart_5y.json"
        value = json.loads((audit / "raw" / name).read_text())
        action = next(iter(value["chart"]["result"][0]["events"]["dividends"].values()))
        action["amount"] = 0.1234567890123
        replace_response(audit, name, canonical(value))
    result = context.prepare_context_data(audit, tmp_path / "prepared", history_sessions=24)
    data, objective = ForecastInputs(result["inputs"]).target
    assert not objective.notices
    actions = [
        v for v in data.snapshot["events"] if v["event"]["kind"] == "CORPORATE_ACTION_NOTICE"
    ]
    assert len(actions) == 1
    assert "forecast_action_effect_or_type_unknown" in str(
        action_assessment(
            actions,
            "SOXL",
            data.contract.calendar.sessions[0].open_at,
            data.contract.calendar.sessions[-1].close_at,
            objective,
        )
    )
