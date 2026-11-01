"""Trading-session horizons across the connected study and retained observations."""

import json
from copy import deepcopy
from datetime import date, datetime, time, timedelta

import pytest
from test_daily_data import CALENDAR, EASTERN, HOLIDAYS, NASDAQ_CALENDAR, Response, chart, install

from quant_rsi.cli import main
from quant_rsi.market import daily_data, market_contracts
from quant_rsi.research import study, study_history
from quant_rsi.research.demo_prices import load_daily_prices
from quant_rsi.research.study_queue import study_arguments, study_options
from quant_rsi.research.study_workbench import Workbench


def command(root, horizon, *, model="ridge"):
    return [
        "research-study",
        "KO",
        "--llm",
        "simulated",
        "--news",
        "simulated",
        "--related",
        "simulated",
        "--prices",
        "synthetic",
        "--model",
        model,
        "--horizon-sessions",
        str(horizon),
        "--no-ui",
        "--output",
        str(root),
    ]


@pytest.mark.parametrize("horizon", [5, 20, 60])
@pytest.mark.parametrize("model", ["ridge", "lightgbm"])
def test_connected_loop_uses_one_horizon_through_both_fits_and_accounts(
    tmp_path, monkeypatch, horizon, model
):
    researcher, contexts = study.simulated_hypothesis, []

    def propose(context, *args, **kwargs):
        contexts.append(context["horizon_sessions"])
        return researcher(context, *args, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", propose)
    root = tmp_path / "study"
    assert main(command(root, horizon, model=model)) == 0
    result = json.loads((root / "result.json").read_bytes())
    assert result["configuration"]["horizon_sessions"] == horizon
    assert len(result["iterations"]["attempts"]) == 2
    assert contexts == [horizon, horizon]
    for attempt in result["iterations"]["attempts"]:
        prediction = attempt["prediction"]
        assert prediction["horizon_sessions"] == horizon
        assert (
            prediction["last_training_label_session"]
            < prediction["development"]["rows"][0]["origin_session"]
        )
        assert prediction["strategy"]["prediction_horizon_sessions"] == horizon
        assert (
            prediction["strategy"]["period"]["end"]
            == prediction["development"]["rows"][-1]["target_session"]
        )
        assert prediction["development"]["metrics"]["sample_count"] == 20
    terminal = result["terminal_evaluation"]
    assert terminal["status"] == "completed"
    assert terminal["strategy"]["prediction_horizon_sessions"] == horizon
    assert (
        json.loads((root / "evaluation_plan.json").read_bytes())["development_last_target_session"]
        < terminal["evaluation"]["rows"][0]["origin_session"]
    )
    assert f"{horizon}-session labels" in (root / "report.html").read_text()


HOLIDAYS_2025 = {
    "2025-01-01",
    "2025-01-20",
    "2025-02-17",
    "2025-04-18",
    "2025-05-26",
    "2025-06-19",
    "2025-07-04",
    "2025-09-01",
    "2025-11-27",
    "2025-12-25",
}


def long_collection(tmp_path, monkeypatch, last, horizon, *, archived=False, nasdaq=False):
    # Explicit fixture calendar and price history, not fetched market evidence.
    calendar = (
        "<h4>All NYSE markets observe the following holidays.</h4>"
        "<p>NYSE Arca Core Trading Session: 9:30 a.m. to 4:00 p.m. ET</p>"
        "<table><tr><th>Holiday</th><th>2025</th><th>2026</th></tr>"
        + "".join(
            f"<tr><td>Holiday {i}</td><td>{date.fromisoformat(a):%A, %B %d}</td>"
            f"<td>{date.fromisoformat(b):%A, %B %d}</td></tr>"
            for i, (a, b) in enumerate(zip(sorted(HOLIDAYS_2025), sorted(HOLIDAYS), strict=True))
        )
        + "</table><p>Each market will close early at 1:00 p.m. on "
        "Thursday, July 3, 2025, Friday, November 28, 2025, Wednesday, December 24, 2025, "
        "Friday, November 27, 2026 and Thursday, December 24, 2026.</p>"
    ).encode()
    days, day = [], date(2025, 1, 2)
    while day <= date.fromisoformat(last):
        if day.weekday() < 5 and day.isoformat() not in HOLIDAYS | HOLIDAYS_2025:
            days.append(day)
        day += timedelta(days=1)
    raw = chart()
    item = raw["chart"]["result"][0]
    item["meta"].update(symbol="KO", instrumentType="EQUITY", exchangeName="NYQ")
    if nasdaq:
        item["meta"]["exchangeName"] = "NMS"
    item["events"] = {}
    item["timestamp"] = [int(datetime.combine(d, time(9, 30), EASTERN).timestamp()) for d in days]
    item["indicators"]["quote"][0] = {
        "open": [60 + i * 0.01 for i in range(len(days))],
        "close": [60.1 + i * 0.01 for i in range(len(days))],
    }
    now = last + "T21:00:00+00:00"
    install(monkeypatch, data=raw, now=now, calendar=CALENDAR if archived else calendar)
    get = daily_data.urlopen
    pdf = b"Explicit offline fixture for the reviewed Nasdaq PDF"
    if nasdaq:
        monkeypatch.setattr(daily_data, "NASDAQ_PAST_CALENDAR_SHA256", daily_data._sha(pdf))

    def sources(request, timeout):
        if request.full_url == daily_data.NASDAQ_PAST_CALENDAR_URL:
            return Response(pdf, request.full_url)
        if request.full_url == daily_data.NASDAQ_CALENDAR_URL:
            return Response(NASDAQ_CALENDAR, request.full_url)
        if request.full_url == daily_data.NYSE_PAST_CALENDAR_URL:
            heading = (
                b"NYSE Group Announces 2025, 2026 and 2027 Holiday and Early Closings Calendar"
            )
            return Response(heading + calendar, request.full_url)
        return get(request, timeout)

    monkeypatch.setattr(daily_data, "urlopen", sources)
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: now)
    return study_history.collection_snapshot("KO", tmp_path, horizon_sessions=horizon)


@pytest.mark.parametrize("horizon", [20, 60])
@pytest.mark.parametrize("archived", [False, True])
def test_long_daily_calendar_publication_and_mature_observation(
    tmp_path, monkeypatch, horizon, archived
):
    prices = tmp_path / "prices"
    snapshot = long_collection(prices, monkeypatch, "2026-06-30", horizon, archived=archived)
    assert len(snapshot["dates"]) == 120 + 3 * (horizon - 5)
    calendar = snapshot["contract"]["calendar"]["sessions"]
    assert len(calendar) == len(snapshot["dates"]) + horizon
    assert snapshot["info"]["latest_target_session"] == calendar[-1]["session_id"]
    if archived:
        evidence = json.loads((prices / "calendar_evidence.json").read_bytes())
        assert evidence["historical_sources"][0]["year"] == 2025
        assert (prices / "nyse_calendar_2025.html").exists()
    history, root = tmp_path / "history", tmp_path / "study"
    args = command(root, horizon)
    args[args.index("--prices") + 1] = "daily"
    assert main(args + ["--prices-dir", str(prices), "--history-dir", str(history)]) == 0
    entry = study_history.history_view(history)["entries"][0]
    assert entry["status"] == "pending"
    assert entry["forecast"]["target_session"] == calendar[-1]["session_id"]
    assert entry["horizon_sessions"] == horizon
    assert study_history.mature_feedback(history, "KO")["history"]["entries"] == []
    assert study_history.recent_factor_recipes(history, "KO") == []
    assert len(study_history.recent_factor_recipes(history, "KO", horizon_sessions=horizon)) == 1
    # A normal observation collection is sufficient after the target matures,
    # even though its bounded Yahoo history request differs from the longer study.
    target = entry["forecast"]["target_session"]
    received = long_collection(tmp_path / "later", monkeypatch, target, 5)
    study_history.observe_prices(history, "KO", received)
    settled = study_history.mature_feedback(history, "KO", horizon_sessions=horizon)["entries"]
    assert len(settled) == 1 and settled[0]["status"] == "settled"
    assert settled[0]["horizon_sessions"] == horizon
    assert settled[0]["strategy"]["strategy"]["risk_statistics"]["daily_return_count"] == horizon
    for old, new in (("range=1y", "range=5y"), ("interval=1d", "interval=1wk")):
        changed = deepcopy(received)
        raw = changed["source_files"]["collection_receipt.json"]
        assert old in raw
        changed["source_files"]["collection_receipt.json"] = raw.replace(old, new)
        with pytest.raises(ValueError, match="frozen source endpoints"):
            study_history.observe_prices(history, "KO", changed)


def test_short_retained_inputs_are_not_overwritten_for_a_longer_horizon(tmp_path, monkeypatch):
    long_collection(tmp_path / "prices", monkeypatch, "2026-06-30", 5)
    receipt = tmp_path / "prices/collection_receipt.json"
    before = receipt.read_bytes()
    with pytest.raises(ValueError, match="fresh --prices-dir"):
        load_daily_prices("KO", receipt.parent, horizon_sessions=60)
    assert receipt.read_bytes() == before


@pytest.mark.parametrize("horizon", [True, 0, 10, 61, "20"])
def test_workbench_rejects_unsupported_horizons(horizon):
    with pytest.raises(ValueError, match="Prediction horizon"):
        study_options({"horizon_sessions": horizon})


def test_workbench_arguments_preserve_horizon(tmp_path):
    options = study_options({"horizon_sessions": 60})
    args = study_arguments(tmp_path / "task", tmp_path, "KO", options)
    assert args[args.index("--horizon-sessions") + 1] == "60"
    assert study_options({})["horizon_sessions"] == 5


def test_saved_template_keeps_horizon_after_restart(tmp_path):
    workbench = Workbench(tmp_path)
    plan = workbench.save_plan(
        {
            "name": "Business horizon",
            "symbols": ["KO"],
            "interval_hours": 24,
            "enabled": False,
            "options": {"horizon_sessions": 60},
        }
    )
    restarted = Workbench(tmp_path)
    batch = restarted.run_plan(plan["id"])
    task = restarted.task(batch["tasks"][1]["id"])
    manifest = json.loads((task / "task.json").read_bytes())
    assert manifest["horizon_sessions"] == 60
    args = manifest["arguments"]
    assert args[args.index("--horizon-sessions") + 1] == "60"
    assert restarted.overview()["tasks"][1]["horizon_sessions"] == 60


def test_long_horizon_keeps_news_and_business_factor_path_runnable(tmp_path):
    root = tmp_path / "business"
    assert main(command(root, 60) + ["--news-research", "--financials"]) == 0
    result = json.loads((root / "result.json").read_bytes())
    for attempt in result["iterations"]["attempts"]:
        assert any(key.startswith("financial.") for key in attempt["factor_ids"])
        assert attempt["prediction"]["horizon_sessions"] == 60


def test_long_calendar_fails_when_published_year_is_missing(tmp_path, monkeypatch):
    install(monkeypatch)
    with pytest.raises(ValueError, match="does not cover 2027"):
        daily_data.collect_daily("SOXL", tmp_path / "calendar", future_sessions=60)


def test_nasdaq_history_uses_reviewed_own_exchange_schedule(tmp_path, monkeypatch):
    snapshot = long_collection(tmp_path, monkeypatch, "2026-06-30", 60, archived=True, nasdaq=True)
    assert len(snapshot["dates"]) == 285
    evidence = json.loads((tmp_path / "calendar_evidence.json").read_bytes())
    assert evidence["historical_sources"][0]["method"] == "reviewed_pdf_transcription"
    assert evidence["historical_sources"][0]["source_url"] == daily_data.NASDAQ_PAST_CALENDAR_URL
    days = {s["session_id"]: s for s in snapshot["contract"]["calendar"]["sessions"]}
    assert "2025-11-27" not in days
    assert days["2025-11-28"]["close_at"] == "2025-11-28T13:00:00-05:00"
    with pytest.raises(ValueError, match="differs from the reviewed"):
        daily_data._past_calendar(b"Changed source", nasdaq=True)
    with pytest.raises(ValueError, match="heading"):
        daily_data._past_calendar(CALENDAR, nasdaq=False)
