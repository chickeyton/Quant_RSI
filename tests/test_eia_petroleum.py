"""Offline official-layout fixtures; these do not establish predictive value."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from quant_rsi.research import eia_petroleum as eia

TABLE = (
    "STUB_1,10/23/26,10/16/26,Difference,10/24/25,Percent Change,10/25/24,Percent Change\n"
    "Commercial (Excluding SPR),410.625,412.517,-1.892,415.966,-1.300,425.509,-3.500\n"
    "Total Motor Gasoline,209.2,205.3,3.9,210.7,-0.7,210.9,-0.8\n"
    "Distillate Fuel Oil,102.8,104.3,-1.5,112.2,-8.3,112.9,-8.9\n"
).encode()
PAGE = eia.ARCHIVE_URL + "2026/2026_10_28/wpsr_2026_10_28.php"
INDEX = f'<a href="{PAGE}">28</a>'.encode()
PAGE_TEXT = (
    "Data for week ending October 23, 2026 | <strong>Release Date:</strong> October 28, 2026"
    '<a href="download/table4.csv">CSV</a>'
).encode()


def test_archive_uses_actual_dated_links_and_rejects_unrelated_hosts():
    raw = INDEX + INDEX + INDEX.replace(b"www.eia.gov", b"example.com")
    assert eia.release_links(raw, date(2026, 10, 30), date(2026, 10, 30)) == [
        (date(2026, 10, 28), PAGE)
    ]
    with pytest.raises(ValueError, match="No EIA"):
        eia.release_links(raw, date(2026, 10, 1), date(2026, 10, 27))


@pytest.mark.parametrize("path", ["download/table4.csv", "csv/table4.csv"])
def test_release_page_resolves_archive_layout_and_checks_publication(path):
    page = PAGE_TEXT.replace(b"download/table4.csv", path.encode())
    url, observation = eia.parse_release_page(page, PAGE, date(2026, 10, 28))
    assert url == PAGE.rsplit("/", 1)[0] + "/" + path
    assert observation == date(2026, 10, 23)
    with pytest.raises(ValueError, match="date differs"):
        eia.parse_release_page(page, PAGE, date(2026, 10, 29))
    with pytest.raises(ValueError, match="CSV link"):
        eia.parse_release_page(
            page.replace(path.encode(), b"https://example.com/table4.csv"), PAGE, date(2026, 10, 28)
        )


def test_changes_use_same_release_comparatives_and_wait_for_publication():
    records = eia.parse_table(TABLE, date(2026, 10, 28))
    snapshot = {
        "records": records,
        "provider": "eia_weekly_petroleum",
        "sha256": "fixture",
        "failures": [],
        "notice": eia.NOTICE,
    }
    cutoffs = [
        "2026-10-23T20:00:00+00:00",  # observation date is not availability
        "2026-10-28T20:00:00+00:00",  # date-only release conservatively waits
        "2026-10-29T05:00:00+00:00",
        "2026-11-12T05:00:01+00:00",  # older than fourteen days
    ]
    bundle = eia.prepare_petroleum(snapshot, cutoffs, [2])
    values = bundle["columns"]["external.eia_commercial_crude_wow"]
    assert values[:2] == [None, None]
    assert values[2] == pytest.approx(410.625 / 412.517 - 1)
    assert values[3] is None
    assert all(item["runnable"] for item in bundle["definitions"])
    assert not any(
        item["runnable"] for item in eia.prepare_petroleum(snapshot, cutoffs, [1, 2])["definitions"]
    )
    assert len(values) == len(cutoffs)  # missing inputs never shrink the sample
    with pytest.raises(ValueError, match="timezones"):
        eia.prepare_petroleum(snapshot, ["2026-10-30T20:00:00"], [0])


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\n",
        TABLE.replace(b"10/16/26", b"10/15/26"),
        TABLE.replace(b"410.625", b"NaN"),
        TABLE.replace(b"412.517", b"0"),
        TABLE.replace(b"Commercial (Excluding SPR)", b"Unidentified inventory"),
        TABLE + TABLE.splitlines()[1] + b"\n",
    ],
)
def test_invalid_tables_do_not_become_factors(raw):
    with pytest.raises(ValueError):
        eia.parse_table(raw, date(2026, 10, 28))


def test_collection_reuses_retained_sources_and_detects_tampering(tmp_path, monkeypatch):
    calls = []

    def download(url):
        calls.append(url)
        return INDEX if url == eia.ARCHIVE_URL else PAGE_TEXT if url == PAGE else TABLE

    monkeypatch.setattr(eia, "_download", download)
    snapshot = eia.collect_petroleum(tmp_path, "2026-10-01", "2026-10-30")
    assert len(calls) == 3
    assert len(snapshot["records"]) == 3
    assert not snapshot["failures"]
    assert all(item["received_at"] for item in snapshot["resources"])
    assert eia.collect_petroleum(tmp_path, "2026-10-01", "2026-10-30") == snapshot
    assert len(calls) == 3
    with pytest.raises(ValueError, match="identity changed"):
        eia.collect_petroleum(tmp_path, "2026-10-02", "2026-10-30")
    (tmp_path / snapshot["resources"][-1]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="resource changed"):
        eia.collect_petroleum(tmp_path, "2026-10-01", "2026-10-30")


def test_failed_release_is_retained_without_fabricating_data(tmp_path, monkeypatch):
    def download(url):
        if url == eia.ARCHIVE_URL:
            return INDEX
        raise OSError("Offline source failure")

    monkeypatch.setattr(eia, "_download", download)
    snapshot = eia.collect_petroleum(tmp_path, "2026-10-01", "2026-10-30")
    assert snapshot["records"] == []
    assert len(snapshot["failures"]) == 1
    bundle = eia.prepare_petroleum(snapshot, ["2026-10-30T20:00:00+00:00"], [0])
    assert all(not item["runnable"] for item in bundle["definitions"])
    assert all(column == [None] for column in bundle["columns"].values())


def test_external_columns_participate_in_two_real_fits_and_revision():
    from quant_rsi.research.demo import load_catalog
    from quant_rsi.research.demo_model import LightGBMDemo

    library = load_catalog(Path(__file__).resolve().parents[1] / "instruments/shared/equities.json")
    model = LightGBMDemo(library, evaluation_sessions=20)
    model.configure_peers({"definitions": [], "columns": {}, "support": []})
    cutoffs = [day + "T20:00:00+00:00" for day in model.data.dates]
    start = datetime.fromisoformat(cutoffs[0]) - timedelta(days=14)
    end = datetime.fromisoformat(cutoffs[-1])
    records = []
    release = start
    week = 0
    while release < end:
        for index, field in enumerate(eia.FIELDS):
            records.append(
                {
                    "field": field,
                    "available_at": release.astimezone(timezone.utc).isoformat(),
                    "value": 100 + (week * (index + 1)) % 17,
                    "prior_value": 100,
                }
            )
        release += timedelta(days=7)
        week += 1
    bundle = eia.prepare_petroleum(
        {
            "records": records,
            "provider": "simulated_eia_layout",
            "sha256": "fixture",
            "failures": [],
            "notice": "SIMULATED industry data; engineering verification only.",
        },
        cutoffs,
        [*model.train, *model.origins],
    )
    # Latest data can expire after the fixed development window. Keep the
    # completed research fits, but do not publish a forecast with missing inputs.
    for column in bundle["columns"].values():
        column[-1] = None
    model.configure_external(bundle)
    assert model.context("XOM")["external_features"]["definitions"] == bundle["definitions"]
    model.price_info = {"latest_target_session": "2030-01-01"}
    initial = bundle["definitions"][0]
    revised = bundle["definitions"][1]
    price = model.library["definitions"][0]
    first = model.predict({"symbol": "XOM"}, [price, initial], "Crude inventory hypothesis")
    feedback = model.observe({"symbol": "XOM"}, first)
    second = model.predict({"symbol": "XOM"}, [price, revised], "Revise to gasoline inventory")
    assert first["factor_ids"] != second["factor_ids"]
    assert first["training_feature_columns"] == second["training_feature_columns"] == 2
    assert first["development"]["metrics"]["sample_count"] == 20
    assert second["development"]["metrics"]["sample_count"] == 20
    assert feedback
    for fit, factor in zip(model._fits, [initial, revised], strict=True):
        np.testing.assert_allclose(
            fit[1][:, 1], np.asarray(bundle["columns"][factor["factor_id"]], dtype=float)
        )
    for prediction, factor in [(first, initial), (second, revised)]:
        usage = next(
            item
            for item in prediction["input_usage"]
            if item["source"] == "publication_dated_external_drivers"
        )
        assert usage["status"] == "used"
        assert usage["factor_ids"] == [factor["factor_id"]]
        assert factor["factor_id"] not in prediction["input_usage"][0]["factor_ids"]
        assert prediction["latest_forecast_unavailable"]["missing_factor_ids"] == [
            factor["factor_id"]
        ]
        assert "latest_forecast" not in prediction
    with pytest.raises(ValueError, match="configured once"):
        model.configure_external(bundle)
    forged = initial | {"recipe": initial["recipe"] | {"metric": "invented"}}
    with pytest.raises(ValueError, match="exact retained usable source"):
        model.predict({"symbol": "XOM"}, [price, forged], "Invalid source")


def test_cli_runs_external_revision_and_exposes_input_check(tmp_path):
    import json

    from test_research_study import command

    from quant_rsi.cli import main
    from quant_rsi.research.study_ui import read_state
    from quant_rsi.research.study_ui_views import visual_output

    root = tmp_path / "study"
    assert main(command(root, "--llm", "simulated", "--external-drivers", "--no-ui")) == 0
    result = json.loads((root / "result.json").read_bytes())
    external_ids = []
    for attempt in result["iterations"]["attempts"]:
        prediction = attempt["prediction"]
        external_ids.append([x for x in prediction["factor_ids"] if x.startswith("external.")])
        assert external_ids[-1]
    assert external_ids[0] != external_ids[1]
    comparison = result["study"]["external_comparison"]
    assert comparison["status"] == "completed" and comparison["extra_fits"] == 1
    assert result["configuration"]["comparison_fit_budget"] == 1
    assert (
        comparison["removed_factor_ids"] == external_ids[result["iterations"]["selected_round"] - 1]
    )
    assert comparison["kept_factor_ids"]
    assert comparison["without_external_drivers"]["forecast"]["sample_count"] == 20
    assert "External drivers removed" in (root / "report.html").read_text()
    view = visual_output("external_comparison", comparison, {}, "KO")
    assert len(view["rows"]) == 2
    assert view["rows"][0]["mae"] == comparison["with_external_drivers"]["forecast"]["mae_price"]
    assert "SIMULATED" in result["study"]["external_data"]["notice"]
    assert "External business drivers" in (root / "report.html").read_text()
    check = json.loads((root / "input_report.json").read_bytes())
    assert len(check["external_drivers"]["definitions"]) == 3
    assert read_state(root)

    preflight = tmp_path / "check"
    # Real provider selected but never invoked during a preflight.
    assert (
        main(
            command(preflight, "--llm", "codex", "--external-drivers", "--check-inputs", "--no-ui")
        )
        == 0
    )
    check = json.loads((preflight / "input_report.json").read_bytes())
    assert all(d["runnable"] for d in check["external_drivers"]["definitions"])
    assert not (preflight / "result.json").exists()
    assert not list((preflight / "checkpoints").glob("fit_*.json"))
    assert not list((preflight / "checkpoints").glob("hypothesis_*.json"))
