"""Small offline EIA HTML fixtures; no live provider or predictive-quality claims."""

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_rsi.research import eia_petroleum as eia
from quant_rsi.research.eia_price_history import PRICE_HISTORY_URLS, TITLES, parse_price_history


def price_page(field="wti", *, release="10/28/2026", last="107"):
    def row(label, values):
        cells = "".join(f"<td class='B3'>{v}</td>" for v in values)
        return f"<tr><td class='B6'>&nbsp;{label}</td>{cells}</tr>"

    return (
        f"<title>{TITLES[field]}</title>"
        + row("2026 Oct-19 to Oct-23", [100, 101, 102, 103, 104])
        + row("2026 Oct-26 to Oct-30", [106, last, "", "", ""])
        + f"<td class='F2'>Release Date: {release}</td>"
        + "<td class='F2'>Next Release Date: 11/4/2026</td>"
    ).encode()


def source_fixture(url):
    from test_eia_petroleum import INDEX, PAGE, PAGE_TEXT, TABLE

    if url in PRICE_HISTORY_URLS.values():
        return price_page(
            next(field for field, value in PRICE_HISTORY_URLS.items() if value == url)
        )
    return INDEX if url == eia.ARCHIVE_URL else PAGE_TEXT if url == PAGE else TABLE


def test_all_quotes_wait_for_one_publication_and_expire():
    record = parse_price_history(price_page(), "wti", "2026-10-29T09:00:00+00:00")
    assert record["observation_date"] == "2026-10-27"
    assert record["prior_observation_date"] == "2026-10-20"
    assert record["value"] == 107 and record["prior_value"] == 101
    assert record["available_at"] == "2026-10-29T05:00:00+00:00"
    snapshot = dict(
        records=[record], price_tables=True, provider="fixture", sha256="", failures=[], notice=""
    )
    bundle = eia.prepare_petroleum(
        snapshot,
        [
            "2026-10-27T20:00:00+00:00",
            "2026-10-28T20:00:00+00:00",
            "2026-10-29T05:00:00+00:00",
            "2026-11-12T05:00:01+00:00",
        ],
        [2],
    )
    assert bundle["columns"]["external.eia_wti_5obs_return"] == [None, None, 107 / 101 - 1, None]


@pytest.mark.parametrize(
    "old,new",
    [
        (b"Dollars per Barrel", b"Cents per Gallon"),
        (b"Release Date: 10/28", b"Release Date: 11/28"),
        (b"Release Date: 10/28/2026", b"Updated: 10/28/2026"),
        (b"Oct-26 to Oct-30", b"Oct-27 to Oct-31"),
        (b">107<", b">NaN<"),
        (b">107<", b">0<"),
        (b"Release Date: 10/28", b"Release Date: 10/26"),
    ],
)
def test_rejects_wrong_series_dates_or_values(old, new):
    with pytest.raises(ValueError):
        parse_price_history(price_page().replace(old, new), "wti", "2026-10-29T09:00:00+00:00")


def test_new_batches_keep_prior_publications_and_replay_without_history(tmp_path, monkeypatch):
    history = tmp_path / "history"
    monkeypatch.setattr(eia, "_download", source_fixture)
    first = eia.collect_petroleum(
        tmp_path / "first", "2026-10-01", "2026-10-30", history_directory=history
    )
    original = (tmp_path / "first/snapshot.json").read_bytes()
    receipts = sorted((history / "eia_price_publications").glob("*.json"))
    assert len(receipts) == 2
    original_receipts = {p.name: p.read_bytes() for p in receipts}

    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 6, tzinfo=timezone.utc)

    monkeypatch.setattr(eia, "datetime", Later)

    def next_publication(url):
        raw = source_fixture(url)
        if url in PRICE_HISTORY_URLS.values():
            raw = raw.replace(b"10/28/2026", b"11/4/2026").replace(
                b"Oct-26 to Oct-30", b"Nov- 2 to Nov- 6"
            )
            raw = raw.replace(b">107<", b">120<")
        return raw

    monkeypatch.setattr(eia, "_download", next_publication)
    second = eia.collect_petroleum(
        tmp_path / "second", "2026-10-01", "2026-11-06", history_directory=history
    )
    assert not second["failures"]
    assert len(second["records"]) == 7
    cutoffs = ["2026-10-30T20:00:00+00:00", "2026-11-06T21:00:00+00:00"]
    columns = eia.prepare_petroleum(second, cutoffs, [0, 1])["columns"]
    assert columns["external.eia_wti_5obs_return"] == [107 / 101 - 1, 120 / 101 - 1]
    assert all(
        (history / "eia_price_publications" / name).read_bytes() == raw
        for name, raw in original_receipts.items()
    )
    assert (tmp_path / "first/snapshot.json").read_bytes() == original
    shutil.rmtree(history)
    monkeypatch.setattr(eia, "_download", lambda _: pytest.fail("Replay must not access provider"))
    assert (
        eia.collect_petroleum(
            tmp_path / "second", "2026-10-01", "2026-11-06", history_directory=history
        )
        == second
    )
    assert eia.collect_petroleum(tmp_path / "first", "2026-10-01", "2026-10-30") == first


def test_source_failure_or_revision_keeps_original_publication(tmp_path, monkeypatch):
    history = tmp_path / "history"
    monkeypatch.setattr(eia, "_download", source_fixture)
    first = eia.collect_petroleum(
        tmp_path / "first", "2026-10-01", "2026-10-30", history_directory=history
    )

    def changed(url):
        if url == PRICE_HISTORY_URLS["brent"]:
            raise OSError("Source unavailable")
        return source_fixture(url).replace(b">107<", b">999<")

    monkeypatch.setattr(eia, "_download", changed)
    second = eia.collect_petroleum(
        tmp_path / "second", "2026-10-01", "2026-10-30", history_directory=history
    )
    assert second["records"] == first["records"]
    assert len(second["failures"]) == 2
    assert "retained original" in second["failures"][0]["reason"]
    receipt = history / "eia_price_publications/wti_2026-10-28.json"
    item = json.loads(receipt.read_bytes())
    (receipt.parent / item["file"]).write_bytes(b"tampered")
    third = eia.collect_petroleum(
        tmp_path / "third", "2026-10-01", "2026-10-30", history_directory=history
    )
    assert not any(row["field"] == "wti" for row in third["records"])
    assert any("publication changed" in row["reason"] for row in third["failures"])


def test_current_prices_survive_missing_archive_and_respect_window(tmp_path, monkeypatch):
    def download(url):
        if url == eia.ARCHIVE_URL:
            raise OSError("Archive unavailable")
        return source_fixture(url)

    monkeypatch.setattr(eia, "_download", download)
    snapshot = eia.collect_petroleum(tmp_path / "current", "2026-10-01", "2026-10-30")
    assert {r["field"] for r in snapshot["records"]} == {"wti", "brent"}
    old = eia.collect_petroleum(tmp_path / "old", "2026-10-01", "2026-10-27")
    assert not old["records"]


def test_new_publication_enables_latest_fit_without_changing_development():
    import numpy as np

    from quant_rsi.research.demo import load_catalog
    from quant_rsi.research.demo_model import LightGBMDemo

    library = load_catalog(Path(__file__).resolve().parents[1] / "instruments/shared/equities.json")
    models = [LightGBMDemo(library, evaluation_sessions=20) for _ in range(2)]
    cutoffs = [day + "T20:00:00+00:00" for day in models[0].data.dates]
    records, release = [], datetime(2019, 12, 27, tzinfo=timezone.utc)
    while release < datetime(2020, 6, 6, tzinfo=timezone.utc):
        records.append(
            {
                "field": "wti",
                "available_at": release.isoformat(),
                "observation_date": release.date().isoformat(),
                "value": 100 + len(records) % 7,
                "prior_value": 100,
            }
        )
        release += timedelta(days=7)
    raw = (
        price_page(release="6/24/2020")
        .replace(b"2026 Oct-19 to Oct-23", b"2020 Jun-15 to Jun-19")
        .replace(b"2026 Oct-26 to Oct-30", b"2020 Jun-22 to Jun-26")
    )
    current = parse_price_history(raw, "wti", "2020-06-25T12:00:00+00:00")
    predictions, bundles = [], []
    for model, rows in zip(models, [records, records + [current]], strict=True):
        model.configure_peers({"definitions": [], "columns": {}, "support": []})
        bundle = eia.prepare_petroleum(
            dict(
                records=rows,
                price_tables=True,
                provider="offline_fixture",
                sha256="fixture",
                failures=[],
                notice="SIMULATED prices; real LightGBM engineering check only.",
            ),
            cutoffs,
            [*model.train, *model.origins],
        )
        bundles.append(bundle)
        model.configure_external(bundle)
        model.price_info = {"latest_target_session": "2020-07-03"}
        factor = next(
            d for d in bundle["definitions"] if d["factor_id"] == "external.eia_wti_5obs_return"
        )
        predictions.append(
            model.predict(
                {"symbol": "XOM"}, [model.library["definitions"][0], factor], "Same hypothesis"
            )
        )
    assert "latest_forecast_unavailable" in predictions[0]
    assert "latest_forecast" in predictions[1]
    assert predictions[0]["development"] == predictions[1]["development"]
    before = [i for i, value in enumerate(cutoffs) if value < current["available_at"]]
    for i in before:
        assert (
            bundles[0]["columns"]["external.eia_wti_5obs_return"][i]
            == bundles[1]["columns"]["external.eia_wti_5obs_return"][i]
        )
    np.testing.assert_allclose(
        models[0]._fits[0][1][models[0].train], models[1]._fits[0][1][models[1].train]
    )


def test_same_release_prefers_latest_observation_independent_of_record_order():
    current = parse_price_history(price_page(), "wti", "2026-10-29T09:00:00+00:00")
    archive = current | {"observation_date": "2026-10-23", "value": 104}
    for records in ([archive, current], [current, archive]):
        bundle = eia.prepare_petroleum(
            dict(
                records=records,
                price_tables=True,
                provider="fixture",
                sha256="",
                failures=[],
                notice="",
            ),
            ["2026-10-30T20:00:00+00:00"],
            [0],
        )
        assert bundle["columns"]["external.eia_wti_5obs_return"] == [107 / 101 - 1]
