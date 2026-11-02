"""Small offline Table 11 layout fixtures; not predictive-quality evidence."""

from datetime import date

import pytest

from quant_rsi.research.eia_prices import parse_prices

TABLE = (
    "STUB_1,STUB_2,STUB_3,Fri 9/25,Fri 10/2,Mon 10/12,Tue 10/13,Wed 10/14,"
    "Thu 10/15,Fri 10/16,Mon 10/19,Tue 10/20,Wed 10/21,Thu 10/22,Fri 10/23\n"
    "Crude Oil,  WTI - Cushing,,999,888,90,91,92,93,100,101,102,103,104,120\n"
    "Crude Oil,  Brent,,777,666,100,101,102,103,110,111,112,113,114,130\n"
).encode()
RELEASE = date(2026, 10, 28)
WEEK = date(2026, 10, 23)


def test_price_returns_exclude_weekly_averages_and_wait_for_release():
    rows = parse_prices(TABLE, RELEASE, WEEK)
    assert [r["field"] for r in rows] == ["wti", "brent"]
    assert [(r["value"], r["prior_value"]) for r in rows] == [(120, 100), (130, 110)]
    assert all(r["available_at"] == "2026-10-29T05:00:00+00:00" for r in rows)
    assert all(r["observation_date"] == "2026-10-23" for r in rows)
    assert all(r["prior_observation_date"] == "2026-10-16" for r in rows)
    assert all(len(r["daily_observation_dates"]) == 6 for r in rows)


def test_missing_daily_quotes_are_not_replaced_with_weekly_averages():
    fewer = TABLE.replace(b"90,91,92,93,100", b"NA,NA,NA,NA,NA")
    assert [r["field"] for r in parse_prices(fewer, RELEASE, WEEK)] == ["brent"]
    no_latest = TABLE.replace(b"104,120", b"104,NA")
    assert [r["field"] for r in parse_prices(no_latest, RELEASE, WEEK)] == ["brent"]
    one_missing = TABLE.replace(b"93,100", b"NA,100")
    assert parse_prices(one_missing, RELEASE, WEEK)[0]["prior_value"] == 100


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        TABLE + TABLE,
        TABLE.replace(b"Mon 10/12", b"Tue 10/12"),
        TABLE.replace(b"Tue 10/13", b"Mon 10/12"),
        TABLE.replace(b"104,120", b"104,nan"),
        TABLE.replace(b"104,120", b"104,-1"),
        TABLE.replace(b"104,120", b"104,0"),
        TABLE.replace(b"WTI - Cushing", b"Unknown"),
    ],
)
def test_invalid_price_layout_or_values_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_prices(raw, RELEASE, WEEK)


def test_price_week_must_match_validated_archive_page():
    with pytest.raises(ValueError, match="archive week"):
        parse_prices(TABLE, RELEASE, date(2026, 10, 16))


def test_daily_dates_can_cross_the_new_year():
    raw = TABLE.replace(
        TABLE.splitlines()[0],
        b"STUB_1,STUB_2,STUB_3,Fri 12/5,Fri 12/12,Mon 12/22,Tue 12/23,Wed 12/24,"
        b"Thu 12/25,Fri 12/26,Mon 12/29,Tue 12/30,Wed 12/31,Thu 1/1,Fri 1/2",
    )
    rows = parse_prices(raw, date(2026, 1, 7), date(2026, 1, 2))
    assert rows[0]["observation_date"] == "2026-01-02"
    assert rows[0]["prior_observation_date"] == "2025-12-26"


@pytest.mark.parametrize("prices", [TABLE, b"invalid price table", None])
def test_collection_retains_prices_and_keeps_inventory_on_price_failure(
    tmp_path, monkeypatch, prices
):
    from quant_rsi.research import eia_petroleum as eia

    page_url = eia.ARCHIVE_URL + "2026/2026_10_28/wpsr_2026_10_28.php"
    page = (
        "Data for week ending October 23, 2026 Release Date: October 28, 2026"
        '<a href="csv/table4.csv">Inventory</a>'
    )
    if prices is not None:
        page += '<a href="csv/table11.csv">Prices</a>'
    inventory = "STUB_1,10/23/26,10/16/26,Difference,x,x,x,x\n" + "".join(
        f"{label},100,99,1,x,x,x,x\n" for label in eia.FIELDS.values()
    )
    urls = {
        eia.ARCHIVE_URL: f'<a href="{page_url}">28</a>'.encode(),
        page_url: page.encode(),
        page_url.rsplit("/", 1)[0] + "/csv/table4.csv": inventory.encode(),
        page_url.rsplit("/", 1)[0] + "/csv/table11.csv": prices,
    }
    calls = []

    def download(url):
        calls.append(url)
        return urls[url]

    monkeypatch.setattr(eia, "_download", download)
    shared = tmp_path / "shared"
    first = eia.collect_petroleum(
        tmp_path / "first", "2026-10-01", "2026-10-30", cache_directory=shared
    )
    second = eia.collect_petroleum(
        tmp_path / "second", "2026-10-01", "2026-10-30", cache_directory=shared
    )
    assert first == second
    assert len(calls) == (3 if prices is None else 4)
    assert len(first["records"]) == (5 if prices == TABLE else 3)
    assert len(first["failures"]) == int(prices not in (None, TABLE))
    cutoffs = ["2026-10-28T20:00:00+00:00", "2026-10-29T20:00:00+00:00"]
    bundle = eia.prepare_petroleum(first, cutoffs, [1])
    values = bundle["columns"]["external.eia_wti_5obs_return"]
    assert values[0] is None
    if prices == TABLE:
        assert values[1] == pytest.approx(0.2)
        retained = next(r for r in first["records"] if r["field"] == "wti")
        assert retained["source_url"].endswith("table11.csv")
        assert len(retained["resource_sha256"]) == 64
    else:
        assert values[1] is None
        assert not bundle["definitions"][-1]["runnable"]
    assert all(d["runnable"] for d in bundle["definitions"][:3])
    expired = eia.prepare_petroleum(first, ["2026-11-12T05:00:01+00:00"], [0])
    assert expired["columns"]["external.eia_wti_5obs_return"] == [None]
