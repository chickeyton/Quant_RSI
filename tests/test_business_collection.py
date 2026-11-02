"""Offline scheduling checks; shared raw bytes and source failures remain auditable."""

import io
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from test_eia_price_history import price_page

from quant_rsi.core.artifacts import digest
from quant_rsi.research import eia_petroleum as eia
from quant_rsi.research import issuer_financials as issuer
from quant_rsi.research.eia_price_history import PRICE_HISTORY_URLS


class ReleaseRequests:
    """The first four distinct releases must overlap; retries do not rejoin the barrier."""

    def __init__(self, releases, failed):
        self.releases, self.failed = releases, failed
        self.barrier = threading.Barrier(4)
        self.lock = threading.Lock()
        self.calls = Counter()
        self.active = self.peak = 0

    def visit(self, url):
        with self.lock:
            self.calls[url] += 1
            first = self.calls[url] == 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if first and url in self.releases[:4]:
                self.barrier.wait(timeout=5)
            if url == self.failed:
                raise OSError("Retained source failure")
        finally:
            with self.lock:
                self.active -= 1


@pytest.mark.parametrize("callers", [1, 2])
@pytest.mark.parametrize("failed", [False, True])
def test_parallel_eia_releases_preserve_order_and_shared_cache(
    tmp_path, monkeypatch, callers, failed
):
    dates = [date(2026, 9, 30) + timedelta(days=7 * i) for i in range(5)]
    releases = [eia.ARCHIVE_URL + d.strftime("%Y/%Y_%m_%d/wpsr_%Y_%m_%d.php") for d in dates]
    source = ReleaseRequests(releases, releases[1] if failed else None)
    calls = Counter()
    lock = threading.Lock()

    def download(url):
        with lock:
            calls[url] += 1
        if url == eia.ARCHIVE_URL:
            return "".join(f'<a href="{u}">Release</a>' for u in releases).encode()
        if url in releases:
            source.visit(url)
            d = dates[releases.index(url)]
            return (
                f"Data for week ending {(d - timedelta(days=5)):%B %d, %Y} "
                f"Release Date: {d:%B %d, %Y} "
                '<a href="download/table4.csv">CSV</a>'
            ).encode()
        if url in PRICE_HISTORY_URLS.values():
            return price_page(next(k for k, v in PRICE_HISTORY_URLS.items() if v == url))
        page = url.split("/download/")[0]
        d = next(d for d, release in zip(dates, releases, strict=True) if release.startswith(page))
        header = (
            f"STUB_1,{d - timedelta(days=5):%m/%d/%y},{d - timedelta(days=12):%m/%d/%y},x,x,x,x,x\n"
        )
        return (header + "".join(f"{v},100,99,1,x,x,x,x\n" for v in eia.FIELDS.values())).encode()

    monkeypatch.setattr(eia, "_download", download)

    def collect(index):
        return eia.collect_petroleum(
            tmp_path / str(index), "2026-10-01", "2026-10-30", cache_directory=tmp_path / "cache"
        )

    with ThreadPoolExecutor(max_workers=callers) as executor:
        snapshots = list(executor.map(collect, range(callers)))
    assert source.peak == 4
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    expected = [d.isoformat() for i, d in enumerate(dates) if not (failed and i == 1)]
    snapshot = snapshots[0]
    assert [
        r["released_on"] for r in snapshot["records"] if r["field"] == "commercial_crude"
    ] == expected
    assert len(snapshot["failures"]) == int(failed)
    assert [r["url"] for r in snapshot["resources"] if r["url"] in releases] == [
        u for u in releases if u != source.failed
    ]
    assert all(n == (callers if url == source.failed else 1) for url, n in calls.items())
    for index, snapshot in enumerate(snapshots):
        for resource in snapshot["resources"]:
            assert (
                digest((tmp_path / str(index) / resource["file"]).read_bytes())
                == resource["sha256"]
            )


@pytest.mark.parametrize("failed", [False, True])
def test_parallel_issuer_releases_keep_precedence_and_identical_documents(
    tmp_path, monkeypatch, failed
):
    index = "https://issuer.example/earnings"
    releases = [f"https://issuer.example/release-{i}" for i in range(5)]
    source = ReleaseRequests(releases, releases[1] if failed else None)
    profile = {
        "issuer": {
            "index_url": index,
            "allowed_hosts": ["issuer.example"],
            "archive": "html",
            "release_pattern": r"release-\d",
            "pdf_pattern": "statement.pdf",
        }
    }

    def download(request, timeout):
        url = request.full_url
        if url == index:
            raw = "".join(f'<a href="{u}">Release</a>' for u in releases).encode()
        elif url in releases:
            source.visit(url)
            raw = (
                f'<meta name="published_time" content="2026-10-{20 + releases.index(url)}">'
                '<a href="/statement.pdf">Statement</a>'
            ).encode()
        else:
            # All releases share the same bytes, exercising concurrent content-addressed writes.
            assert url == "https://issuer.example/statement.pdf"
            raw = b"%PDF-offline-scheduling-fixture"
        return io.BytesIO(raw)

    monkeypatch.setattr(issuer, "build_opener", lambda *_: SimpleNamespace(open=download))
    monkeypatch.setattr(
        issuer,
        "parse_statement",
        lambda *_args, **_kwargs: (
            [{"field": "revenue", "end": "2026-09-30", "value": 100}],
            1,
        ),
    )
    monkeypatch.setattr(issuer, "statement_date", lambda _raw, published, _modified: published)
    snapshot = issuer.collect_issuer("XOM", tmp_path, profile)
    assert source.peak == 4
    expected = [u for u in releases if u != source.failed]
    assert [row["release_url"] for row in snapshot["records"]] == expected
    assert len(snapshot["failures"]) == int(failed)
    assert [r["url"] for r in snapshot["resources"] if r["url"] in releases] == expected
    for resource in snapshot["resources"]:
        assert digest((tmp_path / resource["file"]).read_bytes()) == resource["sha256"]
