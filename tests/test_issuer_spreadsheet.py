"""Original generated XLSX fixtures; public sources are never queried in tests."""

import io
from datetime import date, datetime
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from openpyxl import Workbook
from openpyxl.writer.excel import ExcelWriter
from test_xom_financials import PROFILE

from quant_rsi.core.artifacts import digest
from quant_rsi.research import issuer_financials as issuer
from quant_rsi.research.issuer_spreadsheet import parse_spreadsheet, spreadsheet_date


def workbook():
    book = Workbook()
    book.remove(book.active)
    for section, profile in [
        ("income", PROFILE),
        ("balance", PROFILE["balance"]),
        ("cashflow", PROFILE["cashflow"]),
    ]:
        sheet = book.create_sheet(profile["spreadsheet_sheet"])
        sheet["A1"] = profile["statement_title"]
        sheet["A2"] = "(Preliminary)"
        sheet["A3"] = "Dollars in millions (unless otherwise noted)"
        if section == "balance":
            sheet["B3"], sheet["C3"] = "June 30,\n2026", datetime(2025, 12, 31)
            sheet["C3"].number_format = "mmmm d, yyyy"
            amounts = [
                ("Crude oil, products and merchandise", 240e6, 300e6),
                ("Materials and supplies", 20e6, 30e6),
            ]
        else:
            sheet["B3"] = ("Six" if section == "cashflow" else "Three") + " Months Ended June 30,"
            sheet["B4"], sheet["C4"] = "2026", "2025"
            amounts = (
                [
                    ("Net cash provided by operating activities", 300e6, 200e6),
                    ("Additions to property, plant, and equipment", -90e6, -80e6),
                ]
                if section == "cashflow"
                else [("Sales and other operating revenue", 1.2e9, 1e9)]
            )
        for number, values in enumerate(amounts, start=6):
            for column, value in enumerate(values, start=1):
                cell = sheet.cell(number, column, value)
                if column > 1:
                    cell.number_format = '#,##0,,;(#,##0,,);"—"'
    book.properties.created = datetime(2026, 7, 31, 12)
    book.properties.modified = datetime(2026, 8, 2, 12)
    return book


def encode(book):
    raw = io.BytesIO()
    # Avoid save_workbook's wall-clock metadata rewrite in this deterministic fixture.
    ExcelWriter(book, ZipFile(raw, "w", ZIP_DEFLATED)).save()
    return raw.getvalue()


def test_xlsx_values_are_raw_dollars_and_keep_dates_durations_and_component_cells():
    raw = encode(workbook())
    income, name = parse_spreadsheet(raw, PROFILE)
    assert name == "Statement of Income"
    assert [(r["end"], r["value"], r["cells"]) for r in income] == [
        ("2026-06-30", 1.2e9, ["B6"]),
        ("2025-06-30", 1e9, ["C6"]),
    ]
    cash, _ = parse_spreadsheet(raw, PROFILE["cashflow"], cashflow=True)
    assert [r["value"] for r in cash] == [300e6, 200e6, 90e6, 80e6]
    assert {r["period_months"] for r in cash} == {6}
    balance, _ = parse_spreadsheet(raw, PROFILE["balance"], balance=True)
    assert [r["value"] for r in balance] == [260e6, 330e6]
    assert balance[0]["cells"] == ["B6", "B7"]
    assert [r["value"] for r in balance[0]["components"]] == [240e6, 20e6]
    assert balance[1]["end"] == "2025-12-31"
    assert spreadsheet_date(raw, date(2026, 7, 31), date(2026, 9, 1)) == date(2026, 8, 2)


@pytest.mark.parametrize(
    "problem",
    [
        "scale",
        "formula",
        "text_number",
        "boolean",
        "year",
        "duration",
        "currency",
        "oversized_sheet",
    ],
)
def test_income_spreadsheet_rejects_ambiguous_units_periods_and_non_numeric_values(problem):
    book = workbook()
    sheet = book["Statement of Income"]
    if problem == "scale":
        sheet["B6"].number_format = "#,##0"
    elif problem == "formula":
        sheet["B6"] = "=1200000000"
    elif problem == "text_number":
        sheet["B6"] = "1200000000"
    elif problem == "boolean":
        sheet["B6"] = True
    elif problem == "year":
        sheet["C4"] = "2023"
    elif problem == "duration":
        sheet["B3"] = "Six Months Ended June 30,"
    elif problem == "currency":
        sheet["A3"] = "Euros in millions"
    else:
        sheet["A1001"] = "Out of bound"
    with pytest.raises(ValueError):
        parse_spreadsheet(encode(book), PROFILE)


@pytest.mark.parametrize("problem", ["missing", "negative", "conflicting"])
def test_inventory_spreadsheet_never_uses_a_partial_or_conflicting_total(problem):
    book = workbook()
    sheet = book["Balance Sheet"]
    if problem == "missing":
        sheet["B7"] = None
    elif problem == "negative":
        sheet["B7"] = -20e6
    else:
        for column, value in enumerate(["Materials and supplies", 21e6, 30e6], start=1):
            cell = sheet.cell(8, column, value)
            cell.number_format = sheet.cell(7, column).number_format
    with pytest.raises(ValueError):
        parse_spreadsheet(encode(book), PROFILE["balance"], balance=True)


def test_image_only_pdf_falls_back_to_retained_official_xlsx_without_mixing(tmp_path, monkeypatch):
    release = "https://investor.exxonmobil.com/press-releases/detail/1/exxonmobil-announces-fixture-results"
    host = "https://d1io3yog0oux5.cloudfront.net"
    pdf, supplement, xlsx = (
        host + path
        for path in [
            "/earnings_release/fixture.pdf",
            "/supplement/fixture.pdf",
            "/supplement_xlsx_/fixture.xlsx",
        ]
    )
    html = '<meta name="published_time" content="2026-07-31">' + "".join(
        f'<a href="{url}">Statement</a>' for url in [pdf, supplement, xlsx]
    )
    documents = {
        PROFILE["index_url"]: f'<a href="{release}">Results</a>'.encode(),
        release: html.encode(),
        pdf: b"%PDF-image-fixture",
        supplement: b"%PDF-image-fixture",
        xlsx: encode(workbook()),
    }
    fetched = []

    class Opener:
        def open(self, request, **kwargs):
            fetched.append(request.full_url)
            return io.BytesIO(documents[request.full_url])

    def unreadable(*args, **kwargs):
        raise ValueError("Fixture PDF contains no extractable text")

    monkeypatch.setattr(issuer, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(issuer, "parse_statement", unreadable)
    result = issuer.collect_issuer("XOM", tmp_path, {"issuer": PROFILE})
    assert fetched == [PROFILE["index_url"], release, pdf, supplement, xlsx]
    assert len(result["records"]) == 8
    assert len(result["failures"]) == 6
    assert {r["source_url"] for r in result["records"]} == {xlsx}
    assert {r["statement_id"] for r in result["records"]} == {digest(documents[xlsx])}
    assert {r["available_at"] for r in result["records"]} == {"2026-08-03T05:00:00+00:00"}
    assert {r["sheet"] for r in result["records"]} == {
        "Statement of Income",
        "Balance Sheet",
        "Cash Flow",
    }
    assert all("page" not in r and r["cells"] for r in result["records"])
    assert any(r["file"].endswith(".xlsx") for r in result["resources"])
    for resource in result["resources"]:
        assert digest((tmp_path / resource["file"]).read_bytes()) == resource["sha256"]
