"""Read configured issuer statement cells without evaluating spreadsheet formulas."""

import io
import math
import re
from datetime import datetime
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException


def _workbook(raw):
    try:
        with ZipFile(io.BytesIO(raw)) as archive:
            files = archive.infolist()
            if len(files) > 1000 or sum(f.file_size for f in files) > 32_000_000:
                raise ValueError("Issuer spreadsheet exceeds the expanded byte/member bound")
        return load_workbook(io.BytesIO(raw), read_only=True, data_only=True, keep_links=False)
    except (BadZipFile, InvalidFileException, ParseError, KeyError) as exc:
        raise ValueError("Unsupported issuer spreadsheet") from exc


def spreadsheet_date(raw, published, page_modified):
    book = _workbook(raw)
    try:
        dates = [book.properties.created, book.properties.modified]
        dates = [value.date() for value in dates if isinstance(value, datetime)]
        return max([published] + dates) if dates else max(published, page_modified)
    finally:
        book.close()


def _text(value):
    return " ".join(str(value or "").split())


def parse_spreadsheet(raw, profile, *, cashflow=False, balance=False):
    book = _workbook(raw)
    try:
        name = profile["spreadsheet_sheet"]
        if name not in book.sheetnames:
            raise ValueError("Missing configured financial worksheet")
        sheet = book[name]
        if (
            not isinstance(sheet.max_row, int)
            or not isinstance(sheet.max_column, int)
            or not 1 <= sheet.max_row <= 1000
            or not 3 <= sheet.max_column <= 32
        ):
            raise ValueError("Unsupported financial worksheet dimensions")
        table = list(sheet.iter_rows())
        if not any(
            profile["statement_title"].lower() in _text(r[0].value).lower() for r in table[:5]
        ):
            raise ValueError("Worksheet statement heading differs from its profile")
        unit_rows = [
            i for i, r in enumerate(table[:10]) if "Dollars in millions" in _text(r[0].value)
        ]
        if len(unit_rows) != 1:
            raise ValueError("Expected one explicit USD millions worksheet heading")
        header = unit_rows[0]
        if balance:
            ends = [
                cell.value.date()
                if isinstance(cell.value, datetime)
                else datetime.strptime(_text(cell.value), "%B %d, %Y").date()
                for cell in table[header][1:3]
            ]
            if not 0 < (ends[0] - ends[1]).days <= 380:
                raise ValueError("Ambiguous worksheet balance dates")
        else:
            if header + 1 >= len(table):
                raise ValueError("Missing worksheet year row")
            period = re.fullmatch(
                r"(Three|Six|Nine|Twelve) Months Ended ([A-Z][a-z]+ \d{1,2}),",
                _text(table[header][1].value),
            )
            if not period or _text(table[header][2].value):
                raise ValueError("Ambiguous shared worksheet period heading")
            months = {"Three": 3, "Six": 6, "Nine": 9, "Twelve": 12}[period[1]]
            if not cashflow and months != 3:
                raise ValueError("Income worksheet must provide an explicit three-month period")
            years = [_text(cell.value) for cell in table[header + 1][1:3]]
            if any(not re.fullmatch(r"20\d{2}", year) for year in years):
                raise ValueError("Missing explicit worksheet year columns")
            ends = [datetime.strptime(f"{period[2]} {year}", "%B %d %Y").date() for year in years]
            if not 350 <= (ends[0] - ends[1]).days <= 380:
                raise ValueError("Worksheet columns are not comparable prior-year periods")

        def values(labels):
            aliases = [labels] if isinstance(labels, str) else labels
            matching = [
                row
                for row in table[header + 1 :]
                if _text(row[0].value).casefold() in {_text(label).casefold() for label in aliases}
            ]
            if not matching:
                raise ValueError(f"Missing worksheet row: {labels}")
            for row in matching:
                for cell in row[1:3]:
                    if type(cell.value) not in {int, float} or not math.isfinite(cell.value):
                        raise ValueError("Worksheet amount requires a finite cached numeric value")
                    # These cells store USD but display millions via two scaling commas.
                    number_format = re.sub(r'"[^"]*"', "", cell.number_format)
                    if not re.search(r"[0#],,(?!,)", number_format):
                        raise ValueError(
                            "Worksheet amount does not explicitly display USD millions"
                        )
            pairs = [[float(cell.value) for cell in row[1:3]] for row in matching]
            if any(pair != pairs[0] for pair in pairs):
                raise ValueError(f"Conflicting worksheet rows: {labels}")
            return (
                pairs[0],
                [cell.coordinate for cell in matching[0][1:3]],
                _text(matching[0][0].value),
            )

        records = []
        for field, label in profile["rows"].items():
            if isinstance(label, dict):
                labels = label.get("sum", [])
                if (
                    not balance
                    or field != "inventory"
                    or set(label) != {"sum"}
                    or not isinstance(labels, list)
                    or not 2 <= len(labels) <= 4
                    or any(not isinstance(item, str) or not item.strip() for item in labels)
                    or len({_text(item).casefold() for item in labels}) != len(labels)
                ):
                    raise ValueError("Inventory requires distinct worksheet components")
                parts = [values(item) for item in labels]
            else:
                parts = [values(label)]
            for i, end in enumerate(ends):
                amounts = [part[0][i] for part in parts]
                value = sum(amounts)
                if (balance and any(amount < 0 for amount in amounts)) or (
                    field == "revenue" and value <= 0
                ):
                    raise ValueError("Invalid reported worksheet revenue or inventory")
                row = {
                    "field": field,
                    "tag": " + ".join(part[2] for part in parts),
                    "end": str(end),
                    "value": value,
                    "cells": [part[1][i] for part in parts],
                }
                if balance:
                    row["period_kind"] = "reported_balance_instant"
                if isinstance(label, dict):
                    row["components"] = [
                        {"tag": part[2], "value": part[0][i], "cell": part[1][i]} for part in parts
                    ]
                if cashflow:
                    row.update(period_kind="reported_cashflow_duration", period_months=months)
                    if field == "capital_expenditure":
                        row["value"] *= -1
                        if row["value"] < 0:
                            raise ValueError("Capital expenditure must be a worksheet cash outflow")
                records.append(row)
        return records, name
    except (ParseError, KeyError) as exc:
        raise ValueError("Unsupported issuer financial worksheet") from exc
    finally:
        book.close()
