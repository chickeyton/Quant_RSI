"""Original small fixtures mirror XOM's column layout, not market performance."""

import io
import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import issuer_financials as issuer
from quant_rsi.research.issuer_financials import NOTICE, statement_rows
from quant_rsi.research.study_business import BusinessFactorProposal, business_column
from quant_rsi.research.study_financials import prepare_financials

PROFILE = json.loads(
    (Path(__file__).resolve().parents[1] / "instruments/US/XOM/financials.json").read_bytes()
)["issuer"]
INCOME = """ATTACHMENT
Investor disclosures
Quarterly data summary
CONDENSED CONSOLIDATED STATEMENT OF INCOME
(Preliminary)
Three Months Ended Nine Months Ended
Dollars in millions (unless otherwise noted) September 30, September 30,
2026 2025 2026 2025
Sales and other operating revenue 1,200 1,000 9,000 8,000
Total revenues and other income 1,250 1,050 9,100 8,100
Income/(Loss) before income taxes 200 150 800 700
"""
CASHFLOW = """CONDENSED CONSOLIDATED STATEMENT OF CASH FLOWS
(Preliminary)
Nine Months Ended
Dollars in millions (unless otherwise noted) September 30,
2026 2025
Net cash provided by operating activities 300 200
Additions to property, plant, and equipment (90) (80)
"""
BALANCE = """CONDENSED CONSOLIDATED BALANCE SHEET
(Preliminary)
Dollars in millions (unless otherwise noted) September 30, December 31,
2026 2025
Inventories
Crude oil, products and merchandise 240 300
Materials and supplies 20 30
"""


def test_shared_dates_keep_quarterly_revenue_and_reported_cashflow_duration():
    income = statement_rows(INCOME, PROFILE)
    assert [(r["field"], r["end"], r["value"]) for r in income] == [
        ("revenue", "2026-09-30", 1.2e9),
        ("revenue", "2025-09-30", 1e9),
    ]
    cash = statement_rows(CASHFLOW, PROFILE["cashflow"], cashflow=True)
    assert [r["value"] for r in cash] == [300e6, 200e6, 90e6, 80e6]
    assert {r["period_months"] for r in cash} == {9}
    assert {r["period_kind"] for r in cash} == {"reported_cashflow_duration"}
    assert (
        statement_rows(
            CASHFLOW.replace("plant, and", "plant and"), PROFILE["cashflow"], cashflow=True
        )[-2]["value"]
        == 90e6
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        statement_rows(INCOME, PROFILE | {"shared_month_day": False})


@pytest.mark.parametrize(
    "before,after",
    [
        ("September 30, September 30,", "September 30,"),
        ("2026 2025", "2026 2024"),
        ("in millions", "in thousands"),
    ],
)
def test_shared_headers_reject_ambiguous_dates_and_scale(before, after):
    with pytest.raises(ValueError):
        statement_rows(INCOME.replace(before, after), PROFILE)


def test_inventory_adds_all_components_and_preserves_their_provenance():
    rows = statement_rows(BALANCE, PROFILE["balance"], balance=True)
    assert [r["value"] for r in rows] == [260e6, 330e6]
    assert [r["end"] for r in rows] == ["2026-09-30", "2025-12-31"]
    assert rows[0]["components"] == [
        {"tag": "Crude oil, products and merchandise", "value": 240e6},
        {"tag": "Materials and supplies", "value": 20e6},
    ]


@pytest.mark.parametrize(
    "before,after",
    [
        ("Materials and supplies 20 30", ""),
        ("Materials and supplies 20 30", "Materials and supplies — 30"),
        ("Materials and supplies 20 30", "Materials and supplies (20) 30"),
        (
            "Materials and supplies 20 30",
            "Materials and supplies 20 30\nMaterials and supplies 21 30",
        ),
    ],
)
def test_inventory_never_publishes_a_partial_or_conflicting_total(before, after):
    with pytest.raises(ValueError):
        statement_rows(BALANCE.replace(before, after), PROFILE["balance"], balance=True)


def test_duplicate_inventory_components_are_rejected():
    profile = deepcopy(PROFILE["balance"])
    profile["rows"]["inventory"]["sum"] = ["Materials and supplies"] * 2
    with pytest.raises(ValueError, match="distinct"):
        statement_rows(BALANCE, profile, balance=True)


@pytest.mark.parametrize("primary", ["complete", "partial", "unreadable", "ambiguous"])
def test_document_fallback_preserves_one_version_and_raw_source_identity(
    tmp_path, monkeypatch, primary
):
    release = "https://investor.exxonmobil.com/press-releases/detail/1/exxonmobil-announces-fixture-results"
    host = "https://d1io3yog0oux5.cloudfront.net"
    first, second = host + "/earnings_release/fixture.pdf", host + "/supplement/fixture.pdf"
    html = '<meta name="published_time" content="2026-10-30">'
    html += f'<a href="{first}">Release</a><a href="{second}">Supplement</a>'
    if primary == "ambiguous":
        html += f'<a href="{host}/earnings_release/another.pdf">Ambiguous</a>'
    documents = {
        PROFILE["index_url"]: f'<a href="{release}">Results</a>'.encode(),
        release: html.encode(),
        first: b"%PDF-primary-fixture",
        second: b"%PDF-supplement-fixture",
    }
    fetched = []

    class Opener:
        def open(self, request, **kwargs):
            fetched.append(request.full_url)
            return io.BytesIO(documents[request.full_url])

    # Transport and PDF decoding are fixtures; real row parsing and retention run below.
    def parse(raw, definition, *, cashflow=False, balance=False):
        if raw == documents[first] and (
            primary == "unreadable" or (primary == "partial" and (cashflow or balance))
        ):
            raise ValueError("Fixture document has no readable section")
        text = CASHFLOW if cashflow else BALANCE if balance else INCOME
        return statement_rows(text, definition, cashflow=cashflow, balance=balance), 1

    monkeypatch.setattr(issuer, "build_opener", lambda *args: Opener())
    monkeypatch.setattr(issuer, "parse_statement", parse)
    monkeypatch.setattr(issuer, "statement_date", lambda *args: date(2026, 10, 30))
    result = issuer.collect_issuer("XOM", tmp_path, {"issuer": PROFILE})
    expected = first if primary in {"complete", "partial"} else second
    assert {r["source_url"] for r in result["records"]} == {expected}
    assert {r["statement_id"] for r in result["records"]} == {digest(documents[expected])}
    assert {r["available_at"] for r in result["records"]} == {"2026-10-31T05:00:00+00:00"}
    assert len(result["records"]) == (2 if primary == "partial" else 8)
    assert (second in fetched) == (expected == second)
    assert bool(result["failures"]) == (primary != "complete")
    for resource in result["resources"]:
        assert digest((tmp_path / resource["file"]).read_bytes()) == resource["sha256"]


def test_publication_dated_xom_features_support_business_revisions_without_invented_margin():
    rows = statement_rows(INCOME, PROFILE)
    rows += statement_rows(CASHFLOW, PROFILE["cashflow"], cashflow=True)
    rows += statement_rows(BALANCE, PROFILE["balance"], balance=True)
    records = [
        {
            "priority": 0,
            "start": None,
            "period_kind": "reported_three_months",
            "available_at": "2026-10-31T05:00:00+00:00",
            "statement_id": "current",
            "unit": "USD",
        }
        | row
        for row in rows
    ]
    # Prior September inventory comes from an older release, not December's comparator.
    prior = BALANCE.replace("2026 2025", "2025 2024").replace("240 300", "180 300")
    records += [
        records[0] | row | {"available_at": "2025-11-01T05:00:00+00:00", "statement_id": "prior"}
        for row in statement_rows(prior, PROFILE["balance"], balance=True)
    ]
    snapshot = {
        "symbol": "XOM",
        "provider": "issuer_earnings",
        "records": records,
        "sha256": digest(canonical(records)),
        "notice": NOTICE,
    }
    bundle = prepare_financials(
        snapshot, ["2026-10-31T04:59:59+00:00", "2026-10-31T05:00:00+00:00"], [1]
    )
    columns = bundle["columns"]
    assert columns["financial.revenue_yoy"] == [None, pytest.approx(0.2)]
    assert columns["financial.inventory_yoy"] == [None, pytest.approx(0.3)]
    assert columns["financial.operating_cashflow_yoy"] == [None, pytest.approx(0.5)]
    assert columns["financial.free_cashflow_yoy"] == [None, pytest.approx(0.75)]
    assert columns["financial.operating_margin"] == [None, None]
    assert bundle["coverage"]["runnable_factors"] == 4
    assert {g["factor_id"] for g in bundle["coverage_gaps"]} == {
        "financial.operating_margin",
        "financial.operating_margin_yoy",
    }
    bundle["columns"] = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    for metric, expected in [("inventory_yoy", 0.1), ("free_cashflow_yoy", 0.55)]:
        proposal = BusinessFactorProposal.model_validate_json(
            canonical(
                {
                    "name": metric + "_revenue_gap",
                    "description": "Fixture business revision",
                    "inputs": [
                        {"name": "business", "factor_id": "financial." + metric},
                        {"name": "sales", "factor_id": "financial.revenue_yoy"},
                    ],
                    "expression": {
                        "schema_version": 1,
                        "nodes": [
                            {"name": "a", "op": "leaf", "inputs": [], "leaf": "business"},
                            {"name": "b", "op": "leaf", "inputs": [], "leaf": "sales"},
                            {"name": "gap", "op": "subtract", "inputs": ["a", "b"], "leaf": None},
                        ],
                    },
                }
            )
        )
        definition = proposal.definition(bundle["definitions"])
        values, bindings = business_column(definition, bundle)
        assert values[1] == pytest.approx(expected)
        assert len(bindings) == 2
