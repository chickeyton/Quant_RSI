"""Inventory is a balance, not the cash-flow change or last year-end comparator."""

import json

import pytest
from test_research_study import command
from test_study_financials import snapshot

from quant_rsi.cli import main
from quant_rsi.research import issuer_financials, study, study_financials


def inventory_snapshot():
    data = snapshot()
    data["profile"]["fields"]["inventory"] = ["InventoryNet"]
    rows = [
        {"end": "2023-03-31", "filed": "2023-04-25", "val": 80},
        {"end": "2023-12-31", "filed": "2024-02-25", "val": 90},
        {"end": "2024-03-31", "filed": "2024-04-25", "val": 100},
    ]
    data["companyfacts"]["facts"]["us-gaap"]["InventoryNet"] = {
        "units": {"USD": [r | {"form": "10-Q", "accn": r["filed"]} for r in rows]}
    }
    return data


def test_inventory_uses_mature_balance_and_comparable_year_not_previous_year_end():
    data = inventory_snapshot()
    records = study_financials.reported_records(data)
    before, _ = study_financials.inventory_at_cutoff(records, "2024-04-26T04:59:59+00:00")
    assert before == {}
    values, evidence = study_financials.inventory_at_cutoff(records, "2024-04-26T05:00:00+00:00")
    assert values == {"inventory_yoy": 0.25}
    assert evidence["prior_year"]["end"] == "2023-03-31"
    assert evidence["current"]["start"] is None
    assert evidence["current"]["period_kind"] == "reported_balance_instant"
    # A later amendment changes only cutoffs at which it is visible.
    rows = data["companyfacts"]["facts"]["us-gaap"]["InventoryNet"]["units"]["USD"]
    rows.append(rows[-1] | {"val": 120, "filed": "2024-05-25", "accn": "amendment"})
    records = study_financials.reported_records(data)
    assert study_financials.inventory_at_cutoff(records, "2024-04-27T20:00:00+00:00")[0] == values
    assert study_financials.inventory_at_cutoff(records, "2024-05-26T05:00:00+00:00")[0] == {
        "inventory_yoy": 0.5
    }


@pytest.mark.parametrize("problem", ["zero_prior", "ambiguous", "duration", "wrong_unit", "stale"])
def test_inventory_missing_or_invalid_comparisons_remain_unavailable(problem):
    data = inventory_snapshot()
    units = data["companyfacts"]["facts"]["us-gaap"]["InventoryNet"]["units"]
    rows = units["USD"]
    if problem == "zero_prior":
        rows[0]["val"] = 0
    elif problem == "ambiguous":
        rows.append(rows[-1] | {"val": 101})
    elif problem == "duration":
        for row in rows:
            row["start"] = "2023-01-01"
    elif problem == "wrong_unit":
        units["EUR"] = units.pop("USD")
    cutoff = "2025-02-01T20:00:00+00:00" if problem == "stale" else "2024-04-27T20:00:00+00:00"
    bundle = study_financials.prepare_financials(data, [cutoff], [0])
    definition = next(
        d for d in bundle["definitions"] if d["factor_id"] == "financial.inventory_yoy"
    )
    assert not definition["runnable"]
    assert bundle["columns"]["financial.inventory_yoy"] == [None]


BALANCE = """Example Company
CONDENSED CONSOLIDATED BALANCE SHEETS
(In millions)
June 28, December 31,
2024 2023
ASSETS
Inventories $ 100 $ 90
"""
PROFILE = {"statement_title": "Consolidated Balance Sheets", "rows": {"inventory": "Inventories"}}


def test_issuer_balance_columns_are_instants_not_annual_growth():
    rows = issuer_financials.statement_rows(BALANCE, PROFILE, balance=True)
    assert [(r["end"], r["value"]) for r in rows] == [("2024-06-28", 1e8), ("2023-12-31", 9e7)]
    assert all(r["period_kind"] == "reported_balance_instant" for r in rows)
    assert issuer_financials.statement_rows(BALANCE, PROFILE) is None
    assert (
        issuer_financials.statement_rows(
            BALANCE.replace("BALANCE SHEETS", "STATEMENTS OF CASH FLOWS"), PROFILE, balance=True
        )
        is None
    )


@pytest.mark.parametrize(
    "before,after",
    [
        ("In millions", "In thousands"),
        ("100", "—"),
        ("100", "(100)"),
        ("2024 2023", "2023 2024"),
        ("Inventories", "Inventory adjustment"),
    ],
)
def test_issuer_inventory_rejects_ambiguous_or_invalid_balances(before, after):
    with pytest.raises(ValueError):
        issuer_financials.statement_rows(BALANCE.replace(before, after), PROFILE, balance=True)


@pytest.mark.parametrize("model", ["ridge", "lightgbm"])
def test_inventory_enters_fit_and_can_be_revised_in_existing_loop(tmp_path, monkeypatch, model):
    original = study.simulated_hypothesis

    def hypothesis(context, library, **kwargs):
        result = original(context, library, **kwargs)
        if kwargs.get("previous") is None:
            result["factor_ids"] = [
                f for f in result["factor_ids"] if not f.startswith("financial.")
            ]
            result["factor_ids"].append("financial.inventory_yoy")
        return result

    monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    root = tmp_path / "inventory"
    assert main(command(root, "--llm", "simulated", "--financials", "--model", model)) == 0
    result = json.loads((root / "result.json").read_text())
    first, revised = result["iterations"]["attempts"]
    assert "financial.inventory_yoy" in first["prediction"]["factor_ids"]
    assert isinstance(first["prediction"]["feature_values"]["financial.inventory_yoy"], float)
    assert "financial.inventory_yoy" not in revised["prediction"]["factor_ids"]
    assert first["prediction"]["development"]["metrics"]["sample_count"] == 20
    assert "financial.inventory_yoy" in (root / "report.html").read_text()
