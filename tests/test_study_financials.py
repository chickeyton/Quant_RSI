"""Business facts use filing availability, fixed rows and the existing study loop."""

import json
from copy import deepcopy

import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import study_financials as financials
from quant_rsi.research.study_ui_views import visual_output


def snapshot(symbol="KO"):
    rows = [
        {
            "start": "2023-01-01",
            "end": "2023-03-31",
            "filed": "2023-04-25",
            "form": "10-Q",
            "accn": "old",
            "val": 100,
        },
        {
            "start": "2024-01-01",
            "end": "2024-03-31",
            "filed": "2024-04-25",
            "form": "10-Q",
            "accn": "new",
            "val": 120,
        },
    ]
    facts = {
        "cik": 1,
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": rows}},
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            rows[0] | {"val": 10},
                            rows[1] | {"val": 18},
                        ]
                    }
                },
            }
        },
    }
    return {
        "symbol": symbol,
        "companyfacts": facts,
        "sha256": digest(canonical(facts)),
        "profile": {
            "symbol": symbol,
            "cik": 1,
            "fields": {
                "revenue": ["Revenues"],
                "operating_income": ["OperatingIncomeLoss"],
            },
        },
        "provider": "fixture",
        "notice": financials.NOTICE,
    }


def test_date_only_filing_boundary_and_later_restatement():
    data = snapshot()
    revenue = data["companyfacts"]["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    revenue.append(revenue[-1] | {"val": 150, "filed": "2024-05-25", "accn": "restated"})
    records = financials.reported_records(data)
    before, _ = financials.at_cutoff(records, "2024-04-26T04:59:59+00:00")
    first, evidence = financials.at_cutoff(records, "2024-04-26T05:00:00+00:00")
    later, _ = financials.at_cutoff(records, "2024-05-26T05:00:00+00:00")
    assert "revenue_yoy" not in before
    assert first == pytest.approx(
        {"revenue_yoy": 0.2, "operating_margin": 0.15, "operating_margin_yoy": 0.05}
    )
    assert evidence["current"]["facts"]["revenue"]["accession"] == "new"
    assert later["revenue_yoy"] == 0.5
    assert "operating_margin" not in later  # No mixing incompatible filing versions.
    with pytest.raises(ValueError, match="timezone"):
        financials.at_cutoff(records, "2024-04-26T05:00:00")


@pytest.mark.parametrize("symbol", ["KO", "AAPL"])
def test_shared_calculations_reject_ytd_wrong_unit_and_ambiguous_facts(symbol):
    data = snapshot(symbol)
    units = data["companyfacts"]["facts"]["us-gaap"]["Revenues"]["units"]
    base = units["USD"][-1]
    units["USD"].append(base | {"start": "2023-10-01", "val": 9999})
    units["EUR"] = [base | {"val": 9999}]
    records = financials.reported_records(data)
    assert len(records) == 4
    result, _ = financials.at_cutoff(records, "2024-04-27T00:00:00+00:00")
    assert result["revenue_yoy"] == pytest.approx(0.2)
    units["USD"].append(base | {"val": 333})
    result, _ = financials.at_cutoff(financials.reported_records(data), "2024-04-27T00:00:00+00:00")
    assert "revenue_yoy" not in result


def test_missing_rows_disable_factor_without_dropping_or_filling():
    cutoffs = ["2024-04-01T20:00:00+00:00", "2024-04-27T20:00:00+00:00"]
    data = snapshot()
    bundle = financials.prepare_financials(data, cutoffs, [0, 1])
    assert all(not d["runnable"] for d in bundle["definitions"])
    assert bundle["columns"]["financial.revenue_yoy"] == [None, pytest.approx(0.2)]
    assert len(bundle["evidence"]) == 2
    selected = financials.prepare_financials(data, cutoffs, [1])
    assert {d["factor_id"] for d in selected["definitions"] if d["runnable"]} == {
        "financial.revenue_yoy",
        "financial.operating_margin",
        "financial.operating_margin_yoy",
    }


def test_retained_snapshot_hash_is_checked(tmp_path):
    data = snapshot()
    profile = tmp_path / "profiles/US/KO/financials.json"
    profile.parent.mkdir(parents=True)
    profile.write_bytes(canonical(data["profile"]))
    (tmp_path / "snapshot.json").write_bytes(canonical(data))
    assert financials.collect_financials("KO", tmp_path, tmp_path / "profiles") == data
    changed = deepcopy(data)
    changed["companyfacts"]["cik"] = 9
    (tmp_path / "snapshot.json").write_bytes(canonical(changed))
    with pytest.raises(ValueError, match="identity"):
        financials.collect_financials("KO", tmp_path, tmp_path / "profiles")


@pytest.mark.parametrize("model", ["ridge", "lightgbm"])
def test_business_factors_run_both_rounds_and_same_row_ablation(tmp_path, model):
    root = tmp_path / "study"
    assert main(command(root, "--llm", "simulated", "--financials", "--model", model)) == 0
    result = json.loads((root / "result.json").read_bytes())
    attempts = result["iterations"]["attempts"]
    for attempt in attempts:
        ids = attempt["prediction"]["factor_ids"]
        assert any(identity.startswith("financial.") for identity in ids)
    assert attempts[0]["prediction"]["factor_ids"] != attempts[1]["prediction"]["factor_ids"]
    comparison = result["study"]["financial_comparison"]
    assert comparison["status"] == "completed"
    assert comparison["extra_fits"] == 1
    assert result["configuration"]["comparison_fit_budget"] == 1
    assert len(comparison["development_sessions"]) == 20
    selected = attempts[comparison["selected_round"] - 1]["prediction"]
    assert comparison["with_financials"]["forecast"] == selected["development"]["metrics"]
    assert comparison["without_financials"]["forecast"]["sample_count"] == 20
    assert "Financial" in (root / "report.html").read_text()
    assert "SIMULATED" in result["study"]["financial_data"]["notice"]
    view = visual_output("financial_comparison", comparison, {}, "KO")
    assert len(view["rows"]) == 2
    assert view["rows"][0]["mae"] == selected["development"]["metrics"]["mae_price"]
