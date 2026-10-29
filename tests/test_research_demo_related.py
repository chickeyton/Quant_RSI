"""Configured relationships are contextual evidence, not discovered correlations."""

import json
from pathlib import Path

import pytest

from quant_rsi.cli import main
from quant_rsi.research.demo_related import RelationshipProfile, profile_related


def test_profile_mode_completes_loop_with_simulated_remaining_stages(capsys):
    catalog = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"
    assert (
        main(["research-demo", "ko", "--related", "profile", "--catalog", str(catalog), "--json"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "mixed" and result["summary"]["factors_changed"]
    related = next(item["output"] for item in result["trace"] if item["stage"] == "related_stocks")
    assert related["stocks"] == ["CCEP", "MNST"]
    assert related["reviewed_on"] == "2026-10-29"
    assert related["selection"] == "configured_not_discovered"
    assert "source-backed configuration" in result["notice"]
    assert "News, LLM, prices and predictions are simulated" in result["notice"]


@pytest.mark.parametrize(
    "symbols, error",
    [
        (["KO"], "must be distinct"),
        (["CCEP", "CCEP"], "must be distinct"),
        ([], "at least 1 item"),
        (["A", "B", "C", "D"], "at most 3 items"),
    ],
)
def test_profile_rejects_self_duplicate_empty_and_oversized_links(symbols, error):
    with pytest.raises(ValueError, match=error):
        RelationshipProfile.model_validate_json(
            json.dumps(
                {
                    "instrument": "KO",
                    "reviewed_on": "2026-10-29",
                    "relationships": [
                        {
                            "symbol": symbol,
                            "relationship": "business_partner",
                            "reason": "Test only",
                            "source_urls": ["https://example.com"],
                        }
                        for symbol in symbols
                    ],
                }
            )
        )


@pytest.mark.parametrize("symbol", ["../../KO", "UNKNOWN"])
def test_profile_does_not_fallback_to_invented_relationships(symbol):
    with pytest.raises(ValueError):
        profile_related(symbol)
