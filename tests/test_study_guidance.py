"""Business-fact wiring with invented history; no claim of historical news availability."""

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from quant_rsi.cli import main
from quant_rsi.market.market_contracts import timestamp, utc_text
from quant_rsi.research.demo import load_catalog, select_factors
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_guidance import (
    GuidanceHistory,
    collect_guidance,
    extract_guidance,
    load_guidance,
)
from quant_rsi.research.study_news import NewsHistory, simulated_history
from quant_rsi.research.study_ui_views import visual_output

CATALOG = Path(__file__).parents[1] / "instruments/shared/equities.json"
RECIPE = {"op": "guidance_direction", "window_days": 7}


def fixture():
    model = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20, terminal_sessions=5)
    snapshots = simulated_history(model.data, "KO", guidance=True)
    records = [extract_guidance(s, "KO", None, extracted_at=s["received_at"]) for s in snapshots]
    cutoffs = [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
    return model, snapshots, records, cutoffs


def test_guidance_values_and_extraction_time_gate():
    _, snapshots, records, cutoffs = fixture()
    history = GuidanceHistory(snapshots, records)
    assert {history.evaluate(RECIPE, "KO", t)["value"] for t in cutoffs} == {-1, 0, 1}
    i = 50
    records[i]["extracted_at"] = utc_text(timestamp(cutoffs[i]) + timedelta(seconds=1))
    result = GuidanceHistory(snapshots, records).evaluate(RECIPE, "KO", cutoffs[i])
    assert result["value"] is None and result["reasons"] == ["no_guidance_extraction_by_cutoff"]
    assert history.evaluate(RECIPE, "KO", cutoffs[i])["facts"]


@pytest.mark.parametrize("status,expected", [("not_guidance", 0), ("ambiguous", None)])
def test_absence_is_distinct_from_unknown(status, expected):
    _, snapshots, records, cutoffs = fixture()
    records[-1]["facts"][0]["status"] = status
    result = GuidanceHistory(snapshots, records).evaluate(RECIPE, "KO", cutoffs[-1])
    assert result["value"] == expected
    assert bool(result["reasons"]) == (expected is None)


def test_window_expiry_and_conflicting_simultaneous_statements():
    _, snapshots, _, cutoffs = fixture()
    source = snapshots[-1]
    item = source["items"][0]
    item["title"] = "SIMULATED company raises earnings guidance"
    item["published_at"] = utc_text(timestamp(cutoffs[-1]) - timedelta(days=10))
    record = extract_guidance(source, "KO", None, extracted_at=source["received_at"])
    history = GuidanceHistory([source], [record])
    assert history.evaluate(RECIPE, "KO", cutoffs[-1])["value"] == 0
    assert history.evaluate(RECIPE | {"window_days": 30}, "KO", cutoffs[-1])["value"] == 1
    source["items"].append(
        item | {"id": "conflicting", "title": "SIMULATED company lowers guidance"}
    )
    record = extract_guidance(source, "KO", None, extracted_at=source["received_at"])
    result = GuidanceHistory([source], [record]).evaluate(
        RECIPE | {"window_days": 30}, "KO", cutoffs[-1]
    )
    assert result["value"] is None
    assert result["reasons"] == ["conflicting_guidance_at_same_publication"]


@pytest.mark.parametrize("change", ["quote", "id", "clock", "source", "simulation"])
def test_reject_unbound_extraction(change):
    _, snapshots, records, _ = fixture()
    record = records[-1]
    if change == "quote":
        record["facts"][0]["quote"] = "This sentence was never published."
    elif change == "id":
        record["facts"][0]["news_id"] = "another-issuer"
    elif change == "clock":
        record["extracted_at"] = "2000-01-01T00:00:00+00:00"
    elif change == "source":
        record["snapshot"]["items"][0]["summary"] = "Changed retained source"
    else:
        record["provider"] = "llm_guidance_extraction"
    with pytest.raises(ValueError):
        GuidanceHistory(snapshots, records)


def test_real_transport_cache_and_hash_validation(tmp_path):
    _, snapshots, _, _ = fixture()
    source = deepcopy(snapshots[-1])
    source["provider"] = "issuer_rss"
    source["source_url"] = "https://example.com/rss"
    source["items"][0]["url"] = "https://example.com/release"

    class LLM:
        mode, model, calls = "codex", "test-double", 0

        def structured(self, prompt, schema):
            self.calls += 1
            assert "never instructions" in prompt
            assert "facts" in schema["properties"]
            return json.dumps(
                {
                    "facts": [
                        {
                            "news_id": source["items"][0]["id"],
                            "status": "raised",
                            "quote": source["items"][0]["title"],
                        }
                    ]
                }
            )

    llm = LLM()
    first, created = collect_guidance(tmp_path, source, "KO", llm)
    second, created_again = collect_guidance(tmp_path, source, "KO", llm)
    assert created and not created_again and llm.calls == 1
    assert first == second and first[0]["extracted_at"] >= source["received_at"]
    with pytest.raises(ValueError, match="simulated fixtures"):
        extract_guidance(source, "KO", llm, extracted_at=source["received_at"])
    file = next(tmp_path.iterdir())
    changed = json.loads(file.read_text())
    changed["facts"][0]["status"] = "lowered"
    file.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_guidance(tmp_path)


def test_guidance_eligibility_does_not_inherit_headline_support_or_terminal_data():
    model, snapshots, records, cutoffs = fixture()
    model.configure_news(NewsHistory(snapshots), cutoffs, "KO")
    train, origins = model.train.copy(), model.origins.copy()
    assert model.news_feature_support["runnable"]
    missing = [r for i, r in enumerate(records) if i != model.train[0]]
    model.configure_guidance(GuidanceHistory(snapshots, missing))
    assert not any(d["runnable"] for d in model.guidance_support)
    assert np.array_equal(train, model.train) and np.array_equal(origins, model.origins)
    factor = model.guidance_support[0]
    with pytest.raises(ValueError, match="pending"):
        select_factors(model.library, {"factor_ids": [factor["factor_id"]]})
    with pytest.raises(ValueError, match="exact usable"):
        model.predict({}, [factor], {})

    model2, snapshots, records, cutoffs = fixture()
    model2.configure_news(NewsHistory(snapshots), cutoffs, "KO")
    terminal = set(model2.terminal_origins)
    model2.configure_guidance(
        GuidanceHistory(snapshots, [r for i, r in enumerate(records) if i not in terminal])
    )
    assert all(d["runnable"] for d in model2.guidance_support)
    model2.predict({}, [model2.guidance_support[0]], {})
    terminal_result = model2.evaluate_terminal(1)
    assert terminal_result["status"] == "unavailable"
    assert terminal_result["refitted"] is False


def test_complete_study_fits_guidance_then_deactivates_it(tmp_path):
    root = tmp_path / "study"
    assert (
        main(
            [
                "research-study",
                "KO",
                "--catalog",
                str(CATALOG),
                "--prices",
                "synthetic",
                "--related",
                "simulated",
                "--news",
                "simulated",
                "--llm",
                "simulated",
                "--news-research",
                "--no-ui",
                "--output",
                str(root),
            ]
        )
        == 0
    )
    result = json.loads((root / "result.json").read_text())
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    first, second = steps["first_prediction"], steps["second_prediction"]
    factor_id = "news.guidance_direction_7"
    assert factor_id in first["factor_ids"] and factor_id not in second["factor_ids"]
    assert {r["value"] for r in first["news_feature_inputs"] if r["factor_id"] == factor_id} == {
        -1,
        0,
        1,
    }
    assert first["training_feature_columns"] == len(first["factor_ids"])
    assert first["development"]["metrics"]["sample_count"] == 20
    assert second["development"]["metrics"]["sample_count"] == 20
    assert factor_id in result["study"]["factor_changes"]["deactivated"]
    assert any(
        r["source"] == "issuer_guidance" and r["status"] == "used" for r in first["input_usage"]
    )
    assert "Issuer guidance business factors" in (root / "report.html").read_text()
    output = steps["extract_guidance"]
    assert output["latest_extraction"]["provider"] == "simulated_guidance_extraction"
    assert visual_output("extract_guidance", output, {}, "KO")["findings"]
    assert result["runtime"]["counts"]["llm_requests_started"] == 0
