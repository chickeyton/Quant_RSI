"""Offline source fixtures verify archived-news wiring, not historical market validity."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from quant_rsi.cli import main
from quant_rsi.research import study
from quant_rsi.research.demo import load_catalog, select_factors
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.demo_proposals import NewsFactorProposal, extend_library
from quant_rsi.research.study_news import retain_snapshot, simulated_history

CATALOG = Path(__file__).parents[1] / "instruments/shared/equities.json"


def command(output, archive):
    return [
        "research-study",
        "KO",
        "--llm",
        "simulated",
        "--news",
        "issuer-rss",
        "--related",
        "simulated",
        "--catalog",
        str(CATALOG),
        "--output",
        str(output),
        "--news-archive",
        str(archive),
    ]


def test_current_receipt_is_retained_without_becoming_historical_training(tmp_path, monkeypatch):
    model = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20)
    data = replace(model.data, instruments=("KO",))
    current = simulated_history(data, "KO")[-1]
    # Explicit source double, with its receipt after every mature evaluation origin.
    current["provider"] = "issuer_rss"
    current["source_url"] = "https://example.com/rss"
    current["items"][0]["url"] = "https://example.com/latest"
    current["items"][0]["text"] = current["items"][0]["title"]
    info = {
        "currency": "USD",
        "latest_target_session": "2020-07-03",
        "session_close_times": [data.timestamp(i, "close") for i in range(len(data.dates))],
    }
    monkeypatch.setattr(study, "load_daily_prices", lambda *args: (data, info))
    monkeypatch.setattr(study, "issuer_news", lambda *args, **kwargs: deepcopy(current))
    archive = tmp_path / "news"
    for name in ("first", "second"):
        root = tmp_path / name
        assert main(command(root, archive)) == 0
        result = json.loads((root / "result.json").read_text())
        news = result["study"]["news_factors"]
        assert news["snapshot_count"] == 1 and not news["support"]["runnable"]
        assert news["executed_initial"] == news["executed_revised"] == []
        assert all(not item["runnable"] for item in news["definitions"])
        assert "pending" in (root / "report.html").read_text()
        assert "latest_forecast" in result["summary"]  # Actual selected price factors still run.
        snapshot = json.loads((root / "news_history.json").read_text())[0]
        assert snapshot["received_at"] == current["received_at"]
    assert len(list(archive.iterdir())) == 1  # Same receipt is not rewritten or double counted.


def test_pending_news_selection_rejected_without_price_substitution():
    library = load_catalog(CATALOG)
    proposal = {
        "name": "earnings",
        "keywords": ["earnings"],
        "window_days": 7,
        "description": "Count retained earnings titles",
    }
    hypothesis = {
        "proposed_factors": [{"op": "momentum", "window": 9, "description": "Price"}],
        "proposed_news_factors": [proposal],
        "factor_ids": ["research.momentum_9", "news.earnings_7"],
    }
    extended = extend_library(library, hypothesis)
    with pytest.raises(ValueError, match="pending"):
        select_factors(extended, hypothesis)
    with pytest.raises(ValueError, match="new news factor name"):
        extend_library(
            extended, hypothesis | {"proposed_news_factors": [proposal | {"keywords": ["demand"]}]}
        )


@pytest.mark.parametrize(
    "changes", [{"keywords": [""]}, {"keywords": []}, {"window_days": 31}, {"name": "a.b"}]
)
def test_news_recipe_bounds(changes):
    with pytest.raises(ValueError):
        NewsFactorProposal.model_validate(
            {
                "name": "events",
                "keywords": ["earnings"],
                "window_days": 7,
                "description": "Test",
                **changes,
            }
        )


def test_daily_study_rejects_simulated_archive_before_fitting(tmp_path, monkeypatch):
    model = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20)
    data = replace(model.data, instruments=("KO",))
    archive = tmp_path / "simulated"
    retain_snapshot(archive, simulated_history(data, "KO")[0])
    monkeypatch.setattr(study, "load_daily_prices", lambda *args: (data, {"currency": "USD"}))
    monkeypatch.setattr(study, "issuer_news", lambda *args, **kwargs: {})
    output = tmp_path / "rejected"
    assert main(command(output, archive)) == 1
    assert "Simulated news history" in json.loads((output / "failure.json").read_text())["error"]
    assert not (output / "result.json").exists()
