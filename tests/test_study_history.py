"""Controlled receipt clocks prove continuity, not elapsed real-market performance."""

import json
import shutil
from copy import deepcopy

import pytest
from test_daily_data import chart, install
from test_study_news import snapshot as news_snapshot

from quant_rsi.cli import main
from quant_rsi.core.artifacts import ArtifactStore
from quant_rsi.market import market_contracts
from quant_rsi.research import study, study_history
from quant_rsi.research.demo import simulated_hypothesis
from quant_rsi.research.study_news import load_archive
from quant_rsi.research.study_ui import read_state


def prices(tmp_path, monkeypatch, day, *, name=None, modify=None):
    now = day + "T21:00:00+00:00"
    raw = chart(day)
    source = raw["chart"]["result"][0]
    source["meta"].update(symbol="KO", instrumentType="EQUITY", exchangeName="NYQ")
    source["events"] = {}
    if modify:
        modify(source)
    install(monkeypatch, data=raw, now=now)
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: now)
    root = tmp_path / (name or day)
    return root, study_history.collection_snapshot("KO", root)


def result_for(snapshot):
    forecast = {
        "origin_session": snapshot["dates"][-1],
        "target_session": snapshot["contract"]["calendar"]["sessions"][-1]["session_id"],
        "reference_price": snapshot["closes"][-1],
        "predicted_price": snapshot["closes"][-1] * 1.01,
        "horizon_sessions": 5,
    }
    return {
        "configuration": {
            "symbol": "KO",
            "initial_cash": "10000",
            "commission_bps": "5",
            "slippage_bps": "5",
            "threshold": 0.0,
        },
        "iterations": {
            "selected_round": 1,
            "attempts": [
                {
                    "prediction": {
                        "factor_ids": ["research.momentum_5"],
                        "latest_forecast": forecast,
                    }
                },
                {"prediction": {"factor_ids": ["rejected"]}},
            ],
        },
    }


def command(root, history, collection):
    return [
        "research-study",
        "KO",
        "--llm",
        "simulated",
        "--news",
        "simulated",
        "--related",
        "simulated",
        "--no-related-prices",
        "--prices-dir",
        str(collection),
        "--history-dir",
        str(history),
        "--no-ui",
        "--quiet",
        "--output",
        str(root),
    ]


def test_two_complete_studies_pass_mature_outcomes_to_next_hypothesis(tmp_path, monkeypatch):
    history, first, second = (tmp_path / n for n in ("history", "first", "second"))
    contexts = []

    def hypothesis(context, library, **kwargs):
        contexts.append(deepcopy(context))
        return simulated_hypothesis(context, library, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    collection, _ = prices(tmp_path, monkeypatch, "2026-10-23")
    assert main(command(first, history, collection)) == 0
    frozen = (first / "result.json").read_bytes()
    assert not contexts[0]["prior_outcomes"]["entries"]
    entry = study_history.history_view(history)["entries"][0]
    assert entry["status"] == "pending"
    collection, _ = prices(tmp_path, monkeypatch, "2026-10-30")
    assert main(command(second, history, collection)) == 0
    assert (first / "result.json").read_bytes() == frozen
    prior = contexts[2]["prior_outcomes"]["entries"]
    assert len(prior) == 1 and prior[0]["status"] == "settled"
    assert prior[0]["metrics"]["forecast"]["observations"] == 1
    assert prior[0]["strategy"]["strategy"]["sharpe_ratio"] is not None
    assert prior[0]["strategy"]["buy_and_hold"]["fees"]["commission"] != "0.00"
    current = read_state(first)["history"]["entries"]
    assert [e["status"] for e in current] == ["settled", "pending"]
    assert "Forecast history and later observations" in (second / "report.html").read_text()
    assert json.loads((second / "prior_outcomes.json").read_text())["entries"] == prior
    frozen_history = contexts[2]["prior_outcomes"]["history"]["entries"]
    assert len(frozen_history) == 1 and frozen_history[0]["status"] == "settled"
    view = next(e["view"] for e in read_state(second)["events"] if e["stage"] == "summary")
    assert view["continuity"]["entries"][0]["provided_as_feedback"]
    assert (
        view["continuity"]["selected_factor_ids"]
        == json.loads((second / "result.json").read_text())["study"]["selection"][
            "selected_factor_ids"
        ]
    )
    assert "Continuing the previous research" in (second / "report.html").read_text()


def test_registration_confirmation_clock_and_selected_model_are_frozen(tmp_path, monkeypatch):
    _, snapshot = prices(tmp_path, monkeypatch, "2026-10-23")
    result, history = result_for(snapshot), tmp_path / "history"
    clocks = iter(["2026-10-26T13:29:59+00:00", "2026-10-26T13:30:01+00:00"])
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: next(clocks))
    entry = study_history.register_study(history, tmp_path / "study", result, snapshot)
    assert entry["status"] == "not_prospective"
    assert entry["factor_ids"] == ["research.momentum_5"]
    assert entry["publication"]["post_commit_confirmed_at"].endswith("13:30:01+00:00")
    assert study_history.register_study(history, tmp_path / "study", result, snapshot) == entry
    result["iterations"]["selected_round"] = 2
    with pytest.raises(ValueError, match="different completed result"):
        study_history.register_study(history, tmp_path / "study", result, snapshot)


def test_observation_revisions_are_append_only_and_current_assessment_wins(tmp_path, monkeypatch):
    _, original = prices(tmp_path, monkeypatch, "2026-10-23")
    history = tmp_path / "history"
    entry = study_history.register_study(history, tmp_path / "run", result_for(original), original)
    _, received = prices(tmp_path, monkeypatch, "2026-10-30")
    settled = study_history.observe_prices(history, "KO", received)["entries"][0]
    assert settled["status"] == "settled" and settled["assessment_count"] == 1
    assert study_history.observe_prices(history, "KO", received)["entries"][0] == settled
    revised = deepcopy(received)
    revised["closes"][revised["dates"].index(original["dates"][-1])] += 1
    latest = study_history.observe_prices(history, "KO", revised)["entries"][0]
    assert latest["status"] == "insufficient_evidence" and latest["metrics"] is None
    assert latest["assessment_count"] == 2
    assert not study_history.mature_feedback(history, "KO")["entries"]
    assert ArtifactStore(history / "artifacts").json(entry["result_ref"]) == result_for(original)


def test_future_receipts_wrong_sources_and_clock_rollback_rejected(tmp_path, monkeypatch):
    _, original = prices(tmp_path, monkeypatch, "2026-10-23")
    history = tmp_path / "history"
    study_history.register_study(history, tmp_path / "run", result_for(original), original)
    _, received = prices(tmp_path, monkeypatch, "2026-10-30")
    future = deepcopy(received)
    future["info"]["received_at"] = "2030-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="precedes price receipt"):
        study_history.observe_prices(history, "KO", future)
    wrong = deepcopy(received)
    wrong["contract"]["source_id"] = "another_source"
    with pytest.raises(ValueError, match="frozen source"):
        study_history.observe_prices(history, "KO", wrong)
    study_history.observe_prices(history, "KO", received)
    with pytest.raises(ValueError, match="roll back"):
        study_history.observe_prices(history, "KO", original)


def test_immature_missing_and_late_predictions_never_enter_feedback(tmp_path, monkeypatch):
    _, original = prices(tmp_path, monkeypatch, "2026-10-23")
    history = tmp_path / "history"
    study_history.register_study(history, tmp_path / "run", result_for(original), original)
    assert (
        study_history.observe_prices(history, "KO", original)["entries"][0]["status"] == "pending"
    )
    _, later = prices(tmp_path, monkeypatch, "2026-10-30")
    missing = deepcopy(later)
    missing["dates"][-1] = "2099-01-01"
    entry = study_history.observe_prices(history, "KO", missing)["entries"][0]
    assert entry["reason"] == "missing_target_period_prices"
    late = study_history.register_study(history, tmp_path / "late", result_for(original), original)
    assert late["status"] == "not_prospective"
    assert not study_history.mature_feedback(history, "KO")["entries"]


def test_standalone_observe_cli_requires_prices_and_matches_symbol(tmp_path, monkeypatch, capsys):
    _, original = prices(tmp_path, monkeypatch, "2026-10-23")
    history = tmp_path / "history"
    study_history.register_study(history, tmp_path / "run", result_for(original), original)
    collection, _ = prices(tmp_path, monkeypatch, "2026-10-30")
    assert (
        main(
            [
                "research-history",
                "observe",
                "--json",
                "KO",
                "--history-dir",
                str(history),
                "--prices-dir",
                str(collection),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["entries"][0]["status"] == "settled"
    assert (
        main(["research-history", "status", "CALM", "--history-dir", str(history), "--json"]) == 0
    )
    assert not json.loads(capsys.readouterr().out)["entries"]


def test_shared_archive_preserves_receipts_across_runs(tmp_path, monkeypatch):
    history = tmp_path / "history"
    for day in ("2026-10-23", "2026-10-30"):
        collection, _ = prices(tmp_path, monkeypatch, day)
        current = news_snapshot(received=day + "T21:00:00+00:00")
        current["items"][0]["text"] = "Earnings growth"
        monkeypatch.setattr(study, "issuer_news", lambda *args, current=current, **kwargs: current)
        args = command(tmp_path / (day + "-study"), history, collection)
        args[args.index("--news") + 1] = "issuer-rss"
        assert main(args) == 0
    saved = load_archive(history / "news_archive" / "real" / "KO")
    assert sorted(s["received_at"] for s in saved) == [
        "2026-10-23T21:00:00+00:00",
        "2026-10-30T21:00:00+00:00",
    ]


def test_missing_confirmation_cannot_backdate_a_recovered_publication(tmp_path, monkeypatch):
    _, snapshot = prices(tmp_path, monkeypatch, "2026-10-23")
    history, run = tmp_path / "history", tmp_path / "run"
    original = study_history._confirm
    monkeypatch.setattr(study_history, "_confirm", lambda *args: None)
    first = study_history.register_study(history, run, result_for(snapshot), snapshot)
    assert first["status"] == "not_prospective"
    monkeypatch.setattr(study_history, "_confirm", original)
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2026-10-26T14:00:00+00:00")
    recovered = study_history.register_study(history, run, result_for(snapshot), snapshot)
    assert recovered["status"] == "not_prospective"
    assert recovered["issued_at"] == first["issued_at"]


def test_target_correction_is_another_assessment_not_another_prediction(tmp_path, monkeypatch):
    _, snapshot = prices(tmp_path, monkeypatch, "2026-10-23")
    history = tmp_path / "history"
    study_history.register_study(history, tmp_path / "run", result_for(snapshot), snapshot)
    _, received = prices(tmp_path, monkeypatch, "2026-10-30")
    first = study_history.observe_prices(history, "KO", received)["entries"][0]
    received["closes"][-1] += 1
    second = study_history.observe_prices(history, "KO", received)["entries"][0]
    assert second["assessment_count"] == 2
    assert second["metrics"]["actual_price"] == first["metrics"]["actual_price"] + 1
    assert len(study_history.mature_feedback(history, "KO")["entries"]) == 1
    assert second["forecast"] == first["forecast"]


def test_copying_a_retained_collection_does_not_create_a_new_receipt(tmp_path, monkeypatch):
    root, snapshot = prices(tmp_path, monkeypatch, "2026-10-23")
    copied = tmp_path / "copied"
    shutil.copytree(root, copied)
    assert study_history.collection_snapshot("KO", copied) == snapshot
