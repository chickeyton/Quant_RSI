"""Related prices must reach real fits without changing samples or inventing coverage."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pytest
from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study_peers
from quant_rsi.research.demo import load_catalog, select_factors
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_peers import aligned_closes, peer_feature, prepare_related_prices

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def model_and_bundle(tmp_path, monkeypatch=None, missing_latest=False):
    model = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20)
    model.data = replace(model.data, instruments=("KO",))
    cutoffs = [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
    related = {"stocks": ["PEER"], "provider": "instrument_relationship_profile"}
    if monkeypatch:
        peer = replace(model.data, instruments=("PEER",), closes=model.data.closes * 2)
        times = list(cutoffs)
        if missing_latest:
            times[-1] = times[-1].replace("16:00", "17:00")
        monkeypatch.setattr(
            study_peers,
            "load_daily_prices",
            lambda *a, **kw: (
                peer,
                {"provider": "public_daily_prices", "session_close_times": times},
            ),
        )
    bundle = prepare_related_prices(
        model.data,
        cutoffs,
        list(model.train) + list(model.origins),
        related,
        mode="daily" if monkeypatch else "synthetic",
        directory=tmp_path,
    )
    return model, bundle


def test_peer_recipe_oracles_and_no_future_or_gap_bridging():
    target = np.array([100, 102, 104, 106, 108, 110, 115, 120], dtype=float)
    peer = np.array([50, 51, 52, 53, 54, 55, 56, 57], dtype=float)
    actual = peer_feature(target, peer, "relative_return", 5)
    assert actual[5] == pytest.approx(0)
    assert actual[6] == pytest.approx(115 / 102 - 56 / 51)
    assert peer_feature(target, peer, "peer_return", 5)[5] == pytest.approx(0.1)
    target[7] *= 3
    peer[7] *= 5
    np.testing.assert_array_equal(actual[:7], peer_feature(target, peer, "relative_return", 5)[:7])
    peer[3] = np.nan
    assert np.isnan(peer_feature(target, peer, "peer_return", 5)).all()


def test_alignment_uses_dates_and_real_close_times_without_fill():
    days = ("2026-01-05", "2026-01-06", "2026-01-07")
    peer = SimpleNamespace(dates=(days[2], days[0]), closes=np.array([[30], [10]]))
    cutoffs = [day + "T21:00:00+00:00" for day in days]
    info = {"session_close_times": [days[2] + "T22:00:00+00:00", days[0] + "T16:00:00-05:00"]}
    values = aligned_closes(days, cutoffs, peer, info)
    assert values[0] == 10
    assert np.isnan(values[1:]).all()
    info["session_close_times"][0] = days[2] + "T20:00:00"
    with pytest.raises(ValueError, match="timezones"):
        aligned_closes(days, cutoffs, peer, info)


@pytest.mark.parametrize("estimator", ["lightgbm", "ridge"])
def test_peer_columns_enter_fits_and_revisions_share_baselines(tmp_path, monkeypatch, estimator):
    model, bundle = model_and_bundle(tmp_path)
    model = LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20, estimator=estimator)
    model.configure_peers(bundle)
    fits = []
    if estimator == "lightgbm":
        train = lgb.train

        def captured(params, dataset, **kwargs):
            fits.append(dataset.data.copy())
            return train(params, dataset, **kwargs)

        monkeypatch.setattr(lgb, "train", captured)
    else:
        from quant_rsi.research import demo_model

        ridge = demo_model.StandardizedRidge

        def captured(x, y, alpha):
            fits.append(x.copy())
            return ridge(x, y, alpha)

        monkeypatch.setattr(demo_model, "StandardizedRidge", captured)
    first_id, second_id = "peer.PEER.peer_return_5", "peer.PEER.relative_return_10"
    first = model.predict({}, select_factors(model.library, {"factor_ids": [first_id]}), {})
    second = model.predict({}, select_factors(model.library, {"factor_ids": [second_id]}), {})
    np.testing.assert_array_equal(fits[0][:, 0], bundle["columns"][first_id][model.train])
    np.testing.assert_array_equal(fits[1][:, 0], bundle["columns"][second_id][model.train])
    assert first["development"]["baselines"] == second["development"]["baselines"]
    assert first["development"]["metrics"]["sample_count"] == 20
    assert next(u for u in second["input_usage"] if u["source"] == "PEER")["status"] == "used"
    with pytest.raises(ValueError, match="once before"):
        model.configure_peers(bundle)


def test_missing_latest_peer_does_not_switch_models_or_erase_historical_result(
    tmp_path, monkeypatch
):
    model, bundle = model_and_bundle(tmp_path, monkeypatch, missing_latest=True)
    model.configure_peers(bundle)
    model.price_info = {"latest_target_session": "2027-01-01"}
    prediction = model.predict(
        {}, select_factors(model.library, {"factor_ids": ["peer.PEER.peer_return_5"]}), {}
    )
    assert prediction["development"]["metrics"]["sample_count"] == 20
    assert "latest_forecast" not in prediction
    assert prediction["latest_forecast_unavailable"]["missing_factor_ids"] == [
        "peer.PEER.peer_return_5"
    ]


def test_missing_historical_peer_rejects_selection_without_dropping_rows(tmp_path, monkeypatch):
    model, _ = model_and_bundle(tmp_path)
    peer = SimpleNamespace(
        instruments=("PEER",), dates=model.data.dates, closes=model.data.closes.copy()
    )
    peer.closes[40] = np.nan
    cutoffs = [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
    monkeypatch.setattr(
        study_peers,
        "load_daily_prices",
        lambda *a, **kw: (
            peer,
            {"provider": "public_daily_prices", "session_close_times": cutoffs},
        ),
    )
    before = model.train.copy(), model.origins.copy()
    bundle = prepare_related_prices(
        model.data,
        cutoffs,
        list(model.train) + list(model.origins),
        {"stocks": ["PEER"]},
        mode="daily",
        directory=tmp_path,
    )
    assert bundle["support"][0]["status"] == "unavailable"
    model.configure_peers(bundle)
    with pytest.raises(ValueError, match="pending"):
        select_factors(model.library, {"factor_ids": ["peer.PEER.peer_return_5"]})
    with pytest.raises(ValueError, match="usable retained recipe"):
        model.predict({}, [bundle["definitions"][0]], {})
    np.testing.assert_array_equal(before[0], model.train)
    np.testing.assert_array_equal(before[1], model.origins)


def test_discovered_peer_is_usable_but_remains_unverified(tmp_path, monkeypatch):
    model, _ = model_and_bundle(tmp_path)
    cutoffs = [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
    peer = replace(model.data, instruments=("KOF",))
    monkeypatch.setattr(
        study_peers,
        "load_daily_prices",
        lambda *a, **kw: (
            peer,
            {"provider": "public_daily_prices", "session_close_times": cutoffs},
        ),
    )
    bundle = prepare_related_prices(
        model.data,
        cutoffs,
        list(model.train) + list(model.origins),
        {"stocks": [], "research_candidates": [{"symbol": "KOF", "evidence_ids": ["article-1"]}]},
        mode="daily",
        directory=tmp_path,
    )
    assert bundle["support"][0]["status"] == "available"
    assert bundle["support"][0]["relationship_status"] == "unverified_suggestion"
    model.configure_peers(bundle)
    prediction = model.predict(
        {}, select_factors(model.library, {"factor_ids": ["peer.KOF.relative_return_10"]}), {}
    )
    usage = next(u for u in prediction["input_usage"] if u["source"] == "KOF")
    assert usage["status"] == "used" and usage["relationship_status"] == "unverified_suggestion"


def test_failed_sources_budget_and_simulated_symbols_are_explicit(tmp_path, monkeypatch):
    model, _ = model_and_bundle(tmp_path)
    calls = []

    def unavailable(symbol, directory, **kwargs):
        calls.append(symbol)
        raise ValueError("Unsupported exchange")

    monkeypatch.setattr(study_peers, "load_daily_prices", unavailable)
    related = {"stocks": ["KO", "A", "B", "C", "D", "../escape", "A"]}
    result = prepare_related_prices(model.data, [], [], related, mode="daily", directory=tmp_path)
    assert calls == ["A", "B", "C"]
    assert [p["reason"] for p in result["support"]] == [
        "Unsupported exchange",
        "Unsupported exchange",
        "Unsupported exchange",
        "related_price_collection_limit",
        "invalid_symbol",
    ]
    related["provider"] = "simulated_relationships"
    result = prepare_related_prices(model.data, [], [], related, mode="daily", directory=tmp_path)
    assert all(p["status"] == "context_only" for p in result["support"])
    assert calls == ["A", "B", "C"]


def test_offline_command_retains_peer_inputs_usage_and_disabled_path(tmp_path):
    for enabled in (True, False):
        root = tmp_path / str(enabled)
        extra = [] if enabled else ["--no-related-prices"]
        assert main(command(root, "--llm", "simulated", "--no-ui", *extra)) == 0
        result = json.loads((root / "result.json").read_text())
        saved = json.loads((root / "related_prices.json").read_text())
        steps = {e["stage"]: e["output"] for e in result["trace"]}
        first, second = steps["first_prediction"], steps["second_prediction"]
        assert first["development"]["baselines"] == second["development"]["baselines"]
        for prediction in (first, second):
            peers = [u for u in prediction["input_usage"] if u["source"].startswith("DEMO_PEER")]
            assert any(u["status"] == "used" for u in peers) == enabled
            assert prediction["training_feature_columns"] == (3 if enabled else 2)
        assert bool(saved["inputs"]) == enabled
        assert "Model input usage" in (root / "report.html").read_text()
