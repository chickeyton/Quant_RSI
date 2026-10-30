"""Actual fits on invented prices/headlines; no market predictive evidence."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from quant_rsi.research.demo import load_catalog
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_news import NewsHistory, simulated_history

CATALOG = Path(__file__).parents[1] / "instruments/shared/equities.json"


def model():
    return LightGBMDemo(load_catalog(CATALOG), evaluation_sessions=20)


def cutoffs(data):
    return [data.timestamp(index, "close") for index in range(len(data.dates))]


def factor(keywords=None, window_days=5):
    return {
        "factor_id": f"news.earnings_{window_days}",
        "description": "Retained target-issuer earnings title count",
        "kind": "news",
        "input_kind": "retained_issuer_headlines",
        "recipe": {
            "op": "headline_keyword_count",
            "keywords": keywords or ["earnings"],
            "window_days": window_days,
        },
    }


def test_two_actual_fits_share_frozen_news_mask_and_preserve_feature_evidence(monkeypatch):
    study = model()
    snapshots = simulated_history(study.data, "KO")
    # One ambiguous publication removes the whole row regardless of keywords.
    snapshots[90]["items"][0].update(
        publication_precision="unknown", published_at=None, published_date=None
    )
    history = NewsHistory([s for i, s in enumerate(snapshots) if i not in {35, 50, 100}])
    support = study.configure_news(history, cutoffs(study.data), "KO")
    assert support["runnable"]
    assert support["original_training_rows"] == 46
    assert support["available_training_rows"] == support["retained_training_rows"] == 44
    assert support["original_evaluation_rows"] == 20
    assert support["available_evaluation_rows"] == support["retained_evaluation_rows"] == 18
    assert "publication_time_not_precise" in support["reasons"]
    assert study.origin == 99
    assert study.library["news_feature_support"] == support
    # The model has detached the evidence used by its shared eligibility decision.
    history._snapshots.clear()
    trained = []
    actual_train = lgb.train

    def train(params, data, **kwargs):
        trained.append((data.data.copy(), data.label.copy()))
        return actual_train(params, data, **kwargs)

    monkeypatch.setattr(lgb, "train", train)
    context = {"symbol": "KO", **study.context("KO")}
    price_factors = study.library["definitions"][:1]
    first = study.predict(context, price_factors, {})
    second = study.predict(context, price_factors + [factor()], {})
    assert [x.shape for x, _ in trained] == [(44, 1), (44, 2)]
    np.testing.assert_array_equal(trained[0][1], trained[1][1])
    np.testing.assert_array_equal(trained[0][0][:, 0], trained[1][0][:, 0])
    assert first["development"]["baselines"] == second["development"]["baselines"]
    assert first["development"]["metrics"]["sample_count"] == 18
    assert second["development"]["metrics"]["sample_count"] == 18
    assert first["origin_session"] == second["origin_session"] == study.data.dates[99]
    assert second["feature_values"]["news.earnings_5"] == 1.0
    evidence = second["news_feature_inputs"]
    assert len(evidence) == 44 + 18
    assert all(row["value"] in {0.0, 1.0} for row in evidence)
    assert any(row["value"] == 0.0 for row in evidence)
    assert all(not row["reasons"] for row in evidence)
    assert all(
        datetime.fromisoformat(row["snapshot_received_at"]) <= datetime.fromisoformat(row["cutoff"])
        for row in evidence
    )
    assert {row["origin_session"] for row in evidence}.isdisjoint(
        study.data.dates[i] for i in (35, 50, 90, 100)
    )
    for row in evidence:
        assert bool(row["evidence_news_ids"]) == bool(row["value"])
        assert row["provider"] == "simulated_news_archive"
    assert second["news_feature_evidence"]["news.earnings_5"]["value"] == 1.0
    observed = study.observe(context, first)
    summary = study.summarize(
        context,
        first,
        second,
        observed | {"error": first["predicted_price"] - observed["observed_price"]},
    )
    assert "18 matured origins" in summary["text"]
    assert summary["improvement"] == "not_evaluated"
    assert first["last_training_label_session"] < first["development"]["first_evaluation_session"]


@pytest.mark.parametrize("available", ["future_only", "one_evaluation_origin"])
def test_insufficient_news_preserves_price_sample_and_rejects_news_before_fit(
    monkeypatch, available
):
    study = model()
    snapshots = simulated_history(study.data, "KO")
    selected = snapshots[-1:] if available == "future_only" else snapshots[30:76] + snapshots[81:82]
    support = study.configure_news(NewsHistory(selected), cutoffs(study.data), "KO")
    assert not support["runnable"]
    assert support["available_evaluation_rows"] == (0 if available == "future_only" else 1)
    assert support["retained_training_rows"] == 46
    assert support["retained_evaluation_rows"] == 20
    assert study.origin == 100

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Unsupported news must reject before dispatch")

    with monkeypatch.context() as patch:
        patch.setattr(lgb, "train", forbidden_fit)
        with pytest.raises(ValueError, match="News factor execution is unavailable"):
            study.predict({}, [factor()], {})
    prediction = study.predict({}, study.library["definitions"][:1], {})
    assert prediction["training_rows"] == 46
    assert prediction["development"]["metrics"]["sample_count"] == 20
    assert "news_feature_inputs" not in prediction


def public_fixture(*, missing_latest):
    fixture = model()
    data = replace(deepcopy(fixture.data), instruments=("KO",))
    actual_closes = [date + "T20:00:00+00:00" for date in data.dates]
    snapshots = simulated_history(data, "KO")
    for snapshot in snapshots:
        for entry in (snapshot, *snapshot["items"]):
            for key in ("received_at", "published_at"):
                if entry.get(key):
                    entry[key] = (
                        datetime.fromisoformat(entry[key]) + timedelta(hours=4)
                    ).isoformat()
    history = NewsHistory(snapshots[:-1] if missing_latest else snapshots)
    info = {
        "currency": "TEST",
        "latest_target_session": "2030-01-01",
        "session_close_times": actual_closes,
    }
    return data, info, history


@pytest.mark.parametrize("missing_latest", [False, True])
def test_retained_actual_cutoffs_and_explicit_latest_news_availability(missing_latest):
    data, info, history = public_fixture(missing_latest=missing_latest)
    study = LightGBMDemo(
        load_catalog(CATALOG),
        prices=(data, info),
        evaluation_sessions=20,
        news_history=history,
        news_cutoffs=info["session_close_times"],
        news_symbol="KO",
    )
    context = {"symbol": "KO", **study.context("KO")}
    first = study.predict(context, study.library["definitions"][:1], {})
    second = study.predict(context, [factor()], {})
    assert first["latest_forecast"]["outcome"] == "not_observed"
    assert all("T20:00:00" in row["cutoff"] for row in second["news_feature_inputs"])
    observed = study.observe(context, first)
    summary = study.summarize(
        context,
        first,
        second,
        observed | {"error": first["predicted_price"] - observed["observed_price"]},
    )
    if missing_latest:
        assert "latest_forecast" not in second
        assert "latest_forecast" not in summary
        unavailable = second["latest_forecast_unavailable"]
        assert unavailable["reasons"] == ["latest_news_features_unavailable"]
        assert unavailable["news_feature_evidence"]["news.earnings_5"]["value"] is None
        assert summary["latest_forecast_unavailable"] == unavailable
    else:
        assert second["latest_forecast"]["outcome"] == "not_observed"
        evidence = second["latest_forecast"]["news_feature_evidence"]["news.earnings_5"]
        assert evidence["cutoff"] == info["session_close_times"][-1]
        assert evidence["value"] == 1.0
        assert second["latest_forecast"]["feature_values"]["news.earnings_5"] == 1.0
        assert summary["latest_forecast"] == second["latest_forecast"]


def test_wrong_issuer_synthetic_real_cutoffs_and_late_configuration_reject():
    data, info, history = public_fixture(missing_latest=False)
    with pytest.raises(ValueError, match="actual session close"):
        LightGBMDemo(
            load_catalog(CATALOG),
            prices=(data, info),
            evaluation_sessions=20,
            news_history=history,
            news_cutoffs=cutoffs(data),
            news_symbol="KO",
        )
    study = model()
    study.configure_news(history, cutoffs(study.data), "KO")
    with pytest.raises(ValueError, match="target symbol"):
        study.context("SOXL")
    with pytest.raises(ValueError, match="configured once"):
        study.configure_news(history, cutoffs(study.data), "KO")
    late = model()
    late.context("KO")
    with pytest.raises(ValueError, match="before context"):
        late.configure_news(history, cutoffs(late.data), "KO")
