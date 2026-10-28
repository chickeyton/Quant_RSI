"""No-fit serialized-model equivalence and per-call construction boundaries.

The retained model is the tracked invented-data oracle, not a new forecast fit.
The reference function is the exact 12fd predict_view body, whose source hash is
77e1b9ac5609a3d5be41671ed1a9412536f41247972fb09b2eea6e77d448db54.
"""

from copy import deepcopy
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest
from test_forecast_recursive_market import prepared_inputs

from quant_rsi import asof, forecast_research
from quant_rsi.artifacts import canonical, digest
from quant_rsi.forecast_contracts import ForecastDevelopmentPlan
from quant_rsi.local_research import LocalResearchData, verify_local_result
from quant_rsi.schemas import FactorSpec


def _legacy_predict_view(model_text, view):
    """Inference-only primitive. Every controller-bound key remains in the output."""
    import lightgbm as lgb

    if [row["instrument"] for row in view["rows"]] != view["expected_instruments"]:
        raise ValueError("inference view does not cover the controller-bound instrument grid")
    model = lgb.Booster(model_str=model_text)
    if model.num_feature() != 3:
        raise ValueError("model preprocessing interface mismatch")
    rows = [row | {"score": None} for row in view["rows"]]
    eligible = [index for index, row in enumerate(rows) if row["status"] == "eligible"]
    if eligible:
        values = np.asarray([rows[index]["features"] for index in eligible], dtype=float)
        if values.shape != (len(eligible), 3) or not np.isfinite(values).all():
            raise ValueError("invalid eligible inference features")
        scores = model.predict(values, num_threads=1)
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite inference output")
        for index, score in zip(eligible, scores, strict=True):
            rows[index]["score"] = float(score)
    return {
        "kind": "AsOfPrediction",
        "feature_view_ref": digest(canonical(view)),
        "model_ref": digest(model_text.encode()),
        "session_id": view["session_id"],
        "as_of": view["as_of"],
        "knowledge_basis": view["knowledge_basis"],
        "receipt_watermark": view["receipt_watermark"],
        "rows": rows,
        "qualifications": view["qualifications"],
    }


@pytest.fixture(autouse=True)
def forbid_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("prediction-reuse tests must never fit a model")

    monkeypatch.setattr(lgb, "train", forbidden)


@pytest.fixture
def model_text():
    body = (Path(__file__).parent / "fixtures" / "local_score_model.txt").read_bytes()
    assert digest(body) == "e0b7d1f5e28511f06dfa6dd47501b932fbc8a6567bc00b098a263a927eeec850"
    return body.decode()


def view(statuses=("eligible", "eligible")):
    return dict(
        kind="AsOfFeatureView",
        expected_instruments=["A", "B"],
        session_id="s20",
        as_of="2020-11-08T16:01:00Z",
        knowledge_basis="documented_historical_availability",
        receipt_watermark=3,
        qualifications=["Invented offline view"],
        rows=[
            dict(
                instrument=instrument,
                status=status,
                features=features if status == "eligible" else None,
                source_refs=[digest(instrument.encode())],
                reason=None if status == "eligible" else "missing_quote",
            )
            for instrument, status, features in zip(
                ["A", "B"], statuses, [[0.01, 0.005, 0.003], [0.03, 0.01, 0.01]], strict=True
            )
        ],
    )


def count_model_calls(monkeypatch):
    counts = dict(loads=0, predictions=0)
    constructor = lgb.Booster
    original_predict = constructor.predict

    def load(*args, **kwargs):
        counts["loads"] += 1
        return constructor(*args, **kwargs)

    def predict(self, *args, **kwargs):
        counts["predictions"] += 1
        return original_predict(self, *args, **kwargs)

    monkeypatch.setattr(constructor, "predict", predict)
    monkeypatch.setattr(lgb, "Booster", load)
    return counts


@pytest.fixture
def development_case():
    inputs, _ = prepared_inputs(unknown_action=False, with_signals=False)
    value = deepcopy(inputs["target"]["data"])
    value["specification"].update(train_cutoff=29, seed=0)
    data = LocalResearchData(value)
    factor = FactorSpec(op="momentum", window=3)
    plan = ForecastDevelopmentPlan.model_validate_json(
        canonical(
            dict(
                schema_version=1,
                plan_type="forecast_development",
                training_mode="M0",
                data=data.spec.model_dump(mode="json"),
                objective=inputs["target"]["objective"],
                budget=dict(max_attempts=1, worker_timeout_seconds=20, max_worker_seconds=20),
                primary_metric="negative_mae_pp",
                evaluation_stride=5,
                min_development_sessions=6,
                baseline="unit_baseline",
                proposals=[
                    dict(
                        name="unit_baseline",
                        hypothesis="No-fit prediction-equivalence input",
                        factor=factor.model_dump(mode="json"),
                    )
                ],
            )
        )
    )
    return data, plan, factor


@pytest.mark.parametrize(
    "statuses",
    [
        ("eligible", "eligible"),
        ("eligible", "unresolved"),
        ("unresolved", "unresolved"),
    ],
)
def test_public_view_output_is_canonically_identical_and_fresh_each_call(
    monkeypatch, model_text, statuses
):
    data = view(statuses)
    before = canonical(data)
    counts = count_model_calls(monkeypatch)
    expected = _legacy_predict_view(model_text, data)
    counts.update(loads=0, predictions=0)
    assert canonical(asof.predict_view(model_text, data)) == canonical(expected)
    assert canonical(asof.predict_view(model_text, data)) == canonical(expected)
    assert counts == dict(loads=2, predictions=0 if "eligible" not in statuses else 2)
    assert canonical(data) == before


def test_real_development_views_reuse_one_model_and_preserve_every_output(
    monkeypatch, model_text, development_case
):
    data, plan, factor = development_case
    counts = count_model_calls(monkeypatch)
    with monkeypatch.context() as old:
        old.setattr(
            forecast_research,
            "_view_predictor",
            lambda text: lambda current: _legacy_predict_view(text, current),
        )
        expected = forecast_research.development_predictions(data, plan, factor, model_text)
    origins = list(forecast_research.evaluation_origins(plan))
    assert counts == dict(loads=len(origins), predictions=len(origins))
    counts.update(loads=0, predictions=0)
    actual = forecast_research.development_predictions(data, plan, factor, model_text)
    assert canonical(actual) == canonical(expected)
    assert counts == dict(loads=1, predictions=len(origins))
    again = forecast_research.development_predictions(data, plan, factor, model_text)
    assert canonical(again) == canonical(expected)
    assert counts == dict(loads=2, predictions=2 * len(origins))
    assert all(row["model_ref"] == digest(model_text.encode()) for row in actual)


def test_all_ineligible_development_views_still_load_once_without_predicting(
    monkeypatch, model_text, development_case
):
    data, plan, factor = development_case
    original_view = forecast_research.forecast_feature_view

    def unresolved(*args, **kwargs):
        result = original_view(*args, **kwargs)
        for row in result["rows"]:
            row.update(status="unresolved", features=None, reason="unit_unresolved")
        return result

    monkeypatch.setattr(forecast_research, "forecast_feature_view", unresolved)
    counts = count_model_calls(monkeypatch)
    with monkeypatch.context() as old:
        old.setattr(
            forecast_research,
            "_view_predictor",
            lambda text: lambda current: _legacy_predict_view(text, current),
        )
        expected = forecast_research.development_predictions(data, plan, factor, model_text)
    assert counts == dict(loads=6, predictions=0)
    counts.update(loads=0, predictions=0)
    actual = forecast_research.development_predictions(data, plan, factor, model_text)
    assert canonical(actual) == canonical(expected)
    assert counts == dict(loads=1, predictions=0)


def test_empty_origins_do_not_decode_even_a_bad_model(monkeypatch, development_case):
    data, plan, factor = development_case
    counts = count_model_calls(monkeypatch)
    monkeypatch.setattr(forecast_research, "evaluation_origins", lambda plan: ())
    assert forecast_research.development_predictions(data, plan, factor, "bad model") == []
    assert counts == dict(loads=0, predictions=0)


def two_feature_model(model_text):
    """No fitting: remove unused feature 2 from the fixed model's header only."""
    lines = model_text.splitlines()
    assert {line.partition("=")[2] for line in lines if line.startswith("split_feature=")} == {"0"}
    for index, line in enumerate(lines):
        if line == "max_feature_idx=2":
            lines[index] = "max_feature_idx=1"
        elif line.startswith(("feature_names=", "feature_infos=")):
            key, _, value = line.partition("=")
            lines[index] = key + "=" + " ".join(value.split()[:2])
    return "\n".join(lines) + "\n"


def outcome(call):
    try:
        return "ok", canonical(call())
    except Exception as exc:
        return "error", type(exc).__name__, str(exc)


@pytest.mark.parametrize(
    "case",
    [
        "grid_before_bad_model",
        "shape",
        "nonfinite_feature",
        "bad_model",
        "all_ineligible_bad_model",
        "interface_before_feature",
    ],
)
def test_real_serialized_model_errors_and_order_match_original(monkeypatch, model_text, case):
    data = view()
    expected_loads = 1
    if case == "grid_before_bad_model":
        data["expected_instruments"].reverse()
        model_text = "not a serialized model"
        expected_loads = 0
    elif case == "shape":
        data["rows"][0]["features"] = [0.01]
    elif case == "nonfinite_feature":
        data["rows"][0]["features"][0] = float("nan")
    elif case == "bad_model":
        model_text = "not a serialized model"
    elif case == "all_ineligible_bad_model":
        data = view(("unresolved", "unresolved"))
        model_text = "not a serialized model"
    elif case == "interface_before_feature":
        model_text = two_feature_model(model_text)
        data["rows"][0]["features"] = [float("nan")]
    counts = count_model_calls(monkeypatch)
    expected = outcome(lambda: _legacy_predict_view(model_text, data))
    assert expected[0] == "error"
    counts.update(loads=0, predictions=0)
    actual = outcome(lambda: asof.predict_view(model_text, data))
    assert actual == expected
    assert counts["loads"] == expected_loads
    if case == "grid_before_bad_model":
        assert actual[2] == "inference view does not cover the controller-bound instrument grid"
    if case == "interface_before_feature":
        assert actual[2] == "model preprocessing interface mismatch"


def test_interface_and_finite_output_checks_still_run_on_every_private_view(monkeypatch):
    calls = dict(loads=0, interface=0, predictions=0)

    class Model:
        def num_feature(self):
            calls["interface"] += 1
            return 3

        def predict(self, values, **kwargs):
            assert kwargs == {"num_threads": 1}
            calls["predictions"] += 1
            return (
                np.ones(len(values)) if calls["predictions"] == 1 else np.full(len(values), np.nan)
            )

    def load(**kwargs):
        assert kwargs == {"model_str": "unit-only-counting-stub"}
        calls["loads"] += 1
        return Model()

    monkeypatch.setattr(lgb, "Booster", load)
    predict = asof._view_predictor("unit-only-counting-stub")
    predict(view())
    with pytest.raises(ValueError, match="nonfinite inference output"):
        predict(view())
    assert calls == dict(loads=1, interface=2, predictions=2)


def test_later_grid_error_precedes_interface_and_model_is_not_shared_after_failure(
    monkeypatch, model_text, development_case
):
    data, plan, factor = development_case
    counts = count_model_calls(monkeypatch)
    original_view = forecast_research.forecast_feature_view
    calls = 0

    def broken_later_view(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original_view(*args, **kwargs)
        if calls == 2:
            result["expected_instruments"] = ["wrong"]
        return result

    with monkeypatch.context() as broken:
        broken.setattr(forecast_research, "forecast_feature_view", broken_later_view)
        with pytest.raises(ValueError, match="instrument grid"):
            forecast_research.development_predictions(data, plan, factor, model_text)
    assert counts == dict(loads=1, predictions=1)
    forecast_research.development_predictions(data, plan, factor, model_text)
    assert counts == dict(loads=2, predictions=7)


@pytest.mark.parametrize("corruption", ["metadata", "fit_clock", "prediction"])
def test_local_result_replay_keeps_metadata_time_and_prediction_rejections(
    monkeypatch, model_text, development_case, corruption
):
    data, plan, factor = development_case
    _, _, metadata, _ = forecast_research.training_view(data, plan, factor)
    predictions = forecast_research.development_predictions(data, plan, factor, model_text)
    result = metadata | dict(
        actual_fit_started_at="2026-10-27T00:00:00Z",
        actual_fit_completed_at="2026-10-27T00:00:01Z",
        model_effective_at="2026-10-27T00:00:01Z",
        predictions=predictions,
    )
    # These are a controlled validation input and a retained model, not a claim
    # that this test fitted that model or produced a real worker receipt.
    if corruption == "metadata":
        result["training_rows"] += 1
        reason = "mature-label/preprocessing metadata"
    elif corruption == "fit_clock":
        result["model_effective_at"] = "2026-10-27T00:00:02Z"
        reason = "fit/effective time"
    else:
        result["predictions"][0]["model_ref"] = digest(b"wrong binding")
        reason = "actual serialized model"
    counts = count_model_calls(monkeypatch)
    with pytest.raises(ValueError, match=reason):
        verify_local_result(data, plan, factor, result, model_text)
    assert counts["loads"] == (1 if corruption == "prediction" else 0)
