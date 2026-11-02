"""Frozen-input export plus optional real upstream fit/predict/account integration."""

import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quant_rsi.cli import main
from quant_rsi.core.artifacts import digest
from quant_rsi.research import study_qlib
from quant_rsi.research.demo import load_catalog, select_factors
from quant_rsi.research.demo_features import price_feature
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.study_ui import read_state

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


def command(root, *extra):
    return [
        "research-study",
        "KO",
        "--llm",
        "simulated",
        "--news",
        "simulated",
        "--related",
        "simulated",
        "--prices",
        "synthetic",
        "--no-ui",
        "--compare-qlib",
        "--output",
        str(root),
        *extra,
    ]


def test_export_uses_selected_fit_and_excludes_evaluation_labels():
    catalog = load_catalog(CATALOG)
    model = LightGBMDemo(catalog, evaluation_sessions=20, terminal_sessions=10)
    model.data = replace(model.data, instruments=("KO",))
    first = select_factors(catalog, {"factor_ids": ["equity.price_momentum_5"]})
    second = select_factors(catalog, {"factor_ids": ["equity.price_mean_return_5"]})
    model.predict(model.context("KO"), first, {})
    original = model.comparison_inputs(1)
    model.predict(model.context("KO"), second, {})
    assert model.comparison_inputs(1) == original
    assert model.comparison_inputs(2)["factor_ids"] == ["equity.price_mean_return_5"]
    train = original["segments"]["train"]
    np.testing.assert_array_equal(
        np.asarray(train["features"]).ravel(),
        price_feature(model.data, first[0]["recipe"])[model.train],
    )
    assert max(train["label_sessions"]) <= original["segments"]["development"]["sessions"][0]
    for name in ("development", "terminal"):
        assert "labels" not in original["segments"][name]
    model.data.closes[model.terminal_origins[-1] + model.horizon, 0] *= 2
    assert model.comparison_inputs(1) == original
    # The exported labels are the fitted labels, not a later recomputation.
    model.data.closes[model.train[-1], 0] *= 2
    assert model.comparison_inputs(1) == original
    # A missing terminal feature preserves the complete window as unavailable.
    model._fits[0][1][model.terminal_origins[0], 0] = np.nan
    missing = model.comparison_inputs(1)["segments"]["terminal"]
    assert not missing["available"] and len(missing["features"]) == 10
    assert missing["features"][0] == [None]
    with pytest.raises(ValueError, match="selected fit"):
        model.comparison_inputs(3)


def test_missing_optional_runtime_fails_before_provider_work(tmp_path, monkeypatch, capsys):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    # Keep the simulated missing dependency local to this adapter. Patching the
    # stdlib module can leak through another module's first `from ... import version`.
    metadata = SimpleNamespace(
        version=missing, PackageNotFoundError=importlib.metadata.PackageNotFoundError
    )
    monkeypatch.setattr(study_qlib, "importlib", SimpleNamespace(metadata=metadata))
    root = tmp_path / "missing"
    assert main(command(root)) == 1
    assert "optional Qlib workflow environment" in capsys.readouterr().err
    assert not root.exists()


def test_ridge_comparison_rejected_before_provider_work(tmp_path, capsys):
    root = tmp_path / "ridge"
    assert main(command(root, "--model", "ridge")) == 1
    assert "requires --model lightgbm" in capsys.readouterr().err
    assert not root.exists()


@pytest.mark.parametrize("terminal_sessions", [0, 10])
@pytest.mark.parametrize("horizon_sessions", [5, 20, 60])
def test_actual_upstream_workflow_matches_frozen_inputs_and_common_accounts(
    tmp_path, terminal_sessions, horizon_sessions
):
    pytest.importorskip("qlib", reason="optional upstream Qlib workflow environment is absent")
    pytest.importorskip("mlflow", reason="optional upstream Qlib workflow recorder is absent")
    root = tmp_path / "study"
    assert (
        main(
            command(
                root,
                "--terminal-sessions",
                str(terminal_sessions),
                "--horizon-sessions",
                str(horizon_sessions),
            )
        )
        == 0
    )
    assert_upstream_comparison(root, terminal_sessions)
    inputs = json.loads((root / "qlib-comparison/inputs.json").read_bytes())
    assert inputs["horizon_sessions"] == horizon_sessions


def test_qlib_and_walk_forward_keep_separate_budgets_and_selected_model(tmp_path):
    pytest.importorskip("qlib", reason="optional upstream Qlib environment is absent")
    pytest.importorskip("mlflow", reason="optional upstream recorder is absent")
    root = tmp_path / "combined"
    assert (
        main(
            command(
                root, "--horizon-sessions", "60", "--walk-forward-folds", "2", "--compare-models"
            )
        )
        == 0
    )
    assert_upstream_comparison(root, 10)
    result = json.loads((root / "result.json").read_bytes())
    chosen = result["iterations"]["selected_round"]
    selected = result["iterations"]["attempts"][chosen - 1]["prediction"]
    diagnostic = result["study"]["walk_forward"]
    assert diagnostic["status"] == "completed"
    assert diagnostic["selected_round"] == chosen
    assert diagnostic["extra_fits"] == result["configuration"]["walk_forward_fit_budget"] == 2
    assert len(diagnostic["folds"][-1]["training_sessions"]) > selected["training_rows"]
    assert result["study"]["qlib_comparison"]["selected_round"] == chosen
    paired = result["study"]["model_comparison"]
    assert paired["selected_round"] == chosen
    assert set(paired["models"]) == {"lightgbm", "ridge"}
    for name in ("development", "terminal"):
        assert (
            paired["segments"][name]["models"]["lightgbm"]["metrics"]
            == result["study"]["qlib_comparison"]["segments"][name]["native"]["metrics"]
        )


def assert_upstream_comparison(root, terminal_sessions):
    result = json.loads((root / "result.json").read_text())
    comparison = result["study"]["qlib_comparison"]
    inputs = root / "qlib-comparison/inputs.json"
    assert comparison["inputs_sha256"] == digest(inputs.read_bytes())
    assert comparison["versions"]["pyqlib"] == "0.9.7"
    assert comparison["recorder_id"]
    assert (root / "qlib-comparison/model.txt").stat().st_size > 0
    assert (root / "qlib-comparison/mlflow.db").stat().st_size > 0
    assert result["configuration"]["comparison_fit_budget"] == 1 + int(
        result["configuration"]["financials"]
    ) + int(result["configuration"].get("compare_models", False))
    assert result["runtime"]["counts"]["comparison_model_stages_completed"] == 1 + int(
        result["configuration"]["financials"]
    ) + int(result["configuration"].get("compare_models", False))
    assert result["runtime"]["counts"]["model_stages_completed"] == 2
    selected = result["iterations"]["attempts"][result["iterations"]["selected_round"] - 1]
    assert comparison["factor_ids"] == selected["factor_ids"]
    for name in ("development", "terminal"):
        values = comparison["segments"][name]
        if name == "terminal" and not terminal_sessions:
            assert values["status"] == "unavailable"
            continue
        assert values["prediction_parity"]
        assert values["qlib"]["metrics"] == pytest.approx(values["native"]["metrics"])
        assert values["qlib"]["strategy"] == values["native"]["strategy"]
    assert "Actual Qlib workflow comparison" in (root / "report.html").read_text()
    view = next(e["view"] for e in read_state(root)["events"] if e["stage"] == "qlib_comparison")
    assert view["kind"] == "engine_comparison"
    assert len(view["rows"]) == (4 if terminal_sessions else 2)


@pytest.mark.parametrize(
    "business_factor",
    ["financial.operating_cashflow_yoy", "financial.free_cashflow_yoy", "financial.inventory_yoy"],
)
def test_actual_upstream_receives_selected_business_column(tmp_path, monkeypatch, business_factor):
    pytest.importorskip("qlib", reason="optional upstream Qlib workflow environment is absent")
    pytest.importorskip("mlflow", reason="optional upstream Qlib workflow recorder is absent")
    from quant_rsi.research import study
    from quant_rsi.research.demo import simulated_hypothesis

    def researcher(context, catalog, **kwargs):
        result = simulated_hypothesis(context, catalog, **kwargs)
        definition = next(d for d in catalog["definitions"] if d["factor_id"] == business_factor)
        assert definition["runnable"]
        result["factor_ids"] = [
            key for key in result["factor_ids"] if not key.startswith("financial.")
        ] + [business_factor]
        return result

    monkeypatch.setattr(study, "simulated_hypothesis", researcher)
    root = tmp_path / "business-study"
    assert main(command(root, "--financials")) == 0
    assert_upstream_comparison(root, 10)
    inputs = json.loads((root / "qlib-comparison/inputs.json").read_bytes())
    column = inputs["factor_ids"].index(business_factor)
    values = [row[column] for row in inputs["segments"]["train"]["features"]]
    assert all(isinstance(value, float) and np.isfinite(value) for value in values)
    assert any(value != 0 for value in values)
    comparison = json.loads((root / "qlib-comparison/comparison.json").read_bytes())
    assert business_factor in comparison["factor_ids"]
