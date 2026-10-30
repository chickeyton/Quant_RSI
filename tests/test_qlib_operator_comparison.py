"""Optional conformance run; normal development does not install Qlib."""

import importlib.util
from pathlib import Path

import pytest


def test_actual_pinned_qlib_matches_native_full_window_contract():
    pytest.importorskip("qlib", reason="optional pinned Qlib operator environment is absent")
    path = Path(__file__).resolve().parents[1] / "examples" / "compare_qlib_operators.py"
    spec = importlib.util.spec_from_file_location("compare_qlib_operators", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    report = example.run_comparison()
    summary = report["summary"]
    assert summary["mismatches"] == 0
    assert summary["future_prefix_unchanged"]
    assert summary["future_mutation_observed"]
    assert summary["raw_values_outside_eligibility"] > 0
    assert report["upstream"]["version"] == "0.9.7"
    assert example.qlib_values([10.0, 20.0, 40.0], "qlib_roc_v1", 2) == [None, None, 0.25]
    assert example.qlib_values([10.0, 20.0, 30.0], "qlib_ma_v1", 3) == pytest.approx(
        [1, 0.75, 2 / 3]
    )
    missing = next(
        row
        for row in report["results"]
        if row["case"] == "missing" and row["op"] == "qlib_roc_v1" and row["window"] == 5
    )
    assert missing["qlib_raw"][10] is not None  # Ref only needs its endpoints.
    assert missing["native"][10] is None  # Native requires all consecutive inputs.
    assert missing["qlib_aligned"][10] is None
