"""Compare two native price adapters with pinned, real Qlib operators offline."""

import argparse
import importlib.metadata
import math
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.factors.price_adapters import (
    QLIB_REVISION,
    CandidatePriceRecipe,
    evaluate_price_adapter,
)

QLIB_VERSION = "0.9.7"
QLIB_RELEASE = "da920b7f954f48ab1bb64117c976710de198373e"
QLIB_FILES = {
    "base": "30382dd1a3c47506e61514f37533a3407f63fd01c9a9fd5a12b8c0cce85c5d5c",
    "ops": "6f648355725a85a9f17528d864281fc065f8a4f887261a9909f7495d5db42760",
}


def upstream_identity():
    if importlib.metadata.version("pyqlib") != QLIB_VERSION:
        raise ValueError("this comparison requires pyqlib==" + QLIB_VERSION)
    from qlib.data import base, ops

    hashes = {
        name: digest(Path(module.__file__).read_bytes())
        for name, module in (("base", base), ("ops", ops))
    }
    if hashes != QLIB_FILES:
        raise ValueError("installed Qlib operator sources differ from the pinned wheel")
    return dict(version=QLIB_VERSION, release_commit=QLIB_RELEASE, source_sha256=hashes)


def qlib_values(prices, op, window):
    import pandas as pd
    from qlib.data.base import Feature
    from qlib.data.ops import Div, Mean, Ref

    class FrozenClose(Feature):
        def _load_internal(self, instrument, start_index, end_index, *args):
            return pd.Series(prices, dtype="float64").loc[start_index:end_index].copy()

    # Qlib caches by expression string. The data digest prevents another scenario
    # (especially future mutations) from accidentally reusing the previous result.
    close = FrozenClose("fixture_" + digest(canonical(prices)))
    numerator = Ref(close, window) if op == "qlib_roc_v1" else Mean(close, window)
    values = Div(numerator, close).load("invented", 0, len(prices) - 1, "day")
    return [float(value) if math.isfinite(value) else None for value in values]


def compare_case(name, prices, op, window):
    recipe = CandidatePriceRecipe(op=op, window=window)
    native, eligible = [], []
    for index in range(len(prices)):
        history = prices[max(0, index + 1 - recipe.observations) : index + 1]
        eligible.append(
            len(history) == recipe.observations
            and all(value is not None and math.isfinite(value) and value > 0 for value in history)
        )
        native.append(
            evaluate_price_adapter(recipe, [math.nan if x is None else x for x in history])
        )
    raw = qlib_values(prices, op, window)
    aligned = [value if allowed else None for value, allowed in zip(raw, eligible, strict=True)]
    mismatches = [
        i
        for i, (left, right) in enumerate(zip(native, aligned, strict=True))
        if (left is None) != (right is None)
        or left is not None
        and not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    ]
    errors = [
        abs(left - right)
        for left, right in zip(native, aligned, strict=True)
        if left is not None and right is not None
    ]
    return dict(
        case=name,
        op=op,
        window=window,
        prices=prices,
        eligible=eligible,
        native=native,
        qlib_raw=raw,
        qlib_aligned=aligned,
        raw_values_outside_eligibility=[
            i
            for i, (value, allowed) in enumerate(zip(raw, eligible, strict=True))
            if value is not None and not allowed
        ],
        mismatch_indices=mismatches,
        max_absolute_error=max(errors, default=0.0),
    )


def run_comparison():
    identity = upstream_identity()
    clean = [float(80 + i // 4 + i % 7) for i in range(96)]
    missing = [None if i in {8, 35, 70} else value for i, value in enumerate(clean)]
    mutation_start = 64
    mutated = [value if i < mutation_start else value * 3 for i, value in enumerate(clean)]
    results = [
        compare_case(name, prices, op, window)
        for name, prices in (("clean", clean), ("missing", missing), ("future_mutated", mutated))
        for op in ("qlib_roc_v1", "qlib_ma_v1")
        for window in (1, 2, 5, 30)
    ]
    baseline = {(row["op"], row["window"]): row for row in results if row["case"] == "clean"}
    future_unchanged = all(
        row[key][:mutation_start] == baseline[row["op"], row["window"]][key][:mutation_start]
        for row in results
        if row["case"] == "future_mutated"
        for key in ("native", "qlib_raw")
    )
    future_observed = all(
        any(
            row[key][mutation_start:] != baseline[row["op"], row["window"]][key][mutation_start:]
            for row in results
            if row["case"] == "future_mutated"
        )
        for key in ("native", "qlib_raw")
    )
    summary = dict(
        cases=len(results),
        observations=sum(len(row["native"]) for row in results),
        eligible_observations=sum(sum(row["eligible"]) for row in results),
        mismatches=sum(len(row["mismatch_indices"]) for row in results),
        raw_values_outside_eligibility=sum(
            len(row["raw_values_outside_eligibility"]) for row in results
        ),
        max_absolute_error=max(row["max_absolute_error"] for row in results),
        future_prefix_unchanged=future_unchanged,
        future_mutation_observed=future_observed,
    )
    return dict(
        kind="QlibOperatorComparison",
        schema_version=1,
        upstream=identity,
        native_formula_source_revision=QLIB_REVISION,
        native_adapter_sha256=digest(
            Path(evaluate_price_adapter.__code__.co_filename).read_bytes()
        ),
        environment={
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        eligibility_policy="complete_consecutive_finite_positive_native_window",
        tolerance=dict(relative=1e-12, absolute=1e-12),
        mutation_start=mutation_start,
        scope="offline_operator_numerics_only_no_provider_training_portfolio_or_performance_claim",
        summary=summary,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_comparison()
    atomic_write(args.output, canonical(report))
    print(canonical(dict(output=str(args.output), **report["summary"])).decode())
    if (
        report["summary"]["mismatches"]
        or not report["summary"]["future_prefix_unchanged"]
        or not report["summary"]["future_mutation_observed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
