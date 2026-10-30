# Compare the native price adapters with Qlib operators

This optional comparison executes real Qlib `Ref`, `Mean` and `Div` operators
against the existing native `qlib_roc_v1` and `qlib_ma_v1` adapters on the same
invented close-price arrays. It is an initial numerical compatibility check for
[RFC #39](https://github.com/natureofnature/Quant_RSI/issues/39), not a Qlib research
workflow, speed benchmark, trading simulation or predictive-performance study.

## Pinned, optional environment

Use Linux/Python 3.12 from the repository root. Keep this environment separate
from existing experiment environments; installing dependencies needs network
access, but the comparison itself never fetches data or initializes a provider.

```bash
python3.12 -m venv /tmp/quant-rsi-qlib
/tmp/quant-rsi-qlib/bin/python -m pip install \
  -r requirements-dev.lock -r examples/requirements-qlib-operators.txt
/tmp/quant-rsi-qlib/bin/python -m pip install --no-deps pyqlib==0.9.7
/tmp/quant-rsi-qlib/bin/python -m pip install --no-deps -e .
mkdir -p runs/qlib-operators
/tmp/quant-rsi-qlib/bin/python examples/compare_qlib_operators.py \
  --output runs/qlib-operators/comparison.json
/tmp/quant-rsi-qlib/bin/python -m pytest -q tests/test_qlib_operator_comparison.py
```

The optional requirements cover the imports needed by the operators. This is
not a full Qlib application installation: MLflow, Jupyter, portfolio optimization
and other unused Qlib dependencies are intentionally absent. `pip check` may
therefore report those missing application dependencies. Normal Quant RSI
installation and CI do not require Qlib; the optional test explicitly skips when
Qlib is absent.

The runtime is the published [pyqlib 0.9.7 wheel](https://pypi.org/project/pyqlib/0.9.7/),
whose selected source files match official release commit
[`da920b7`](https://github.com/microsoft/qlib/tree/da920b7f954f48ab1bb64117c976710de198373e).
The example rejects another installed version or changed `base.py`/`ops.py` file
hashes. The native formula definitions retain their separate source pin,
`54355232463878d2eebb91fe0ee5fa7fa1f5976c`; this comparison does not rewrite it.
Version 0.9.7 is a deliberately fixed wheel baseline, not a claim of latest Qlib.
The JSON records both identities, installed dependency versions and the native
adapter's source hash.

## What is aligned

An in-memory Qlib `Feature` supplies the arrays to actual upstream expressions:
`Ref(close, d) / close` and `Mean(close, d) / close`. It replaces data access only;
it does not replace or monkeypatch the arithmetic operators. Qlib's expression
cache key includes the complete fixture's digest so a changed future cannot
reuse an earlier result accidentally.

The fixture contains 96 closes, clean and missing-input variants, and a variant
whose values change from index 64 onward. Both formulas use windows 1, 2, 5 and
30. Every native result sees only its preceding registered-size window.

Raw upstream outputs are retained. Qlib's
[rolling implementation](https://github.com/microsoft/qlib/blob/da920b7f954f48ab1bb64117c976710de198373e/qlib/data/ops.py)
uses partial windows and may ignore a missing value when computing a mean.
`Ref` needs its endpoints; the native ROC contract additionally requires all
intermediate observations. An independently calculated mask therefore requires
the complete consecutive window of finite positive prices before numerical
comparison. It is not derived from whether the native result happened to pass.
This reconciles an intentional eligibility difference without pretending raw
operator behavior is identical.

## Recorded engineering result

The Linux/Python 3.12 operator run on October 29, 2026 produced:

| Measurement | Result |
| --- | --- |
| Case/formula/window combinations | 24 |
| Evaluated positions | 2,304 |
| Positions eligible under the common full-window contract | 1,907 |
| Numerical mismatches after eligibility alignment | 0 |
| Largest absolute difference | 2.22 × 10⁻¹⁶ |
| Raw Qlib values outside native eligibility | 249 |
| Earlier native and raw Qlib outputs unchanged after future mutation | Yes |
| Both engines actually respond within the changed future segment | Yes |

Comparison tolerance is `1e-12` relative and absolute. The optional test passed
with the actual pinned runtime; the ordinary environment reported one explicit
skip. The script writes the full inputs, eligibility masks and raw/aligned/native
outputs to the requested JSON artifact and exits nonzero on a mismatch or failed
future-prefix check. The result is reproducible evidence for these finite cases,
not proof of equivalence over every floating-point input.

The comparison does not exercise Qlib's data provider, availability/receipt
rules, processors, factor search, model fitting, portfolio backtest or broker
integration. Overflow/underflow and other extreme-price behavior remain covered
by native tests, not by this upstream comparison. Broader same-data, same-budget
research comparisons still need a separately frozen task and validation design.
