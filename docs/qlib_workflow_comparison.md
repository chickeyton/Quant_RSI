# Compare an actual Qlib workflow on the selected study inputs

`research-study --compare-qlib` runs the upstream Qlib workflow after native
research and selection. It uses actual `StaticDataLoader`, `DataHandlerLP`,
`DatasetH`, `LGBModel.fit/predict` and a local MLflow recorder. It is separate from
the older [operator-only conformance example](qlib_operator_comparison.md).

## Optional environment

Use Linux and Python 3.12 from the repository root. Keep the normal development
environment unchanged; Qlib brings additional dependencies including MLflow and
Pandas. The optional requirements pin Qlib 0.9.7, MLflow 3.17.2 and Pandas 2.3.3
alongside the native numerical/test pins.

```bash
python3.12 -m venv /tmp/quant-rsi-qlib-env
source /tmp/quant-rsi-qlib-env/bin/activate
python -m pip install -r examples/requirements-qlib-workflow.txt
python -m pip install --no-deps -e .

quant-rsi research-study XOM --news-research --compare-qlib \
  --output /tmp/xom-qlib-study
```

Codex authentication and public service access follow the
[connected-study guide](connected_research_study.md). Use a new output directory.
Missing Qlib dependencies or `--model ridge` fail before news collection or LLM
calls. This first comparison supports LightGBM only. When reusing an optional environment
after updating the repository, rerun its requirements installation before starting a
new study; financial inputs also require the current pinned PDF/font dependencies.
Existing study artifacts keep their original runtime identities.

For an offline demonstration with actual native and Qlib fits:

```bash
quant-rsi research-study KO --llm simulated --news simulated --related simulated \
  --prices synthetic --compare-qlib --no-ui --output /tmp/qlib-offline-study
pytest -q tests/test_study_qlib.py tests/test_qlib_operator_comparison.py
```

The core CI job runs the complete regression suite without Qlib and explicitly skips
the optional upstream integration cases. A separate **Qlib / Python 3.12** job installs
the pinned optional environment, requires the real runtime, and runs just the two
modules above with retained timing and JUnit artifacts. It runs alongside the core
suite without adding Qlib to the normal development environment. Run the command above
locally to exercise the real upstream model and recorder; mocks do not
satisfy this check. The optional regressions include both cash-flow factors
with actual native/Qlib fits, common-account parity and separate financial-removal
and Qlib comparison budgets. Their researcher, news and financial facts are explicit
offline fixtures. The missing-dependency test scopes its stub to the Qlib adapter
so that importing another module cannot retain a fake global version lookup.

## What is held equal

The selected native fit supplies its exact feature order, feature values,
mature training labels, and development/terminal session lists. Current news
and peer features keep their existing eligibility; no second feature pipeline
recomputes or substitutes inputs. Qlib gets evaluation features without their
labels. No Qlib preprocessing, validation segment or early stopping changes the
frozen model configuration: both fits use 24 boosting rounds and seed 7. The
adapter checks Qlib's prepared rows, columns and training labels before fitting.

The selected native model remains frozen. Qlib receives one additional fit,
declared as `comparison_fit_budget: 1`, counted separately from the research
`--max-fits` allowance. For the default two-round study this is three fits total.
Comparison results never feed back into native selection or replace its latest
forecast. Disabled or unavailable terminal rows remain unavailable; the adapter
does not drop them to manufacture a comparison. A later comparison failure
leaves the study failed with its completed native stages retained in the trace.

Both engines' forecasts enter the same Quant RSI account evaluator with the
same next-open policy, costs, cash, threshold and buy-and-hold benchmark. The
HTML report and final UI node show forecast MAE, net return, Sharpe and drawdown.
The complete account histories remain in `qlib-comparison/comparison.json`.

`qlib-comparison/inputs.json` retains the common numerical inputs and hashes of
the native prediction and market prices. `model.txt` retains the upstream fitted
booster. `mlflow.db` and `artifacts/` hold local workflow tracking; no remote
tracking service is required. Recorded package versions and the input hash
appear in the comparison report. Existing completed study directories are not
rewritten; run a fresh study to create a comparison.

## Interpretation

This is a workflow semantics comparison of the same selected LightGBM problem.
Matching forecasts are expected when the adapters preserve its semantics;
agreement is reported with absolute tolerances of `1e-10` for log returns and
`1e-8` for prices. Differences are reported rather than treated as an automatic
win. Reusing factors selected by the native research loop is not a matched-budget
comparison of research policies or independent predictive evidence.

This does not run Qlib's portfolio/backtest engine, Alpha158/360, a cross-sectional
benchmark, or an autonomous Qlib researcher. Both account reports use the common
native execution adapter. It provides the first actual upstream workflow path
on which broader comparisons can be built, without changing the study's existing
historical-data and LLM-knowledge limitations.

Upstream implementations: [data loader](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/data/dataset/loader.py),
[data handler](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/data/dataset/handler.py),
[LightGBM model](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/contrib/model/gbdt.py).
## Cash-flow integration observation (2026-11-01)

A follow-up run on `a7b7bc2` used public AAPL/related daily prices and retained official
AAPL statements, with an explicitly simulated acceptance researcher and simulated news.
The first round selected `financial.operating_cashflow_yoy`; the revision selected
`financial.free_cashflow_yoy`. The first native round was retained. Actual Qlib 0.9.7
then fitted the selected common inputs using LightGBM 4.7.0 and MLflow 3.17.2.
All 20 development and 10 reserved-terminal predictions matched the native model:
maximum absolute differences were zero for both log return and price. The common
input hash was `e62771a8a2bf0255c4783df6ea756b0adde7f4b80eb6a3ed0104c89d1e21d26d`.
This checks business-column transport and workflow semantics. It does not test a real
LLM proposal, compare research policies, or establish predictive advantage. The
separate real-Codex cash-flow observation is recorded in the
[connected-study guide](connected_research_study.md#financial-business-factors).
