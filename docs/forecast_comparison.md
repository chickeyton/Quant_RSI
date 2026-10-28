# Paired Forecast Comparison

This bounded experiment compares price features and publishes a separately fitted
model. News and financial statements can be archived alongside the inputs, but
**B0/B1/B2 do not use them as model features**.

## Frozen inputs and feature recipes

Use a validated, versioned ForecastInputs JSON envelope with one target and
independent industry and market sources. SOXL/XSD/SPY are the initial source
roles; XSD is a semiconductor industry proxy, not an exact replication of SOXL's
underlying index. Preserve each source's raw evidence, calendar, action
interpretation, receipt times and declared historical availability assumptions.

B0 retains the original three columns. B1 contains B0 plus 20/60-session returns
and population volatility. B2 contains B1 plus 5/20-session industry and market
returns and relative returns: 3, 7 and 15 columns respectively. Missing prices or
misaligned closes are exclusions, with no filling. See
[the input contract](forecast_context_inputs.md).

## Preparing actual inputs

The source adapter accepts a saved audit package and a new output directory:

~~~python
from quant_rsi.forecast.forecast_context_data import prepare_context_data

prepared = prepare_context_data("evidence/market_audit", "runs/context_inputs")
~~~

It writes the standalone input envelope to runs/context_inputs/inputs.json.
The audit manifest must bind original price responses, their HTTP receipts, and
official exchange calendars. The adapter preserves original HTTP times and
performs ingestion/export in its own child-process workspace; registry receipt
times reflect that actual ingestion. No fit occurs. Optional collected signals
and all supplied evidence can be passed through the signals argument. Keep
generated inputs and verbose evidence outside version control.

## Evaluation and publication

The production grid uses 756 completed sessions. Zero-based evaluation origins
are 378–498, 504–624 and 630–750, stepping by five: 25 per fold.
Before availability/action exclusions, training origins are 60–373, 60–499 and
60–625: 314/440/566 observations. All recipes share each fold's frozen training
and evaluation masks, including recorded exclusions.

A training label must end on or before the fold-origin close and be knowable by
that boundary. The LocalDataSpec.train_cutoff field belongs to the imported
container; this experiment uses its explicit per-fold boundaries instead.

Each fold refits the same deterministic LightGBM configuration (seed 7 for the
initial dataset, 24 boosting rounds). Selection uses pooled MAE in percentage
points across every eligible frozen fold. Exact ties prefer B0, B1, then B2.
Incomplete candidates cannot win. RMSE, fold metrics and zero-return/training-mean
benchmarks remain available in the immutable fold reports. This development
comparison is not independent performance confirmation.

The selected recipe receives a separate publication fit using labels actually
received and mature at the frozen publication boundary. With complete coverage,
691 origins (60–750) are available. Its model has a separate identity and does
not inherit the evaluated models' measured performance.

## Commands and recovery

Set QUANT_RSI_WORKSPACE to the absolute isolated workspace used for this
experiment. Save the input envelope as forecast_inputs.json.

~~~bash
python -m quant_rsi forecast-compare prepare forecast_inputs.json --experiment-id context-v1
python -m quant_rsi forecast-compare run context-v1 --protocol-ref FROZEN_PROTOCOL_HASH --max-fits 1
python -m quant_rsi forecast-compare status context-v1
python -m quant_rsi forecast-compare replay context-v1
~~~

Preparation freezes inputs, masks, feature views, recipes and runtime without
fitting. Run requires the exact returned protocol reference. The existing worker
and SQLite Journal charge each dispatch separately: at most nine evaluation
fits and one publication fit, each reserving 180 seconds, totaling 1,800 slot
seconds. Failed or unknown attempts remain charged; there is no automatic
redispatch. Completed worker output can be recovered without fitting again.
Changing a publication request cannot create a new fit; the experiment ID binds
the cumulative ledger.

A committed publication receives a separate post-commit timestamp. A report is
forward only if that confirmation precedes the next registered open. Otherwise
it remains retrospective or abstained. Unknown actions or unavailable prediction
inputs cause abstention. Repeated runs and replay use saved artifacts without
collection, proposal generation or fitting.

## Validation

~~~bash
python -m pytest -q tests/test_forecast_comparison.py
~~~

Integration fixtures exercise the real worker entry point with a substitute
model, so lifecycle tests perform no real model fits. They cover common masks,
pooled selection, incomplete candidates, separate publication, charged failures,
crash recovery, immutable budgets and post-commit timing.
