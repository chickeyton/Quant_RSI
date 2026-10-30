# Observe issued five-session forecasts

`forecast-observe` connects an existing, confirmed forecast comparison publication
to later market receipts. It performs no collection, fitting, model selection, or
trading. The original experiment, forecast, model, objective, source identities,
calendar and fit charges remain unchanged. Outcomes use a separate append-only
Journal under the same trusted `QUANT_RSI_WORKSPACE`.

## Workflow

First complete a `forecast-compare` experiment (including its separate publication
fit). Use its registered experiment ID, including when it was produced by an
enhanced forecast workflow or factor discovery round:

```bash
quant-rsi forecast-observe register example-experiment
quant-rsi forecast-observe status example-experiment
```

Registration binds the original publication artifact, post-commit confirmation,
model/input/protocol references and experiment manifest/runtime identity. Its
clock cannot manufacture a prospective forecast. The original publication must
already have been issued and confirmed after the origin close and before the
next registered session opens; otherwise observations are explicitly
`not_prospective` and have no evaluation metrics. Registration itself may occur
later, so this is an individual forecast receipt, not a preregistered unbiased
cohort or independent confirmation.

Import later quote receipts through the existing `data import` path using the
**same target data contract** as the experiment. The new snapshot must extend the
original receipt prefix in the same workspace registry. A replacement provider,
different calendar, renamed source, or freshly initialized registry is rejected.
Then explicitly record the outcome:

```bash
quant-rsi forecast-observe settle example-experiment \
  --snapshot-ref TARGET_SNAPSHOT_SHA256 --request-id target-close-received
quant-rsi forecast-observe replay example-experiment
```

There is no polling or scheduler. Neither `--as-of` nor an observation timestamp
override is accepted: receipt first-seen clocks and the controller clock are
authoritative. A source publication timestamp cannot backdate newly imported
quotes. Every outcome stores the snapshot, source/contract/registry references,
receipt watermark, observed time, frozen forecast reference, label evidence and
the previous outcome reference. The limit is 256 observations per experiment;
changing request IDs cannot start a new ledger or renew that limit.

## Outcomes and corrections

- `immature`: the target close has not elapsed, or the kernel has not yet obtained
  a mature target/anchor label. The reason distinguishes elapsed time from absent
  quotes.
- `settled`: a forward publication has a mature eligible label. The record includes
  actual target price, signed price error, absolute/RMS percentage-point return
  errors and the unchanged-price control on the same label. A single row's RMSE
  equals its absolute error; it is not an aggregate study result.
- `insufficient_evidence`: an unsupported corporate action, revised origin anchor,
  or another label exclusion prevents scoring.
- `not_prospective`: the original publication was retrospective or abstained.

Repeat a request with the same snapshot to receive its original immutable result,
even after time passes. Use a new request ID for another observation. Reusing an ID
with a different snapshot fails. New observations cannot roll back clocks or
source receipt watermarks.

A later target-price correction produces another linked outcome; it never edits
the first observed outcome. The latest row is the latest recorded assessment,
including when a correction makes a previously settled label ineligible. Do not
count revisions as additional forecasts or select whichever revision gives the
best score. If the **origin** price changes, the outcome is
`published_origin_price_revised` with no metrics: the issued forecast's anchor and
return denominator are never silently replaced. New corporate-action notices
outside the frozen objective's recognized notice references conservatively make
the label ineligible; this command cannot rewrite the objective after issuance.

## Replay and limits

Keep the original experiment and market registry. Registration and status can
verify an existing publication from an older runtime without changing its saved
identity. Observation mutations and outcome replay require the observation
workspace's recorded source/runtime. Outcome replay validates raw market receipts
and recomputes each label and metric at its saved clock. It does not rerun the
original fit; original model replay remains `forecast-compare replay` under the
original experiment's recorded runtime.

These are developer-visible prospective records with source qualifications, not
production promotion, calibrated intervals, evidence of alpha, or an independent
evaluation. Historical training assumptions remain disclosed by the publication.
This slice does not implement campaign-level cohort selection, minimum sample
requirements, confidence intervals, trial corrections, automatic researcher
feedback, or policy evolution. Those require separately registered protocols.
