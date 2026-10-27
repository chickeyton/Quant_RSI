# Configurable forecast workflow: market demonstration

The bounded two-instrument run completed successfully on October 27, 2026. It demonstrates the configurable workflow, shared accounting and terminal repeatability. Both selected models have higher MAE than both naive controls; predictive advantage remains unestablished.

## Frozen setup

Source commit: [a4d7b4d](https://github.com/natureofnature/Quant_RSI/commit/a4d7b4d27ff193d070742aee7acb415bb06f0fa2). The [workflow guide](enhanced_forecast_workflow.md) describes the supported contracts. The [machine-readable evidence](evidence/enhanced_forecast_20261027.json) records input, protocol, runtime, grant, model and receipt references, source clocks, masks and exact metrics. It is a summary; replay requires the original preserved workspace and recorded runtime.

| Run | Target / industry / market | History | Evaluation | Publication training |
| --- | --- | --- | --- | --- |
| r1-soxl-20261027-v1 | SOXL / XSD / SPY | 756 sessions | 3 × 25 origins | 691 rows |
| r1-ko-20261027-v1 | KO / XLP / SPY | 756 sessions | 3 × 25 origins | 691 rows |

History spans October 20, 2023 through October 26, 2026. B0/B1/B2 share each fold's dates and eligibility masks, with 314, 440 and 566 training rows. Selection minimizes pooled development MAE. Each run then fits a separate publication model; evaluation scores are not assigned to that model.

One campaign, **r1-market-proof-20261027**, registered both runs before collection. Each received ten attempts and 1,800 reserved timeout-slot seconds, for an aggregate ceiling of twenty attempts and 3,600 seconds. These are conservative reserved slots, not elapsed training time.

## Development comparison

Errors below are percentage points of five-session simple returns. Each row uses 75 observations; lower is better.

| Target | Model or control | MAE | RMSE |
| --- | --- | ---: | ---: |
| SOXL | B0 | 15.605010 | 18.957833 |
| SOXL | B1 | 15.336110 | 18.967063 |
| SOXL | B2 (selected) | 15.335722 | 19.051510 |
| SOXL | Unchanged price | 14.576806 | 18.029765 |
| SOXL | Training-label mean | 14.865512 | 18.214248 |
| KO | B0 | 1.882867 | 2.476648 |
| KO | B1 | 1.940417 | 2.521565 |
| KO | B2 (selected) | 1.838277 | 2.427914 |
| KO | Unchanged price | 1.816333 | 2.439320 |
| KO | Training-label mean | 1.830616 | 2.426162 |

SOXL B2 loses to both controls on both metrics. KO B2 beats unchanged price on RMSE only; it loses to the training-label mean on RMSE and to both controls on MAE. Selection among three candidates is development evidence, not independent statistical confirmation. The SOXL setup repeats the earlier historical comparison to exercise the new workflow; it is not an independent replication of predictive performance.

## Separate publication outputs

These unvalidated point outputs target the November 2, 2026 close, five trading sessions after the October 26 origin. They do not establish the future realized error.

| Target | Origin price (USD) | Forecast price (USD) | Forecast return | Post-commit confirmation (UTC) |
| --- | ---: | ---: | ---: | --- |
| SOXL | 90.500000 | 89.190300 | -1.447182% | 2026-10-27T09:47:47.811408+00:00 |
| KO | 92.699997 | 94.125759 | +1.538039% | 2026-10-27T09:48:08.572667+00:00 |

Both actual issue and post-commit confirmation clocks precede the first target session's 13:30 UTC open. Their forward timing classification concerns issuance only, not forecast accuracy. Each publication has its own model and result reference, and **publication_model_validation_metrics** is null.

## Accounting and repeatability

- All twenty LightGBM jobs completed: nine evaluation fits and one publication fit per instrument. Failed jobs, replacement runs and extra allocations: zero.
- The original child journals reconcile to twenty reservations and 3,600 reserved seconds. Total measured worker-process time was 13.514269 seconds; it is a different quantity from reserved timeout slots.
- Repeated **prepare**, **run**, **status** and **replay** for both terminal workflows, plus campaign **status** and **replay**, returned the original reports. All 13,774 workspace files (96,283,240 bytes) retained identical hashes. Additional reservations and reserved seconds: zero.
- Exact-source [hosted CI](https://github.com/natureofnature/Quant_RSI/actions/runs/40477159469) passed 428 offline tests, lint and formatting checks.

## Limits

The models use price and context features; news and fundamentals are not inputs. Original HTTP and registry first-seen clocks are preserved, but historical availability and raw-price units remain explicit assumptions rather than certified point-in-time evidence. This demonstration does not implement recursive research, calibrated confirmation, production promotion or a profitability claim. The previous completed experiment remains a separate, closed record.
