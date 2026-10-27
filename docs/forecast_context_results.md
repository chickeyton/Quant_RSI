# First Price-and-Context Forecast Batch

The experiment **soxl-context-20261027-v1**, executed on October 27, 2026 UTC,
completed the [paired forecast workflow](forecast_comparison.md). The engineering
pipeline passed; **the models did not beat either naive control**.

## Inputs and historical comparison

SOXL is the target, XSD the semiconductor-industry proxy, and SPY the market
context. Each source contains 756 complete daily sessions from October 20, 2023
through October 26, 2026. Historical availability and raw USD/share units are
explicit assumptions, not certified historical point-in-time evidence.

Three frozen folds use 314/440/566 training observations and 25 evaluation origins
each. Every recipe shares the same masks and labels. B0/B1/B2 contain 3/7/15 price
features. Independently calculated vectors at origins 378 and 755 match exactly,
and all 75 origin/target date pairs agree with the independent calendar check.

Errors below are percentage points of five-session simple return, pooled over
the same 75 observations.

| Recipe or control | MAE | RMSE |
| --- | ---: | ---: |
| B0 | 15.6050 | 18.9578 |
| B1 | 15.3361 | 18.9671 |
| B2 | 15.3357 | 19.0515 |
| Unchanged price | 14.5768 | 18.0298 |
| Fold training-label mean | 14.8655 | 18.2142 |

B2 was selected by the predeclared minimum-MAE rule. Its MAE improvement over B1
is only 0.000388 percentage points, while its RMSE is higher. Both naive controls
perform better than all three models. These development results do not establish
predictive or economic advantage, independent statistical confirmation, or
completed recursive self-improvement.

## Separate publication model

The selected B2 recipe was fitted once more on 691 mature observations. This
publication model has a different identity from every evaluation model; it does
not inherit their measured scores as its own validation.

- Origin: October 26, 2026; raw closing price **$90.50**.
- Target: November 2, 2026, at 21:00 UTC, five registered trading sessions later.
- Engineering point forecast: **$89.1903**, or approximately **−1.4472%**.
- Frozen training boundary: October 27 at 03:34:12.273409 UTC.
- Publication confirmed after commit at 03:43:02.307659 UTC, before the next open
  at 13:30 UTC. Its forward classification describes timing, not reliability.

The 142 archived sidecar events contain 20 news items and 122 financial facts.
Both sources have partial coverage. Their receipt, query, payload and original
response evidence is retained; **none is used as a model feature**.

## Validation and reproducibility

111 relevant offline tests and complete lint/format checks passed. All nine
evaluation fits and the separate publication fit completed. The experiment
consumed its full 10-attempt/1,800-slot-second reservation budget, with no retry
or additional fit.

Repeated execution, status and replay equal the original report. Verification
actively prohibited training, worker dispatch, subprocess launch and collection;
no prohibited call occurred, and every experiment-file fingerprint remained
unchanged.

The model source was frozen at commit e3e17353eca46179be0df8fc2ab74e1328054c5e.
Local evidence resides in runs/soxl_context_20261027_v1/; generated data and
artifacts are not tracked in Git. The primary audit files are
freeze_summary.json, input_oracle_check.json, worker_input_preflight.json,
fit_10_report.json and actual_repeat_replay_summary.json.

Forecast artifact: 596753f2e896c2cdd5c10437ecd654011cd53b7b6728c7c9fdbc6dadbe166162.
Protocol artifact: 3ac53f288c1c9f5c0f64e15ae577d84a2d6e8d2e196490ca380057081356607d.
