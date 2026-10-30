# Execution requirements for RFC #31

These tags classify survey requirements; they do not add operators to the RFC's
bounded grammar. Multiple tags may apply to one definition. See also the [upstream API patterns](runtime_patterns.md). No model fits or
factor-search allowances are created by collecting these records.

| Mode | Representative candidates | Required capability / acceptance evidence |
| --- | --- | --- |
| `elementwise` | Qlib KMID/KLEN | Typed units, denominator policy, finalized input availability |
| `rolling_window` | Qlib ROC/MA/CORR/CORD, Parkinson | Session/calendar window, minimum history, NaNs, no future leakage |
| `recursive_state` | EMA, RSI, ATR, OBV, KAMA, Supertrend | Seed/reset identity, ordered updates, persisted state, chunk/full replay equality |
| `cross_section` | Qlib CSRankNorm | Frozen as-of universe, tied ranks, common eligibility and missing members |
| `multi_series_alignment` | GS market beta, rolling OLS | Benchmark/regressor identity, clock alignment, missing index policy |
| `rolling_fit` | GS rolling OLS | Mature responses, fit cutoffs, parameters and training lineage; fitted values are not forecasts |
| `report_asof` | Valuation/quality ratios, supplied revenue growth | Fiscal period, publication/receipt times, revisions, currency/TTM basis |
| `multi_output` | Bollinger, Ichimoku, Supertrend, OLS | Stable named output order and types; selector-based outputs also need explicit binding |
| `shifted_components` | Ichimoku | Distinguish calculation, observation and chart display timestamps |
| `centered_noncausal` | GS seasonal trend | Reference-only; reject for causal materialization without a separately versioned redesign |

## Suggested integration order

1. Map KMID/KLEN and the reciprocal ROC example into supported arithmetic/lag
   operators. Explicitly record a capability gap if the frozen grammar lacks one;
   do not execute arbitrary formula strings from this directory.
2. Exercise rolling windows and recursive state separately with a moving average
   and OBV. Confirm both timestamp eligibility and restart determinism. Named
   similarities to existing momentum/volatility recipes do not establish identity.
3. Add benchmark alignment, cross-sectional ranking and report-as-of inputs only
   when their evidence scopes and availability clocks can be represented.
4. Defer learned rolling fits, composite delayed indicators and remaining stateful
   variants until the necessary contracts exist. Keep the centered decomposition
   blocked rather than assigning it an artificial lag and calling it causal.

For each implemented adapter, freeze parameters, source commit, implementation
hash, dependency versions, adjustment basis, output names/units, warmup, missingness
and eligibility before materialization. Compare a short deterministic series to
the pinned upstream implementation, including non-default settings and edge cases.
Then perturb future inputs, test missing coverage, replay state and preserve the
same eligible comparison rows. Numeric zero is permitted only where supported by
the transformation and observed inputs; unavailable evidence remains missing.

## Gaps left explicit

This batch does not supply news extraction, event activation/decay/retraction,
learned intermediate predictors, unrestricted generated code, order-book streams
or cross-frequency publication-time joins. In particular, conventional technical
indicators are not evidence that event-driven discovery is implemented. #31 owns
those contracts and the existing retained-news evidence path remains authoritative.

No candidate has independent predictive validation. Engineering acceptance can
succeed with negative economic results; predictive/trading promotion still follows
the repository's separate evidence and budget requirements.
