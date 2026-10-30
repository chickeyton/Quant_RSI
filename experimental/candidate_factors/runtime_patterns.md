# Upstream API patterns

Execution modes describe dependencies; these API patterns describe how the
selected upstream implementations expose them. References are pinned in
[sources.json](sources.json), with entry points in the [candidate index](catalog.md).
This is an integration survey, not a requirement to install four runtimes.

| Source | Observed API pattern | What the shared framework needs to represent |
| --- | --- | --- |
| Qlib | Alpha158DL/Alpha360DL build named expression strings; operators evaluate rolling/lagged series. CSRankNorm is a separate panel preprocessing step. | Typed expression nodes, explicit operator semantics and preprocessing scope. Strings in this survey are data, not trusted executable code. |
| GS Quant | Decorated functions consume pandas Series and return Series/DataFrames; rolling regressions use estimator objects. Windows may be observation counts or date offsets with separate ramp behavior. | Alignment, calendar/count windows, readiness, output naming and fitted-state identity. A common Python signature does not imply one execution mode. |
| LEAN | C# indicator objects consume sequential Update inputs, retain state, expose IsReady/WarmUpPeriod and Reset; composite outputs are properties. | Ordered event updates, initialization and reset/replay contracts, component output selection. Port selected behavior or write a separately validated bridge; do not require the entire trading engine just to describe a candidate. |
| QuantDinger | A registry associates metadata/defaults with DataFrame-to-scalar callbacks; a panel helper iterates symbols; catalog adapters dispatch by namespace. | Separate a definition from a configured binding and output selector. Adapter availability and upstream package licenses remain explicit dependencies. |

## Semantics worth preserving or deliberately changing

- Qlib's temporal rank supports partial windows with `min_periods=1`; its
  cross-sectional rank groups instruments by datetime. A full-window local policy
  must be explicit, not silently described as exact upstream behavior.
- GS Bollinger bands return two columns in lower/upper order. The adapter must name
  and freeze them before entering a model recipe; a middle band is not a third
  upstream output of that function.
- LEAN Ichimoku's main indicator value is the current close, while the named
  components hold the calculated lines. Binding only the main value would create
  a different feature. Its `senkouAPeriod` argument appears in the constructor/name
  but is not used by the calculation at the pinned commit.
- QuantDinger `_numeric` removes nonfinite observations. `_require` uses that
  helper, so some rolling calculations operate on retained observations instead
  of elapsed bars. Independently cleaned fields also need alignment checks.
- QuantDinger `compute_panel_factor` skips missing/insufficient-history instruments
  and omits nonfinite results. Importing that behavior without a frozen population
  policy would silently change the cross-sectional comparison set.
- QuantDinger's catalog includes TA-Lib only when available and Qlib families only
  when requested. Catalog listing, successful computation and permission to
  redistribute a dependency are three separate checks.

These observations are source inspection, not runtime equivalence evidence. Keep
upstream behavior and the proposed Quant RSI contract separately documented. Any
intentional change to cleaning, initialization or timing needs its own versioned
semantics and adapter tests under RFC #31.
