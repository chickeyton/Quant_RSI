# Configurable enhanced forecasting

This workflow implements a bounded delivery slice of [RFC #1](https://github.com/natureofnature/Quant_RSI/issues/1): explicit target, industry proxy and market proxy inputs, shared campaign allowances, collection, freezing, historical comparison, and a separate publication fit. It does not implement feedback-dependent recursive research or establish predictive advantage.

## Install and register one shared campaign

Use Python 3.12 and the pinned development dependencies. All instruments use the same engine and canonical workspace:

~~~sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
export QUANT_RSI_WORKSPACE="$PWD/runs/enhanced-workspace"
mkdir -p "$QUANT_RSI_WORKSPACE" runs/enhanced
python -m quant_rsi forecast-campaign init examples/forecast_campaign.json
python -m quant_rsi forecast-workflow prepare soxl-context-example \
  > runs/enhanced/soxl-prepared.json
python -m quant_rsi forecast-workflow prepare ko-context-example \
  > runs/enhanced/ko-prepared.json
~~~

Campaign initialization performs no network access or fitting. It freezes the complete roster, normalized configurations, and parent budget. The example reserves two child ceilings of ten fits and 1,800 worker-seconds each, against a total of twenty fits and 3,600 worker-seconds. These seconds are reserved worker timeout slots, not measured elapsed training time. Unused and failed allocations remain reserved; there is no transfer, rollover or replacement-ID mechanism. Edit a new plan before initialization to choose a different finite roster.

Campaign and run IDs are immutable; run IDs are unique across the workspace. An identical repeated campaign plan returns its existing grant. Changed limits, profiles or roster entries fail. Preparation derives each configuration from that registration and requests **zero fits**. Instrument profile files under **instruments/US/** are organization aids, not separate engines or workspaces.

## Supported inputs and meaning

Each profile requires three distinct explicit uppercase tickers. The example supplies SOXL/XSD/SPY and KO/XLP/SPY. Source metadata must separately validate the supported instrument contract; a configured symbol alone does not establish support or proxy suitability. XSD is a context proxy, not an exact SOXL benchmark or leverage replica.

Sources are five-year Yahoo daily charts and published NYSE Group schedules, including the January 9, 2025 closure. Provider-identified NYQ/PCX/ASE USD equities and ETFs are supported subject to complete validated data and calendar coverage. NASDAQ and other markets fail explicitly; proxies are never substituted automatically.

Configuration freezes 756–1000 historical sessions, a five-session horizon, raw USD/share prices, an explicit historical-availability assumption, and a receipt-age limit of at most 24 hours. Default history is 756 sessions. Actual HTTP receipts and registry first-seen timestamps remain separate. News and fundamentals are not collected or used. The evidence remains previously exposed historical development evidence; a new directory or campaign does not make it an independent or hidden test.

## Execute, inspect and replay

~~~sh
python -m quant_rsi forecast-workflow run soxl-context-example --max-fits 1
python -m quant_rsi forecast-workflow status soxl-context-example
python -m quant_rsi forecast-workflow replay soxl-context-example
python -m quant_rsi forecast-campaign status example-context
python -m quant_rsi forecast-campaign replay example-context
~~~

Repeat execution to advance, or use **--max-fits 10** for a child's entire fixed batch. Optional **--config-ref** and **--protocol-ref** flags assert externally recorded references; a mismatch fails before dispatch. The verified parent and child authorities supply the normal references.

Each child has nine evaluation jobs and one separate publication fit. Failed or interrupted dispatched jobs remain charged in its original experiment journal. The campaign derives actual totals from those journals; it does not keep a competing fit ledger. Its upfront allocations bound the aggregate even across interruptions. This guarantee covers registered managed forecast-comparison runs only. Legacy basic forecasts and C2 research have separate allowances and cannot serve as a campaign-funded fallback. A batch may terminate without publication.

B0/B1/B2 reuse the fixed model and common eligible samples. Inspect actual fold and publication row counts; unsupported data may exclude rows. Reports separate observed origin price, target session, predicted price, actual model features, development errors, unchanged-price/training-mean controls, and publication timing. A **false** control comparison means the selected model did not beat that control on the specified error metric. Candidate selection is not predictive validation. Publication does not inherit evaluation scores; no calibrated probability or confidence interval is provided.

## Recovery and preservation

Repeat preparation for the same registered run after interrupted collection or input freezing. Repeat execution after interruption. Completed requests return saved results without collection or fitting, including after data becomes stale. Status and replay do not dispatch jobs.

Changed configuration, missing initialized authority, corrupted receipts or altered inputs fail explicitly. Restore original saved state; deleting ledgers or renaming directories does not renew an allocation. Before new dispatch, frozen receipts must satisfy the age limit and the origin must remain the latest complete registered session. Stale inputs are not silently refreshed. Already-dispatched results can still be recovered and confirmed, with late publication reported honestly.

The managed identity is enforced at both the workflow and shared experiment interfaces. Direct comparison commands cannot omit a managed parent binding to obtain standalone allowance. Campaign initialization does not adopt an existing legacy experiment workspace; keep those completed runs and their original semantics intact.

Keep the complete canonical workspace, including **.quant_rsi/forecast_campaign_identity.json**, **forecast_campaigns/**, **enhanced_forecast_workflows/** and **forecast_experiments/** under **.quant_rsi/**. Preparations include their own first-seen registries. Preserve the frozen source and dependency versions. Restoring only an output JSON is insufficient. For isolated replay, restore the complete workspace at its recorded absolute path, install its recorded runtime, set QUANT_RSI_WORKSPACE and run replay. Relocation or migration requires a separately designed contract.

The existing standalone forecast and forecast-compare interfaces remain available in legacy workspaces. Managed historical forecasting does not grant access to, replace, or reset C5 protected-evidence authority.
