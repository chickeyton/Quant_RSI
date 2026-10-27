# Configurable enhanced forecasting

This workflow implements a bounded delivery slice of [RFC #1](https://github.com/natureofnature/Quant_RSI/issues/1): explicit target, industry proxy and market proxy inputs, shared campaign allowances, collection, freezing, historical comparison, and a separate publication fit. It does not implement feedback-dependent recursive research or establish predictive advantage.

## Install and register one shared campaign

Use Linux, Python 3.12, and the pinned development dependencies. All instruments use the same engine and canonical workspace:

~~~sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
export QUANT_RSI_WORKSPACE="$PWD/runs/enhanced-workspace"
mkdir -p "$QUANT_RSI_WORKSPACE"
quant-rsi forecast-campaign init examples/forecast_campaign.json
~~~

Campaign initialization performs no network access or fitting. It freezes the complete roster, normalized configurations, and parent budget. The example reserves two child ceilings of ten fits and 1,800 worker-seconds each, against a total of twenty fits and 3,600 worker-seconds. These seconds are reserved worker timeout slots, not measured elapsed training time. Unused and failed allocations remain reserved; there is no transfer, rollover or replacement-ID mechanism. Edit a new plan before initialization to choose a different finite roster.

Campaign and run IDs are immutable; run IDs are unique across the workspace. An identical repeated campaign plan returns its existing grant. Changed limits, profiles or roster entries fail. Preparation derives each configuration from that registration and requests **zero fits**. Instrument profile files under **instruments/US/** are organization aids, not separate engines or workspaces.

## Preflight and execute an explicit profile

Run preflight after registration. It checks the supported Python/Linux environment, workspace access, original campaign authority, exact registered profile, remaining allocation, and any saved workflow runtime and data freshness. It neither collects data nor creates training reservations. Before preparation, source access is reported as **not checked**; no extra connectivity probe or fallback source is used.

~~~sh
quant-rsi forecast-workflow preflight soxl-context-example \
  --campaign-id example-context --profile instruments/US/SOXL/forecast.json
quant-rsi forecast-workflow execute soxl-context-example \
  --campaign-id example-context --profile instruments/US/SOXL/forecast.json \
  --max-fits 10 --output runs/enhanced/soxl
quant-rsi forecast-workflow preflight ko-context-example \
  --campaign-id example-context --profile instruments/US/KO/forecast.json
quant-rsi forecast-workflow execute ko-context-example \
  --campaign-id example-context --profile instruments/US/KO/forecast.json \
  --max-fits 10 --output runs/enhanced/ko
~~~

`execute` checks the environment and original registration, prepares that run, then advances its existing comparison by at most `--max-fits` jobs. This argument is required and must be 0–10; it does not increase the cumulative allowance. Zero allows preparation, including network collection, but dispatches no new training. To inspect source access and freeze inputs before choosing a training increment, use `forecast-workflow prepare ID --campaign-id example-context --profile PATH`, followed by `forecast-workflow run ID --max-fits 1`.

The output directory contains `forecast.json` and a standalone `forecast.html`; open the HTML in a browser. Without `--output`, `execute` saves successful results under `$QUANT_RSI_WORKSPACE/reports/ID/`. Repeating the same completed request returns the saved result without collection or fitting. Keep the same campaign, run ID, profile, and workspace when recovering an incomplete request.

## Read the report

~~~sh
quant-rsi forecast-workflow report soxl-context-example --output runs/enhanced/soxl
quant-rsi forecast-workflow report soxl-context-example --output runs/enhanced/soxl --replay
~~~

`report` reads the verified existing workflow and campaign without collecting or training. `--replay` additionally checks the saved computation and runtime. The HTML is a projection of those reports, with no separate model, estimator, or budget ledger.

The report distinguishes the observed origin price, target session, predicted price and return, selected recipe, actual feature names, development sample counts, same-observation naive controls, and publication timing. Failed control comparisons remain visible. Late, abstaining, incomplete, and unavailable outcomes do not become valid forecasts because the command returned successfully. Inspect `status`, `experiment`, and `failure` in JSON. Reserved worker seconds are timeout-slot accounting, not elapsed training time; remaining slots do not guarantee a publication.

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

Workflow errors return structured JSON with `failure.category`, `stage`, retained identity, and verified accounting when available. Errors return exit code 1; a handled keyboard interruption returns 130. With `--output`, the CLI also attempts to save the failed report as JSON and HTML. An unreadable authority means **unknown** usage, never zero.

- For transient network errors, timeouts, or interruption, follow the exact `failure.recovery_command` when present. It preserves the original invocation and workspace; wait for another active controller before retrying.
- For stale inputs, do not recollect into the frozen run. If a dispatched result or completed publication output is pending, the supplied recovery command uses the original comparison and `--max-fits 0` to reconcile it without a new job.
- For exhausted allowances, changed profiles, unsupported inputs, missing or corrupt authority, or a runtime mismatch, no automatic recovery command is supplied. Read the error and restore the original required state where applicable. A new ID is not a retry or a renewed allocation.

Repeat preparation for the same registered run after interrupted collection or input freezing. Repeat execution after interruption. Completed requests return saved results without collection or fitting, including after data becomes stale. Status and replay do not dispatch jobs.

Changed configuration, missing initialized authority, corrupted receipts or altered inputs fail explicitly. Restore original saved state; deleting ledgers or renaming directories does not renew an allocation. Before new dispatch, frozen receipts must satisfy the age limit and the origin must remain the latest complete registered session. Stale inputs are not silently refreshed. Already-dispatched results can still be recovered and confirmed, with late publication reported honestly.

The managed identity is enforced at both the workflow and shared experiment interfaces. Direct comparison commands cannot omit a managed parent binding to obtain standalone allowance. Campaign initialization does not adopt an existing legacy experiment workspace; keep those completed runs and their original semantics intact.

Keep the complete canonical workspace, including **.quant_rsi/forecast_campaign_identity.json**, **forecast_campaigns/**, **enhanced_forecast_workflows/** and **forecast_experiments/** under **.quant_rsi/**. Preparations include their own first-seen registries. Preserve the frozen source and dependency versions. Restoring only an output JSON is insufficient. For isolated replay, restore the complete workspace at its recorded absolute path, install its recorded runtime, set QUANT_RSI_WORKSPACE and run replay. Relocation or migration requires a separately designed contract.

The existing standalone forecast and forecast-compare interfaces remain available in legacy workspaces. Managed historical forecasting does not grant access to, replace, or reset C5 protected-evidence authority.

## Validation and evidence

The offline CLI tests cover both explicit profiles, read-only preflight, JSON/HTML generation, network failure, interrupted-job recovery, stale completed-publication recovery, stale inputs, authority loss, and completed duplicate requests. They use deterministic fixtures and a dummy worker; they do not establish live-source availability or predictive quality.

~~~sh
pytest -q tests/test_forecast_usability.py tests/test_forecast_report.py
~~~

The [retained two-instrument demonstration](enhanced_forecast_results.md) separately records the earlier live-source run and its original runtime. Neither selected model beat both naive controls on MAE. Its saved report can be displayed as historical evidence; it is not a fresh acceptance run of this CLI.
