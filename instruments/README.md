# Instrument Profiles and Local Outputs

All instruments share the engine in `src/quant_rsi/`. Keep instrument-specific configuration here; do not copy source modules into instrument directories.

## Layered Factor Specifications

Shared stock definitions live in [`shared/equities.json`](shared/equities.json).
Each stock selects them and declares its own sources in `<MARKET>/<SYMBOL>/factors.json`:
see [CALM](US/CALM/factors.json) and [KO](US/KO/factors.json). CALM's egg-price and
disease-event definitions stay inside CALM; they are not stock-wide defaults.

Run `quant-rsi factors validate US CALM` to resolve and inspect the configuration.
This is read-only specification validation, not data collection or forecasting.
The [factor-layer guide](../docs/factor_layers.md) explains numeric/text types,
ownership, examples and the remaining execution boundary.

## Tracked Profiles

Profiles follow `instruments/<MARKET>/<SYMBOL>/forecast.json`. The current US examples explicitly bind SOXL/XSD/SPY and KO/XLP/SPY as target/industry/market roles. They also declare the supported source, calendar, history, horizon, price units, and availability assumption.

To configure another supported instrument, copy a profile into its symbol directory and edit the explicit roles. Validate it with `forecast-workflow preflight` against a registered campaign. The profile must match the frozen campaign configuration; changing a JSON file does not modify an existing run. Use a new registered workflow identity when inputs change. See the [enhanced workflow guide](../docs/enhanced_forecast_workflow.md) for registration and supported contracts.

The [CALM factor tree](US/CALM/factor_tree.json) is a specification-only factor-discovery ledger, not a runnable forecast profile or development plan. See its [worked example](../docs/calm_factor_discovery_example.md) for the node meanings and unresolved evidence.

## Ignored Data and Reports

Use these destinations for new local exports and reports:

| Path | Contents |
| --- | --- |
| `data/US/<SYMBOL>/<collection-id>/` | Explicitly exported inputs and source evidence |
| `runs/US/<SYMBOL>/<workflow-id>/` | Human-readable reports and other requested output copies |
| `$QUANT_RSI_WORKSPACE/.quant_rsi/` | Shared authoritative registry, campaigns, journals, and bound workflow artifacts |

The first two paths are organization conventions, not automatic storage routing. Select destinations through existing output arguments when available. The guided workflow retains its authoritative collection and execution state in the shared workspace, even when reports are exported elsewhere. All three locations are ignored by Git.

For the campaign registered in the guide:

```bash
export QUANT_RSI_WORKSPACE="$PWD"
quant-rsi forecast-workflow preflight soxl-context-example \
  --campaign-id example-context --profile instruments/US/SOXL/forecast.json
quant-rsi forecast-workflow execute soxl-context-example \
  --campaign-id example-context --profile instruments/US/SOXL/forecast.json \
  --max-fits 10 --output runs/US/SOXL/soxl-context-example
```

Use globally unique workflow and collection IDs across instruments. A report destination does not isolate authority, reset charges, or renew a budget. Keep historical data, workspaces, output paths, and their recorded source environments intact.
