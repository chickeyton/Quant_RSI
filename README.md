# Quant RSI

**Run one stock research study: context, factor proposals, models, strategy comparison and a report.**

After [installation](#installation-and-usage), run from the repository root:

```bash
quant-rsi research-study KO --output /tmp/ko-study
```

This connects configured company relationships, issuer news, daily prices, two
Codex hypothesis calls and two real LightGBM fits. Open `/tmp/ko-study/report.html`
for prediction baselines, initial/revised strategies versus buy-and-hold, costs,
risk and the latest forecast. Use a new output path for each study.

To try the same report offline, with real model fits on synthetic inputs:

```bash
quant-rsi research-study KO --output /tmp/ko-study-offline \
  --llm simulated --related simulated --news simulated --prices synthetic
```

The [connected study guide](docs/connected_research_study.md) explains prerequisites,
options, artifacts and interpretation. Feedback revises bounded numeric recipes;
both attempts use the same development observations, so a better second result
is not independent evidence of improvement. The latest forecast has no observed
outcome. RSI means **recursive self-improvement** here; testing whether research
feedback improves independent results remains the longer-term goal.

The smaller `research-demo` entry point retains its existing defaults and
[replaceable-service guide](docs/simulated_research_loop.md). The separate bounded
engine retains its budget accounting, durable evidence and recovery capabilities.

[Quick start](#installation-and-usage) · [Capabilities](#current-capabilities) ·
[Guides](#documentation) · [Roadmap](#roadmap) · [Contributing](#development-and-validation)

## Architecture

Shared definitions live outside individual stocks. Each stock selects shared
templates and owns its source bindings, local hypotheses and factor relationships.

```mermaid
flowchart TD
    A["Common numeric / text factor contracts"] --> B["Shared stock factor templates"]
    A --> C["Instrument-specific definitions and bindings"]
    B --> C
    C --> D["Configuration validation and resolution"]
    D -->|"bounded numeric / event expression path"| E["Shared experiment engine"]
    P["Existing fixed recipes and research plans"] --> E
    M["Market data, calendars and availability rules"] --> E
    E --> F["Training and evaluation"]
    F --> G["Forecasts, reports and observation"]
    E --- H["Budget ledger, evidence and checkpoints"]
```

Validation alone does not make a definition a model input. The opt-in
[factor discovery path](docs/factor_discovery.md) executes selected numeric bindings
and registered event indicators; general text tasks and factor trees remain specifications.
Existing forecasts retain their registered recipes.
CALM's egg-price relationships belong to its own configuration; KO can reuse the
same stock-price template without inheriting CALM's business assumptions.
See [factor ownership and types](docs/factor_layers.md).

| Location | Responsibility |
| --- | --- |
| `src/quant_rsi/` | Shared data, factor contracts, research, forecast, evaluation and execution code |
| `instruments/shared/` | Reusable stock factor templates |
| `instruments/<MARKET>/<SYMBOL>/` | Instrument profiles, factor bindings and local relationships |
| [`experimental/candidate_factors/`](experimental/candidate_factors/README.md) | Source-pinned open-source factor survey, licenses and integration examples; not runtime registrations |
| `examples/` and `tests/` | Runnable plans, small fixtures and offline regression tests |
| `docs/` | Operating guides, design specifications and experiment reports |
| `data/`, `runs/`, `.quant_rsi/` | Ignored local data, outputs and authoritative workspace state |

## Research workflow

The connected study follows one visible path:

```mermaid
flowchart TD
    A["Stock + relationship/news/price context"] --> B["LLM proposes bounded numeric recipes"]
    B --> C["LightGBM fit + development forecasts"]
    C --> D["Prediction baselines + costed strategy / buy-and-hold"]
    D --> E["Feedback to LLM: revise recipes"]
    E --> F["Second fit on the same training and development windows"]
    F --> G["HTML report + full trace + unobserved latest forecast"]
```

The study defaults to 20 matured five-session forecasts, with training labels
strictly before the first evaluation origin. The revision sees those results,
so its comparison remains development analysis. Current news informs research
choices and is not backfilled as historical features. Strategy orders execute at
the next open in a hypothetical account; this is not live order submission.

For registered inputs and real LightGBM workers, use the separate
[bounded research workflow](docs/research_workflow.md). It connects discovery,
per-fold strategy/buy-and-hold reports and linked forecast/paper observation.
Its development forecast MAE drives discovery; account returns do not select
candidates. These capabilities can be connected to the product loop one step
at a time.

## Current capabilities

| Capability | Status and boundary |
| --- | --- |
| Connected research study | One command produces a trace and HTML/Markdown report with bounded recipe proposals, two model fits, prediction baselines and costed strategy comparisons; development feedback is reused |
| Simulated product loop | One stock-symbol command shows context, catalog-based factor selection, prediction, feedback, revised selection and a second prediction; each simulated service can be replaced independently |
| Live LLM hypotheses | Optional Codex CLI or OpenAI Responses API makes two independent calls with explicit feedback; current news is research context, not historical training data |
| Company relationships | Optional KO profile lists source-backed CCEP bottling and MNST distribution links; configured relationships, not automatic discovery or statistical correlations |
| Live news in the product loop | Routes target and configured related issuers to registered RSS feeds; KO is available, CCEP/MNST are explicitly `not_configured`; no general search or article fetching |
| LightGBM in the product loop | Numeric feature calculation and two fits on synthetic or collected daily prices; feedback informs a development reestimate at the same origin, with no independent improvement claim |
| Daily prices in the product loop | Collect once or reuse verified daily evidence; report development errors separately from the latest-bar five-session forecast, whose outcome is unobserved |
| Bounded research workflow | One profile runs bounded discovery, costed per-fold strategy/buy-and-hold reports and linked forecast/paper observation; feedback uses development forecast MAE only |
| Five-session price forecasting | Available for explicitly supported sources/markets; includes collection, fixed-model research, model sealing and replay |
| Prospective forecast observation | Offline registration and append-only outcome settlement from observed market receipts; real prospective accumulation and outcome-driven policy updates remain outstanding |
| Enhanced forecast comparison | Available: three predefined price/context recipes, frozen folds, naive controls and a separate publication fit |
| Bounded factor research | Available: momentum/mean-return neighbor search and limited, registered research-program changes |
| Shared and instrument-specific factors | Numeric bindings can enter bounded discovery; general text definitions remain specifications |
| Automatic factor discovery | Finite multi-leaf compositions, feedback-dependent exploration and common multi-fold comparisons; no general proposer-program evolution |
| Candidate factor execution | Opt-in [Qlib ROC and MA close-price adapters](docs/candidate_factor_runtime.md) feed bounded discovery; the remaining survey candidates are not executable |
| Dynamic event factors | Registered title instances, observed lifecycle transitions, decay and archival; KO RSS observations can enter this path |
| Multi-branch factor discovery | CALM is a worked specification; its factor tree is not executable |
| News and financial evidence | Retained KO title indicators can enter opt-in comparisons/discovery; other evidence remains sidecar-only |
| News collection and extraction | [Bounded KO official RSS observations](docs/news_collection.md), retained-title extraction and revision tracking; provider-side search/full-text extraction remain open in [RFC #28](https://github.com/natureofnature/Quant_RSI/issues/28) |
| Statistical confirmation and promotion | Incomplete; protected evaluation is engineering-only and retains the incumbent |
| Intraday bars, model and IBKR paper trading | Standalone, offline-tested: IBKR adapter with a paper-only guard, intraday snapshots, fixed LightGBM development evaluation, replay simulator and paper-account episodes. Not yet wired into the C2/C4/C5 controller or `quant-rsi`; see the [intraday guide](docs/intraday_ibkr.md) |
| Live trading and general self-modification | Not implemented |

Older guides use project-local labels: **M0** is fixed-model execution, **C2** is
the factor research loop, **C4** is a research-program paired experiment, and
**C5** is protected evaluation. These are not standard quant/AI terms.

## Installation and Usage

Use **Linux and Python 3.12**. Process control uses POSIX locks, signals and resource
limits; Windows is not supported. Dependency installation needs network access.
Run from a fresh clone:

```bash
git clone https://github.com/natureofnature/Quant_RSI.git
cd Quant_RSI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
```

Run a connected study using your existing Codex CLI login and public data:

```bash
quant-rsi research-study KO --output /tmp/ko-study
```

For an offline study with the same model/report path:

```bash
quant-rsi research-study KO --output /tmp/ko-study-offline \
  --llm simulated --related simulated --news simulated --prices synthetic
```

See the [study guide](docs/connected_research_study.md) for cached prices, API mode,
account assumptions and report files. The output directory must be new and outside
the checkout. Failures retain completed stage traces without silently substituting
simulated services. To inspect the smaller original interaction, run
`quant-rsi research-demo KO`; its flags and independent worktree setup remain in
[the demo guide](docs/simulated_research_loop.md).

When ready to run real model workers, the bounded engine has a separate example:

```bash
export QUANT_RSI_WORKSPACE="$PWD"
mkdir -p runs/workflow-demo
python examples/prepare_research_workflow.py --fixture --output runs/workflow-demo
quant-rsi research-workflow run runs/workflow-demo/profile.json --max-fits 10 \
  > runs/workflow-demo/run.json
quant-rsi research-workflow report rfc39-demo > runs/workflow-demo/report.md
quant-rsi research-workflow replay rfc39-demo > runs/workflow-demo/replay.json
```

The default profile has two research rounds, two folds and a ten-fit allowance.
It reports strategy costs, risk and buy-and-hold comparisons on separate fold
accounts. Its invented historical prices establish no predictive or prospective
advantage. See the [workflow guide](docs/research_workflow.md) for existing inputs,
bounded resume, detailed status and later observations.

For the smaller synthetic kernel demonstration:

```bash
quant-rsi run examples/development.json --output runs/demo
quant-rsi status runs/demo
quant-rsi replay runs/demo
quant-rsi resume runs/demo
```

Expect two completed candidates and one deliberately rejected future-shift
candidate. `run` needs a new directory; use `resume` for an initialized experiment.
Commands return JSON: exit code zero can include candidate failures or budget
exhaustion, so inspect `execution_status` and candidate statuses.

Inspect the layered factor specifications without collecting data or training:

```bash
quant-rsi factors validate US CALM
quant-rsi factors validate US KO
```

These commands report `specification_only` and `runnable: false`. CALM validation
does not enable CALM forecasting; its exchange is outside the current adapter.
For market-data forecasts, follow the [baseline guide](docs/forecast_baseline.md)
or the [enhanced workflow guide](docs/enhanced_forecast_workflow.md). They cover
source access, supported contracts, explicit budgets and recovery.

## Results and limits

The recorded SOXL and KO enhanced experiments completed their engineering
workflows. The selected models did **not beat both naive controls on development
MAE**. See the [experiment report](docs/enhanced_forecast_results.md) for values,
sample sizes and qualifications. These are development results, not independent
confirmation of predictive advantage or trading profitability.

Historical source availability includes explicit assumptions. Preserve original
evidence and distinguish publication time from actual first-seen time. Failed
trials and unresolved costs remain recorded; replay does not create new evidence.
The worker is a process boundary for trusted computation, not a sandbox for
arbitrary generated code. Detailed contracts are in the
[workflow and recovery guide](docs/experiment_workflows.md).

## Roadmap

1. Use the optional live LLM hypothesis step to inspect selection and feedback
   behavior while keeping the rest of the loop runnable with simulated services.
2. Use issuer RSS and LightGBM on synthetic or collected daily prices, then replace remaining
   simulated data and services individually, inspecting each change.
3. Compare the resulting research methods under frozen tasks and budgets, using
   the native engine alongside compatible mature engines; measure useful results
   and the effort required to test a hypothesis.

Track the [architecture RFC #24](https://github.com/natureofnature/Quant_RSI/issues/24),
[factor-tree issue #8](https://github.com/natureofnature/Quant_RSI/issues/8),
[news RFC #28](https://github.com/natureofnature/Quant_RSI/issues/28),
[factor discovery RFC #31](https://github.com/natureofnature/Quant_RSI/issues/31),
[minimal-loop RFC #39](https://github.com/natureofnature/Quant_RSI/issues/39) and
[broader roadmap #1](https://github.com/natureofnature/Quant_RSI/issues/1).
The [architecture specification](docs/architecture.md) describes the target design;
it is not a declaration that every capability is implemented.

## Documentation

| Start here when you want to… | Guide |
| --- | --- |
| Run a complete study and inspect prediction/strategy reports | [Connected research study](docs/connected_research_study.md) |
| Run the stock-symbol product loop and replace simulated services | [Simulated research loop](docs/simulated_research_loop.md) |
| Run one profile through research, strategy comparison and observation | [Minimal research workflow](docs/research_workflow.md) |
| Run detailed commands or recover an experiment | [Experiment workflows and recovery](docs/experiment_workflows.md) |
| Run bounded discovery and manage event instances | [Factor discovery](docs/factor_discovery.md) |
| Understand shared factors and single-stock bindings | [Factor layers](docs/factor_layers.md) |
| Use the first source-pinned candidate adapters | [Candidate factor runtime](docs/candidate_factor_runtime.md) |
| Compare native price formulas with actual pinned Qlib operators | [Optional Qlib operator comparison](docs/qlib_operator_comparison.md) |
| Inspect a company-specific factor tree | [CALM worked example](docs/calm_factor_discovery_example.md) |
| Run a five-session baseline or enhanced comparison | [Baseline](docs/forecast_baseline.md), [enhanced workflow](docs/enhanced_forecast_workflow.md) |
| Run bounded recursive market research | [Recursive market exploration](docs/recursive_market_exploration.md) |
| Import local data or observe a sealed model | [Local data and observation](docs/local_observer.md) |
| Register a published forecast and settle its observed outcome | [Forecast observations](docs/forecast_observations.md) |
| Collect IBKR intraday bars or run an IBKR paper episode | [Intraday and IBKR guide](docs/intraday_ibkr.md), [design](docs/intraday_ibkr_design.md) |
| Understand protected engineering evaluation | [Evaluation protocol](docs/c5_protocol.md) |
| Preserve versions and old experiment environments | [Version control](docs/version_control.md) |

The reference [R&D-Agent-Quant paper](2505.15155v2.pdf) informs the research
direction; its published results are not evidence for this implementation.

## Development and Validation

From the activated environment:

```bash
ruff check src tests examples/profile_validation.py
ruff format --check src tests examples/profile_validation.py
pytest -q
```

[CI](.github/workflows/ci.yml) uses pinned dependencies and offline fixtures,
including real LightGBM worker tests. It checks engineering behavior, not market
predictive quality. Start with tests relevant to a change, then run the required
broader checks; no coverage-percentage threshold is configured. For reproducible
timings and bounded logs, use `python examples/profile_validation.py --output
/tmp/quant-rsi-validation` with a fresh directory. Optional `--shards 2` keeps each
module's shared fixtures in one process; CI uses this two-process mode.
[Validation cost and test tiers](docs/validation_cost.md)
describes focused feedback and final-revision full-suite evidence. CI preserves
these profiles as artifacts for 14 days.

### Contributor Delivery

Use four-space Python indentation, standard naming conventions and Ruff formatting
(100-character lines). Keep dependency pins, commands and documentation in sync.
Submit changes through a temporary branch and a PR targeting `main`; include the
behavior change, related issues, actual validation results and checks not run.
Merge after required checks and review, then remove the temporary branch. Public
documentation, issues and PRs are written in English. See [AGENTS.md](AGENTS.md).

Keep credentials, datasets and verbose logs out of version control. Preserve
completed experiments and their recorded source/dependency environments; do not
rewrite manifests to make historical runs match new code.
