# Quant RSI

See [Development trajectory](docs/development_trajectory.md) for entry points, budgets, and recovery of bounded development runs.

A quantitative research system based on the [agreed architecture](docs/architecture.md). It currently provides an **M0 fixed-model kernel, C2 dynamic research loop, C4 bounded research-package experiments, and C5 protected engineering evaluation** on synthetic data, plus [local data, the same C2 research path, standalone inference, and persistent baseline observation](docs/local_observer.md). Programs remain fixed within each run. A [five-session price forecast baseline](docs/forecast_baseline.md) is also available; calibrated statistical promotion and full recursive self-improvement remain unfinished.

## Installation and Usage

Validated on Linux with Python 3.12. Process control relies on POSIX `flock`, signals, and resource limits; Windows is not supported. Installing dependencies requires network access. Synthetic experiments need no external services. Automatic data collection for five-session forecasts requires access to public market data and exchange calendars, without an API key.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .

quant-rsi run examples/development.json --output runs/demo
quant-rsi status runs/demo
quant-rsi replay runs/demo
quant-rsi resume runs/demo
```

`run` requires a new output directory. Once an experiment is initialized, use `resume`; `run` cannot overwrite it. The example contains two valid candidates, `baseline_return_1` and `momentum_5`, and an `invalid_future_shift` candidate that should be rejected. The expected result is two `completed` candidates and one `rejected` candidate; that rejection is part of the intended validation.

Commands return JSON. `summary.json` stores the summary produced by a run or resume, while `status` reads the current state from the ledger. The CLI returns 0 when it successfully handles a request, including candidate failures or budget exhaustion; callers must inspect `execution_status` and each candidate's `status`. Configuration errors, validation failures, and run-lock conflicts return a nonzero exit code.

## Five-Session Price Forecasts

~~~bash
export QUANT_RSI_WORKSPACE="$PWD"
quant-rsi forecast run SOXL --request-id soxl-first --output runs/soxl-first
quant-rsi forecast status soxl-first
quant-rsi forecast replay soxl-first
~~~

One command collects daily bars, preserves source evidence, runs fixed-model research, seals the model, and persists a forecast. Reusing a request ID and output directory reuses the original inputs and result; use a new ID and directory for a new forecast. This is an engineering baseline; practical forecasting quality has not been established. See the [forecast baseline guide](docs/forecast_baseline.md) for supported markets, data assumptions, output interpretation, and recovery.

A [paired forecast comparison](docs/forecast_comparison.md) adds longer price history and industry/market context, three frozen evaluation folds, and a separate publication fit. Its B0/B1/B2 models use price features; collected news and financial statements remain sidecar evidence. Each experiment has a cumulative limit of nine evaluation fits plus one publication fit. The [first completed batch](docs/forecast_context_results.md) passed engineering validation but did not outperform the naive price baselines.

## Synthetic Kernel Conventions

- Data is a complete price panel generated with a fixed random seed. It uses weekdays and synthetic UTC timestamps, not an exchange calendar, corporate actions, or actual tradable instruments.
- `train_cutoff` is the zero-based training cutoff index. The label at close `t` is `open[t+2] / open[t+1] - 1`; only labels that have matured by the cutoff are eligible for training. LightGBM parameters are fixed, with one fit per candidate and no refitting during development.
- The bounded factor language supports only `momentum` and `mean_return`, with windows of 1–30 sessions. Features comprise the factor, the one-session return, and the current session's open-to-close return. Candidate-supplied Python code is not executed.
- Whole-share targets and orders are formed at close `t` and filled at open `t+1`. Orders use only the cash, holdings, closing prices, and predictions known at the decision time. A subsequent opening-price gap can affect the quantity filled.
- The portfolio targets equal weights across a long-only top-k selection. Sells precede buys, and orders on each side execute in instrument order. Insufficient cash produces partial fills. Fees and slippage are charged per side in basis points. Positions are liquidated after the final label interval, with exit fees and slippage deducted.
- The primary metric is **arithmetic mean net active return × annualization**, relative to a zero-interest cash benchmark; it is not CAGR. Maximum drawdown uses portfolio equity at each interval's ending open. Results that breach the limit are retained but excluded from development incumbent selection.
- `development_incumbent` is only the best feasible candidate in the current development comparison. All results are marked `development_only`, `capacity_status=unverified`, and `promotion_decision=not_requested`.

## Experiment Ledger and Recovery

```text
runs/demo/
  manifest.json       # Content hashes for the plan, data, and runtime
  journal.sqlite      # Candidate states, budget reservations, transactional events
  artifacts/<sha256>  # Data, model text, predictions, accounting, logs, and other artifacts
  work/0000/          # Worker input, raw logs, and atomically committed output
  summary.json        # Readable summary
```

Each candidate reserves one attempt before execution; invalid factors also count. Before a valid candidate starts, a full `worker_timeout_seconds` slot is charged against `max_worker_seconds`. Completion, failure, and interruption do not refund it. This is a conservative worker-slot budget. Actual elapsed time is reported separately as `elapsed_seconds`, or `null` when recovery cannot determine it.

Creation first writes `initialization.json` atomically, fixing the plan and runtime. Ledger tables, identity, and the `run_created` event are committed in one SQLite transaction. The initialization marker is then removed before any candidate can start. After interrupted creation, `resume` or `run` with the same plan can finish initialization; a different plan is rejected. If interruption occurred before the first record was published, retry `run` on the empty directory, which may contain a lock or temporary files. An initialized experiment with a missing ledger, conflicting identity, or damaged structure is never rebuilt automatically.

Worker input contains one fully hashed computation binding. Plans and data are loaded by content hash, and the factor must match the candidate in the bound plan. The worker verifies the binding before computation. Before accepting a result, the parent checks the persisted input and returned identity so a mismatched payload cannot reuse the original proposal's identity for scoring or replay.

Only one controller may own an experiment. The worker inherits the run lock and has its own deadline signal. If the parent is interrupted, recovery is rejected while that worker remains alive; retry `resume` later. Complete, correctly bound worker output is reused for verification and accounting. Without valid output, the attempt is recorded as `interrupted`. Terminal candidates are never rerun implicitly.

`replay` verifies content hashes, computation bindings, and runtime identity. It recomputes all orders, fills, and net accounting from saved predictions and requires an exact match with the stored report; it neither retrains nor creates new evidence. `resume` and `replay` reject changes to source code or installed package versions; restore the recorded environment before running them. `status` can still read old results. Experiment directories may be moved as a whole, but their internal files must not be edited. Hashes detect corruption and inconsistency; they are not signatures protecting against someone with write access.

## Development and Validation

```bash
pytest -q
ruff check src tests
ruff format --check src tests
```

Tests cover end-to-end execution with real LightGBM workers, independently calculated accounting, future-data perturbations, complete prediction coverage, budget exhaustion, invalid output, lock exclusion, artifact corruption, and crash recovery. No coverage-percentage threshold is configured.

Directory and module responsibilities:

| Path | Purpose |
| --- | --- |
| `src/quant_rsi/schemas.py` | Strict DevelopmentPlan schemas and bounded factor definitions |
| `src/quant_rsi/data.py`, `model.py`, `worker.py` | Synthetic data, causal features, fixed training, and worker entry point |
| `src/quant_rsi/accounting.py` | Independent accounting and primary-metric calculation used by the controller |
| `src/quant_rsi/controller.py`, `journal.py`, `artifacts.py` | Scheduling, lifecycle, budgets, and artifact verification |
| `market_contracts.py`, `market_data.py`, `asof.py`, `local_research.py` (under `src/quant_rsi/`) | Explicit local contracts, immutable receipts, causal views, and local inputs for the same C2 kernel |
| `baseline.py`, `execution_account.py`, `observer*.py` (under `src/quant_rsi/`) | Model sealing, standalone inference, shared accounting, and finite-step observers |
| `tests/` | pytest tests named `test_*.py` |
| `examples/development.json` | Runnable development plan |
| `docs/`, `discussion_boards/` | Agreed designs, review discussions, and historical records |

Use four-space indentation in Python, `snake_case` for functions and modules, and `PascalCase` for classes. Ruff handles formatting and static checks with a line length of 100. Dependencies are pinned in `requirements-dev.lock`; update and validate the lock file when changing dependencies. Use concise imperative commit subjects. Pull requests should explain the purpose, behavior changes, and validation results. This README is the reference for current commands; the existing `AGENTS.md` is preserved as requested.

## C2: Finite-Step Dynamic Research

```bash
quant-rsi research run examples/research_development.json --output runs/research_demo --steps 1
quant-rsi research status runs/research_demo
quant-rsi research resume runs/research_demo --steps 3
quant-rsi research replay runs/research_demo
```

For each run or resume, `--steps` is an integer from 1 to 100, defaulting to 1. It limits the number of newly committed observation/checkpoint pairs in that invocation. Finishing a pending step without a committed observation consumes one step; previously committed steps are not counted again. When an invocation returns `paused`/`step_limit` at a commit boundary, no background controller remains and no further proposals or dispatches occur. Only an explicit `resume` continues the run. This finite-step interface is the stopping mechanism; there is no asynchronous stop or cancel command. Budget exhaustion or a research stopping condition returns `completed` with a specific `stop_reason`. Resuming a terminal run does not start new research.

After a process interruption, `running` in `status` means state remains unfinished; it does not prove that a process is still alive. Use `research resume` to recover the original step. A surviving worker's inherited lock and deadline prevent concurrent recovery. Committed proposals and results are reused. Unpublished deterministic proposal computation may only be repeated for the same request and parent state, without reserving another attempt. Missing valid worker output preserves the actual `interrupted`/`failed` outcome and original charges. Recovering an old step, continuing with a new valid step, and read-only replay are distinct operations.

The frozen `neighbor_search_v1` policy starts from the baseline. It checks the current incumbent's same-operator window +1, window −1, then the other operator at the same window, skipping out-of-range, underpopulated, or previously visited neighbors. A feasible candidate becomes the development incumbent and resets no-progress only when its primary metric improves by strictly more than the finite, nonnegative `min_improvement`; ties do not move the incumbent. A valid baseline initializes the incumbent with no-progress = 0; otherwise research stops. Subsequent failures, duplicates, ineligible candidates, and non-improvements each increment no-progress once. Research stops when that limit is reached, the registered neighborhood is exhausted, or the remaining budget is insufficient.

For example, after trying `momentum(6)`, an improvement may lead to `momentum(7)`, while no improvement may lead back to `momentum(4)`. This updates memory and search state under a fixed program; it does not rewrite the researcher package. Each candidate receives a newly sealed singleton DevelopmentPlan, leaving the template and earlier plans unchanged. Models, data, and accounting still use the original kernel. Only committed development observations visible to the current task may influence proposals. Gains from repeated development selection are not independent validation evidence.

One SQLite ledger covers the entire campaign. One attempt is reserved before each proposer call, including the baseline. `attempts_reserved = kernel_attempts + unmapped_proposal_reservations`; reservations for unmaterialized candidates or generation failures remain recorded. Each actual dispatch charges a full timeout slot, with no refunds for a crash before launch, failure, or interruption. Dispatch is refused if less than one complete slot remains. Checkpoints cannot overwrite budget allowances, and resuming does not reset them. No LLM is connected, so no fabricated token or billing costs are reported.

Program code and configuration are fixed as a ResearcherPackage. MemorySnapshot retains observations; ResearchStateCheckpoint records the parent state, incumbent, visited candidates, no-progress count, continuation, and ledger identity. Requests, actual responses, decisions, experiment manifests, and provenance are content-addressed artifacts. Original proposal responses are also saved under `proposals/`. Candidate indices and the kernel plan's `proposal_index` are identified separately to avoid confusing singleton-plan indices with global attempt IDs.

Cache acceptance and semantic replay verify expected proposals against the frozen pure policy. A cached candidate that is valid but differs from the request's expected candidate is explicitly rejected. Its response artifact and reservation remain, and no substitute candidate is trained. An actual proposer failure retains a missing score. Applying an observation, comparing and swapping the parent checkpoint, and recording events happen in one transaction to prevent learning from an observation twice.

Reports include data, package, and checkpoint references; input evidence; hypotheses and failure reasons; development deltas against the fixed baseline and previous incumbent; selection results; remaining budgets; and stopping reasons. Per-attempt `elapsed_seconds` is measured, or `null` if recovery cannot determine it; it is distinct from the logical worker-slot allowance. `status` and `replay` do not publish research state. Replay additionally recomputes kernel accounting and research-state transitions without creating requests.

The added modules are `researcher.py` for the frozen policy, `research.py` for campaign orchestration and reporting, and `research_journal.py` for reservations and observation transactions in the same database. Regression tests are in `tests/test_research.py`. Keep source and dependency snapshots of accepted versions (M0 is under `runs/accepted_source/m0_f54aa55ddba7`) to replay runs bound to those versions. Never overwrite old run artifacts with a newer implementation.

## C4: Bounded Research-Package Experiments

First generate replayable base-package evidence with the current source, then explicitly start one patch experiment:

```bash
quant-rsi research run examples/research_development.json --output runs/c4_evidence --steps 100
quant-rsi package run --evidence runs/c4_evidence --output runs/c4_trial --state-policy reset --steps 1
quant-rsi package status runs/c4_trial
quant-rsi package resume runs/c4_trial --steps 1
quant-rsi package resume runs/c4_trial --steps 100
quant-rsi package replay runs/c4_trial
```

A package campaign contains one change request and at most one parent/candidate research pair. ChangeProposer permanently reserves that request before generation. The source run must use the current implementation's `window_first` base package and have no pending observations. The entry point verifies the complete committed chain and kernel accounting, then seals a read-only evidence copy. Older C2 evidence cannot bypass version checks; replay it with its original snapshot or create new research under the current engine.

The frozen rule scans committed observations for the most recent eligible group sharing the same actual incumbent: a neighboring window with the same operator, and the other operator at the same window. If the alternate operator's score is strictly higher than the group's last window candidate, it proposes an `operator_first` patch. Otherwise it returns an explicit `no_patch`, including when evidence is insufficient. This is a development hypothesis; reusing historical evidence is not independent validation. The actual response is saved separately. Validation covers the exact base, source and watermark, interface, reset policy, sole allowed field and value, and the frozen program's complete expected output. Invalid or mismatched cached responses retain their original text, rejection reason, and charged request; no replacement response is sampled.

The sole intervention is `policy.neighbor_order`: `window_first` checks window +1, window −1, then the alternate operator; `operator_first` checks the alternate operator, window +1, then window −1. The search space, model, data, costs, risk rules, metric, and allowances stay fixed. The candidate package binds its parent/patch/proposer/request lineage. Parent and candidate runs bind their respective packages and restart from the same baseline with empty memory. State migration, checkpoints from another package, and independently resuming a child arm through `research resume` are rejected; use `package resume` for coordinated recovery.

C2 evidence from local CSV data can use the same `package run` entry point. Both arms read the same LocalResearchData and account rules from the evidence copy, without rereading the original CSV or latest registry. Launch checks verify the actual C4 grant, locks, package, inputs, and persisted round allowances. An eligible descendant model can be sealed with `baseline seal runs/c4_trial/arms/candidate` and then used with `infer`. Without evidence satisfying the actual pairing conditions, the result remains `no_patch`. Local C5 uses separately registered tasks, restricted development views, and the original account rules; see [Local C5 engineering comparison](docs/local_c5.md). See also the [local C4 operating guide](docs/local_observer.md#6-本地-c4-开发比较).

Before initializing either child arm, the outer controller atomically allocates two complete grants: total research-attempt and worker-slot limits are each twice the per-arm plan. Grants bind a directory, ID, plan, and package and are never reclaimed. Each arm charges its actual consumption to its original ledger. `research_budget` reports limits, granted, consumed, outstanding, and unallocated amounts separately, maintaining:

- `limit = granted + unallocated`;
- `granted = consumed + outstanding`.

Historical source-evidence consumption and patch requests are reported separately. Worker slots are conservative logical allowances, not measured elapsed time. Unstarted or unresolved grants are not released, and failures are not represented as zero returns.

`--steps` limits newly committed observations **per arm** for the current invocation. Each round's target counts are sealed first. After a crash, resume the unfinished round with its original `--steps`; committed observations are not repeated. If the previous round was fully committed, another resume starts a new finite continuation round. Child-arm initialization receipts are persisted before execution. Initialized ledgers that are missing or corrupted are never rebuilt. Workers inherit both outer and arm locks, so recovery cannot bypass an orphan worker that still holds them.

`status` and `replay` only inspect state; they do not commit missing outer receipts or manufacture artifacts. `committed_completion_ref` identifies the committed outer result. If both arms have finished but this field is `null`, an explicit resume is required to finalize the campaign. Resuming a terminal campaign creates no requests, experiments, rounds, or events. There is no `promote` command. `accepted_pointer` is a read-only engineering baseline reference for this campaign, not a C5 promotion registry; candidate development scores cannot update it.

The C4 engine generates a new base-package identity and records its engineering ancestry relative to accepted C2; it does not attach an old digest to new source code. The original C2 source is preserved under `runs/accepted_source/c2_16b85204c890/`. New modules are `changes.py`, `package_journal.py`, and `package_experiment.py`. Mechanism, negative-case, initialization/commit-window, and orphan-lock tests are in `tests/test_packages.py`. In an executed synthetic example, the parent and candidate completed 6 and 8 experiments respectively, with a final development delta of 0; this does not demonstrate improved returns.


## C5: Protected Engineering Evaluation

See the [C5 execution and recovery guide](docs/c5_protocol.md) and [local C5 input and accounting boundaries](docs/local_c5.md). Set `QUANT_RSI_WORKSPACE` explicitly to an absolute path. Its single authority records evidence exposure, development grants, scoring slots, reports, and the registry. Changing output directories or labels does not renew allowances.

```bash
export QUANT_RSI_WORKSPACE="$PWD"
quant-rsi research run examples/research_development.json --output runs/c5_evidence --steps 100
quant-rsi package run --evidence runs/c5_evidence --output runs/c5_packages --state-policy reset --steps 100
quant-rsi evidence init examples/evidence_program.json
quant-rsi compare register examples/comparison.json --output runs/c5_comparison
quant-rsi compare run runs/c5_comparison --steps 100
quant-rsi compare replay runs/c5_comparison
quant-rsi compare decide runs/c5_comparison --request-id first-retain
```

The initial version supports only `engineering_only`: it executes development searches and fixed-model scoring but requires retaining the incumbent. `confirmatory` is rejected before reading scoring data or allocating a new grant. `initial` aliases the same run, model, and score as `parent`; it does not add budget consumption or economic-quality evidence. Each task has 80 development sessions, 40 scoring decision sessions, and 2 execution/liquidation tail sessions.

New source code requires a new runtime/package identity. The original C4 r2 source is preserved under `runs/accepted_source/c4_b4ad09985d3e/`; replay historical runs with the matching snapshot, without modifying their manifests. C5 modules are `evidence.py`, `comparison.py`, `protected_scoring.py`, and `promotion.py`. Failure and accounting regression tests are in `tests/test_comparison.py` and `tests/test_protected_accounting.py`.

## Current Limitations and Future Work

The local adapter supports explicitly mapped event CSV files. Observers support predictions alone or indicative accounts under bounded rules; these do not constitute a completed real-market pilot or broker paper trading. Beyond public daily-data collection, there is no live provider/broker execution integration, Qlib/RD-Agent adapter, LLM proposer, self-modifying code, calibrated confirmatory inference, production promotion, or M1 scheduled retraining. C5 reports `null` for confidence intervals, p-values, and power, with specific reasons why calibration is absent; the formal-test count is 0. Unsupported plans and modes are rejected. The worker is a process boundary for trusted computation, not a security sandbox for arbitrary generated code.

Extend capabilities according to the [architecture's dependencies and acceptance criteria](docs/architecture.md). Returns on synthetic data cannot establish strategy effectiveness.

## Local Data and Persistent Observation

Complete contracts and plans using the [local observation guide](docs/local_observer.md). Markets, calendars, providers, historical availability, instrument universes, currencies, and costs must be chosen explicitly. Placeholders in `examples/local_*.json.template` are rejected by validation and cannot be used as market defaults.

```bash
export QUANT_RSI_WORKSPACE="$PWD"
quant-rsi data ingest contract.json events.csv --request-id import-001 --evidence-file availability.bin
quant-rsi research run local-research.json --output runs/local-research --steps 1
quant-rsi baseline seal runs/local-research
quant-rsi infer BUNDLE_SHA256 SNAPSHOT_SHA256 SESSION_ID
quant-rsi observer register observation.json --episode-id baseline-v1
quant-rsi observer run baseline-v1 --request-id poll-001 --steps 8
quant-rsi observer replay baseline-v1
quant-rsi observer stop baseline-v1 --request-id stop-001
```

`data ingest` and `baseline seal` return the hash references required by subsequent plans. Use a new poll request after each new market-data import; use the original request when recovering an interruption. `status` and `replay` neither repair nor write state and do not call training or proposal generation. If capital, costs, or settlement rules have not been chosen, use a `prediction_only` plan without financial fields. The market calendar, instrument identities, knowledge basis, and full DataContract must still be supplied explicitly.
