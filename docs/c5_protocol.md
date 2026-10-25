# C5 engineering comparison

This implementation exercises the evidence protocol with actual synthetic development
searches and their selected LightGBM models. It does not supply a calibrated statistical
promotion method. Production decisions always retain the registered parent.

## Workspace and immutable authority

Set `QUANT_RSI_WORKSPACE` to an existing absolute directory. All C5 commands locate
`.quant_rsi/authority.json` and `.quant_rsi/authority/control.sqlite` there, regardless
of the current directory, output label, or source snapshot. There is no ordinary
per-run authority override. Changing this trusted configuration selects another
administrative workspace; this is not protection against a filesystem administrator.

`evidence init PROGRAM.json` accepts a descriptive name, a completed current-runtime
C4 package campaign, and `task_count` from 1 to 4. The example program points to
`runs/c5_packages`; resolve that path before changing directories. Bootstrap persists
one UUID before generating the source or allocating work. Interrupted bootstrap
resumes that identity; an initialized missing/corrupt anchor, ledger or required
artifact fails closed. The private generator seed is removed after initialization
and never appears in a research view. The full source is generated once, then sliced.

Each authority has one program, one generation and one batch. Names and output handles
cannot reset these limits. Do not delete authority state to obtain more evidence.
This directory is ignored by Git and contains private scoring data plus durable costs.

## Register, run and inspect

Follow the complete command sequence in [README](../README.md#c5工程保护评估).
`compare register PLAN --output ROOT` compiles the frozen input plan, task grids,
training/eligibility rules and expected coverage before granting development work.
The only accepted runnable mode is `engineering_only`. A `confirmatory` request fails
preflight with `blocked_uncalibrated` before any grant or protected-data read.

Use `compare run ROOT --steps N` or `compare resume ROOT --steps N`, with 1–100 steps.
One step is one durable operation: one development observation for a fixed unique run,
sealing all selections, OPENED, one scoring-attempt outcome, or report finalization.
An invocation commits an absolute cursor target. If interrupted, resume with the same
`--steps`; a different value is rejected while that invocation remains pending.
Finishing already committed child work only completes its receipt, without extra work.

`compare status ROOT` and `compare replay ROOT` verify stored identities and coverage;
replay also recomputes fixed-model inference and accounting. Neither fits a model,
proposes a strategy nor publishes new artifacts. Closed runs remain read-only on resume.
Use the recorded source/runtime snapshot to replay after an engine update.

`compare cancel ROOT` retires only an unexposed RESERVED batch. Development grants
remain spent; cancellation cannot make the authority AVAILABLE again. OPENED cannot
be cancelled for a refund. An exposed completed batch is retired automatically.

## Coverage, selection and budgets

Each task has separate parent and candidate runs with empty initial search state.
The initial anchor is the parent alias: the same run, grant, selection, model and score,
with no additional economic weight. Expected keys are generated from the frozen plan.
Missing, duplicated, substituted or conflicting keys prevent OPENED.

Each unique run receives 8 development attempts of 20 seconds, irrevocably granting 160
seconds. Only terminal, committed development incumbents can supply models. Selection
sealing reconstructs training rows, mature labels, feature hashes and prediction
consistency from the actual permitted view. It binds the model, preprocessing, complete
development plan, checkpoint and fresh execution state. An authoritative terminal
`no_valid_incumbent` alone permits the registered zero-interest cash fallback.
Missing models, corrupt data or broken ledgers never become cash.

Only after OPENED can scoring atomically register an attempt ID, exact computation
binding, 20-second debit and outbox entry. Known unissued attempts keep their identity.
A valid published output is reused. A possibly dispatched attempt without valid output
keeps its charge and permits one newly charged retry of the same sealed computation.
Semantic failures and malformed outputs are terminal; there is no third attempt.
The controller lock is inherited by workers, so orphan workers block competing recovery.

With four tasks, the maximum is 64 development attempts/1280 granted seconds plus 8
scoring keys × 2 attempts × 20 seconds = 320 scoring seconds. Unused development grants
are not refunded. Parent/initial aliases never duplicate these totals. Reports separately
show actual development consumption and the historical C4 source costs; importing those
costs does not bill them again as newly performed C5 work.

## Scoring and decision interpretation

Each nonoverlapping task contains 122 weekday sessions and four synthetic instruments.
Development uses indices 0–79, train information cutoff 40, and development decisions 40–77.
Scoring decisions 80–119 fill at opens 81–120 and are valued at opens 82–121. The final
valuation includes liquidation at open 121 and its costs. Exactly 40 returns enter
`252 * mean(net_return)`, relative to zero-interest cash, with no external flows.
The initial denominator is cash before the first scoring fill. The two tail sessions
do not create additional samples. Report subdivision preserves the same account path,
terminal cost and economic mass; resetting accounts defines a different design.

All tasks share one conservative dependence cluster and one economic group; task
weights sum to 1. Execution seeds are not independent market histories. Risk failures
retain their path and costs. Reference/fallback technical failure means insufficient
evidence; candidate failure prevents promotion; integrity damage makes evidence invalid.
Incomplete keys remain present and aggregate differences are null, without reweighting.

Positive quotes, long-only holdings and cash constraints do not generate exact
bankruptcy in this source. An isolated accounting fixture supports a declared total-loss
ledger event: return−1 followed by zero returns on the original grid. Production plans
register an empty event list and reject injection. Missing quotes or invalid denominators
remain failures. This does not implement real-market bankruptcy processing.

`compare decide ROOT --request-id ID` performs a production retain transaction. The
authority checks registered mode/plan/report, then compares package plus monotonic
revision and atomically records pointer, report and decision. Repeating the same ID and
binding returns the original result; changed bindings or stale revisions fail. A forged
positive report cannot authorize promotion. Positive/negative/inconclusive/invalid CAS
fixtures use a separate in-memory `TestRegistry` with no production file or CLI access.

CI, p-value and power are null with the missing-calibration/feasibility reason;
`formal_tests_executed=0`. Engineering exposure retires the earmarked error allowance
without claiming a statistical test occurred. Statistical calibration, real data,
paper execution and complete recursive self-improvement remain unfinished capabilities.
