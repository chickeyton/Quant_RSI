# Run the minimal research loop

`research-workflow` joins the existing factor-discovery campaign, verified
forecast results, costed strategy accounts and forecast observation ledger under
one frozen profile. It is the minimum workflow slice of
[RFC #39](https://github.com/natureofnature/Quant_RSI/issues/39), not completion of
the broader platform or evidence of predictive advantage.

## Offline start

Use the Linux/Python 3.12 editable installation in the [README](../README.md).
Run these commands from the repository root in the same existing workspace:

```bash
export QUANT_RSI_WORKSPACE="$PWD"
mkdir -p runs/workflow-demo
python examples/prepare_research_workflow.py \
  --fixture --campaign-id rfc39-demo --output runs/workflow-demo
quant-rsi research-workflow run runs/workflow-demo/profile.json --max-fits 10 \
  > runs/workflow-demo/run.json
quant-rsi research-workflow status rfc39-demo > runs/workflow-demo/status.json
quant-rsi research-workflow report rfc39-demo > runs/workflow-demo/report.md
quant-rsi research-workflow replay rfc39-demo > runs/workflow-demo/replay.json
```

The example generates invented KO-labelled prices and an invented daily calendar;
it does not download market data. It registers 105 sessions, uses the first 100
for research and retains five further calendar sessions. It binds the shared
Qlib ROC/MA price factors and runs real LightGBM workers. Its old dates make the
publication retrospective: it cannot produce prospective evidence or backdated
paper fills.

The default profile requests two rounds and two folds (`[[80],[90]]`). Each
round reserves two recipes per fold plus one publication fit: five fits per round,
ten total. The next candidate uses the existing discovery frontier and permitted
development feedback. A failed fit remains an attempt in the ledger; a completed
command does not imply that every candidate succeeded.

For shorter invocations, use `--max-fits 1`, then continue the same campaign:

```bash
quant-rsi research-workflow resume rfc39-demo --max-fits 1
```

`prepare PROFILE` freezes a profile without fitting; `run PROFILE` prepares it
and advances its existing allowance. Repeating commands or changing output paths
does not renew the campaign's fit grant. A workflow must be registered before
its same-ID discovery campaign starts. Preserve its original workspace and source
runtime; do not copy authority files or edit a registered profile to resume it.

## Supply existing inputs

Use an existing `ForecastInputs` envelope with a single KO target:

```bash
python examples/prepare_research_workflow.py inputs.json \
  --campaign-id ko-exploration --output runs/ko-exploration \
  --folds '[[80,85],[95,100]]' --max-rounds 2
```

Choose folds that fit the registered input history and have mature five-session
labels. The generator's default account has initial cash 10,000 in the target
currency, one position, full allocation, unit lots, a 10,000-unit fill cap,
5 bps commission and 5 bps slippage on each side, zero transaction tax and a
20% drawdown suspension threshold. These are explicit experimental assumptions,
not brokerage estimates. Review the account, folds and threshold in `profile.json`
before `prepare` or `run`. General source acquisition and live scheduling are
outside this command.

## Read the report

`report` prints Markdown; `status` and `replay` return JSON with the detailed
artifacts and account records. Replay verifies recorded work without fitting or
fetching new data. The report includes every discovery round and both `B0` and
`D1` evaluation recipes, with a strategy and same-target buy-and-hold account
for each fold. `B0` is the fixed forecast baseline; `D1` is the discovery
candidate. Neither label means buy-and-hold.

- A prediction above the frozen threshold requests long; negative/below-threshold
  or missing predictions request cash. Orders target the next registered open.
- Each fold starts with its own capital and retains every close from its first
  origin through its last origin plus five sessions. Gaps between folds are not
  simulated. These episodes are not one continuous portfolio equity curve.
- Both arms use the same dates, capital, costs, lots, fill caps and terminal
  inventory-preserving valuation. The report retains fees, turnover, drawdown,
  exposure, orders, fills, cash, positions and unresolved reasons.
- Missing valuations or revised consumed evidence suppress aggregate metrics;
  a later valid mark does not erase a missing period. Read the metric evidence
  status as well as the processing status.

The five-session prediction is close-to-close. The trade starts at the following
open and ends at a later eligible open or the final close mark; these are different
intervals. See [strategy replay semantics](strategy_replay.md) for details.

**Development forecast MAE drives the next research candidate.** Account return
and buy-and-hold differences are diagnostic outputs; they neither choose the
candidate nor tune its frozen threshold. Reusing development feedback does not
make the result an independent test of the researcher or selected model.

## Observe an actually issued forecast

The completed workflow registers each available publication in the existing
[forecast observation ledger](forecast_observations.md) and prepares/confirms a
hypothetical paper account. Publication and actual paper submission must both
finish before the next registered open. Late initialization is explicitly
non-prospective, even if the forecast itself was issued on time.

Import new receipts through `quant-rsi data ingest` using the same registered
contract and workspace. Then supply its returned snapshot reference:

```bash
quant-rsi research-workflow observe ko-exploration \
  --snapshot-ref TARGET_SNAPSHOT_SHA256 --request-id observation-001
quant-rsi research-workflow report ko-exploration
quant-rsi research-workflow replay ko-exploration
```

`observe` connects that publication's forecast outcome and paper account in one
record. It performs no fitting or collection. The snapshot must preserve prior
receipts; controller and first-seen clocks are authoritative. Repeating an ID
with the same snapshot returns its original result. Use a new ID for a later
assessment; reusing an ID with a different snapshot fails. Resume an unfinished
request before submitting another. The workflow has a fixed 256-request allowance.
Corrections append assessments and preserve earlier fills and valuations.

## What the evidence establishes

The offline example exercises historical research with real workers. Separate
clock-controlled tests use deterministic worker doubles to verify timely
publication, durable submission, later quote receipts, fills, valuation and
outcome linkage. Those tests establish engineering behavior, not elapsed
prospective performance or a realistic trading simulator.

There is no broker, real order, online scheduler, independent confirmation,
account-return-driven research policy or established predictive advantage here.
The next decision should use actual usability, validation cost and matched-budget
comparisons with simple baselines and compatible mature engines. Negative returns
and insufficient evidence remain legitimate outcomes.
