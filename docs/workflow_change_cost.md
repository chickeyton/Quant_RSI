# Cost of an ordinary strategy hypothesis change

This October 29, 2026 measurement addresses the usability gate in
[RFC #39](https://github.com/natureofnature/Quant_RSI/issues/39) and
[G0 in #40](https://github.com/natureofnature/Quant_RSI/issues/40). It uses merged
`main@1af6e1ba2d92ce9c7b1bb6e401c66320125f20b1`, invented prices and real LightGBM
workers. It measures changing a hypothesis on the same software version, not a
software-upgrade speedup or market performance.

## Frozen question and allowance

Before either arm ran, the local plan froze a baseline threshold of zero and a
candidate threshold of `0.02` in predicted five-session **log-return** units.
The question was whether a reader can change that one assumption, obtain all
strategy/buy-and-hold reports and replay them without source changes or manual
repair of registered artifacts. The threshold was not changed after results.

The fixture was generated once. The candidate profile is an exact copy with
only two scalar changes: the threshold and the new campaign identity required
for a separately frozen run. Input envelope, receipt snapshot, factor definitions,
folds, model settings, account rules and exposure classification are identical.
Both arms have two rounds, two folds and ten fit attempts; the combined ceiling
is twenty. The predeclared small-task feedback target was under five minutes per
arm, with failures retained and no new allowance on replay.

## What changed and what it cost

| Item | Observed result |
| --- | --- |
| Production modules edited | 0 |
| User configuration change | One added candidate profile; one semantic field plus campaign ID |
| Registered artifact repair | None |
| Recipe/model/score changes | None: expression refs, model refs and numerical predictions matched |
| Research feedback changes | None: development MAE feedback matched |
| Account decisions changed | All eight historical fold/recipe decisions, from long to cash |
| Fit attempts | Baseline 10; candidate 10; replay added zero |
| Failed fits | Zero in both arms; no attempts omitted |
| Documented workflow operations | 10: generate once, derive candidate once, four CLI operations per arm |
| Hands-on human time | Not measured; command wall time is not a substitute |

This is a logical workflow count, not a measured count of human interactions.
The actual wrapper invoked nine external commands and derived the candidate
configuration internally. Environment setup and worker subprocesses are excluded;
actual manual interactions were not measured. The existing
`status` output is verbose JSON; the Markdown `report` is the readable view.

| Operation | Baseline | Changed threshold |
| --- | ---: | ---: |
| Run research and assemble account report | 6.443 s | 5.900 s |
| Read status | 0.273 s | 0.246 s |
| Render Markdown report | 0.265 s | 0.258 s |
| Replay recorded work | 0.743 s | 0.665 s |

One-time fixture/profile generation took 0.535 s. All commands exited zero;
both arms completed within the small-task target. These are single observations,
not evidence that the candidate is faster. Runs were sequential, baseline first,
with fresh processes/workspace, uncontrolled OS caches and possible concurrent
agent package installation. The Linux host reported 16 logical CPUs; Python was
3.12.3. `PYTHONHASHSEED=0` and OMP/OpenBLAS/MKL thread counts were one. Source and
runtime identity did not change during measurement.

## Keep the losing and no-trade outcomes

Each row is a separate six-close account episode. Returns include the declared
5 bps commission and 5 bps slippage on the executed side. The changed-threshold
account stayed in cash, while the baseline and buy-and-hold bought 75 units in
fold 0 and 76 units in fold 1.

| Round | Fold | Recipe | Baseline net return | Candidate net return | Buy-and-hold net return |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | 0 | B0 | -1.5982% | 0.0000% | -1.5982% |
| 0 | 0 | D1 | -1.5982% | 0.0000% | -1.5982% |
| 0 | 1 | B0 | +1.4212% | 0.0000% | +1.4212% |
| 0 | 1 | D1 | +1.4212% | 0.0000% | +1.4212% |
| 1 | 0 | B0 | -1.5982% | 0.0000% | -1.5982% |
| 1 | 0 | D1 | -1.5982% | 0.0000% | -1.5982% |
| 1 | 1 | B0 | +1.4212% | 0.0000% | +1.4212% |
| 1 | 1 | D1 | +1.4212% | 0.0000% | +1.4212% |

Buy-and-hold metrics matched exactly between arms. For fold 0, baseline fees
were USD 9.82 and maximum drawdown was approximately -2.9585%; for fold 1,
USD 9.88 and -2.9102%. Cash-only candidate fees, drawdown and exposure were zero.
The threshold avoided a loss in one fold and missed a gain in the other. Do not
pool repeated rounds/folds as independent observations or concatenate their equity.

## Reproduce the operation with existing commands

Use the pinned editable installation of the recorded source. Start in the
repository root; choose a fresh external output directory and workspace.
Generate the fixture only once: regenerating it with another campaign ID creates
another source identity and would invalidate this comparison.

```bash
export QUANT_RSI_WORKSPACE=/tmp/ordinary-change-workspace
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir /tmp/ordinary-change-workspace /tmp/ordinary-change-output
python examples/prepare_research_workflow.py --fixture \
  --campaign-id change-baseline --output /tmp/ordinary-change-output/prepared
python - <<'PY'
import json
from pathlib import Path
p = Path('/tmp/ordinary-change-output')
baseline = json.loads((p / 'prepared/profile.json').read_text())
(p / 'baseline.json').write_text(json.dumps(baseline))
candidate = json.loads(json.dumps(baseline))
candidate['discovery']['campaign_id'] = 'change-candidate'
candidate['strategy_threshold'] = 0.02
(p / 'candidate.json').write_text(json.dumps(candidate))
PY
for arm in baseline candidate; do
  quant-rsi research-workflow run /tmp/ordinary-change-output/$arm.json --max-fits 10 \
    > /tmp/ordinary-change-output/$arm-run.json
  quant-rsi research-workflow status change-$arm \
    > /tmp/ordinary-change-output/$arm-status.json
  quant-rsi research-workflow report change-$arm \
    > /tmp/ordinary-change-output/$arm-report.md
  quant-rsi research-workflow replay change-$arm \
    > /tmp/ordinary-change-output/$arm-replay.json
done
```

For the recorded run, a local standard-library subprocess wrapper measured each
command with `time.monotonic`, retained both output streams and recorded exit
codes before proceeding. The frozen plan, exact commands, profile hashes, source
identity, reports and comparison are retained at
`/tmp/rfc39-ordinary-change-20261029/`; that local path is not a public dataset.

## Decision and remaining limits

The ordinary configuration-change path works without modifying production code.
It also exposes an avoidable-looking cost: **changing only the account threshold
still repeats ten fits**, even though all eight historical model refs and scores
were unchanged. Keep the current evidence; evaluate whether an explicitly
retrospective strategy comparison over frozen predictions is worthwhile before
adding reuse machinery. Do not bypass frozen profile identity or recharge old
campaigns to simulate reuse.

This is one small strategy change, not a factor-implementation change or a
measurement of typical real-market workloads. No equivalent integrated command
existed before #42, so there is no measured pre-#42 hands-on baseline. Human effort,
larger-task latency, real-data usability and the full validation cost gate remain
open. The earlier 16% timestamp-workload improvement still missed its independent
20% target. These limitations are reasons to run focused evidence-producing
tasks before expanding the framework.
