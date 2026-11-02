# Bounded C2 exploration

The original `neighbor_search_v1` policy stops when it reaches `max_no_progress`
or exhausts the incumbent's immediate neighbors. Existing plans retain that behavior.
For a new schema-1 `ResearchPlan`, select `neighbor_exploration_v1` before running:

```bash
python - <<'PY'
import json
from pathlib import Path

plan = json.loads(Path("examples/research_development.json").read_text())
plan["policy"]["algorithm"] = "neighbor_exploration_v1"
Path("/tmp/research-exploration.json").write_text(json.dumps(plan, indent=2))
PY
quant-rsi research run /tmp/research-exploration.json \
  --output /tmp/research-exploration --steps 2
quant-rsi research resume /tmp/research-exploration --steps 100
quant-rsi research replay /tmp/research-exploration
```

Use a fresh output directory. This changes the frozen research package; do not edit
an existing run's plan, package, manifests or checkpoints to switch algorithms.
Historical runs still require their original source/dependency environment.

## Selection and stopping rules

1. Evaluate the registered baseline first. A missing/ineligible baseline still stops
   the campaign; exploration cannot bypass that gate.
2. Try the incumbent's immediate neighbors in the registered `neighbor_order`.
3. If `no_progress >= max_no_progress`, or no eligible unvisited neighbor remains,
   search the full registered range in ascending window order, with `momentum`
   before `mean_return` at each window. Select the first unvisited feasible recipe.
   No random draws, new factors or out-of-range windows are introduced.
4. Retain the incumbent on ties, inadequate improvements and failures. A successful
   strict improvement resets `no_progress` and returns to local-neighbor preference.
5. Every dispatched exploration uses the same proposal/attempt and worker-second
   reservations as a local step. Budget limits take precedence. If the finite space
   is exhausted, stop with `registered_search_space_exhausted`.

The existing `visited`, `incumbent` and `no_progress` checkpoint fields determine the
next step. No additional registry, RNG state or parallel budget is created. Invalid,
visited and structurally immature recipes are filtered before proposal reservation.
Actual missing/invalid data can still make a dispatched attempt fail, which is retained
and charged normally. Forecast plans account for five-session label maturity.

Schema-3 recursive cap/frontier campaigns keep their separately registered
`neighbor_search_v1` semantics and reject this algorithm. This change addresses the
schema-1 C2 plateau gap in [#5](https://github.com/natureofnature/Quant_RSI/issues/5);
it does not satisfy the two mechanism-generation requirement in #3, change the C5
production promotion gate, or establish that exploration improves research quality.

## Verification

Pure tests cover ties, failed trials, subsequent strict improvement, finite-space
exhaustion, maturity and both budget limits. The integration test runs actual workers
with a predeclared high improvement threshold to exercise a plateau without modifying
accounting scores. It pauses before exploration, interrupts observation commitment,
resumes without redispatch or duplicate charges, and replays the finished campaign.
Legacy-policy and recursive-cap regressions remain separate compatibility checks.
