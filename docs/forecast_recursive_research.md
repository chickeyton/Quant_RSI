# Bounded Recursive Forecast Research

Version 3 connects the existing five-session forecast worker to a registered trajectory
with at most two cap advances. It searches target-price momentum and mean-return recipes
using M0 training and negative MAE in simple-return percentage points. This development
mechanism does not establish predictive advantage, promotion, or profitability.

## Register the task and allowance

Use one absolute QUANT_RSI_WORKSPACE and an already registered local engineering snapshot.
This implementation labels its exposure engineering_fixture. R1 comparison allowances
do not authorize this work.

Prepare research-v3.json with ResearchPlan schema_version 3, forecast development,
an explicit RegisteredNeighborDomain, and ForecastRecursiveProfile.from_development(...).
The domain fixes the operators, global window bounds, initial cap, step 1 and maximum
two advances. The initial policy uses window_first and the registered initial cap.
The full development template, objective, seed and paired-arm budget stay unchanged.
A smaller source allowance belongs in the outer grant.

This supported API starts after data ingestion and task preparation. Paths, IDs and
snapshot references must identify the actual registered input. The example assumes the
full development template allows 8 attempts / 1,440 seconds with 180-second worker slots;
it is documentation, not an execution grant.

```python
import os
from pathlib import Path

from quant_rsi.artifacts import canonical
from quant_rsi.forecast_recursive_campaign import init_recursive_campaign
from quant_rsi.research import create_research
from quant_rsi.researcher import ResearchPlan
from quant_rsi.trajectory import init_trajectory, resume_trajectory
from quant_rsi.trajectory_state import TrajectoryPlan

workspace = Path(os.environ["QUANT_RSI_WORKSPACE"]).resolve()
plan = ResearchPlan.model_validate_json(Path("research-v3.json").read_bytes())
grant = {
    "kind": "ForecastRecursiveCampaign",
    "schema_version": 3,
    "campaign_id": "recursive-example",
    "source_directory": str(workspace / "source"),
    "source_campaign_id": "recursive-source",
    "research_plan": plan.model_dump(mode="json"),
    "source_allowance": {"max_attempts": 4, "max_worker_seconds": 720},
    "budget": {"max_attempts": 36, "max_worker_seconds": 6480},
    "max_proposal_stages": 2,
    "max_generations": 2,
}
init_recursive_campaign(grant)
source = create_research(plan, workspace / "source", steps=4)
trajectory = TrajectoryPlan.model_validate_json(canonical({
    "schema_version": 3,
    "name": "Bounded forecast development",
    "source_directory": str(workspace / "source"),
    "source_checkpoint_ref": source["checkpoint_ref"],
    "state_policy": "reset",
    "max_proposal_stages": 2,
    "max_generations": 2,
    "budget": {"worker_attempts": 36, "registered_worker_seconds": 6480},
}))
init_trajectory(trajectory)
result = resume_trajectory(steps=100)
```

Registration also supports the CLI command
python -m quant_rsi forecast-campaign recursive-init grant.json.
Existing research and trajectory run/resume/status/replay commands use the same authority.
Registration itself performs no ingestion or fitting.

## Interpret decisions and costs

Each request binds its immediate source's final checkpoint, selection, event tail, complete
allowed feedback, domain and original development template. Only a qualifying final
same-operator improvement at the current cap can propose the next registered cap.
Parent and candidate restart identically with full reset allowances. A completed valid
pair can advance the experimental head even with a negative paired delta; advancement
is not promotion.

Source charges count once. Each stage contributes its latest cumulative child receipts
once; reading that child as the next source adds no duplicate charge. Failed and unused
commitments remain occupied. Reports distinguish proposals, charged attempts, confirmed
fit receipts, unknown fit outcomes, paired stages and semantic advances.

No-patch creates no research descendants. Domain, generation, capacity and failed-operation
terminals remain distinct. Resume reconciles the original pending operation before another
reservation. Workers inherit the original ownership descriptors.

## Seal and replay the selected forecast

After the trajectory seals, call baseline.seal_forecast on its actual selected source:
the registered source for generation 0, or
.quant_rsi/trajectory/children/stage_<generation-1>/arms/candidate for later generations.
Other child selections reject. Pass the resulting bundle_ref to
forecast_requests.issue_forecast with the actual snapshot and origin session.

The bundle preserves the selected M0 model's training cutoff and complete frozen trajectory,
model, mask and ancestor closure. It performs no publication refit. Verification replays
the original sealed ledger in memory without mutable research sources. Late forecasts
remain labeled late.

Pure semantic and admission tests do not establish a continuous actual two-transition run.
Record that proof separately, including failures and naive controls.
