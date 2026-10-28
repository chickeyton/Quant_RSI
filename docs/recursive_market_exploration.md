# Recursive Market Exploration

This path prepares historical market inputs for bounded target-price research. It uses
the existing five-session M0 worker, explicit instrument profiles and registered
allowances. Development improvements and generation advances do not establish
predictive advantage.

## Prepare an explicit profile

Follow the Python 3.12/Linux installation instructions in the
[enhanced workflow guide](enhanced_forecast_workflow.md). Select
[SOXL/XSD/SPY](../instruments/US/SOXL/forecast.json) or
[KO/XLP/SPY](../instruments/US/KO/forecast.json). Run from the repository root:

```bash
quant-rsi forecast-campaign recursive-prepare instruments/US/SOXL/forecast.json \
  --output runs/recursive-soxl \
  --campaign-id recursive-soxl --source-campaign-id recursive-soxl-source
```

Preparation collects and imports data, then writes `research_plan.json`,
`campaign_plan.json` and `market_evidence.json`. It does not register an allowance
or train a model. Review those files and the returned workspace before execution.
The output directory must be fresh on the first call. Repeating the command must
retain its original profile, policy, IDs and collected evidence.

The initial supported profile requires 756 quoted historical sessions and five
future calendar sessions. Its training boundary is index 721; its six development
origins are 721, 726, 731, 736, 741 and 746. The latest forecast origin is index 755,
with target index 760. All indices are zero-based.

Only the target-price data enters the research model. Industry and market proxies,
signals and their original evidence remain in the frozen input bundle as retained,
unused context. The target projection changes only its training cutoff and seed.
Registration verifies that the ordinary market exporter produces exactly that
projection, retaining the original receipt bytes and availability clocks.

## Register and execute the reviewed plan

Use the preparation's workspace for every command:

```bash
export QUANT_RSI_WORKSPACE="$(pwd)/runs/recursive-soxl/preparation/workspace"
quant-rsi forecast-campaign recursive-init runs/recursive-soxl/campaign_plan.json \
  --market-preparation runs/recursive-soxl/preparation
quant-rsi research run runs/recursive-soxl/research_plan.json \
  --output "$QUANT_RSI_WORKSPACE/source" --steps 4 > runs/recursive-soxl/source.json
```

Registration imports the verified evidence closure into the original authority;
it performs no training. The source command can consume at most four attempts and
720 worker-seconds. The total reserved campaign allowance is 36 attempts and
6,480 worker-seconds: source plus two paired stages, each with two arms capped at
eight attempts and 1,440 worker-seconds. Unused and failed commitments remain
occupied. These limits are separate from enhanced comparison allowances.

After the source has a final checkpoint, refresh its verified report, including after
any source recovery. This replaces an incomplete or stale redirected report before
the trajectory plan reads its checkpoint:

```bash
quant-rsi research status "$QUANT_RSI_WORKSPACE/source" > runs/recursive-soxl/source.json
```

Then create the trajectory plan:

```python
import json
import os
from pathlib import Path

output = Path("runs/recursive-soxl")
source = json.loads((output / "source.json").read_text())
plan = {
    "schema_version": 3,
    "name": "Bounded historical market exploration",
    "source_directory": str(Path(os.environ["QUANT_RSI_WORKSPACE"]) / "source"),
    "source_checkpoint_ref": source["checkpoint_ref"],
    "state_policy": "reset",
    "max_proposal_stages": 2,
    "max_generations": 2,
    "budget": {"worker_attempts": 36, "registered_worker_seconds": 6480},
}
(output / "trajectory.json").write_text(json.dumps(plan, indent=2) + "\n")
```

```bash
quant-rsi trajectory init runs/recursive-soxl/trajectory.json
quant-rsi trajectory run --steps 1
quant-rsi trajectory status
```

Use `trajectory resume --steps 1` to reconcile and continue the same trajectory.
Inspect each result; a finite run may stop without advancing. Source recovery uses
`research resume "$QUANT_RSI_WORKSPACE/source" --steps 1`. Retain the original
workspace, authority and IDs. Missing authority or changed frozen configuration
does not authorize replacement collection, a new allowance or another fit.

## Policy identity and interpretation

The outer campaign grant is schema 4 and binds the exact
`terminal_frontier_first_divergence_v1` proposer descriptor. ResearchPlan,
ForecastRecursiveProfile, trajectory mathematics and the eight-field root binding
remain schema 3. The grant's policy identity must agree with the program, stage
authorization, evidence capsule, request and patch lineage. Missing or mixed
selectors reject; existing schema-3 grants retain their original policy.

The new policy considers authenticated terminal development feedback at the current
cap and the preceding cap. A supported improvement must also permit an affordable
first divergence in the reset search at the next cap. The original selected parent
remains unchanged. Protected outcomes are not proposal inputs. Cap limits are
3 → 4 → 5; no-patch, failed-stage and exhausted-budget outcomes remain terminal
evidence.

The market exposure is `historical_market_exploration`. Historical provider
availability is an explicit assumption; first-seen receipts remain intact.
Engineering fixtures use the separately declared `engineering_fixture` exposure.
Neither mode establishes statistical promotion or profitability.

## Seal, replay and report evidence

After the original trajectory is SEALED, use its actual selected source to seal and
issue the forecast. Run this example from the repository root in the same environment
and workspace. Change `runs/recursive-soxl` when using another prepared campaign.
Sealing retains the M0 model and training cutoff without a publication refit.

```sh
python - <<'PY'
import json
import os
from pathlib import Path

from quant_rsi.execution.baseline import seal_forecast
from quant_rsi.forecast.forecast_requests import issue_forecast
from quant_rsi.research.trajectory import inspect_trajectory

out = Path("runs/recursive-soxl")
workspace = Path(os.environ["QUANT_RSI_WORKSPACE"])
terminal = inspect_trajectory()
if terminal["status"] != "SEALED":
    raise SystemExit("Resume this trajectory to SEALED before sealing its forecast.")

campaign = json.loads((out / "campaign_plan.json").read_text())
generation = terminal["generation"]
source = (
    Path(campaign["source_directory"])
    if generation == 0
    else workspace / ".quant_rsi/trajectory/children"
    / f"stage_{generation - 1}" / "arms/candidate"
)
bundle = seal_forecast(source)
inputs = json.loads((out / "preparation/inputs.json").read_text())
receipt = json.loads((out / "preparation/preparation.receipt.json").read_text())
request_id = campaign["campaign_id"] + "-final"
report = issue_forecast(
    bundle["bundle_ref"],
    inputs["target"]["data"]["specification"]["snapshot_ref"],
    receipt["origin_session_id"],
    request_id=request_id,
)
for name, value in (
    ("terminal.json", terminal), ("sealed_forecast.json", bundle),
    ("forecast.json", report),
):
    (out / name).write_text(json.dumps(value, indent=2) + "\n")

forecast = report["forecast"]
print(json.dumps({
    "request_id": request_id,
    "status": report["status"],
    "observed_origin_session": forecast["origin_session"],
    "forecast_target_session": forecast["target_session"],
    "target_close_at": forecast["target_close_at"],
    "currency": forecast["currency"],
    "rows": [
        {
            "instrument": row["instrument"], "status": row["status"],
            "origin_price": row["origin_price"],
            "forecast_price": row["forecast_price"],
            "forecast_simple_return_percent": (
                None if row["forecast_return"] is None else 100 * row["forecast_return"]
            ),
            "reasons": row["reasons"],
        }
        for row in forecast["rows"]
    ],
}, indent=2))
PY

quant-rsi forecast replay recursive-soxl-final > runs/recursive-soxl/forecast-replay.json
```

Reuse the same request ID and binding on retry; do not create a new request ID to avoid an existing failure. For another campaign, the replay ID is its campaign ID followed by `-final`. The full `forecast.json` also retains the actual issue/confirmation times, deadline, knowledge basis, qualifications, historical evaluation, and benchmark reports. A top-level `abstain` status can override the timing label; inspect its row reasons and confirmation fields. Historical data or a long research run may yield a late forecast. If sealing rejects a missing eligible incumbent, retain the error; this recipe does not authorize a replacement fit or collection.

Record forecast timing, charged attempts, confirmed fits, unknown outcomes, actual
generations and any faults exercised. Keep an early no-patch result as a result; it
does not establish a successful two-generation run.

Use `quant-rsi forecast replay recursive-soxl-final` to verify the issued forecast.
Use `quant-rsi trajectory replay` to verify the recursive receipts, head, terminal
state, and frozen authority/evidence, and report the recorded cumulative accounting.
Trajectory replay does not reread the mutable shared campaign registry; check
current registry and ledger counts separately when verifying unchanged accounting.

`forecast-campaign status` and `forecast-campaign replay` apply only to comparison
campaigns. They reject recursive grants, so do not use them for `recursive-soxl`.
Preserve any failed invocation and its error when reporting replay outcomes.

A replay performed with original source paths present does not prove independence
from those paths. Pure policy tests and zero-fit registration tests do not
establish a continuous live execution.
