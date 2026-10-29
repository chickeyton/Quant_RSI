# Bounded factor discovery and event instances

This implements the first executable path of [RFC #31](https://github.com/natureofnature/Quant_RSI/issues/31): shared numeric definitions and observed event indicators can feed a finite expression grammar, a feedback-dependent candidate frontier, and the original forecast worker. It is a development research facility, not evidence of predictive advantage.

## What runs

- Foundational leaves use the existing `NumericFactorDefinition` contracts. Explicit catalog bindings preserve shared/local ownership and source entities as input evidence. KO selects the shared momentum and mean-return definitions; unsupported text specifications remain rejected.
- Leaves can refer to target, industry or market price roles, the registered retained-news indicator, or a frozen event-instance snapshot. No new source is silently selected or fetched.
- The grammar generates single leaves and distinct two-leaf `add`, `subtract` and `multiply` expressions. These definitions are generated from the configured leaves, not individually listed in the catalog. Canonical identity removes node-name and commutative-order duplicates. The expression validator accepts at most seven nodes and depth three; the current proposer deliberately generates only depth-two compositions. Learned intermediate models are not implemented.
- Each candidate runs B0 versus D1 (B0 plus the candidate column), using two or three chronological folds and one publication fit through the existing Journal/controller/LightGBM worker. Legacy B0/B1/B2 and B0/N1 identities remain unchanged.
- All registered leaves must be available, even if a candidate uses only some of them. Numerical representability is checked across the entire registered two-leaf grammar, so an overflowing candidate cannot change its sample population. Consequently every candidate and baseline uses the same input-eligibility population, horizon, folds, training-label maturity and publication boundary. This conservative policy may return insufficient evidence.
- The fixed proposer starts with a library leaf, then prefers multiplication following a positive development delta and subtraction otherwise. It retains all unvisited candidates; failure does not discard non-incumbent branches. This is feedback-dependent candidate selection, **not evolution of the proposer program** or completion of #1/#3's two executed mechanism-change acceptance.

## Prepare and run

Use the README's Linux/Python 3.12 editable environment. Begin with an existing `ForecastInputs` envelope, such as the `inputs.json` produced by the enhanced market preparation. Preserve its evidence, runtime and qualifications. The example below reads existing data and freezes definitions; it performs no collection or fitting:

```bash
python examples/prepare_factor_discovery.py /absolute/path/to/preparation/inputs.json \
  --output runs/ko-discovery-inputs --campaign-id ko-discovery \
  --exposure historical_market_exploration

export QUANT_RSI_WORKSPACE="$(pwd)/runs/ko-discovery-workspace"
quant-rsi factor-discovery init runs/ko-discovery-inputs/plan.json \
  --inputs runs/ko-discovery-inputs/inputs.json
quant-rsi factor-discovery resume ko-discovery --max-fits 1
quant-rsi factor-discovery status ko-discovery
quant-rsi factor-discovery replay ko-discovery
```

The default example uses the existing 756-session, three-fold grid and three candidate rounds: a fixed ceiling of **21 fits / 3,780 reserved worker-seconds**. `resume --max-fits` bounds new fit dispatches in that call (0–50); repeat the same identity to continue. Failed fits remain charged. A real source's historical availability assumptions remain assumptions. Newly observed news cannot fill historical rows.

For the small offline test envelope (100 sessions plus five calendar sessions), use `--folds '[[80],[90]]' --exposure engineering_fixture`. The fixture command is:

```bash
pytest -q tests/test_factor_discovery.py
```

The tests construct offline inputs, exercise the CLI and interruption paths, and include a real small LightGBM comparison. Fixture outcomes do not constitute a historical market study.

The Python API is `initialize(envelope, DiscoveryPlan)`, `resume(campaign_id, max_fits=...)`, and `inspect(campaign_id, replay=...)` in `quant_rsi.factors.discovery`. `bind_catalog` explicitly maps instrument binding IDs to existing input roles; it does not infer that a source exists. Inline typed definitions can also express instrument-local hypotheses. All output artifacts remain under the registered workspace authority.

## Dynamic event lifecycle

`quant_rsi.factors.lifecycle` provides versioned `EventTemplate`, `create`, `transition`, `snapshot`, `expire`, `attach_instance` and `value_at` interfaces. Instance identities are instrument-specific; templates contain no issuer binding. The finite states are:

```text
proposed → validated → experimental → active_for_registered_research
    ↓           ↓           ↓                    ↓
 rejected ← rejected      inactive ←────────── inactive
    ↓                       ↓
 archived                archived

experimental / active_for_registered_research → suspended → archived
```

Admission means the instance may be used as an experimental feature, not that its predictive effect has been demonstrated. Each transition preserves supporting artifact references, its reason and the trusted observation clock. No caller-supplied backdated transition time is accepted. Reuse of an event ID with different content rejects.

An instance outputs a dimensionless indicator while admitted. Its registered TTL is at most 30 days; decay is either none or linear. Expiration yields zero under that instance's explicit policy; `expire` persists inactivity/rejection and archival when due. Unknown/not-yet-observed/not-admitted instances remain missing. An unreliable revision suspends an admitted instance with a missing value, which stays missing after archival. Archiving a never-admitted instance does not turn it into a verified zero. Inactive or suspended instances do not implicitly reactivate: a new occurrence requires a new explicit instance identity.

For retained KO headlines, automatic template instantiation and correction handling is available after `run_news`:

The [prospective RSS collection workflow](news_collection.md) can now supply registered
retained headlines and call this observer directly. Its source coverage remains partial;
it does not backfill historical observed evidence or authorize new fits.

```python
from quant_rsi.factors.event_observer import observe
from quant_rsi.factors.lifecycle import EventTemplate, snapshot, attach_instance
from quant_rsi.core.artifacts import canonical, digest

template = EventTemplate(
    template_id="issuer.earnings_release_title",
    description="Registered exact earnings-title research indicator",
    ttl_hours=168,
)
observation = observe(search_result, template)
instance_id = next(
    item["instance_id"] for item in observation["outcomes"]
    if item["status"] == "active_for_registered_research"
)
instance = snapshot(instance_id)
envelope = attach_instance(envelope, instance)
event_leaf = {
    "name": "release_event", "role": "event",
    "instance_ref": digest(canonical(instance)),
}
# Include event_leaf with numeric leaves in a NEW DiscoveryPlan.
```

The observer verifies that the entire input result was sealed by the registered news campaign. A positive title match creates and admits an indicator; a valid later revision no longer matching the grammar deactivates it. A revision that cannot be reliably extracted suspends an admitted instance instead of treating uncertainty as false. Older revisions cannot resurrect it. This is a rule-specific indicator correction, not general semantic event truth, retraction resolution or full-text analysis. Distinct document IDs create distinct instances; the existing aggregate news leaf supplies semantic quarter-level deduplication when that is desired.

`attach_instance` freezes a snapshot and its evidence into the input envelope. New campaigns require the complete current instance history. Existing campaigns keep their original snapshot even after later transitions; historical evaluation sees only transitions observed by that row's decision. A current correction must be used in a new frozen study/model version, not applied silently to an already sealed model. Lifecycle scheduling is explicit through `observe`/`expire`; no background service is installed.

## Authority, masks and recovery

The parent campaign stores immutable input/plan/runtime references, the full child roster and total allocations in the original artifact/Journal infrastructure. Before each child initializes, it receives an immutable expression and feedback binding. Every version-3 child checks that allocation on prepare, open, dispatch and replay; another experiment ID cannot claim it. Input/fold/definition changes require a deliberately new campaign, not an automatic retry.

News-backed discovery has a distinct owner binding in the existing news ledger. Its frozen evidence cannot also authorize an extra legacy B0/N1 comparison. No news, lifecycle transition or new candidate replenishes the parent fit allowance. The returned report distinguishes allocated fits, attempted fits, conservatively charged worker seconds and proposal count.

An interruption after proposal commitment reuses the original candidate. An interruption in a fit uses the original worker recovery. No complete set of eligible rows seals `insufficient_evidence` without fitting; the exclusion masks are retained and replay does not attempt preparation again. Missing authority fails closed. Restore the original source/runtime for semantic replay; changing a file path is not a migration protocol.

Every candidate report preserves fold errors, naive controls, exclusions, lineage, expression novelty and its development delta. `selected_expression_ref=null` means retain the baseline, not an error. Publication fits are already included in each child allocation. A failed publication remains a failed child outcome; a successful development selection does not claim an issued usable forecast or production promotion.

## Open-issue delivery mapping

| Issue | Delivered here | Still separate |
| --- | --- | --- |
| #24, #8 | Shared/local bindings, bounded executable multi-leaf expressions | CALM graph execution, arbitrary recursion and learned intermediate predictors |
| #3, #5 | Feedback-dependent finite frontier; exploration after failed candidates | Proposer-program evolution and two executed policy changes |
| #6, #11 | Existing price objective/worker, frozen chronological multi-fold comparisons | Independent real prospective study and changes to older trading paths |
| #28 | Registered retained-news reuse and automatic indicator instances | Live provider search, general/full-text extraction and historical coverage |
| #12 | Complete trial history, comparable per-fold reports and explicit unavailable diagnostics | Calibrated overfitting diagnostics, trading cost stress and applicable Sharpe/PBO methods |
| #1 | CLI, recovery, source/evidence/model lineage and clear outcomes | Independent prospective acceptance and the broader roadmap |
| #2, #4, #7, #9, #10 | Existing gates/objectives preserved | Production promotion, calibrated inference, multi-horizon and costed trading evaluation |

No related issue is automatically closed by this engineering slice. No live collection, paid/model API calls or predictive-performance experiment is part of its test evidence.
