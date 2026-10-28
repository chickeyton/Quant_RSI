# Retained-corpus news factor: first bounded slice (RFC #28)

This opt-in implementation connects **KO → registered retained-corpus search → typed
headline evidence → one scalar → the existing paired worker experiment**. It is not a
live search provider, an article downloader, or evidence of a useful trading signal.
Legacy B0/B1/B2 comparisons and their version-1 identities remain unchanged.

## Reviewed decisions (before research)

- **Instrument:** KO, already supported by the existing forecast adapter. No exchange
  support is changed; CALM and NVDA are not enabled by this work.
- **Source:** `retained_issuer_news`, capability `retained_corpus_search`. A caller supplies
  an already retained `SignalSnapshot` and its original content-addressed receipts,
  queries and JSON title payloads. No credential, paid service, polling substitution,
  network request, or retention permission is implied. The caller must have permission
  to retain the input. Coverage means **only this supplied corpus**, never all issuer news.
- **Task:** [shared typed task](../instruments/shared/issuer_title_task.json), with
  [KO aliases and source binding](../instruments/US/KO/news.json). The finite rule matches
  exactly an issuer alias followed by `Reports <quarter> Quarter <year> Results`.
  Aliases are escaped literal strings, not arbitrary executable patterns. The rule
  extracts a boolean title indicator and a semantic issuer/year/quarter event key;
  it does not infer earnings values, confirm a real-world event, or assign confidence.
- **Search:** deterministic entity/source query, one bounded locally searched page,
  ordered by document ID and revision. The requested publication interval selects
  documents, then retains their visible revision lineage. Unknown publication times
  remain unknown. No adaptive query generation or unrestricted source discovery.
- **Allowances:** default cumulative campaign cap 128 document revisions, 1 MiB of
  selected original source evidence (payloads, receipts and query records), 128 rule
  calls. Upper schema bounds are 1,024 documents/calls and 16 MiB. Generated result,
  query and extraction metadata is additional local artifact storage, not provider
  bytes. Network requests are exactly zero; unknown token/monetary costs are `null`,
  not invented zero-dollar receipts. Fit reservations remain separate worker seconds.
- **Feature:** `issuer_earnings_release_count_7d`, a deduplicated count of positive
  registered title indicators over `[cutoff - 7 calendar days, cutoff)`. The cutoff is
  the latest source-coverage endpoint visible at the decision, no older than 24 hours;
  extraction can finish between cutoff and decision without backdating evidence. Same
  issuer/year/quarter is counted once across duplicate titles or metadata variants.
  Different reports and revision evidence remain preserved. This is not a general
  semantic syndication, contradiction, or retraction resolver.
- **Missingness:** complete visible corpus coverage with no match gives zero. Partial,
  missing, unknown-time, malformed, rejected, unavailable, or late-derived evidence
  excludes the row from **both** arms. Date-only publication is conservatively
  unresolved, never assigned an invented midnight. An explicit source quantity zero
  is not extracted by this boolean task and cannot be confused with its count.
- **Availability:** observed receipts **and actual trusted extraction completion**.
  A newly searched historical corpus cannot fill historical training rows, even when
  empty. Production uses the existing trusted clock; only offline fixtures replace
  it with a controlled clock. Accumulate prospective observations before comparison.
- **Comparison:** schema-version 2 opt-in pair B0 versus N1 (B0 plus one ordered count
  column), identical pre-fit common masks, mature labels, controls, model settings,
  folds, and 180-second reservations per fit. Selection is development-only. The
  winning baseline, a zero improvement, and insufficient evidence are valid outcomes.

## Python API

There is deliberately no live collection CLI. This new API requires an **editable checkout**
(`pip install -e .`) with its `instruments/` directory retained; the new task configuration
is not bundled into a regular wheel. Existing wheel-based legacy forecast paths are unchanged.
From that editable checkout and a configured
absolute `QUANT_RSI_WORKSPACE`, use the same artifact store and input envelope interfaces
as the existing forecasts:

```python
from pathlib import Path
from quant_rsi.core.artifacts import ArtifactStore
from quant_rsi.factors.news import NewsRequest, run_news, attach_news_result
from quant_rsi.forecast.forecast_experiment import prepare_experiment, resume_experiment

# snapshot: existing SignalSnapshot with retained_issuer_news / KO records.
# evidence: {sha256: raw_bytes}, including original payload/query/receipt evidence.
# times: actual observed interval and as-of, not backdated acquisition claims.
request = NewsRequest.model_validate_json(request_json)
store = ArtifactStore(Path(workspace) / "news_artifacts")
result = run_news(request, snapshot, evidence, store)
forecast_envelope = attach_news_result(forecast_envelope, result)

# Accumulate enough observations with distinct request IDs in the SAME campaign.
# Then freeze exactly one comparison; no fits are performed by preparation.
prepared = prepare_experiment(
    forecast_envelope, "ko-news-first-pair", news_factor=True,
    fold_origins=((80, 85, 90),),  # illustrative only; pre-register your own grid
)
# If prepared["status"] reports insufficient_evidence, stop: no fits were authorized.
# Only for a successfully prepared protocol, with an explicitly authorized fit budget:
report = resume_experiment(
    "ko-news-first-pair", protocol_ref=prepared["protocol_ref"], max_fits=3,
)
```

Use `request_id` for immutable query identity and `campaign_id` for cumulative search
and extraction accounting. Input documents must have JSON payloads containing `title`
identical to `SignalEvent.fields['title']`. This adapter does not reinterpret an arbitrary
RSS XML response as a JSON document: an original receipt and reviewed retained-document
representation must exist first. Content extent is always `retained_title_only`.

`attach_news_result` adds immutable evidence and preserves the first committed extraction
for an existing document revision. A changed payload needs a new revision. The resulting
version-1 `ForecastInputs` envelope uses its existing signal/evidence extension points;
new task/schema/configuration records are hash-bound, not silently ignored envelope keys.

## Authority, recovery, and replay

The news ledger lives only at the existing trusted authority's
`news_campaigns/<campaign_id>`. Output paths cannot locate or renew allowances. Its
anchor binds the canonical campaign and artifact-store paths and fixed limits. A missing
anchor or ledger fails closed. Intent and conservative usage reservations are atomically
persisted before rule execution, and each completed extraction is checkpointed. A crash
may repeat an uncommitted **local** rule under its existing reservation; it cannot trigger
an external call or uncertain provider charge. Terminal calls read the saved result.

Exactly **one** paired experiment identity/input/fold binding is permitted per campaign.
Renaming the experiment, changing the input/query version, or retrying an unsuccessful
initialization cannot mint another fit allowance. Changes need a deliberately new research
campaign, not an automatic retry. Resume/replay verifies the original registered search
outputs, frozen campaign receipt, and original experiment ledger. Existing managed enhanced
forecast campaigns are not silently broadened to admit these signals.

Search/extraction usage and comparison fit accounting are both retained; they have different
units. Later searches retain campaign cumulative usage but cannot replace a frozen comparison's
inputs, earlier extracted revisions, or historical feature views.

## Engineering acceptance and limitations

The offline tests include hand-annotated fixed headline examples, evidence tampering,
malformed output, future tense, adversarial instruction text, incomplete/empty coverage,
unknown publication time, late extraction, revised publication dates, duplicate semantic
headlines, request/allowance identity changes, interrupted output recovery, and terminal
idempotence. Integration tests pass actual pipeline evidence into the existing worker,
including a small synthetic LightGBM pair, and verify common-mask exclusions and replay.
These are invented engineering fixtures, **not** historical return experiments or alpha claims.

This is an initial A/B/C vertical slice, not completion of every RFC checkbox: there is no
live provider integration, full-article retention, generalized entity linking, quantity
extraction, general retraction interpretation, broad causal factor tree, independent
statistical confirmation, or production promotion. It cannot supply a historical corpus
that does not exist. Independent statistical confirmation remains separate under issue #4.
