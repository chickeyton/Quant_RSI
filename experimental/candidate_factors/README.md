# Candidate factor survey

This is an experimental, source-pinned collection for [RFC #31](https://github.com/natureofnature/Quant_RSI/issues/31).
It supplies factor semantics and execution requirements while the RFC owns the
runtime contracts, materializer and discovery lifecycle. Nothing here is loaded
by the production factor catalog, registered as a model feature or promoted.

The bounded discovery baseline introduced in PR #32 is documented in the
[factor discovery guide](../../docs/factor_discovery.md). This survey does not
extend its operator grammar or automatically bind any of these candidates.

## Contents

| Source | Detailed candidates | Raw inventory entries | Selected-file license |
| --- | ---: | ---: | --- |
| Microsoft Qlib | 11 | 54 expression templates | MIT |
| Goldman Sachs GS Quant | 6 | 57 public functions/classes | Apache-2.0 |
| QuantConnect LEAN | 5 | 167 indicator-directory filenames | Apache-2.0 |
| QuantDinger | 10 | 63 built-in declarations | Apache-2.0 |

The 341 inventory entries are **not 341 distinct validated factors**: helpers,
parameterized expressions and duplicates remain in the raw survey. Only the 32
[detailed candidates](catalog.md) have semantic annotations; one is explicitly
noncausal and reference-only. This is a first collection, not an exhaustive import.

- [sources.json](sources.json): exact upstream commits, source-file SHA-256 values
  and preserved license/notice fingerprints.
- [inventories/](inventories/): static indexes, extracted without importing upstream
  Python or running third-party software. LEAN's broad index is filenames only;
  license/file-level review is limited to the seven listed source files.
- [definitions/](definitions/): survey records with source anchors, example
  parameters, input/output units, state, timing constraints and unresolved edges.
  `survey_version` describes this research metadata, **not** a second runtime schema.
  Example parameters are not a claim about all upstream defaults.
- [execution_modes.md](execution_modes.md): requirements and suggested small
  integration sequence for #31, including gaps this collection does not cover.
- [fixtures/](fixtures/): invented arithmetic oracles and proposed integration
  contracts. No real price datasets or upstream example datasets are included.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/):
  attribution and verbatim license/notice files; retain these with redistribution.

## What is verified

The automated checks validate reference integrity, pinned URLs, local license
hashes, inventory links and arithmetic consistency of eight synthetic examples.
With local upstream snapshots, a second audit verifies their commits and original
source/license bytes against the manifest. Upstream packages have **not** been
installed or executed; computational conformance is `not_run` and predictive
quality is `not_assessed` for every candidate. The integration-contract cases are
requirements for future adapter tests, not completed materialization tests.

Run from the repository root with Python 3.12 and the development environment
installed as described in the [project README](../../README.md):

```bash
python -m pytest -q tests/test_candidate_factor_survey.py
```

By default the optional snapshot audit is skipped. To reproduce it, create four
local checkouts under a cache directory named exactly `qlib`, `gs-quant`, `Lean`
and `QuantDinger`. Use each repository URL and full commit from `sources.json`;
checkout the listed source and license files (not just the current default branch).
No network access is performed by the tests:

```bash
QUANT_RSI_UPSTREAM_CACHE=/path/to/checkouts \
  python -m pytest -q tests/test_candidate_factor_survey.py
```

## Collection and reuse rules

1. Pin a commit before collecting; inspect the actual selected files, license
   headers, applicable subtree/REUSE declarations and NOTICE files. Repository
   popularity and GitHub's top-level license label are insufficient.
2. Record source paths and byte hashes. Keep original notices and licenses. These
   JSON files select/restructure source metadata and paraphrase semantics; Qlib
   inventory expression strings and QuantDinger declaration metadata are extracted
   from the source. The upstream executable modules are not vendored.
3. Keep unclear-license or restricted code out of the import. Optional TA-Lib/Qlib
   adapters in QuantDinger do not grant licenses to their dependencies; those
   implementations are not included here. No claim is made about unselected
   QuantDinger frontend/mobile files or other third-party assets.
4. Give materially different seeds, windows, price bases and outputs distinct
   identities. Qlib BETA is a price/time slope, unlike GS market beta. Qlib ROC is a
   reciprocal price ratio, unlike standard return. LEAN VWAP uses OHLC4; QuantDinger
   uses HLC3 and defaults to distance. These names are not equivalence proofs.
5. Implement through #31 only after fixing missingness, readiness, availability,
   units and dependency versions. Start with deterministic fixtures, then upstream
   equivalence and point-in-time/replay tests. A failed predictive experiment is a
   valid recorded outcome, not grounds for deleting the candidate.

Accepted reusable definitions belong in `instruments/shared/`; instrument-specific
bindings belong under `instruments/<MARKET>/<SYMBOL>/`. Keep rejected or deferred
survey records here with their provenance. Promotion, training, and profitability
claims are separate decisions; collecting a factor authorizes none of them.
