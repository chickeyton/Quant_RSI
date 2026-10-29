# Retained KO market account diagnostic

The first attempt to reuse real-market forecasts in the merged strategy adapter
produced **insufficient evidence, not a usable return comparison**. This is a
real-use finding for [RFC #39](https://github.com/natureofnature/Quant_RSI/issues/39)
and [#40 G0](https://github.com/natureofnature/Quant_RSI/issues/40). It should keep
the real-data usability gate open even though the synthetic workflow works.

## Source and frozen operation

The October 29, 2026 diagnostic reuses the previously exposed KO forecasts from
the [enhanced market experiment](enhanced_forecast_results.md). It performs no
collection, new fitting, candidate selection or publication-model backfill.
All nine saved evaluation jobs are retained: three folds times B0/B1/B2. These
recipe labels belong to the old experiment, not the newer B0/D1 discovery pair.

The original experiment is `r1-ko-20261027-v1`, in campaign
`r1-market-proof-20261027`. Its preserved source is
`a4d7b4d27ff193d070742aee7acb415bb06f0fa2`; its runtime reference is
`1a3700bf54f6cd9693f108b410658fab8eec9ebf8ec8b73a44dc7ae0be1dca61`.
The original environment was checked against every recorded source/dependency
identity before replay. The new pure-account adapter uses merged `main@1af6e1b`.
The old workspace's identity was not rewritten to pretend it ran new code.

The extraction policy was saved before account performance was computed:

- Input reference: `313d5d70b49716687006b1eec36bbde16ac1f1d0c18fcbc1d9fff408254eccab`.
- Protocol reference: `149057a27020f0e1e17eb4838740b69861e67e4a860134c086894f5c1e258e0c`.
- Keep all 25 original step-five origins per fold: 378–498, 504–624 and 630–750.
- Mark every close through the last origin plus five: 126 valuations per fold.
- Independently initialize USD 10,000 per arm/fold; threshold zero, long-or-cash,
  unit lots, 10,000-unit fill cap, 5 bps commission plus 5 bps slippage per side,
  20% drawdown suspension and inventory-preserving terminal valuation.
- Apply identical accounting assumptions to the strategy and buy-and-hold arm.
  Keep every missing/action-related restriction. Do not concatenate fold equity.

## Observed outcome

| Fold | Evaluation recipes retained | Prior action notices | In-episode dividend notices | Valuations per episode | Strategy and buy-and-hold |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | B0, B1, B2 | 14 | 2 | 126 | insufficient_evidence |
| 1 | B0, B1, B2 | 16 | 2 | 126 | insufficient_evidence |
| 2 | B0, B1, B2 | 18 | 2 | 126 | insufficient_evidence |

All nine episodes and eighteen arms remain in the report. All had zero fills;
aggregate net return and maximum drawdown are **null**, not zero. Zero turnover
here means suspended execution, not a valid finding that cash beats the model.
No arm supplies an interpretable costed performance comparison.

The retained snapshot has 756 open and 756 close observations and 20 cash-dividend
notices. Its corporate-action coverage is explicitly `unknown`. The current
adapter passes all known action notices to the account's unresolved-action
suspension. Notices before a new episode therefore already prevent entry. More
importantly, each episode also contains two dividend notices: merely filtering
the old notices would still not justify complete returns. Neither deleting
notices nor relabeling raw prices as total-return data is an acceptable fix.

## Verification and cost

The original saved models were replayed under their exact original runtime in
2.155 seconds of subprocess wall time. The derived diagnostic measured 35.406
seconds inside its process for runtime checks, input parsing, policy recording
and nine account calculations; that excludes interpreter startup and final JSON
writing. These intervals have different scopes and are not comparative benchmarks.
The raw report is
about 1.47 MB, so inspection used its small summary instead of console dumping.

Before/after content hashes of all **13,774** original workspace files matched.
The original KO ledger remained at ten fit reservations and 1,800 reserved timeout
seconds. Additional fits, additional reserved seconds and newly collected data
were all zero. Old identities, evaluation models and original reports were retained.

Local evidence is retained in `/tmp/rfc39-retained-ko/`: the extraction policy,
input/protocol envelopes, saved evaluation references, original replay and
verification receipts, `replay_accounts.py`, account policy/runtime,
`account-report.json`, `account-summary.json`, `derived-run-receipt.json` and
`derived-verification.json`. The preserved source workspace is
`Quant_RSI_worktrees/configurable_forecast_workflow/runs/r1_market_proof_20261027`.
These local source artifacts are not included in a fresh clone; this report is
a recorded operator diagnostic, not a portable public benchmark.

## Next decision

Real-use evidence supports a narrower next task: define account episode boundaries
and a supported corporate-action policy before broadening factors or models.
Separate irrelevant pre-episode notices from actions affecting positions/orders
inside an episode; require sufficient action amount/timing/coverage evidence for
any supported cash or share adjustment. Missing information must still produce
an explicit unresolved result. Freeze that task and its fixtures before using
the retained data to evaluate the change.

This report does not show a losing model, a profitable strategy or independent
forecast validity. It shows that the current conservative account contract cannot
yet provide the intended real-market comparison on this retained KO task. That
is useful acceptance evidence and a concrete reason to postpone broader framework
expansion.
