# Versioned Forecast Inputs and Feature Recipes

This interface is the first implementation slice after the engineering forecast baseline. It performs no collection, model fitting, selection, or publication. The original three-feature inference path and existing forecasts remain unchanged.

## Frozen inputs

ForecastInputs wraps one existing LocalResearchData envelope for the target and independent envelopes for optional industry and market contexts. Each source has its own ForecastObjective for currency, raw-price units, and source-bound corporate-action interpretations. Evidence bytes are supplied as hash-to-hex mappings and checked on admission. Context instruments are explanatory inputs, never additional forecast targets.

Every input role contains exactly one instrument. The initial interface requires a common currency and provides no implicit FX conversion. Context bars align to the target calendar's exact registered closing timestamps; matching text session IDs or filling from a later/previous session is insufficient. Availability checks use each source's declared historical basis or observed receipts, explicitly recorded in the feature view.

## Ordered recipes

| Recipe | Ordered features | Count |
| --- | --- | --- |
| B0 | Existing momentum-5, one-session close return, and open-to-close return | 3 |
| B1 | B0, 20/60-session returns, and 20/60-session return volatility | 7 |
| B2 | B1, industry then market 5/20-session returns and target-minus-context returns | 15 |

Volatility is population standard deviation of simple daily returns, without annualization. The recipe hash binds names, order, units, windows, and conventions. build_feature_view(inputs, recipe, origin, as_of) returns eligibility, ordered values, source references, and knowledge boundaries. ordered_feature_values checks the entire input/recipe/order/unit binding before model consumption; it rejects missing, nonfinite, or incorrectly sized inputs.

B0 preserves the original numerical definition. B1 includes all B0 columns, and B2 includes all B1 columns. Unknown actions, missing closes, unavailable context, or inadequate history produce explicit ineligibility instead of zero-filled features.

## Shared masks before fitting

build_common_mask(inputs, recipes, origin_indices, label_asof, purpose=...) records all supplied origins, every recipe's feature-view reference, the target label, and exclusions. Only origins eligible for every recipe and with mature labels enter the shared mask. The mask and its ordered origins are content-addressed. It never inspects prediction errors.

Training label maturity remains governed by the existing five-session kernel: a label from origin u is usable at training boundary b only when u+5 is no later than b and all source information is available. Origins beyond the label knowledge boundary are rejected. Mask exclusions remain visible, and publication or evaluation code must not recompute a favorable mask after seeing scores.

The [paired comparison workflow](forecast_comparison.md) uses the separately frozen origin grids 378..498, 504..624, and 630..750, in steps of five. This module does not launch those experiments or reserve their proposed fit budget.

## News, fundamentals, and coverage

SignalEvent records source, entity, event identity/revision, query hash, publication time or date, precision, source availability, actual first receipt, report-period end, original payload/receipt references, extractor version, extracted string fields, and units for every field. Fields such as SEC filing and acceptance times retain their actual names; neither is silently relabeled as proven publication.

Publication precision is second, day, or unknown. A date-only record has no invented timestamp. Visibility is gated by the latest known publication/source-availability/first-receipt timestamp; retrieval of an older article does not make it visible before its real receipt.

SignalCoverage is specific to source, entity, query, and the exact [start_at, end_at) interval. It records complete, partial, or unknown status, actual receipt, and a reason when incomplete. A later coverage revision affects only later knowledge boundaries. Complete empty coverage permits a count of zero; missing or partial coverage yields None. Date-only or unknown publication times remain explicitly unresolved and do not become a precise-window news count.

signal_view is a pure view of a supplied snapshot. It does not certify externally supplied clocks or publication claims. A source adapter must capture actual receipt times and preserve the original responses and query evidence. Historical backfill is not automatically point-in-time data.

## Validation

Run the two focused fixture suites:

~~~bash
python -m pytest -q tests/test_forecast_inputs.py tests/test_forecast_features.py
~~~

They cover legacy-feature equivalence, nested recipes, future-price perturbations, independent calendar IDs, late/missing context, shared warmup/maturity masks, preprocessing mismatches, evidence corruption, receipt-gated news, coverage identity and revisions, and date-only publication. These are interface and causality checks, not predictive-performance evidence.
