# Single-target strategy replay

The RFC #39 adapter in `execution/strategy_replay.py` converts verified forecasts
into a costed, hypothetical long/cash account and a separate same-target
buy-and-hold account. It calls the existing `execution_account` functions for
orders, affordability, lot rounding, fills, fees, cash, holdings and valuation.
It does not fit models, verify worker authority, acquire data, persist a journal
or submit real orders. The workflow controller owns those evidence and lifecycle
boundaries.

## Frozen policy and comparable accounts

The workflow freezes the threshold, five-session horizon and `AccountRules`
before fitting. A finite prediction strictly above the threshold requests long;
a nonpositive/below-threshold or missing prediction requests cash. A long signal
enters only while flat. An existing position is retained without topping up a
partial fill. Cash requests sell held units at the next registered open, subject
to the same costs, fill cap, cash and unresolved-input rules as every other order.
A capped sale can leave inventory; the report retains it.

Buy-and-hold independently requests one entry at the first origin, regardless of
forecast sign or availability. Both accounts use identical calendar dates,
initial capital, allocation, raw-price basis, lot size, costs, fill caps and
terminal close valuation. Both preserve remaining inventory at the end; neither
receives an artificial free liquidation. The shared account's internal
`cash_zero_interest` comparison label is unchanged. The adapter's separate
`buy_and_hold` account is the actual same-target benchmark.

A five-session forecast predicts origin-close to target-close change. Its order
fills at the following registered open. For example, a positive origin `s80`
forecast followed by a missing `s85` forecast requests entry at `s81` open and
exit at `s86` open. The forecast's `s80`–`s85` close horizon and the trade's
`s81`–`s86` open interval are different. The final episode retains inventory and
marks it at the terminal close.

## Historical fold replay

```python
report = replay_strategy(
    inputs.target[0],
    verified_prediction_rows,
    rules,
    origin_sessions=["s80", "s85", "s90"],
    end_session="s95",
    threshold="0",
)
```

Rows bind `origin_session`, `score` (or `None` and `reason`), `model_ref`,
`result_ref` and `feature_as_of`. Additional provenance fields are retained.
The caller must verify those references against the experiment's committed
results; an arbitrary caller-supplied dictionary is not research evidence.
Origins must be a contiguous step-five registered calendar grid. An omitted row
becomes an explicit missing/cash decision. Every registered close from the first
origin through the last origin plus five is retained, including unresolved
valuations. Historical open resolution uses the registered decision delay after
the open; close valuation uses the same delay after the close.

Historical confirmation is simulated at the origin decision time. The report is
labeled `historical_development_simulated_issuance`: it does not claim that a
model fitted today actually existed at those historical dates. Model/result
references and original metadata remain available for audit. The controller
reports folds as separate funded episodes, with any unmodeled gaps explicit;
these episodes must not be stitched into a continuous out-of-sample equity curve.

## Incremental paper observation

`advance_paper_strategy` supports one frozen publication and its five-session
future calendar interval. In addition to the row fields above, it requires
`available_at` and `confirmation_at` from the actual publication. Its source
knowledge basis is always `observed_receipts`.

1. Call with a trusted `observed_at` before the first target open. The function
   returns prepared hypothetical orders. Persist that state in the workflow's
   existing journal.
2. Call again with that committed state as `previous` and a new trusted clock
   witness. This confirms submission. Persist the confirmed state. Initialization
   or confirmation at/after the target open never backdates an order.
3. Supply subsequent verified snapshots, `previous` and `observed_at`. The
   controller checks receipt provenance and append-only lineage, persists each
   assessment, and connects it to the existing forecast outcome settlement.

No open or close is processed twice. An open can wait for its receipt until its
session close; if still missing, its order remains unresolved and is not silently
filled when a later receipt arrives. A missing close waits until the next
registered open (or the terminal session's decision time when no later calendar
session exists), then records an unresolved valuation. A revision to previously
consumed evidence suspends the account and suppresses aggregate return metrics;
previous fills and valuation entries remain unchanged. The returned state is
serializable; the controller, not this pure adapter, owns durable idempotency.

A future calendar can extend beyond the historical rows in `ForecastInputs`.
This is necessary to observe an actually issued forecast without inventing its
future outcomes. Clock-controlled tests establish the connection mechanically;
they are not elapsed prospective performance evidence.

## Report interpretation

Reports retain decisions, order and fill references, cash, positions, fees,
marks and every valuation date. Net return is terminal marked NAV divided by
initial cash minus one. Maximum drawdown uses the shared account's running high
water mark. Turnover is absolute raw fill notional divided by initial cash.
Mean close exposure is the mean marked invested fraction over post-origin
valuation dates, not a continuous-time estimate.

Any unresolved valuation suppresses whole-episode net return and maximum
drawdown even if a later mark is complete. Known fill costs and cash/holdings
remain visible. Paper episodes also suppress terminal return metrics while the
five-session observation is pending or submission was late. Unresolved actions
preserve liabilities; the report does not claim dividend-adjusted total return.
A negative result, no trade or insufficient evidence is a valid result.
