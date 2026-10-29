# Intraday Bars and IBKR Paper Trading

This guide covers the standalone intraday path described in the
[design](intraday_ibkr_design.md):
- collect IBKR intraday bars into immutable snapshots;
- evaluate a fixed LightGBM leaf set with the replay simulator;
- run a prospective episode against an IBKR **paper** account.

Section 7 of the design has diagrams of the RSI cycle: the steps implemented so far, one
paper session, who sees which outcomes, the forward schedule, the two decisions, and how
one episode hands over to the next.

It is a development facility. Engineering fixture results do not establish predictive
quality, and paper fills are simulated by IBKR.

## What is and is not connected

| Part | Status |
| --- | --- |
| IBKR session, paper-only guard, pacing, contract qualification | Implemented; tested offline with a scripted fake transport |
| Intraday snapshots (decision, 1-minute execution and bid/ask grids) | Implemented; deterministic content-addressed column blocks |
| Timing, labels, features, fixed LightGBM, session folds | Implemented |
| Learnable trading policy: cost-aware entry, edge-scaled sizing, nested calibration | Implemented |
| Replay simulator and development evaluation | Implemented; `development_only` evidence |
| Paper episodes: journal, scheduled submission, deadlines, flatten, stop and handover, reconciliation, risk stops, shadow report | Implemented |
| Forward session schedule and temporal-ordering check | Implemented as library functions |
| C2/C4/C5 controller integration, sealing registry, budget ledger, `quant-rsi` subcommands | **Not yet**: they depend on the POSIX controller and need a Linux environment |
| IBKR daily data into the five-session forecast envelope | Daily collection is implemented; envelope integration with a Yahoo action-notice sidecar is not |

The intraday modules do not import the POSIX-only controller, so they run on Windows as well
as Linux. The rest of Quant RSI still requires Linux.

## Prerequisites

1. Install the project as described in the [README](../README.md).
2. Install the IBKR client from your TWS API distribution. It is GPL-licensed, so it is
   never copied into this repository:

   ```bash
   python -m pip install ".[ibkr]"
   python -m pip install "/path/to/TWS API 1050.02/source/pythonclient"
   ```

3. In TWS or IB Gateway, logged in to the **paper** account:
   - enable socket clients;
   - turn read-only API off (orders are needed);
   - note the paper port (7497 for TWS, 4002 for Gateway).
4. Copy [`examples/ibkr.json.template`](../examples/ibkr.json.template) to
   `.quant_rsi/ibkr.json`, which is ignored by Git. Set `expected_account` to your `DU…`
   paper account.
5. Market data: API historical bars need the matching IBKR subscriptions, shared to the
   paper account. Use `--delayed` only for development snapshots.

## Commands

All commands print JSON. Paths are examples.

```bash
# Connection, clock skew and paper guard (places no orders)
python -m quant_rsi.intraday ibkr-check --config .quant_rsi/ibkr.json

# Qualify the profile's instruments to exactly one IBKR contract each
python -m quant_rsi.intraday resolve --config .quant_rsi/ibkr.json \
  --profile instruments/US/KO/intraday.json --output runs/ko/bindings.json

# Collect completed sessions into a snapshot
python -m quant_rsi.intraday collect --config .quant_rsi/ibkr.json \
  --profile instruments/US/KO/intraday.json --bindings runs/ko/bindings.json \
  --calendar runs/ko/calendar.json --store runs/ko/store

# Development evaluation, then a fit on the whole snapshot
python -m quant_rsi.intraday evaluate --store runs/ko/store --snapshot SNAPSHOT_SHA256 \
  --plan examples/intraday_development.json
python -m quant_rsi.intraday fit --store runs/ko/store --snapshot SNAPSHOT_SHA256 \
  --plan examples/intraday_development.json --output runs/ko/model.txt \
  --policy-output runs/ko/signal.json
```

### Learned trading policy

The model only predicts the return over the next H decision bars (30 minutes with the
example plan); the trading policy decides whether and how
much to buy (design section 5.7). With `policy_search` in the plan, the policy is learned
from a registered grid:
- **`evaluate`:** each fold learns its policy from its own training sessions. It holds out
  the last `calibration_sessions`, replays every candidate on them, and scores its test
  sessions with the winner. Each fold's `policy` entry shows the winner and every
  candidate's calibration result.
- **`fit`:** learns the policy on the snapshot's last `calibration_sessions`, then fits the
  final model on all sessions. It prints the chosen `signal` and `signal_sha256`, and
  `--policy-output` writes the `signal`. Copy both into the paper plan; a `signal` that
  does not match `signal_sha256` is refused.

The example plan uses `entry_rule: net_of_cost`, which only buys when the prediction
exceeds the estimated round-trip commission and spread plus `entry_threshold`. Without
`policy_search`, the plan's `signal` is used as written. Declined entries are counted as
`policy_declined_entries` in evaluation and journaled as `entry_declined` on paper.

Two profiles are checked in, each with a matching development plan:

| Profile | Traded instrument | Context | Development plan |
| --- | --- | --- | --- |
| [`instruments/US/KO/intraday.json`](../instruments/US/KO/intraday.json) | KO (NYSE) | XLP, SPY | [`examples/intraday_development.json`](../examples/intraday_development.json) |
| [`instruments/US/SPY/intraday.json`](../instruments/US/SPY/intraday.json) | SPY (ARCA), for S&P 500 exposure | QQQ, IWM | [`examples/intraday_development_spy.json`](../examples/intraday_development_spy.json) |

The SPY plan assumes a 0.5 bps spread and uses lower entry and exit thresholds than KO,
because SPY's spread and 30-minute moves are both smaller. These are starting assumptions;
replace `assumed_spread_bps` with the spread measured from your bid/ask snapshot. The
`ctx:` leaves in a plan must name the profile's context instruments.

`calendar.json` is an explicit `CalendarSpec`: calendar ID, IANA time zone and every
session's open and close. The collector never infers sessions, and it refuses sessions that
have not closed yet. Each raw IBKR response is stored with its request before
normalisation.

How collection is organised:
- **Requests:** only 1-minute bars (plus bid/ask when configured), one request per
  instrument and session. The 5-minute decision bars are aggregated locally from the
  1-minute bars.
- **Duration:** with the KO profile (3 instruments, 250 sessions, bid/ask) that is about
  1,500 requests, roughly 5 hours at the conservative pacing limit.
- **Resuming:** an interrupted run resumes. Re-running the same command reuses every stored
  response with its original first-seen receipt, and requests only what is missing.
- **Pacing violations** back off and retry.

Multi-day requests are not used, because IBKR's maximum duration for 1-minute bars has not
been verified against a live session.

### Paper episode

Copy [`examples/intraday_paper.json.template`](../examples/intraday_paper.json.template)
and fill in:
- the snapshot reference;
- the model SHA-256 printed by `fit`;
- the `signal` and `signal_sha256` printed by `fit` (or the `--policy-output` file);
- the schedule's `anchor_monday` (the Monday of the first paper week);
- `expires_at`.

Then:

```bash
python -m quant_rsi.intraday paper-register --config .quant_rsi/ibkr.json \
  --store runs/ko/store --paper-plan runs/ko/paper.json --model runs/ko/model.txt \
  --journal runs/ko/paper.sqlite

# Run one regular session; session.json is today's MarketSession
python -m quant_rsi.intraday paper-run --config .quant_rsi/ibkr.json \
  --store runs/ko/store --paper-plan runs/ko/paper.json --model runs/ko/model.txt \
  --journal runs/ko/paper.sqlite --session runs/ko/session.json --request-id run-20261030

python -m quant_rsi.intraday paper-status --journal runs/ko/paper.sqlite --episode ko-intraday-p1
python -m quant_rsi.intraday paper-stop --journal runs/ko/paper.sqlite \
  --episode ko-intraday-p1 --request-id stop-001
```

## Operating rules enforced by the code

- **Paper only.** Orders are refused unless all of these hold:
  - the port is a registered paper port;
  - every managed account starts with `DU`;
  - the expected account is managed by the session;
  - the session is the registered order client.

  There is no option to disable this check.
- **One active episode per paper account.**
  - Registration is refused while another episode on the account is registered, active,
    stopping or suspended.
  - Registration is also refused if the account already holds any of the episode's
    instruments.
  - To hand over, stop the old episode. It finishes its session flat and terminates.
- **Timing.**
  - Decisions happen `decision_latency_seconds` after each 5-minute bar ends.
  - Orders are held and submitted at the start of the next 1-minute fill bar.
  - An intent not submitted before its deadline is never sent late.
  - Entries stop so that every planned exit fits before the flatten time. Any position
    still open at the close suspends the episode.
  - A position carried over a close exits at the next session's first opportunity.
- **Crash safety.** An intent is committed as `submitting` before it is sent.
  - After a crash it is never re-sent automatically.
  - Reconciliation matches it to IBKR's orders and executions by `orderRef`.
  - If IBKR has no record by the deadline, it is abandoned.
  - A broker order still working for an intent the episode gave up on is cancelled.
- **Cancels are not final until IBKR confirms them.** At the deadline a working order
  moves to `cancelling` and still counts as active, so no replacement order is sent while
  the old one could still fill. If the old order fills instead, that fill is kept.
- **Order-ID error codes.**
  - Only genuine rejections (for example 201 and 203) mark an order rejected.
  - 202 confirms a cancel.
  - Warnings such as 399 and cancel-not-possible codes such as 161 and 10148 are notices;
    the order's status reports remain authoritative.
  - A late error never overwrites a final status.
- **Ledger consistency.** Fills are applied in the order received. A sell with no
  position, or larger than the position, suspends the episode as `position_mismatch`.
- **Re-checks at submission.**
  - Open orders, executions and positions are reconciled before each submission.
  - Each entry is re-checked when it is submitted. A risk block, a stop or the flatten time
    arriving after the decision abandons the queued entry.
  - A stop cancels working entries and exits open positions at the next fill bar.
- **Reconnects.**
  - `paper-run` reconciles before trading and after every reconnect.
  - After a (re)subscription, bars are not finalised until IBKR's backfill has arrived, or
    `max_lag_seconds` passes (recorded as `backfill_timeout`). Restarts therefore don't
    turn past bars into stale ones.
  - Connectivity lost (1100) blocks entries.
  - Connectivity restored (1101 or 1102) triggers a reconcile; 1101 also renews the data
    subscriptions.
  - A disconnect is a pause: the runner reconnects, re-checks the paper guard, reconciles,
    and re-subscribes in recovery mode.
  - Replayed bars fill the panel but are excluded from the arrival-delay statistics.
    Decisions whose submission window has already passed are skipped.
  - If reconnecting keeps failing, the episode is suspended.
- **Restart-safe risk state.** The day's starting equity and the reject streak are read
  from the journal, so a restart doesn't reset the daily-loss stop.
- **Previous sessions.**
  - Each live session's bars are archived at the close and used as previous sessions for
    `gap` and `vol_z_20`.
  - If the latest known session is more than 5 days old, those leaves are missing rather
    than computed from stale history.
- **Missing data.**
  - A missing feature records `no_prediction` and opens nothing.
  - A silent feed records `stale_feed` and blocks entries.
  - The episode records how late each bar arrives, and flags
    `availability_assumption_violated` when its 99th percentile exceeds the buffer the
    backfill assumed.
- **Risk stops** block entries but never exits:
  - gross and position notional;
  - order rate;
  - daily loss;
  - consecutive rejects;
  - clock skew;
  - stale feed;
  - model expiry;
  - planned restart window.

## Tests

```bash
pytest -q tests/test_intraday_core.py tests/test_ibkr_adapter.py \
  tests/test_intraday_paper.py tests/test_intraday_schedule.py tests/test_intraday_cli.py \
  tests/test_intraday_review_regressions.py tests/test_intraday_second_review.py \
  tests/test_intraday_policy.py
```

These tests use synthetic bars and a scripted fake TWS. They need neither `ibapi` nor a
running TWS, and they check engineering behaviour only. A manual check against a live paper
session is still required:
1. Run `ibkr-check`.
2. Collect one session.
3. Place and cancel one order.
4. Reconcile.
