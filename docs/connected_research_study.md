# Run a connected stock research study

After the [Linux/Python 3.12 installation](../README.md#installation-and-usage),
run from the repository root with a new output directory outside the checkout:

```bash
quant-rsi research-study KO --output /tmp/ko-study
```

An interactive terminal opens the live research UI automatically; the saved report is
also available at `/tmp/ko-study/report.html`. The default uses Codex CLI saved authentication,
KO's configured relationship profile, registered issuer RSS and public daily
prices. Install/login to Codex first; the existing LightGBM dependency performs
two actual fits by default. The smaller `research-demo` command retains its defaults.

The default factor library resolves the stock's shared/local bindings and bounded
compositions. `--factor-root` changes the factor configuration root; explicit `--catalog`
bypasses profile resolution. Source availability is checked on the common research
sample, and unsupported inputs stay pending. See [factor execution and provenance](factor_layers.md#execution-in-the-connected-study).

## Local workbench and recovery

```bash
quant-rsi research-ui --workspace /tmp/quant-rsi-workbench
# Keep the server in the terminal without automatically opening a browser:
quant-rsi research-ui --workspace /tmp/quant-rsi-workbench --no-open --port 8765
# Resume a CLI-created study directly:
quant-rsi research-resume /tmp/ko-interrupted-study --ui
```

The workbench binds only to `127.0.0.1`, prints a random-token URL, and opens the browser.
It accepts a comma-separated watchlist of up to 20 unique symbols, Codex/API/simulated
researcher, optional LLM model, LightGBM/Ridge, real/synthetic inputs, news-research switch,
2–10 rounds, 2–10 fits per stock per cycle, optional patience and MAE/Sharpe selection.
These limits use the same CLI iteration semantics (the smaller round/fit limit applies).
The optional **Research question** field accepts up to 1,000 characters and applies to
each stock in the cycle. It is preserved in saved templates, task cards, the live view,
reports and forecast history. Loading an older template clears the field to general research.
API mode requires a model name and
`OPENAI_API_KEY` in the server environment; Codex uses the existing local login.
No credentials are accepted or saved by the form. The page shows running/completed/failed
studies and queued/cancelled tasks, links to the existing animated graph and reports,
and offers **Continue study** and cancellation of queued tasks. One worker runs at a time
per workbench; additional tasks wait durably. Failed workers do not block the queue.
This is a local single-user tool, not a hosted job service.

Before a new-evidence research task starts its study, the task card and live view
show **collecting prices**, **collecting news**, **collecting financials** (when enabled),
and **checking new evidence**. These stages come from the worker's existing status file
and survive reopening the workbench. They describe the current operation, not a
percentage or a download heartbeat. Research nodes take over when the study starts;
skipped/completed tasks clear the preparation stage. A failed or interrupted preparation
retains its last stage alongside the error, without claiming that research has begun.

**Queue research** starts one watchlist cycle. **Save template** persists its name,
watchlist, settings and optional 1–168 hour recurring interval. **Check & run** starts a
saved template immediately and respects its evidence setting; **Force run** repeats
research even on unchanged evidence. Both work while scheduling is paused. Saving/enabling/editing
schedules the next cycle one interval from that action; it does not run immediately.
Recurring runs can consume real service/API budgets and require explicit enablement.
**Pause schedule** stops future cycles; cancel already queued tasks separately if needed.
Loading/editing a template affects future cycles only; queued inputs remain frozen.

Scheduling and queue dispatch require this server to remain running (use `--no-open`
for a headless terminal). Missed intervals coalesce into one cycle on reopening. A
template cannot start another cycle while its previous tasks are queued/running.
This is wall-clock interval scheduling, not an exchange-session scheduler; holidays
or missing prices may leave forecasts pending. There is no automatic retry of failed
paid calls: continue a resumable research task explicitly or run a fresh cycle.

Each cycle queues **observe → research** for each symbol. The separate observation
worker collects fresh public daily evidence when prior forecasts need assessment,
before any news/LLM work. No pending forecasts means no observation download; synthetic
inputs are never treated as real matured outcomes. Collection failures remain failed
tasks and do not manufacture outcomes or block research/other symbols. The existing
study path also observes its freshly collected prices and passes only settled outcomes
to the LLM as prior feedback. A later news/LLM failure cannot erase completed observations.
New-evidence mode checks prices (including revisions), the newest selected target/related
issuer RSS items, quarterly financial facts when enabled, and settled forecast outcomes.
It also fingerprints research settings, source/profile files and runtime dependencies.
Receipt timestamps and new pending forecasts alone do not trigger research. A matching
previous **completed** task causes a visible `skipped / unchanged_evidence` result before
LLM calls or fits; failed tasks do not consume the trigger. Source-check failures remain
failed tasks, never false "unchanged" results. `trigger.json` and `preflight/` retain the
decision evidence; checked prices/financials are reused by that study. RSS is fetched
again by the ordinary research path and may change in between.

This setting defaults on in the new browser form, off for legacy API/options/templates.
It is a periodic public-data check, not a trading-calendar close event or an exhaustive
web-news monitor. New web-only articles outside the selected RSS window may require
**Force run**. Observation still runs before this gate. Local notices show the latest
five completion, failure, interruption or skip transitions while the page is open;
they do not send email, push notifications or monitor a stopped server.

The comparison table shows each selected round's development MAE, unchanged-price MAE,
net Sharpe with daily sample count, net return, buy-and-hold and terminal availability.
Use the report for dates, settings, terminal metrics and risk-statistic limitations.
The separate outcome table shows the latest 50 published receipts per input mode,
including pending/unavailable evidence and actual matured errors. Task/comparison lists
show the latest 100 tasks; older directories remain intact. These tables do not pool
different stocks into a portfolio, an accuracy score or an independent benchmark.

Research workers call the existing CLI in `WORKSPACE/tasks/ID/study`; `worker.log` captures
output. Observation workers save `observation.json` and retained prices in their own task
directory. `task.json` freezes each queued job, `worker.json` records its exit, and
`plans.json` stores reusable templates. Shared forecast/news history lives under `WORKSPACE/history`, separated by real
and synthetic inputs. The workspace owns its task list: arbitrary old CLI directories are
not silently imported. Use `research-resume PATH` for such directories created by this version.
Closing the browser or stopping the workbench leaves launched workers running. Reopening
the same workspace restores the list and continues pending dispatch. An inherited task
lock covers worker startup as well as research, so reopening cannot launch an overlapping
worker during initialization. An absent worker with unfinished state is marked
interrupted, rather than remaining permanently "running". A run lock prevents two writers.

Recovery preserves the experiment configuration and completed inputs. Run-local JSON
checkpoints bind successful operations to their inputs, source/profile hashes and Python,
NumPy, LightGBM and Pydantic versions. No pickle or arbitrary serialized Python is loaded.
Completed issuer snapshots, searches, page reads, validated LLM hypotheses and fits are reused;
failed operations are retried on explicit continuation. The deterministic orchestration
reconstructs features and predictions from those receipts. Completed LightGBM boosters and
Ridge coefficients are restored without fitting again. An interrupted fit can run again.
Incomplete target collections remain intact; managed collection retries use a new directory.
Explicit external `--prices-dir` inputs are never silently refreshed or repaired.

Before continuation, old runtime/trace/failure records are copied to `attempts/N`. Current
`runtime.json` counts describe the current attempt; `previous_attempts` retains earlier
counts, reported/unknown tokens and elapsed times, while `reused_operations` identifies
cache hits. Failures still cost time/calls, and restarting does not erase those costs.
The fit allowance bounds logical candidate fits, not failed transport/fit attempts or money.
Completed runs are immutable. Runs from older versions without checkpoints cannot be resumed;
code, profile, configuration or bound-input changes require a fresh study. Keep the original
checkout/runtime available to continue an old interrupted run after an upgrade.

### Local Codex operation ownership

On Linux, each local Codex request has a small guardian process holding a separate
`.provider.lock`. Normal study unlock, Ctrl-C and owner SIGKILL cannot release that
operation's ownership. A continuation refuses while the guardian is cleaning up.
Timeout/owner loss terminate the provider process group and adopted detached descendants;
TERM escalates to KILL after 250 ms. Offline fixtures must disappear within five seconds.
If the OS cannot reap a process, the guardian keeps ownership rather than allowing an
overlapping retry. This does not provide a distributed lease or protection against
independently killing the guardian itself.

`provider_attempts/*.json` retains local start/finish/failure state without prompts or
credentials. `remote_completion` remains `unknown`: local termination cannot prove
server cancellation or reconcile billing. A local response followed by interruption
before checkpoint commit may be requested again on explicit continuation. Completed
checkpoints and previous attempt records remain intact; there is no exactly-once claim.

## Research question

```bash
quant-rsi research-study KO --news-research --financials \
  --question "Could inventory growth indicate weaker demand? Which available variables can test it?" \
  --output /tmp/ko-inventory-question
```

The trimmed question is frozen in the study configuration. Active news search, article
review and its bounded follow-up receive it, as do the initial hypothesis and every
feedback-driven revision. The researcher is asked to consider contrary evidence and
identify parts that cannot be measured with the supplied data. The existing factor
catalog and enabled sources determine what can execute; a question does not invent
new data or change the configured horizon, selection metric, tool permissions or budgets.
No extra LLM call or model fit is added just to carry the question through the flow.
Simulated providers still return fixture hypotheses; they do not interpret the question.

Reports show the question and the **selected** hypothesis, including when a later
revision was rejected. Forecast results evaluate selected factors; they do not establish
a causal answer to a business question. Missing business variables must not be described
as tested merely because a price proxy ran.

For a nonempty question, each Codex/API hypothesis returns a bounded `question_map`
with one to four claims: `claim`, selected `factor_ids`, `rationale` and `limitations`.
An unmeasured part uses an empty factor list and explains the missing inputs. The
validator rejects bindings to unselected or invented factors before fitting. The
researcher chooses the factors; the user need not name recipes or prescribe a revision.
The live hypothesis node shows **proposed inputs**. After each completed fit, the
prediction node and report check those IDs against the recorded model columns and show
**inputs fitted**, **inputs not fitted**, or **no measured input**. These labels describe
execution, not support for a causal claim. Claims and limitations remain explicitly
attributed to the LLM, whose interpretation can be wrong. Catalog inputs marked
unavailable, including business variables, retain their recorded reasons in the
evidence path instead of disappearing from the explanation.

The final report puts the selected attempt's map next to the question, while retaining
maps for rejected rounds below. This uses the existing hypothesis calls and adds no
training or selection step. Simulated researchers and older study records have no
structured map; the report says so rather than inventing coverage or rewriting history.

Cross-run feedback and remembered recipes use the same symbol, prediction horizon and
exact trimmed question. Repeating a saved template continues that focus; changing its
question starts a separate feedback scope and changes the new-evidence fingerprint.
Case and punctuation are retained. Blank questions and older records share the general
research scope. All forecasts remain in the global history and can still be observed.
Failed studies preserve the question on `research-resume`; changing it requires a new study.

### Unprescribed factor selection observation (2026-11-01)

At `28e9c05`, real Codex received only this business question, with no named factor
recipe or prescribed revision: “Could changes in US petroleum inventories help explain
near-term XOM returns, and how might company fundamentals change that hypothesis?
Use the available evidence to investigate the question and identify what cannot be tested.”
The run reused 252 retained public daily sessions, issuer financials, EIA archive data
and issuer RSS. Active news search and related-price collection were disabled; it was
not a fresh online discovery benchmark.

Codex selected all three supplied EIA inventory changes, company revenue/inventory/free
cash-flow growth, momentum and volatility. After development feedback it substituted
operating cash-flow growth and a moving-average price recipe. Both rounds supplied four
question-map entries, including one explicitly unmeasured claim about missing causal,
news or peer evidence. All bound IDs were present in their actual model inputs.

Two Codex calls, two real LightGBM research fits and two source-removal diagnostic fits
completed in 87.9 seconds on the local machine. Over 20 development observations,
price MAE was 4.0271 initially and 4.0823 after revision, versus 3.6075 for unchanged
price. The configured MAE rule retained round 1 despite revised net return rising
from 3.60% to 3.74%; buy-and-hold returned 4.98%. Annualized daily Sharpe was
1.7910 initially, 1.6919 revised and 2.3644 for buy-and-hold, using only 24 daily
returns. Terminal evaluation remained unavailable because selected industry features
were missing on reserved dates; the latest forecast had no observed outcome.
These are negative development results and one integration timing, not independent
predictive evidence or a claim that the explanation changes improved performance.

### Question-driven integration observation (2026-11-01)

A real Codex run asked whether inventory growth indicated weaker Coca-Cola demand,
using retained public daily prices and issuer financials plus current RSS and active
news research. It completed two searches, one available article out of four attempted
reads, two LightGBM research fits and one financial-removal diagnostic fit in about
208 seconds. Unavailable pages remained explicit. Both hypotheses selected
`financial.inventory_yoy`; the revision retained it and revenue growth while replacing
margin change and ten-session volatility with momentum and twenty-session volatility.
The researcher challenged the question's premise using the supplied declining inventory
balance, considered contrary demand evidence and named missing turnover, sell-through
and downstream inventory data. This demonstrates question propagation and execution,
not independently verified business reasoning.

Development price MAE changed from **1.0532 to 0.9148**, versus unchanged-price **0.9460**,
so the existing MAE rule selected the revision. Its hypothetical net return was
**-0.25%**, versus buy-and-hold **0.47%**; annualized Sharpe was **-2.1940** on only
24 daily returns. The reserved retrospective window had ten observations, price MAE
**1.9762** versus unchanged-price **2.0310**, and net return **1.39%** versus buy-and-hold
**3.77%**. These observations are not causal findings, prospective validation or evidence
of reliable predictive gain. The completed run retains its development source identity.

## Prediction horizon

### Price history and financial variation

Use daily-price history independently of the prediction horizon:

```bash
quant-rsi research-study KO --historical-sessions 252 --financials --check-inputs \
  --output /tmp/ko-year-input-check
```

The workbench exposes the same **Minimum price history** field and retains it in
templates, queued tasks and resumed studies. Leave it blank for the existing automatic
window. With an explicit value, the minimum is `120 + 3 * (horizon - 5)` plus
`(folds - 1) * (evaluation_sessions + horizon)` when the time-window diagnostic is enabled.
The maximum is 445 sessions, so a 60-session horizon with three diagnostic folds requires
445. Invalid combinations fail before acquisition or paid work. Synthetic mode requires
the automatic window.

The value is a collection minimum, not a truncation instruction: a longer retained
`--prices-dir` is used in full. A shorter retained collection is rejected with a fresh-directory
instruction, preserving its files. Workbench evidence preparation requests the same window
as the study. The source must still supply every required price and exchange-calendar
session; this control does not repair corporate actions or extend financial/news coverage.

Input checks show actual price sessions, frozen training rows and development rows. For
each financial factor, the input report, live stages, researcher context and final report
show available training rows and distinct training/development values. Counts exclude
terminal/latest-forecast rows and do not alter factor admission, missing values or selection.
Repeated daily values are not independent financial observations; distinct values are not
a count of filings. A constant training feature cannot explain differences between training
rows. A longer window may add financial changes, but neither coverage nor variation proves
predictive value. Inspect coverage before running paid research; removing `--check-inputs`
then runs the normal two-fit/revision flow with the same options.

### Choose the prediction target

The study supports one target horizon per run: 5 (default), 20 or 60 **trading
sessions**. The workbench form, saved templates and task history expose the same
setting. For example:

```bash
quant-rsi research-study KO --horizon-sessions 20 --news-research --financials \
  --output /tmp/ko-20-session-study
```

The horizon reaches the LLM context, training labels, maturity exclusions,
reserved-terminal split, latest target date, strategy replay and later outcome
observation. Both native estimators and the optional Qlib comparison use it.
The target is still close-to-close log return; changing the prediction horizon
does not change factor lookback windows or make the strategy hold each trade for
exactly that many sessions. Trades follow the existing next-open signal policy.

Fresh daily collections retain 120, 165 or 285 completed sessions respectively.
The extra history preserves the training allowance while separating training,
development and terminal label intervals. Longer studies request two years of
source bars and require a published exchange calendar covering the full future
horizon. Missing 2025 historical sessions can use the official
[NYSE archive table](https://ir.theice.com/press/news-details/2024/NYSE-Group-Announces-2025-2026-and-2027-Holiday-and-Early-Closings-Calendar/default.aspx)
or [Nasdaq 2025 calendar](https://www.nasdaqtrader.com/content/technicalsupport/2025tradingcalendar.pdf).
Both documents are downloaded and retained with their hashes. The Nasdaq grid's
10 scheduled closures and three 13:00 early closes are a reviewed transcription
bound to SHA-256 `fb761082d449ef76461d2ca8448ccebb79862a0acd124fad2e914469c9086037`;
changed bytes require review. These annual schedules do not certify emergency
closures; missing expected price sessions still fail. Each exchange uses its own
source. The old Nasdaq URL containing `2025holidayandtradinghours.pdf` redirects
to a 2026 document and is deliberately not used for historical dates.

Unsupported calendar years or missing history fail explicitly. On 2026-11-01,
real KO 60-session input checks succeeded with 285 bars; AAPL 60-session checks
stopped because the current Nasdaq source did not cover the 2027 target year.
This is an explicit data-coverage limit, not a synthetic fallback. Related-price
features need the longer historical window but only the normal five-session future
calendar, because they do not publish a separate long-horizon forecast.

Old short collections are preserved; use a fresh output or `--prices-dir` for a
longer study. Later observation receipts may use the normal one-year chart request
and omit the registered historical-calendar download. Other endpoints and settings
remain bound, and every session in the published forecast interval must still
match its frozen calendar.

Mature feedback and remembered recipes are filtered by symbol **and horizon**.
The workbench still displays all studies, with their horizons labelled. Existing
five-session artifacts keep their original identities. A resumed study uses its
saved horizon and cannot switch it while reusing fits.

This is configurable single-horizon research, not simultaneous multi-horizon
optimization or a weighted score. Longer horizons do not establish better
predictions. Quarterly business factors may motivate a longer target, but their
usefulness still requires observation. Retrospective development and terminal
results retain the existing news/LLM knowledge limitations.

### Horizon integration observations (2026-11-01)

A real Codex active-news study on KO used 285 public daily bars, retained official
issuer financials and actual LightGBM fits at a 60-session horizon. It completed
two news searches and read one of four attempted articles; failures remain in the
report. The initial factors included operating-margin change and the KO/KOF
composition. Feedback led to revenue growth, a moving-average recipe and
20-session volatility while retaining the composition. Development MAE changed
from 1.3902 to 1.3355 (unchanged-price baseline 6.4880), so round two was selected.
Both strategies and buy-and-hold returned 5.89% with net Sharpe 0.9409 over the
same 79 daily returns. The ten reserved-terminal observations had selected-model
MAE 2.8089 versus 6.3300 for unchanged price. The latest forecast targets
2027-01-28 from 2026-10-30 and remains unobserved.

A separate AAPL 20-session study used a **simulated researcher**, actual issuer
RSS/financial data, public prices and real native/Qlib fits. Initial and revised
development MAE were 14.7887 and 15.7833; the initial model was retained. Its
strategy return was -1.14% against 0.35% buy-and-hold, despite a better forecast
MAE than unchanged price. The selected native and upstream Qlib predictions matched
on both development and terminal rows. The common-input hash was
`89af464581eca09f788efc706566ebfc7f04440ae997e5f0fdfe42038199c3cc`.

These runs demonstrate the connected interfaces and preserve negative outcomes.
They do not establish predictive gains: current news and pretrained LLM knowledge
were not isolated from historical outcomes, and development feedback was reused.
The Qlib comparison holds model and inputs equal; it is not an independent research
strategy comparison. No elapsed-market outcome is claimed for either latest forecast.

## Frozen-recipe time-window diagnostic

Add `--walk-forward-folds 2` or `3` to the same connected-study command, or select
**Time-window diagnostic** in the workbench. The default is disabled. For example:

```bash
quant-rsi research-study KO --horizon-sessions 20 --news-research --financials \
  --walk-forward-folds 3 --output /tmp/ko-time-windows
```

The windows are fixed in `walk_forward_plan.json` before hypothesis generation.
After research selects a recipe, the diagnostic keeps exactly those factor columns
and model settings. It trains an expanding-window model for each window using only
labels that mature strictly before its first prediction origin. The final window
ends at the planned development boundary; all diagnostic labels and account
valuations remain before the reserved terminal window. No additional LLM calls,
factor search, threshold changes or selection feedback occur in this stage.

Each window contains the configured `--evaluation-sessions` origins (20 by
default). The previous window's last target precedes the next window's first
origin; overlapping labels within each window remain dependent. With three
windows the run retains 170/245/445 price sessions for a 5/20/60-session horizon,
respectively. Two windows require 145/205/365 sessions. The extra older history
is excluded from the original research training allowance: with the same retained
recent prices, enabling this diagnostic does not itself change the research fit.
The diagnostic refits explicitly use the longer, expanding history.

A short retained collection cannot serve this expanded plan; use a fresh output
or `--prices-dir`. Published calendar and source-history coverage are still
required. Related stocks collect matching historical coverage. A longer window
can expose missing financial or news data: unavailable training rows are listed,
and any missing evaluation row makes that entire window unavailable. No missing
window is silently removed from a pooled score.

The report and live view show each window's forecast errors and separate costed
strategy/buy-and-hold accounts, including Sharpe and drawdown. `walk_forward.json`
retains predictions, baseline predictions, exact training dates, exclusions and
account evidence. Pooled **forecast** metrics are produced only if every window
completes; account returns and Sharpe are never stitched or averaged into a
purported continuous portfolio. Each account starts with the configured cash.

The diagnostic allows at most one additional fit per window, separately from the
research fit budget and optional financial/Qlib comparisons. Completed diagnostic
fits use the existing checkpoint mechanism; `research-resume` reuses them after a
later failure. A failed attempt remains recorded and can consume actual compute
beyond the count of successful logical fits, as in the existing study recovery
contract. The selected forecast and prospective history still belong to the
original selected research model.

This is **post-selection retrospective sensitivity analysis**, not independent
out-of-sample research validation. The factor recipe was selected using later
information, current news and potentially pretrained LLM knowledge. Chronological
refitting prevents future labels entering a fit; it does not undo that recipe
selection exposure. The older registered C2/C5 fold requirement in issue #11
remains a separate unresolved contract.

### Time-window integration observation (2026-11-01)

A KO 60-session, three-window run used a **simulated researcher**, real retained
public prices and issuer financials, and actual LightGBM fits. The selected recipe
combined a moving-average price factor, free-cash-flow growth and CCEP relative
return. All three chronological windows completed, with 45, 125 and 205 mature
training rows and 20 prediction origins each. Three extra diagnostic fits were
separate from the two research fits and one financial-removal comparison.

| Account window | Model MAE | Unchanged-price MAE | Net strategy return | Net Sharpe | Buy-and-hold return |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-08-08 to 2025-12-01 | 1.8553 | 2.3240 | 1.12% | 1.0760 | 2.24% |
| 2025-12-02 to 2026-03-27 | 6.7685 | 6.6130 | 1.03% | 0.8469 | 6.66% |
| 2026-03-30 to 2026-07-23 | 1.5157 | 6.4880 | 5.89% | 0.9409 | 5.89% |

Each account has 79 daily returns and starts independently. Window two is worse
than the unchanged-price forecast baseline; windows one and two trail buy-and-hold
on strategy return. These are retained negative comparisons, not evidence of
independent forecasting skill. The run preserves its development-worktree identity;
later presentation changes do not rewrite that completed experiment.

## Financial business factors

```bash
# Real public sources; identify your application/contact to SEC where required:
export SEC_USER_AGENT='Your application and contact address'
quant-rsi research-study KO --news-research --financials --output /tmp/ko-business-study
# Same pipeline, explicitly simulated inputs, real model fits:
quant-rsi research-study AAPL --financials --llm simulated --prices synthetic \
  --news simulated --related simulated --model ridge --no-ui --output /tmp/aapl-business-offline
```

`--financials-source auto` (default with `--financials`) tries the official
[SEC company-facts API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
then the configured official issuer archive if SEC fails. `sec` and `issuer` explicitly
require that provider. A snapshot uses one provider; rows are never silently mixed.
KO uses its [financial information archive](https://investors.coca-colacompany.com/financial-information);
AAPL uses the public financial-report feed linked by its
[investor relations site](https://investor.apple.com/investor-relations/).
The bounded collector retains up to sixteen releases and extracts only the explicitly
reported three-month revenue and operating-income columns from their financial PDFs.
A separate cash-flow section reads explicitly labelled 3/6/9/12-month operating cash
flow and purchases of property, plant and equipment. It preserves each reported
duration, rejects ambiguous numbers and retains per-section failures; an unusable
cash-flow section does not discard a usable income statement, or vice versa.
The PDF reader, font decoder and spreadsheet reader are installed by the normal dependency lock.
`instruments/US/{KO,AAPL,XOM}/financials.json` holds company CIK and ordered US-GAAP tag
mappings. The shared implementation computes `financial.revenue_yoy`,
`financial.operating_margin` and `financial.operating_margin_yoy` (year-over-year margin
change), adds runnable definitions to the existing LLM factor library, executes selected
columns in the existing model, and allows revision after the same error/strategy feedback.
Cash-flow definitions extend the same path:

| Factor | Calculation and period |
| --- | --- |
| `financial.operating_cashflow_yoy` | Change in reported operating cash flow divided by the absolute prior-year amount, over matching 3/6/9/12-month periods |
| `financial.free_cashflow_yoy` | The same scaled change for operating cash flow minus purchases of property, plant and equipment; this is the specific free-cash-flow definition used here |

These are **reported-duration comparisons**, not single-quarter cash-flow estimates.
Nine months compare only with nine months. When SEC reports multiple durations ending
on the latest date, the longest reported duration is used. Issuer PDFs provide an explicit
duration and end date; the parser does not invent a start date. SEC rows retain their
reported starts, which must also match the prior-year period. No YTD subtraction occurs.
Zero prior amounts disable the corresponding ratio; negative amounts use the absolute
prior-year denominator, so a transition from -100 to 150 is +2.5. Free cash flow requires
operating cash flow and nonnegative capital expenditure from the same statement/version.

`financial.inventory_yoy` extends this path with **reported inventory balances**:
current inventory divided by comparable prior-year inventory, minus one. The SEC
adapter accepts USD `InventoryNet` instants; KO/AAPL issuer profiles read the
`Inventories` row only from the balance sheet. Cash-flow inventory adjustments are
never substituted. Both printed balance-sheet dates are retained, but an interim
balance compared with the previous fiscal year-end is not a year-over-year pair.
The feature searches the collected releases for a date 350–380 days earlier.
The larger release allowance supplies earlier balance-sheet comparisons for long
training windows; an archive may still offer fewer releases or unreadable statements.

As with other financial inputs, each value uses only versions available at the
session cutoff. A missing, ambiguous, stale or nonpositive prior balance disables
the factor for the affected rows; no zero fill, row dropping or price proxy is used.
The report retains current/prior statements and per-session coverage. Inventory
growth alone does not establish excess stock, demand strength or an expected share
price direction; the researcher must justify its hypothesis using supplied evidence.
No extra LLM call or model fit is introduced by making this factor available.

Use the existing `--financials` option and a **fresh financial collection directory**
to fetch the newly configured balance rows. Existing retained collections and completed
studies are not rewritten. Collections must match their original profile; a collection
made with an older profile cannot be silently upgraded by reusing its directory.

Inventory integration was checked on 2026-11-01 with retained real KO/AAPL prices
and newly collected issuer statements. Sixteen releases yielded 32 KO and 26 AAPL
inventory rows; three older AAPL balance sections were rejected because PDF text
extraction split numeric glyphs. Those section failures remain in the snapshots.
The checked KO five-session and AAPL twenty-session study plans had no missing
required inventory rows. The earlier eight-release KO collection was insufficient
for the reused long price history; that run correctly excluded inventory and is
preserved separately.

A real Codex study with the expanded KO collection completed both hypotheses and
LightGBM fits. Inventory was available but was not selected; Codex chose revenue and
margin inputs, and retained round one when the revision worsened development MAE
from 1.2108 to 1.4647. A separate **simulated-researcher, real-input** integration
run explicitly selected inventory growth (feature value -0.085596 at its origin),
then replaced it with operating-margin change after feedback. Both actual LightGBM
fits completed; development MAE changed from 1.6747 to 1.1215, still worse than
unchanged price at 0.9460. These are integration observations with exposed development
selection, not evidence that inventory predicts prices or improves independent results.
Issuer cash-outflow signs are normalized; missing, ambiguous or stale facts remain unavailable.

The LLM selects ratios; it does not invent their numerical values. Synthetic mode uses
explicitly marked fixtures through the same parser/calculations.

Cash-flow acceptance on 2026-11-01 used freshly collected official KO/AAPL statements
(64 normalized records per company, no section failures after explicit KO row aliases).
Both companies supplied all five financial factors at the two checked cutoffs. A real
AAPL Codex study with public daily prices and LightGBM selected
`financial.free_cashflow_yoy` in round 1 (latest feature value 0.524564), then changed to
`financial.operating_margin_yoy` after feedback. Development price MAE worsened from
8.1859 to 16.0872, versus 6.9440 for unchanged price, so round 1 was retained. The
same-row financial-removal diagnostic completed. Reserved terminal MAE was 5.7510,
versus 2.7780 for unchanged price; no predictive gain is claimed. Active news issued
two searches, but all four article reads failed and remained disclosed. This is
execution evidence on retrospectively retained data, not prospective validation.


The collector retains `financials/snapshot.json`: raw company facts, content hash, mapping,
provider URL and receipt time. `--financials-dir PATH` reuses that retained snapshot after
symbol/CIK/mapping/hash validation. Issuer snapshots additionally retain content-addressed
HTML/PDF resources, normalized records, statement page numbers and resource hashes.
Every retained resource hash is checked before reuse. Financial mappings are part of
the snapshot identity: snapshots collected with an older mapping remain historical
evidence, and new cash-flow studies need a fresh collection directory. `source_attempts.json` records
failed source requests even when neither provider succeeds. It requires `--financials` and a path outside the
checkout. `financial_features.json` records per-session values, chosen fact accessions,
reporting periods, filing dates, cutoff and missing reasons. Runtime/configuration
identity and completed receipts follow the existing resume contract.

For SEC, only directly reported 70–110 day USD quarters from 10-Q/10-K filings and
amendments are eligible. Annual/YTD values are not converted, so Q4 may be absent. A date-only
filing becomes available at **05:00 UTC the following day**, after the entire Eastern
filing date; fiscal-period end is never treated as publication time. Later amendments
cannot affect earlier cutoffs. The latest visible quarter must be no more than 270 days
old; the comparable prior-year period must match within 15 days (350–380 days apart).
Revenue/income used for a margin must share the period and accession or issuer PDF. Ambiguous values
are unavailable. No quarterly interpolation, row dropping or zero imputation is used.

The fixed train/development rows determine each ratio's eligibility. Missing terminal
or latest values retain the existing unavailable-prediction behavior. This is a
retrospective reconstruction from today's retained public documents, **not** a complete
point-in-time vendor archive or independent evidence of predictive gain. Restatements,
tag coverage and missing reported quarters limit coverage. Unsupported stocks or failure of all requested
sources produce an explicit unavailable financial section; the
ordinary study can continue without those columns. In new-evidence mode, a failed
preflight source check fails that task instead of treating missing data as unchanged.

Issuer availability is the day after the later of the release date and PDF-declared
creation/modification dates, at 05:00 UTC. Website maintenance dates are stored separately;
if PDF dates are absent, the page's disclosed revision date is used conservatively.
For example, Apple historical pages can share a site-wide maintenance date while their
financial PDFs retain distinct publication-day metadata. PDF metadata is provenance,
**not proof of historical first receipt**. Current PDFs could differ from an original
release; this source is unsuitable for claiming an independently verified point-in-time
benchmark. Quarterly starts are not invented. Current publication dates are never replaced
by fiscal-period ends.

### XOM business data

```bash
quant-rsi research-study XOM --financials --financials-source issuer --news-research \
  --question "Does inventory growth exceed revenue growth, and does cash flow support that hypothesis? Revise the factor after feedback." \
  --output /tmp/xom-business-study
```

XOM uses the same financial-feature and business-expression execution path. Its
[official financial-results archive](https://investor.exxonmobil.com/earnings/financial-results)
links dated releases and financial PDFs. The collector first tries the earnings-release
PDF; if it yields no usable configured sections, it tries the release's financial
supplement PDF, then the explicitly linked Excel supplement. This supports the
image-only second-quarter 2026 PDF without OCR. Each release contributes one
document/version, with its own hash, PDF page or worksheet/cell references,
publication and document revision dates. A partially usable primary document is retained
without filling its missing sections from another version. Failed sections remain visible.

The profile maps **sales and other operating revenue**, reported operating cash flow,
and **additions to property, plant, and equipment**. Quarterly revenue is separate from
YTD revenue; cash-flow features preserve their reported cumulative duration. Total
inventory is the sum of **crude oil, products and merchandise** and **materials and
supplies**, with both component values retained. A missing, negative or conflicting
component makes the balance section unavailable; a partial total is never used.
Prior-year inventory comes from a comparable balance date in an older release, not
the previous December comparator printed beside an interim balance.

The configured reports do not separately report operating income. Consequently XOM's
operating-margin and margin-change factors remain unavailable; income before taxes
and non-GAAP segment earnings are not substitutes. Oil prices, production volumes,
refining margins and inventory quantities are not supplied by this profile. A dollar
inventory growth rate is not an inventory-volume or demand measurement. If neither
the configured PDFs nor the spreadsheet are readable, that release remains unavailable.

The spreadsheet adapter reads the configured income, balance and cash-flow worksheets,
with explicit dates and B/C current/prior columns. It reads cached numeric values;
it does not evaluate formulas or follow external workbook links. XOM's cells store
**raw USD** while number formats display millions. Both the dollar/millions heading
and scaling formats are checked; values are not multiplied by a million again.
Formula cells without cached values, text amounts, ambiguous dates and inconsistent
units are rejected. Workbook creation/modification dates use the same conservative
next-day availability rule as PDF metadata. Source failures remain in the report,
even when the structured fallback succeeds.

The SEC profile uses current parent CIK **2115436**. ExxonMobil's
[successor registration filing](https://investor.exxonmobil.com/sec-filings/all-sec-filings/content/0001193125-26-292453/d54886dposasr.htm)
records the July 1, 2026 redomiciliation and successor registrant. The SEC path does not
combine company-facts payloads from predecessor CIK 34088. The issuer path uses the
continuous archive and reported comparative figures linked above; it remains a
retrospective reconstruction, not proof of historical first receipt. Choose `issuer`
explicitly for that archive; `auto` retains the existing SEC-first behavior and can
have less historical coverage when SEC succeeds with a shorter successor payload.

### Business factor expressions

Codex/API hypothesis rounds can propose up to two run-local business expressions
alongside their price recipes. This connects a business hypothesis to actual feature
calculation, a model fit, feedback, and a changed recipe in the next round:

```bash
quant-rsi research-study KO --llm codex --financials --news-research \
  --question "Test inventory growth minus revenue growth as a demand hypothesis. After feedback, consider a cash-flow-based revision and explain contrary evidence." \
  --output /tmp/ko-business-expressions
```

Each `proposed_business_factors` entry contains a `name`, `description`, two to four
`inputs` binding local aliases to supplied `financial.*` or `external.*` factor IDs, and an
`expression`. The expression uses the existing `schema_version: 1` grammar:
topologically ordered `leaf`, `add`, `subtract`, or `multiply` nodes, at most seven
nodes and depth three. Every binding must be used; the last node is the output.
There is no generated Python, arbitrary field lookup, division, or new data fetch.
The researcher can return an empty list when no relevant combination is supported.

For example, aliases `inventory` and `revenue` can bind to
`financial.inventory_yoy` and `financial.revenue_yoy`; subtracting their leaf nodes
creates a relative-growth feature. Its ID is `business.<name>`. A changed recipe
requires a new name, and omitting an old ID deactivates that factor for the round
without deleting its recorded definition. Selection must still include at least
one proposed price recipe under the existing study contract.

With `--external-drivers`, a proposal can also compare two EIA inventory changes
or combine an industry change with a financial variable. For example:

```bash
quant-rsi research-study XOM --llm codex --financials --external-drivers \
  --historical-sessions 252 \
  --question "Test a composition of EIA crude inventory change and XOM revenue growth. After feedback, test distillate inventory instead; explain the different periods and limits." \
  --output /tmp/xom-mixed-business-study
```

The inventory source is a weekly industry aggregate, while revenue growth refers
to the company's reporting period. Arithmetic does not align these periods or
measure XOM's own inventory. Industry-only expressions do not require enabling
financial data. Existing financial-only recipes retain their recorded
`financial_expression` representation; recipes with an industry input use
`business_expression`, through the same grammar and evaluator.

All inputs must be supplied financial or external definitions. Missing historical inputs keep
a proposal unavailable; feature computation propagates missing values without
filling or dropping rows. Input availability uses the original publication cutoffs.
Combining factors does **not** align different reporting periods or turn growth
rates into inventory turnover, demand measurements, or causal evidence. The
researcher must explain those limitations.

Both model implementations execute the resulting column. The live factor cards
and final report show the input aliases and arithmetic nodes as tables. Predictions
retain the underlying source definitions. Each optional source-removal diagnostic
removes every expression depending on that family, together with its direct columns.
A mixed expression is therefore removed in both the financial and external diagnostics.
These separate comparisons do not isolate an interaction effect and remain development
diagnostics, not independent validation. Each enabled diagnostic uses at most one
additional fit; neither changes selection or the latest forecast.

Selected business recipes can be offered in a later study through existing factor
memory. They are rebound to the current source definitions and availability;
old feature values and runnable flags are never imported. Business inputs absent
from the new study cannot be replaced with price proxies.

### Mixed-source integration observation (2026-11-01)

A real Codex run on source `fa2b7d1` used retained XOM prices and issuer financials,
retained issuer RSS, and the EIA snapshot described below. The explicit question
requested a crude-inventory-change minus revenue-growth composition, followed by
a new distillate-inventory-change minus revenue-growth recipe after feedback.
Both rounds kept the same momentum control and actually trained LightGBM.

Two Codex calls, two research fits and two source-removal diagnostic fits completed
in 57.93 seconds, excluding the original data collection. The 183 recorded source
and configuration hashes matched that run. On the same 20 development observations,
MAE increased from 3.5950 to 3.7040. The system retained the first recipe and kept
the unsuccessful revision visible. Both source-removal diagnostics removed the
selected mixed column; they are not two independent tests of an interaction.
The original runtime stage counter predates the external-stage accounting fix;
both additional fits were recorded in their diagnostic artifacts.

Terminal evaluation remained unavailable because an industry input expired on
one reserved date. The latest forecast's outcome was unobserved. This verifies
prompted composition, execution, revision and rejection on retained inputs; it
does not establish autonomous discovery or predictive improvement. The different
weekly and reporting-period meanings remain a limitation of this example.

Local evidence: `/tmp/quant-rsi-mixed-business-codex-study/report.html` and
`/tmp/quant-rsi-mixed-business-codex-validation.json`. These paths are not shipped
with the package, and the original run identity and output were not rewritten.

### Check inputs before research

```bash
quant-rsi research-study AAPL --financials --financials-source issuer --check-inputs \
  --no-ui --output /tmp/aapl-input-check
```

The CLI and workbench **Check inputs** action save `input_report.json` and an HTML table.
They collect target prices/current issuer news and compute financial coverage on the fixed
training/development rows, without constructing an LLM client, fitting a model, or registering
a forecast. Peer prices, active searches and article reads are explicitly outside this check.
The action is not a successful research receipt and cannot suppress a later research task.
Normal studies also compute and display this check before their first LLM request. Missing
financial data is visible and does not masquerade as a complete business-data study.

When fixed training/development rows lack a financial value, `coverage_gaps` groups
one primary reason per missing row, with a count, an actual example cutoff and a
suggested next step. The preflight HTML, live financial/input nodes, hypothesis context
and final report reuse those diagnostics. Reasons distinguish missing visible periods,
stale statements, missing comparable years and incompatible values. They do not claim
to diagnose every upstream cause: a missing supported fact can reflect absent tags,
conflicting filings or parser/source failures, which remain separately visible.

For example, missing prior-year inventory suggests collecting comparable balances
350–380 days apart, not substituting the previous fiscal year-end. Incompatible cash
flows require matching reported durations; a zero comparison base remains unavailable.
Use a fresh collection and study when changing inputs. More data does not guarantee a
valid feature, and diagnostics never fill values, move availability dates or change
admission. Non-required rows do not contribute to these counts. The final financial
table distinguishes **Selected**, **Available, not selected** and **Unavailable**;
selection is not evidence of importance or business causality.

After selection, if the selected fit contains financial and nonfinancial columns, one
additional diagnostic fit removes only the financial columns, retaining identical
training/development sessions, labels, estimator settings and cost policy. The report
compares price MAE, net return and Sharpe, saves `financial_comparison.json`, and declares
the extra fit separately from `--max-fits`. It does not alter selection, the frozen
terminal evaluation or the latest forecast. No applicable selected columns means zero
extra fits and an explicit reason. `--compare-qlib` remains independent and can add its
own declared fit.

Validation for this addition uses offline fixture data and actual Ridge/LightGBM fits,
actual CLI/Codex-transport fault fixtures, and a two-stock workbench repeat/force cycle.
Live SEC requests for both KO and AAPL returned HTTP 403 in the implementation environment;
no successful live SEC-to-model result is claimed. A real Codex session did complete
two hypothesis calls over simulated prices/news/financials with real Ridge fits: it
selected revenue growth initially and margin change after feedback, then retained
the initial round because development MAE worsened. The separate financial-column
removal diagnostic also completed. This is mixed-mode integration evidence only.

### Real-service acceptance (2026-11-01)

Two retained runs used public daily prices through 2026-10-30, official issuer
financial PDFs, current RSS, active news search/read, Codex session authentication,
and real LightGBM fits. Each completed two research fits, one revision attempt and
one additional financial-removal diagnostic. All three financial definitions covered
the fixed 45 training and 20 development rows for both symbols.

| Stock | Initial financial factor | Revised financial factor | Initial / revised price MAE | Unchanged-price MAE | Selected round |
|---|---|---|---:|---:|---:|
| KO | Operating-margin YoY change | Revenue YoY growth | 1.4078 / 2.6096 | 0.9460 | 1 |
| AAPL | Revenue YoY growth | Operating-margin YoY change | 7.2904 / 7.3042 | 6.9440 | 1 |

Both revisions worsened development MAE and were rejected. The selected development
net Sharpe was -0.9308 for KO and 3.0171 for AAPL, each over only 24 daily returns;
these short retrospective samples do not establish long-term performance. Each
reserved terminal evaluation used 10 observations without refitting; those tasks are
still retrospective and their current news/LLM knowledge is not historically isolated.

KO completed seven LLM calls, two searches and two successful article reads out of four;
AAPL completed seven LLM calls, two searches and four reads out of six. Failed article
reads remained visible. Financial factors appeared in the selected model's numerical
feature values, and the removal diagnostic completed for each. This validates execution,
not an advantage over the unchanged-price forecast or a causal effect of a news story.

Local retained evidence (not repository fixtures):

- `/tmp/quant-rsi-issuer-live-v3/{KO,AAPL}/snapshot.json`: eight release PDFs per stock,
  32 normalized fact records each, no rejected statements. Earlier auto-source probes
  retained the SEC 403 before successful issuer fallback.
- `/tmp/quant-rsi-{ko,aapl}-issuer-study/report.html`: complete reports; `result.json`,
  `financial_features.json`, raw source directories and checkpoint identities preserve
  the corresponding inputs/runtime. Runs reused collected target prices and financial
  snapshots; their approximately 199/206 seconds exclude those earlier collections.
- `/tmp/quant-rsi-issuer-browser/`: browser acceptance screenshots. A simulated-input
  watchlist completed two input checks followed by two observations and two real Ridge
  studies, with no browser script errors. Input checks produced no forecasts or research
  deduplication receipts. This browser exercise is engineering evidence, not market evidence.

These paths are machine-local evidence, not files shipped with the package. Automated
regressions use small original offline fixtures rather than downloading issuer documents.

## External business drivers

`--external-drivers` connects the same research workflow to archived EIA weekly
US petroleum inventory and oil-price releases. The workbench exposes the same optional setting;
saved templates and queued tasks preserve it. The normal researcher chooses whether
to use it. After selection, a diagnostic removes the selected external columns and
refits on the same rows and model settings (up to one additional fit, outside the
research fit budget). Reports compare MAE, net return, Sharpe and drawdown. This
diagnostic does not change selection or the latest forecast; it is not independent
validation. If no external columns or no other controls were selected, no fit is
performed and the report explains why.

The workbench gives every daily-data batch a fresh EIA resource cache, shared by its
research or input-check tasks. Within that batch, requests with identical history bounds
download each successful archive/page/table URL once. The same path is used with the
new-evidence check enabled or disabled and for forced runs. A different batch or date
window collects afresh; this is not a permanent URL cache that could hide new releases
or source corrections. Failed requests are not cached as successful evidence.

Each task copies the original bytes, hashes and receipt timestamps into its own retained
snapshot, so deleting the shared batch cache does not break completed-task replay or
relabel historical receipts as newly received. A resource lock protects concurrent cache
writers; source hashes are checked before reuse. A standalone CLI caller can opt into
the same mechanism with `--external-drivers-cache-dir PATH`, using a new path for each
research cycle. `--external-drivers-dir` continues to identify a task's retained snapshot.
Both directories must remain outside the source checkout. Synthetic runs do not download
or populate the cache, and the cache does not add LLM calls or model fits.

Validation at `6bed759` ran two native CLI input checks over a preseeded cache of
retained public XOM prices, RSS and EIA resources. Both checks preserved all 109
resource hashes/receipts, 159 records and the same two excluded source-date failures;
all three external factors were usable. Both task snapshots remained readable after
the shared cache was removed. This made no new network requests, LLM calls or fits.
A separate offline two-stock batch fixture measured six resource requests before
sharing versus three after; the next batch requested all three resources again.
These checks demonstrate request reuse and provenance, not a live-service latency
or total-storage improvement. Each task deliberately keeps its own raw evidence copy.

```bash
quant-rsi research-study XOM --llm codex --news-research --external-drivers \
  --historical-sessions 252 \
  --question "Could changes in US petroleum inventories help explain XOM returns?" \
  --output /tmp/xom-industry-study

quant-rsi research-study XOM --external-drivers --historical-sessions 252 \
  --check-inputs --output /tmp/xom-industry-input-check
```

The three inventory recipes calculate current / prior-week stocks minus one for
commercial crude oil excluding the Strategic Petroleum Reserve, total motor
gasoline, and distillate fuel oil. Both values come from the same release's
[Table 4](https://www.eia.gov/petroleum/supply/weekly/archive/2026/2026_10_28/download/table4.pdf).
These are US industry inventories in million barrels, not company financial
inventory and not oil prices. They express candidate explanatory variables, not
an established causal relationship or evidence of predictive improvement.

Fresh collections also follow the archive's actual Table 11 CSV link when present.
`external.eia_wti_5obs_return` and `external.eia_brent_5obs_return` calculate the last
daily quote divided by its fifth preceding available daily observation, minus one.
The recent-price section begins with **weekly averages**; these are excluded.
The parser recognizes the final two Monday-Friday daily blocks, including CSVs
with one omitted weekly-average column or an extra empty column.
At least six valid daily quotes and the latest dated daily quote are required. The
original daily dates, USD/barrel quotes, source URL and hash are retained. These
five oil-price observations need not coincide with five equity trading sessions.
Like inventories, the factor becomes visible after release, not on the quote date.

The inspected 2026 archive contains Table 11 links through October 15; October 21
and 28 pages have no such link. A missing link leaves the price factors unavailable
once the last release expires. The collector neither guesses retired table URLs nor
backfills this gap with a current spot-price snapshot. Price-table failures do not
discard successfully collected inventory records, and inventory-table failures do
not discard valid prices from the same dated page. Older retained inventory-only
snapshots are not upgraded or rewritten; collect into a new directory to include prices.

The collector follows dated [EIA archive links](https://www.eia.gov/petroleum/supply/weekly/archive/)
and verifies the page's publication date and the CSV's observation week. It
supports both older `csv/` and newer `download/` layouts. Conflicting dates,
missing fields, and failed downloads remain explicit failures. Each raw document
and receipt time is retained; `--external-drivers-dir` reuses the original snapshot
for the same requested date interval without refreshing it.

Date-only releases become available at 05:00 UTC the following day. The latest
published inventory or price change is usable for at most 14 days; subsequent missing data
remain unavailable. Required training/development rows determine factor admission,
without dropping rows or consulting terminal outcomes. A selected factor that
expires before the latest forecast prevents that forecast while preserving
completed development fits. Reports state that no forecast was published and list
the unavailable selected factors, alongside the original reason codes.

The source is a retrospective reconstruction from official archived documents,
not proof that this exact file version was observed historically. Current news
and LLM knowledge still influence hypothesis selection. Synthetic-price mode
uses explicitly marked simulated inventory and oil-price data. Financial and external-driver
diagnostics remove their respective input families separately; requesting both
can consume two additional diagnostic fits.

Workbench preparation fingerprints the retained external records and failures,
then passes that same snapshot to the study. A changed receipt time alone does
not trigger new research. A first collection can require up to three requests per weekly
release, plus the archive index; it is bounded to 104 releases. Existing retained
inputs are reused within the task, with the batch-scoped resource sharing described above.

### Oil-price integration observation (2026-11-02 UTC)

The annual retained-input check on `5b2664a` preserved all 159 existing inventory
records and added 96 oil-price records from 160 retained resources. It rejected a
December price table with repeated/conflicting daily dates and the two already
identified release-page date conflicts. January CSV layout differences were
checked against the dated PDF before supporting them. The original input snapshot
was preserved; reparsing its raw resources into a new directory required no new
downloads. WTI covered every required training/development row; Brent lacked four
rows and was therefore unavailable to the researcher.

A real XOM study asked about oil prices, petroleum inventories and company
fundamentals without prescribing factor IDs or a revision. It used retained target
prices/RSS/financials/EIA and collected related-stock prices. Two Codex calls, two
LightGBM research fits and two source-removal diagnostic fits completed in 113.59
seconds. Both hypotheses selected WTI returns, all three inventory changes and
company factors; the revision changed the price controls. MAE worsened from
3.7698 to 4.0003, versus 3.6075 for unchanged price and 3.3455 for the training-label
mean. The preselected MAE rule retained the first attempt despite strategy Sharpe
rising from 0.1305 to 1.6919. Each Sharpe estimate used only 24 daily account returns.

The report correctly withheld the October 30 latest forecast because the selected
WTI input had expired; terminal evaluation was also unavailable because of missing
selected features. No rows were dropped or alternative model silently substituted.
All 184 recorded source/configuration hashes matched the original commit. This
demonstrates an executable business hypothesis and honest missing-data handling,
not independent predictive gain or a completed continuous oil-price feed.

### External-driver integration observation (2026-11-01)

An XOM run on source `0b306c1` used 252 retained public daily observations,
retained issuer RSS, real Codex calls, and the EIA archive. The explicit research
question requested commercial-crude inventory with a momentum control first,
then distillate inventory with the same control after feedback. This verifies
prompted factor substitution, not autonomous discovery of the hypothesis.

The retained collection contained 53 usable releases and two publication-date
conflicts, which remained excluded. All three external columns covered the
177 training and 20 development rows. Two Codex calls, two research fits and
one diagnostic refit completed in 55.95 seconds with retained inputs; this
duration excludes the initial archive collection. No active news search was
requested. All 183 recorded source/configuration file hashes matched the run.

| Development model | Price MAE | Net strategy return | Sharpe |
|---|---:|---:|---:|
| Initial: crude inventory + momentum | 4.6817 | — | — |
| Revised: distillate inventory + momentum | 2.7388 | 9.18% | 5.2512 |
| Diagnostic: revised model without inventory | 3.1692 | 6.00% | 2.7512 |

These are small retrospective development comparisons after feedback, not
independent gains. In particular, the Sharpe values are short-window diagnostics.
Terminal evaluation was unavailable because the selected inventory feature had
expired on October 15 after an earlier release was excluded for conflicting
dates. No row was dropped and no replacement model was selected. The latest
forecast was available, but its outcome was not observed.

The machine-local report is
`/tmp/quant-rsi-external-codex-ablation-study/report.html`; its separate audit is
`/tmp/quant-rsi-external-codex-ablation-validation.json`. These artifacts are not
shipped with the package. Offline regressions use original small fixtures.

## Nasdaq example: AAPL

```bash
quant-rsi research-study AAPL --news-research --output /tmp/aapl-study
```

[`instruments/US/AAPL`](../instruments/US/AAPL) supplies a complete profile: shared price
bindings, Apple's official [Newsroom Atom feed](https://www.apple.com/newsroom/rss-feed.rss),
a QCOM peer binding, and an AAPL-minus-QCOM five-session return composition. The dated
[Qualcomm agreement](https://investor.qualcomm.com/news-events/press-releases/news-details/2023/Qualcomm-Announces-Agreement-with-Apple-for-Chip-Supply-09-11-2023/default.aspx)
covers Snapdragon modem-RF supply for smartphone launches in 2024–2026; it is business
context, not evidence of a return relationship or a claim about every Apple device.
QCOM has no configured issuer feed, so related-news coverage remains explicitly partial.
Atom `updated` orders current context but is never substituted for original publication
or historical availability. Entries without `published` retain unknown publication time.

The daily collector accepts Yahoo's Nasdaq NMS/NGM/NCM metadata alongside NYSE-family
USD listings. Nasdaq collections retain the official
[holiday schedule and regular market hours](https://www.nasdaq.com/market-activity/stock-market-holiday-schedule)
and its hash, closures, 13:00 early closes, and 09:30–16:00 America/New_York sessions.
It does not substitute a NYSE calendar or extrapolate unpublished years. A changed/incomplete
calendar fails explicitly. Existing raw-price/corporate-action checks still apply.

Live collection checks on 2026-10-31 loaded 120 sessions through 2026-10-30 for AAPL,
QCOM and CCEP, with five-session targets on 2026-11-06. MNST passed exchange collection but
its retained Yahoo evidence contains a 2:1 split dated 2026-08-11, so the model loader rejected
that window as an unresolved corporate action. This is provider evidence, not independent
verification of the split or a reason to bypass price-continuity checks. Nasdaq listing
support does not guarantee every symbol/window is usable or certify point-in-time availability.

A connected live AAPL smoke run on that date completed seven Codex calls, two bounded
search requests and two LightGBM fits in 202.28 seconds. The initial fit used
`bound.price_momentum`, `aapl.relative_supplier_momentum` and generated 20-session
volatility; the revision used `bound.qlib_ma` and 15-session momentum. A separate KO
study with a simulated researcher and real retained daily prices fitted
`peer.CCEP.peer_return_5`, then `peer.CCEP.relative_return_10`. These demonstrate
provider/feature execution, not predictive superiority. Offline failure-injection tests
and browser checks cover continuation and retained-fit reuse; the live smoke did not
inject a real provider outage.


## Second-stock example: XOM

The same entry point supports XOM through three files in
[`instruments/US/XOM`](../instruments/US/XOM): issuer RSS, company relationships and
factor bindings. No stock-specific orchestration or model is required.

```bash
quant-rsi research-study XOM --news-research \
  --history-dir /tmp/xom-research-history --output /tmp/xom-study
```

This uses real Codex CLI calls, ExxonMobil's official press-release RSS, public
XOM/CVX daily prices and two actual model fits by default. Keep the history directory
for later studies and use a fresh output directory each time. As with KO, current
news guides hypotheses; new guidance extractions cannot fill historical feature
rows that precede their availability. Missing news history remains pending.

The configured CVX relationship comes from
[Tengizchevroil's year-end 2025 fact sheet](https://tengizchevroil.com/docs/default-source/factsheets/eng/2025/tco---factsheet---eng-q4.pdf?sfvrsn=4a4fcd5c_1).
It is a dated joint-venture link, not discovered return correlation or evidence of
causality. Both target and partner reuse the shared five-session momentum recipe;
`xom.relative_partner_momentum` subtracts the aligned CVX return from XOM's return.
The LLM may select, omit or revise these inputs. CVX has no registered issuer RSS
here, so feed coverage remains explicitly partial; active search does not silently
turn that missing feed into complete coverage.

Public data and search availability can vary. Examine the saved source coverage,
failed requests, actual selected factors and development/terminal tables before
interpreting a completed run. Neither a successful second-stock run nor lower
development error establishes predictive advantage.

## Bounded research iterations

To compare the selected LightGBM fit with actual upstream Qlib, use
[`--compare-qlib`](qlib_workflow_comparison.md) in the optional environment.
This adds one separately counted fit after native selection, reuses the same
features and evaluation rows, and adds comparison tables to this report/UI.
The research-round and `--max-fits` limits below govern native research fits.

Use `--rounds 5 --max-fits 4 --patience 2` to request up to five rounds with a four-fit
allowance and early stopping after two consecutive non-improvements. `--rounds` and
`--max-fits` accept 2–10; the fit allowance defaults to the round limit. `--patience`
accepts 1–10 and is disabled by default. One round is one hypothesis and one fit,
including the initial baseline. Configuration is recorded before any provider work.

Each successful trial is compared with the incumbent using the preselected development
metric. Ties, missing metrics and worse results retain the incumbent. Subsequent LLM
calls receive its hypothesis/feedback plus all previous trial hypotheses, factor IDs,
development metrics and decisions, including rejected trials. The same training rows,
development origins, strategy costs and baselines are used throughout. The final
latest forecast comes from the selected fitted model, even when an intermediate round
wins; it is not refitted or silently replaced with the last trial.

The UI reuses the feedback/revision/prediction nodes on each round and exposes every
stage in its selector. Decision cards show fit usage and the incumbent. The final
UI/HTML/Markdown report includes a round table and an explicit `round_limit`,
`fit_budget_exhausted` or `no_progress_limit` stopping reason. `result.json` retains
all attempts under `iterations`; `trace.jsonl` retains incremental stage outputs.
Provider/model failure stops the study and retains completed stages plus `failure.json`;
it does not retry, dispatch another fit or produce a successful final report. Failed
attempts are visible in runtime started/completed counts. New study directories can be explicitly continued with `research-resume` or the workbench;
historical directories are never migrated. See [recovery](#local-workbench-and-recovery).

The fit limit is not a currency, token or whole-run wall-clock budget. Active-news
research retains its separate bounded search/read behavior; LLM requests retain their
per-call timeout. Candidate selection uses **development** observations only. Terminal
evaluation follows the frozen selection; it does not influence another revision or
perform production promotion or modification of the researcher's own program (#3).
Complete live profiles are available for KO, XOM and AAPL; other symbols do not
automatically gain relationship profiles or registered news feeds.

## Reserved terminal evaluation

`research-study` reserves `--terminal-sessions 10` by default (accepted range 0–20).
`--terminal-sessions 0` explicitly restores the development-only interval and skips
terminal evaluation. `research-demo` retains its original behavior. The development
observation count still uses `--evaluation-sessions` (default 20); enabling reservation
moves development and training earlier, so results differ from older default studies.

The interval is written to `evaluation_plan.json` before news/LLM research. The last
development label **and strategy account valuation** precede the first terminal origin;
the selected prediction horizon is accounted for in the split. News/peer eligibility and
research masks use training/development availability, not terminal availability.

After the last research decision, `selected_attempt.json` records the chosen round,
factor IDs and development score. The `terminal_evaluation` stage calls that round's
already-fitted LightGBM or Ridge predictor once, without fitting another model or
calling the LLM again. The terminal outcome cannot change the selection. Every
candidate still uses the same development rows, and rejected candidates stay recorded.

`terminal_evaluation.json`, `result.json`, the final UI node and HTML/Markdown report
separate the following results:

- Development metrics used for feedback and selection.
- Reserved terminal predictions versus unchanged-price and training-label-mean baselines;
  costed strategy versus same-period buy-and-hold, including net return, Sharpe and drawdown.
- The selected model's latest forecast, when available, whose outcome is unobserved.

Terminal accounts start with their own initial cash and the same configured costs and
policy; do not concatenate development and terminal NAV curves. Missing selected news
or peer features make terminal evaluation explicitly unavailable on the fixed reserved
rows. The system neither drops rows nor switches to another model. Failure retains the
selection receipt and completed trace but does not produce a successful final report.

This is **retrospective evaluation reserved from feedback and selection**, not proof of
independent predictive improvement. Current news and pretrained LLM knowledge are not
historically isolated. Five-session outcomes overlap and these short-sample risk metrics
are descriptive; no multiple-testing correction or prospective validation is claimed.

For a complete offline example with real training and synthetic data:

```bash
quant-rsi research-study KO --llm simulated --news simulated --related simulated \
  --prices synthetic --rounds 4 --terminal-sessions 10 --no-ui \
  --output /tmp/ko-terminal-study
```

For Ridge, add `--model ridge` and use a new output directory. This illustrates the
same connected path, not a controlled comparison of separately selected models.

## Live research UI

```bash
quant-rsi research-study KO --news-research --output /tmp/ko-active-study-next
```

The command opens a read-only local browser view. Ten nodes connect market data, company
context, news research, the first hypothesis and factor selection, prediction, feedback,
factor revision, the second prediction and the final report. The active node glows; completed
nodes remain lit. Click a node to inspect its stage outputs, or return to live following.
Outputs appear when a stage finishes; this is not token streaming. Fast stages can complete
between refreshes and remain available in the history.

The task panel follows the current stage. The final node completes only after the research
artifacts are saved, then the page shows the development selection, conclusion and full
prediction/strategy report. Failures remain failures: the page retains previous outputs and
shows the failed stage. Simulated inputs and development-only evidence are labelled.

Node outputs use visual views rather than a raw JSON wall:

- Company context shows a relationship diagram; discovered suggestions stay marked unverified.
- News cards show sources, publication/receipt times and evidence IDs. Hypotheses link to
  retained evidence; no confidence scores or relationship weights are invented.
- Failed reads show the reason and final requested URL; application-triggered recovery
  queries are distinguished from LLM-proposed follow-ups.
- Prediction and conclusion views trace news citations → hypothesis → fitted inputs.
  Each round lists actual fitted columns separately from unselected or unavailable news
  definitions, with coverage reasons. This is provenance, not feature attribution or a
  measured causal news effect. Rejected trials remain visible; only the selected model
  supplies the latest forecast.
- Factor cards show executable recipes. Revision views distinguish additions, removals,
  parameter changes and recipe replacements with different IDs.
- Prediction charts compare observed prices, both attempts and the unchanged-price baseline,
  aligned by origin and target session. A keyboard slider inspects each observation;
  an expandable table gives exact values. Feedback ranks the largest absolute errors.
- Final views compare net return, Sharpe, drawdown and account NAV against buy-and-hold.
  Missing values stay unavailable or appear as chart gaps; higher MAE is labelled higher error.

Each output retains a collapsed **Inspect raw JSON** section. News excerpts and large
catalogs are bounded for readability; the original artifacts retain full detail. Visuals
only project retained outputs and do not change factor execution or research selection.

- `--no-ui` keeps the command terminal-only.
- `--ui` explicitly enables the view, including when output is redirected.
- Noninteractive commands and `--json` default to no UI. `--quiet` only hides terminal stage
  messages; it does not override the UI setting.
- The printed URL binds to `127.0.0.1` with a random per-run path. If a browser cannot open,
  use that URL on the same machine (or forward its port over SSH). Nothing binds publicly.
- The viewer uses existing `runtime.json` and `trace.jsonl` files and cannot submit research
  or trading actions. Closing the browser does not cancel research. Its helper exits after
  two minutes without requests; a completed page retains its results in the browser.
- After completion or a handled failure, open `live.html` directly for the saved view,
  without a server, network access, frontend build or additional dependencies. Long stage
  outputs have a bounded preview; `trace.jsonl` retains the full output.

## Offline example

For a fully offline run after installation, keep real model fitting and replace
the external context and prices:

```bash
quant-rsi research-study KO --output /tmp/ko-study-offline \
  --llm simulated --related simulated --news simulated --prices synthetic
```

## Inputs and controls

### Continue research across market days

Opt into one shared history directory while keeping each study output directory fresh:

```bash
quant-rsi research-study KO --news-research \
  --history-dir /tmp/ko-research-history --output /tmp/ko-study-day-one

# On a later day, collect new prices and assess older predictions before new research.
quant-rsi research-study KO --news-research \
  --history-dir /tmp/ko-research-history --output /tmp/ko-study-day-two

# Observe without another LLM call or model fit. Use a new collection directory.
quant-rsi research-history observe KO --history-dir /tmp/ko-research-history \
  --prices-dir /tmp/ko-observation-prices --ui
quant-rsi research-history status KO --history-dir /tmp/ko-research-history
```

The first command saves the **selected round's** latest forecast, factor IDs,
training cutoff, costs and original result in the existing content-addressed store
and Journal. A post-commit clock witness uses the existing publication confirmation
rule: issuance and confirmation must occur after the origin close and before the
next registered open. Late or unconfirmed publications remain `not_prospective`;
synthetic or unavailable latest forecasts cannot become real observations.

The next study assesses earlier forecasts against its newly received daily collection
before making any hypothesis calls. Only mature, eligible outcomes from the latest
assessment enter `context.prior_outcomes`; the exact input is also saved in
`prior_outcomes.json`. This is cross-run research feedback, separate from within-run
development revisions. Once supplied to the researcher, it is no longer untouched
confirmation evidence. The existing retrospective terminal-window limitations remain.

Later public collections have new source versions. The adapter matches provider
endpoints, instrument, raw-price/currency conventions and the six frozen calendar
sessions. It does not claim these separate collections extend an old market registry's
receipt prefix. Revised origin prices, changed calendars and missing periods cannot
produce a scored outcome. Corrections append assessments; the latest assessment wins,
including an invalidating correction. Repeating an identical receipt is idempotent,
and corrections are never counted as additional predictions. Original studies and
their saved reports are not rewritten.

Mature outcomes include price error, unchanged-price forecast control, costed return,
Sharpe, drawdown and buy-and-hold through the existing account evaluator. These are
**hypothetical replays of the frozen signal on later bars**, not durable paper orders
or broker fills. Five daily returns do not establish a stable Sharpe estimate. There
is no new scheduler, automatic order submission or guarantee of predictive improvement.

The existing UI adds a forecast-history table. Opening it again reads current history;
`research-history ... --ui` opens the latest study view after observing. Saved HTML
and each study report remain snapshots. The table shows the latest 50 entries for the
symbol; at most 20 mature outcomes from that view enter the LLM context. All receipts
and corrections remain in the journal. Without `--history-dir`, studies stay isolated.

The **Continuing the previous research** view uses the history snapshot retained in
`prior_outcomes.json` and the target trace at this run's start. It shows which settled
outcomes were supplied to the LLM, pending targets without invented errors, and changes
between the previous run's selected factors and this run's selected factors. The LLM
is asked to explain how mature outcomes motivate its hypothesis; supplying feedback
does not prove it used that feedback. This frozen explanation stays separate from the
live history table, which may receive new observations after the run completes.

### Reusing selected factor recipes

With `--history-dir` (already provided by the workbench), a new study reads selected
recipes from the latest five studies of the same stock. It reads immutable history
artifacts, not the prior run's mutable filesystem path. Older entries can recover the
selected definitions from their frozen catalogs without rewriting those entries.

Bounded generated price and headline recipes are offered to the next LLM and can enter
its actual feature matrix. Their numeric values are recomputed from current inputs;
news eligibility is recalculated from the current retained archive. Financial, peer and
composed factors must already exist in the current configured library. A disabled
financial source, missing peer binding, conflicting definition or unsupported recipe
cannot be revived by an old `runnable=true` flag. Current catalog definitions take
precedence, and the most recently selected version of each remembered ID is considered.

`factor_memory.json`, the live factor stage and the **Recipes from previous studies**
report table record the source study/result reference, input mode, recipe hash,
current availability and whether the recipe was selected again. Pending/unavailable
forecast statuses are shown without invented outcome metrics; only the existing mature
outcome path supplies observed errors. Unselected candidates remain in retained history.
This is not global factor deletion, automatic promotion, or an independent assessment
of factor effectiveness. Studies without shared history remain isolated.
New memory alone does not count as new external evidence for scheduled research.

A real-service check on 2026-11-01 registered the retained KO study from the financial
source acceptance into a separate history at the current receipt time, then ran a new
Codex study with public prices/current RSS and retained issuer financials. The first
fit selected the remembered `research.momentum_20`, `financial.operating_margin_yoy`
and `ko.relative_bottler_momentum` recipes. Its revised combination changed to momentum
10 and revenue growth. Development MAE was 1.4078 initially and 2.8777 after revision
(unchanged-price baseline 0.9460), so round 1 was retained. There were two actual Codex
hypothesis calls, two LightGBM research fits and one financial-removal diagnostic;
this second check did not request active news search. The prior forecast was still
pending, and no realized error for it was supplied. This proves the connection works,
not that reuse improves prediction. Local artifacts are
`/tmp/quant-rsi-ko-factor-memory-study/` and `/tmp/quant-rsi-factor-memory-real-history/`.
Offline tests additionally verify unavailable news, recipe conflicts, symbol isolation,
legacy artifact recovery and unchanged completed study files.

#### Event-specific recipe lifetimes

News proposals declare `reuse_for_hours`: `null` keeps a recurring headline recipe
reusable; an integer from 1 to 720 limits an event-specific hypothesis. For example,
the LLM can propose a short-lived hypothesis about a particular earnings release:

```json
{
  "name": "release_demand",
  "keywords": ["demand", "earnings"],
  "window_days": 7,
  "description": "Test the demand hypothesis motivated by the supplied release.",
  "reuse_for_hours": 48
}
```

The study freezes `recipe_as_of` in its existing operation cache after collecting
inputs and before proposing factors. The first admitted definition records this
clock, its deadline and the hypothesis's supplied news evidence IDs. These IDs document
the hypothesis context; they do not turn current articles into historical observations.
`window_days` still controls the historical headline count independently of this
lifetime. The numerical feature and its receipt-time checks are unchanged.

A new study at or after the original deadline marks that recipe `expired_event_recipe`
and excludes it from selection, even if current news coverage is complete. Repeated
selection retains the original deadline. Within the current catalog, re-proposing the same ID cannot change its
recipe or lifetime, and cannot reactivate it. A new event hypothesis requires a distinct
name and an explicit proposal with supplied news evidence; automatic semantic identity
or source-retraction detection is not implemented. The latest-five-study memory
is not a permanent global reservation of factor names; a separately proposed instance
after a recipe falls outside that memory has its own recorded admission and evidence. The LLM is instructed not to rename
an expired recipe just to extend it.

Existing records without a declared lifetime keep their prior behavior. Unselected
recipes and expired definitions remain in immutable history; no old result, feature
column or forecast is deleted or rewritten. Resuming an already-admitted study uses
its original clock and completed fits, including after the deadline, rather than
changing that study's frozen experiment. A new study checks the current clock again.
Expiry controls research candidate admission, not broker positions or quality promotion.
It does not itself trigger scheduled research. Reports and live factor cards show the
original deadline and expired candidates. These are engineering controls, not proof
that an event's market effect lasts for the chosen number of hours.

Acceptance on 2026-11-01 used two real Codex sessions with synthetic prices/news and
actual Ridge training. The recurring-demand fixture selected `reuse_for_hours: null`.
A separately labelled, fictional two-day plant-maintenance fixture caused Codex to
propose a 48-hour event recipe, select it for the first fit, and omit it after feedback.
Both sessions completed two hypothesis calls and two fits. These are transport and
execution checks, not real company news or market-performance evidence. Artifacts:
`/tmp/quant-rsi-event-recipe-codex-study/` and
`/tmp/quant-rsi-event-recipe-codex-fixture-study/`.
Controlled-clock tests additionally run complete studies across the deadline, preserve
old results, reject renewal, and resume an interrupted study without changing its
clock or completed fit. Browser checks cover the lifetime card and expired-candidate
report table. Actual elapsed-market expiry is not claimed by these fixtures.

#### Source retention and later observations

With shared history, issuer RSS snapshots default to
`HISTORY/news_archive/real/SYMBOL`; synthetic-price history uses a separate `simulated`
namespace. An explicit `--news-archive` still overrides this path. Original receipt
times and existing archive size/count limits are preserved; current news is never
backdated to manufacture historical training rows. Ordinary experiments can record
pending results immediately; actual maturity requires elapsed market time. Controlled
clock tests exercise this workflow without pretending that such time has elapsed.

A 2026-10-31 real Codex acceptance run completed six LLM calls, two news searches
and two LightGBM fits, selected round 2 and registered a pending KO forecast from
2026-10-30 to 2026-11-06 (reference 94.0800, predicted 93.3007). One issuer RSS
snapshot was retained in the shared archive. Its outcome was not yet observed.
A separate controlled-clock, synthetic-source acceptance ran two complete studies,
settled the first prediction and supplied its error/account metrics to the second
hypothesis. This proves engineering continuity; it is not an elapsed real-market result.

Daily prices are collected under `OUTPUT/prices` unless you supply a retained
collection with `--prices-dir PATH`. Populated collections are verified and reused,
not refreshed or overwritten. Freshness is bounded by their receipt. These are
raw USD prices, not dividend-adjusted total returns; unsupported corporate actions
reject the input. See [price details](simulated_research_loop.md#use-collected-daily-prices).

KO's profile supplies CCEP/KOF bottling and MNST distribution relationships. Only KO
has a configured RSS feed: related coverage is explicitly partial, with CCEP/MNST/KOF
`not_configured`. The default remains configured relationships and RSS; opt-in
[active news research](#active-news-research) adds search, page reads and related
company suggestions. RSS news limits apply per issuer; see
[coverage details](simulated_research_loop.md#use-configured-company-relationships).

| Option | Default and meaning |
| --- | --- |
| `--llm` | `codex`; also `api`, `deepseek`, `cursor` or `simulated` |
| `--llm-model`, `--llm-timeout` | Optional Codex or DeepSeek model (DeepSeek default `deepseek-chat`); API model required; timeout 180 seconds per call |
| `--cursor-dir` | Directory for `--llm cursor` request and response files; default `OUTPUT/cursor-llm` |
| `--model` | `lightgbm` by default; `ridge` uses a reviewed regularized linear model on the same rows and features |
| `--ridge-alpha` | Positive finite penalty for Ridge, default 1.0; rejected when supplied with LightGBM |
| `--related`, `--news`, `--prices` | `profile`, `issuer-rss`, `daily`; each also accepts `simulated`, except prices use `synthetic` |
| `--related-prices` / `--no-related-prices` | Enabled by default; collect at most three distinct related symbols once, with no silent replacement of unavailable data |
| `--related-prices-dir` | Optional root of retained daily collections, one subdirectory per peer symbol; otherwise `OUTPUT/related_prices`. Requires daily prices and enabled related prices |
| `--news-limit` | 5 items per issuer, range 1–20 |
| `--news-research` | Off by default; enable LLM-led current search, article reading and source-cited related company candidates |
| `--news-archive` | Optional shared snapshot directory; otherwise `OUTPUT/news_archive` |
| `--evaluation-sessions` | 20 historical origins, range 2–20 |
| `--initial-cash` | 10000 USD per strategy/baseline account |
| `--commission-bps`, `--slippage-bps` | 5 basis points each, on each executed side |
| `--threshold` | 0: a predicted five-session log return above this threshold chooses long, otherwise cash |
| `--selection-metric` | `mae` by default (lower is better); optionally `sharpe` (higher is better) for development-only adoption |
| `--quiet` | Hide stderr progress; timing and usage artifacts are still saved |

API mode requires `OPENAI_API_KEY` and `--llm-model MODEL_ID`. DeepSeek mode requires
`DEEPSEEK_API_KEY` and calls `https://api.deepseek.com/chat/completions` with JSON
mode; omit `--llm-model` to use `deepseek-chat`. Cursor mode does not call a model
service: each hypothesis writes `round-N.request.json` under `--cursor-dir` and waits
until `round-N.response.json` is replaced into that directory. News-research web
search stays on Codex or the OpenAI API. See
[LLM prerequisites](simulated_research_loop.md#use-a-real-llm-for-hypotheses).
Service errors stop the study without silent fallback. Explicit `research-resume` or
**Continue study** retries the failed work using retained checkpoints.
Optional related-price collection failures are recorded per symbol and leave those peers
unavailable; they do not stop an otherwise usable target-price study.

### Related stocks as executable inputs

After optional news discovery, the study tries configured related symbols first, then
unique suggested symbols, with a maximum of three collection attempts. Successful market
data access does not verify a suggested corporate relationship. Simulated relationships
and simulated discovered symbols are never sent to a live market-data service. In synthetic
mode, peer prices are deterministic invented series, explicitly labelled as such.

The run-local catalog offers two recipes for each usable peer, at 5, 10 and 20 sessions:

- `peer.SYMBOL.peer_return_N`: peer close at the decision session divided by its close
  N target sessions earlier, minus one.
- `peer.SYMBOL.relative_return_N`: target simple return minus that peer simple return.

The LLM may select these alongside its proposed target-price recipe and change the chosen
peer, operator or supplied window after feedback. Catalog availability alone does not mean
a feature was selected or trained. The simulated hypothesis exercises a peer return first,
then a relative return after feedback.

Prices align by exact session date and an aware peer close timestamp no later than the
target decision close. Every close in the trailing window must be present and positive.
There is no forward fill, zero substitution, or peer-dependent sample removal. A recipe is
selectable only when it is finite on every already-frozen training and development row;
both attempts and their baselines retain the same rows. A missing latest peer window makes
the selected attempt's latest forecast unavailable without discarding its historical result
or switching to another model.

The daily adapter supports NYSE, NYSE Arca, NYSE American and Nasdaq USD listings.
KO's CCEP/MNST peers now pass the exchange filter and use Nasdaq's own calendar;
CCEP's current retained collection is usable, while MNST's current window is excluded
by a separate unresolved split check. The profile also includes KOF, the NYSE-listed ADS
of Coca-Cola FEMSA. Compatible peers can enter features when their retained collections
cover every required row; otherwise the study continues with its other executable factors.
See [Nasdaq coverage](#nasdaq-example-aapl) for evidence and limits.

`related_prices.json` retains aligned closes, source metadata/hashes, availability and
recipe definitions. Daily collection directories preserve their original receipts and are
reused without refreshing populated directories. UI prediction nodes and HTML/Markdown
reports show each input as **used**, **not selected**, **unavailable**, or **context only**.
Current news remains hypothesis context; only eligible archived headlines enter news columns.

## Choose the model before research

The default remains LightGBM. To use the simple linear alternative with the same
news, factor revision, strategy evaluation and selection stages:

```bash
quant-rsi research-study KO --model ridge --ridge-alpha 1.0 \
  --output /tmp/ko-ridge-study
```

For an offline check, append `--llm simulated --related simulated --news simulated
--prices synthetic`. Both attempts then perform real Ridge fits on synthetic prices
and retained synthetic news; the report identifies those sources.

The version-1 model configuration is saved before collection and passed to both
hypothesis calls. It records the model name, parameters, preprocessing, target and
fixed mature-label training policy. Neither model type nor penalty changes after
feedback. `--ridge-alpha` must be finite and positive, and is rejected with LightGBM
instead of being silently ignored.

Ridge centers each input using its training mean and scales by its training
population standard deviation; constant columns use scale one. It minimizes
`sum((y - intercept - Z @ coefficients)^2) + alpha * sum(coefficients^2)` for
five-session log returns, with an unpenalized intercept. A NumPy augmented
least-squares solve supports duplicate/constant columns without adding a dependency.
The trace records means, scales, coefficients and intercept in `factor_ids` order.
Evaluation rows and latest forecast inputs never fit the scaler. The selected
attempt's model and scaler produce its latest forecast without a third fit.

LightGBM retains its existing deterministic parameters and 24 boosting rounds,
without scaling. Both model choices retain the same maturity rules, common news
mask and unchanged-price/training-mean controls. Separate CLI runs can receive
different LLM factors: merely running each model once is not a controlled model
comparison. Freeze data, factor columns, origins and information budgets before
making that claim. This completes the alternative-model interface in #39 B; it
does not complete multi-profile, external-engine or independent validation work.

## Active news research

```bash
quant-rsi research-study KO --news-research --output /tmp/ko-active-study
```

The same Codex CLI login used for hypotheses now also supports discovery; there is
no separate search API key. This still uses fresh ephemeral CLI calls, not the
conversation history of a previous Codex session. API mode uses `OPENAI_API_KEY`
and `--llm-model MODEL_ID`; choose a model supporting Responses `web_search` and
structured outputs. Provider/model errors are reported, not silently replaced.
See official [Codex web search](https://learn.chatgpt.com/docs/web-search) and
[Responses web search](https://developers.openai.com/api/docs/guides/tools-web-search).

The prelude makes one discovery request: the LLM chooses a focused query and
returns up to three source URLs. The application reads those public pages, then
asks the LLM to review the extracted text, cite article IDs and suggest related
listed companies. It may request one follow-up search. If it declines but the first
search has no readable results or a failed read, the application uses that same
follow-up allowance to search for accessible official HTML coverage of the missing
story. It preserves the LLM's null response and records the recovery query separately.
The second review cannot extend the loop. There are at most two search requests and
six page-read attempts; recovery is not a guarantee that a replacement will be readable.
The prompts request one native search per request; Codex controls internal tool
calls, so this is not an exact provider-call or billing budget. API searches also
set `max_tool_calls=2`. The actual search-tool metadata is retained; a response
without a recorded search is rejected.

Read failures guide the follow-up query. HTTP 401, 403 and 429 add that exact
normalized hostname to this run's `avoid_hosts`, asking the LLM for alternative
accessible official/primary sources. If a result still uses a denied host, the
application records `skipped_denied_host` without making another page request;
its receipt time, text and hash remain absent. A 404 or timeout does not suppress
the entire hostname. This is a run-scoped exclusion/cooldown, not a permanent
blocklist or an attempt to bypass access restrictions.

Only discovery requests enable web search. Hypothesis and review requests keep
tools disabled; shell, apps and multi-agent tools remain disabled for all Codex
calls. Web pages and search results are treated as untrusted input data.

The reader accepts public HTTP(S) HTML/text pages, validates and pins public DNS
addresses, and revalidates every redirect. Gzip responses are supported, with separate
1 MiB limits on received and decompressed bytes; the hash identifies received bytes.
Each page is limited to 1 MiB with up
to three redirects and a 15-second socket-operation timeout. The reader strips
navigation/scripts and prefers article/main content, saving at most 16,000 text
characters; LLM context uses at most 8,000 per page. It does not execute JavaScript
or bypass paywalls, and does not assert that extracted text is the complete article.
Failures, truncation, hashes and actual receipt times remain visible. Publication
times are left unknown rather than inferred from search snippets or model memory.
Failed redirects retain the last validated requested URL so a denying destination
host can guide recovery. PDF and JavaScript-only sources remain unsupported; searches
prefer issuer/primary HTML or plain-text sources instead of adding a browser scraper.

Search summaries and tool actions are kept separately from independently retrieved
page text. Findings and related-company suggestions must cite successfully read
article IDs. The review response schema restricts both citation lists to the currently
readable IDs. When none are readable, it requires empty findings and company suggestions
while still allowing the permitted follow-up query. Runtime checks also reject invalid
citations from providers that ignore the schema; no citation is silently repaired.
Those citations establish which sources were supplied, not automatic
verification of each claim or ticker mapping. Suggestions enter both hypothesis
contexts alongside the existing configured profile, but do not rewrite that
profile, declare measured correlations or automatically add peer price features.
When at least one page is readable, each hypothesis must cite a discovered article
and explain its relevance to factor selection. If all reads fail, the study records
unavailable discovery and can continue using its original RSS/simulated context.

`news_research.json` saves search results, page text and final findings. All search,
read and review stages, including the intermediate follow-up decision, are in
`trace.jsonl`. The HTML/Markdown report shows sources, failures, company suggestions,
and article IDs cited by the initial and revised factor choices. Discovery is
current research context only: articles never enter `news_history.json` or the
receipt-gated historical headline-count matrix.

To exercise this exact path offline:

```bash
quant-rsi research-study KO --news-research --output /tmp/ko-active-offline \
  --llm simulated --related simulated --news simulated --prices synthetic
```

This uses invented search results, articles and a supplier candidate, explicitly
labeled simulated. It completes a follow-up query and changes momentum to trailing
volatility after feedback, with two real LightGBM fits on synthetic data.

## News archive and features

Use the same study command with a shared archive to retain RSS snapshots across runs:

```bash
quant-rsi research-study KO --output /tmp/ko-study-next --news-archive /tmp/ko-news
```

The archive stores content-addressed JSON snapshots, deduplicates identical
content and preserves each source's original `received_at`. Limits are 512 files,
8 MiB total and 512 KiB per file. These recorded receipts are not independently
authenticated timestamps. The run also saves its original snapshot bundle in
`news_history.json`; no additional ledger is created.

The workbench's new-evidence check passes its prepared RSS snapshot to the study,
just as it passes prepared price and financial collections. The trigger decision,
initial hypothesis and revision therefore share the same selected news. There is
one initial RSS acquisition per prepared task, not another fetch when research
starts. Explicit `--news-research` searches and article reads still run normally.

To reuse a retained RSS context directly:

```bash
quant-rsi research-study KO --news issuer-rss \
  --news-snapshot /tmp/ko-prepared-news.json --output /tmp/ko-reused-news
```

Use the JSON snapshot saved by preparation or a matching issuer RSS archive entry
that includes the original headline `text`. The target must match; the snapshot
must contain target news and remain within the normal 512 KiB bound and the
configured per-issuer `--news-limit`. Oversized selections are rejected rather
than truncated. All original source/item receipt timestamps and selected items
are preserved. `--news-snapshot` cannot be combined with simulated news.

The existing acquisition checkpoint retains the loaded snapshot. Continuing a
failed study reuses that completed acquisition even if the external input file
has since changed or been removed. To use newer news, start a new study. This
does not change historical news availability or refresh old receipts to the
current time. The collection stage records `acquisition: retained_snapshot`.

### Issuer guidance as an executable business factor

`--news-research` also makes one bounded structured extraction call over the selected
target issuer's RSS titles and summaries. Each item receives a classification:
`raised`, `lowered`, `reaffirmed`, `not_guidance`, or `ambiguous`, with an exact
source quote. This is an LLM interpretation of the supplied text, not independent
verification of the statement. Analyst targets and past earnings growth are not
management guidance. Full-page search results remain hypothesis context in this slice.

Two run-local catalog factors, `news.guidance_direction_7` and
`news.guidance_direction_30`, encode the latest explicit guidance statement within
7 or 30 calendar days: +1 raised, -1 lowered, 0 reaffirmed. A fully inspected selected
corpus with no explicit statement in the window also yields zero; the evidence
distinguishes these cases. The sign is a business fact representation, not a stock
return forecast. Conflicting simultaneous directions or ambiguous statements are
unavailable. The LLM can select either eligible window or remove the factor in a
later round, using the same existing selection and fitting path.

The feature requires a fresh issuer snapshot, precise publication times, and an
extraction completed by each historical session close. It must cover **every already
frozen training and development row**; it does not drop rows to make a feature usable
and does not inspect terminal outcomes to decide eligibility. New extractions made
today cannot fill yesterday's missing feature. A first real run will normally show
these factors as pending while price research continues.

`guidance_extractions.json` preserves the source snapshots, classifications, quotes,
provider and completion clocks. The UI news node shows the extraction and its sources;
the report shows eligibility and executed feature values. Real extraction records
are also retained in a sibling `<news-archive-name>_guidance` directory, with the same
file/byte bounds as news snapshots. Reusing identical input preserves its original
extraction time and avoids an extra LLM call. `--history-dir` therefore shares both
archives across runs. The archive is local evidence, not an authenticated publication
service, and prior completed studies are not rewritten.

For the complete extraction → numeric feature → fit → feedback → deactivation path:

```bash
quant-rsi research-study KO --news-research --llm simulated --news simulated \
  --related simulated --prices synthetic --no-ui --output /tmp/ko-guidance-offline
```

This fixture explicitly invents historical issuer statements and extraction clocks.
It performs real model fits and verifies engineering wiring; it supplies no evidence
of historical availability or prediction quality. Real sources with a simulated LLM
do not manufacture guidance extractions and remain unavailable.

A 2026-10-31 real Codex/issuer RSS/public-price acceptance run completed seven LLM
calls, two searches and two fits in about 197 seconds. It classified the retained
KO release's "Raises Full Year Guidance" text as `raised` and the other four selected
items as `not_guidance`. Both guidance windows correctly stayed pending because
their extraction occurred after the historical research cutoffs. The complete
synthetic example fitted guidance values {-1, 0, +1}, then removed that factor on
revision. These are separate extraction and engineering checks, not evidence that
real guidance features improved predictions.

### Headline-count recipes

The LLM can propose one or two news recipes with `name`, `keywords`, `window_days`
and `description`, identified as `news.<name>_<window_days>`. Each counts distinct
target-issuer headline IDs matching any of up to five case-insensitive literal
substrings (80 characters each) in a 1–30 calendar-day window. It counts the
**retained selected feed corpus**, not all news, summaries or full articles.

A row uses the newest snapshot/source receipt at or before its market-close
cutoff, no more than 24 hours old. Publications must have precise timestamps
inside the window and before the cutoff. Missing, stale, unavailable or date-only
evidence produces a missing value, not zero. A usable corpus with no keyword
matches produces zero. Daily prices use the real exchange-close instants; current
news received after those closes is never backdated.

Before either LLM call, the model freezes a common availability mask for both
attempts and their baselines. Executable news features need at least 12 usable
training rows and two evaluation rows; otherwise news proposals remain `pending`.
Selecting a pending factor is rejected, without silently replacing it with a price
feature. Eligible news factors are optional: either attempt may omit or deactivate
them. Reports distinguish eligibility from actual feature execution.
A first live collection will normally leave historical news factors
pending while price-only research remains usable. If selected news inputs are
missing at the latest origin, the latest forecast is suppressed with a reason.

With both `--prices synthetic` and `--news simulated` and an empty archive, the
offline command creates explicitly invented receipt history and fits actual news
columns. Simulated history is never mixed into a daily-price study. This exercises
the feature path; it does not establish news prediction value on market data.

## What the study does

1. Collect/read context and prices. The LLM proposes one to three numeric recipes
   using `momentum`, `mean_return`, `qlib_roc_v1` or `qlib_ma_v1` with windows 1–30,
   or `log_return_volatility` with windows 2–30.
   Recipes receive IDs such as `research.momentum_5`. News recipes follow the
   availability rules above and are recorded even when pending.
2. Fit the configured model once. All training labels mature strictly before the first
   evaluation origin. Predict the same set of matured outcomes at the selected horizon for
   the initial and revised attempts.
3. Report prediction errors against unchanged-price and training-label-mean
   baselines; run the corresponding long/cash strategy and buy-and-hold accounts.
4. Give development feedback to the second LLM call. Retain the proposed,
   activated, deactivated and retained factors in the trace, fit the revised recipe set and
   compare both attempts over the same training/evaluation windows.
   Adopt the revision only on strict improvement of the metric chosen before the run:
   lower price MAE by default, or higher net strategy Sharpe with `--selection-metric sharpe`.
   Ties and unavailable/nonfinite comparisons retain the initial attempt; another
   diagnostic cannot override the chosen metric.
5. In daily-price mode, report a separate latest-bar forecast with an unobserved five-session outcome.
   It uses the **selected** attempt's already fitted model, not a refit including newer matured labels;
   inspect its explicit training cutoff and retained price date.
   If that attempt lacks usable latest inputs, report it unavailable; do not silently
   switch to the other attempt. Both forecasts remain in their original stage records.

Recipes execute bounded operators. The study does not generate Python or execute
arbitrary code. Current news can inform recipe selection; archived headline counts
enter historical feature columns only when their receipt-time evidence is usable.

`research.log_return_volatility_N` adds a study-local numeric recipe with
`op=log_return_volatility`, window 2–30. It computes the unannualized sample standard
deviation (`ddof=1`) of N trailing daily close-to-close log returns, requiring N+1
complete finite positive closes. Incomplete windows remain unavailable; future
closes are not read. This is historical price volatility, not extracted news
sentiment. The protected bounded-engine factor contract is unchanged.

## Read the report

Prediction tables show MAE, RMSE, MAPE and directional agreement with the sample
count. These are historical **development** metrics, not an out-of-sample accuracy
claim: the revised proposal has already seen feedback from these same outcomes.
A smaller error or larger return in the revision does not establish independent gain.

Each attempt's strategy uses close-time signals and next-session open fills.
Positive signals enter long while flat; cash signals exit; unchanged long signals
hold existing units. Positions remain between supplied signals and through the
last forecast target close. Five-session prediction labels therefore differ from
the actual signal-to-signal trade intervals. Buy-and-hold uses the same period,
capital, lot size and costs, buying at the first eligible open. End inventory is
marked at the final close, without an invented terminal sale or fee.
Account quotes are rounded half-even to 12 decimal places to satisfy the existing
account contract; source prices and model inputs are unchanged.

The report includes net return, annualized Sharpe and volatility, maximum drawdown,
turnover, commission/slippage,
ending inventory and an HTML NAV plot from daily account valuations. Orders and
fills are hypothetical, using the existing account helpers; their timestamps do
not prove live or durable submission. Price forecasts and account returns answer
different questions. Read the development comparison separately from the latest
forecast, whose outcome remains `not_observed`.

Sharpe uses the mean of **daily costed simple NAV returns** divided by their sample
standard deviation (`ddof=1`), multiplied by `sqrt(252)`. Annualized volatility is
that standard deviation multiplied by `sqrt(252)`. The risk-free rate is explicitly
assumed to be **0%** for all three accounts, consistent with the zero-interest cash
simulation; it is not a current market rate. These statistics include cash days and
every daily valuation between signals, exclude the initial pre-trade cash mark, and
do not treat overlapping five-session prediction labels as daily returns.

The report displays the actual daily return count (normally 24 for 20 consecutive
forecast origins). The two strategy accounts and buy-and-hold use identical dates
and assumptions. Fewer than two returns, missing/nonpositive NAV, or zero volatility
make Sharpe `null` in JSON and `N/A` in reports, with an explicit status; flat
accounts still have zero volatility. Missing valuations are not silently skipped.
Short-sample annualization is descriptive and does not establish a stable long-term
Sharpe or independent improvement. These metrics also reach the second LLM call
through `feedback.strategy_diagnostics`; they do not establish independent
performance. The explicit
`--selection-metric` option now controls development adoption: it is saved in
`config.json` before collection/fitting and supplied to both hypotheses. The report
and `summary.selection` identify the chosen attempt, scores, factors and decision.
Trying a revised recipe is distinct from accepting it. Selection on these reused
development observations is not independent confirmation, even when the score improves.

## Progress and research cost

The CLI prints each running/completed stage and its elapsed seconds to **stderr**;
`--json` stdout stays machine-readable. `--quiet` suppresses only that display.
`runtime.json` is updated at stage boundaries and when usage arrives, so it also
records the active or failed stage when a run does not finish. It is observational
metadata; recovery uses separately validated `checkpoints/` receipts.

The report lists preparation/research stage durations, total measured research
wall time, started/completed LLM/search/model-stage counts and token usage. Counts
refer to application stages; they do not claim an exact count of provider-internal
tool calls or billing events. Simulated searches remain labeled by the simulation
mode and do not count as real LLM requests. Timing ends before report serialization.

Codex `turn.completed` events and API response usage supply only allowlisted numeric
input, cached-input and output token counts. Cached input is already included in
input totals. Missing fields remain unknown; partial totals are explicitly labeled.
No token estimates, authentication logs, private reasoning or currency costs are
inferred or saved. See the official [Codex JSON event example](https://learn.chatgpt.com/docs/non-interactive-mode).

## Saved output

| File | Contents |
| --- | --- |
| `config.json` | Requested run settings |
| `runtime.json` | Live/final stage timings, attempt counts and reported/unknown token usage; retained on failures |
| `inputs.json` | Price inputs and their provider/evidence metadata |
| `news_history.json` | Original retained news snapshot bundle used by this run |
| `related_prices.json` | Aligned peer closes, source evidence, availability and bounded factor definitions |
| `news_research.json` | Optional current discovery: search metadata/results, bounded page text/receipts, final findings and related-company candidates |
| `trace.jsonl` | Completed stage outputs, including proposals and feedback |
| `result.json` | Full structured results and account records |
| `report.md`, `report.html` | Human-readable comparisons; HTML includes NAV visualization |
| `live.html` | Saved interactive workflow view, when UI is enabled; final state and embedded report |
| `ui-server.log` | Local viewer diagnostics, when UI is enabled |
| `checkpoints/`, `attempts/` | Bound completed operations and preserved prior failed/interrupted attempts |
| `failure.json` | Written for failures after run initialization, retaining the completed trace for diagnosis |

Each study needs a fresh directory. Keep outputs and raw collections out of Git;
continue an interrupted new-version study with `research-resume PATH`, preserving its evidence.

## Relation to mature engines

[Qlib's workflow](https://github.com/microsoft/qlib/blob/main/docs/introduction/quick.rst)
connects datasets, model training, backtesting and evaluation.
[QuantDinger](https://github.com/OpenByteInc/QuantDinger) presents an integrated
research/strategy/backtest product. They are references for connecting useful
steps into one inspectable flow, not evidence that this study matches their scope
or results. By default this command uses local Qlib-derived ROC/MA adapters.
The optional [`--compare-qlib` path](qlib_workflow_comparison.md) also executes an
actual upstream dataset/model/recorder workflow on the selected common inputs.
It reuses the native account evaluator; broader portfolio-engine and research-policy
comparisons remain separate work. No Qlib/QuantDinger performance superiority is claimed.

## Recorded validation

On 2026-10-29, a price-feature-only live Codex study completed 14 stages, two model fits and 20
matured evaluation origins. It used three KO news items; CCEP/MNST remained
`not_configured`. The revision changed the selected recipes, but development MAE
worsened from `2.270949` to `2.411026`. Strategy net return fell from `2.8682%` to
`2.4878%`, versus buy-and-hold's `3.8996%`. These are retrospective development
results with reused feedback, not independent evidence of gain. This run verifies
the connected workflow while preserving an unfavorable revision outcome.
It did not validate historical news-feature training; that requires usable
receipt-time history rather than a newly fetched current feed.

On 2026-10-30, two further Codex studies each completed 14 stages and two LightGBM
fits. The synthetic study used 128 explicitly invented news snapshots, 46 training
rows and 20 evaluation rows. Codex selected `news.demand_mentions_7` in both
attempts; each fitted matrix contained a price column and an actual news-count
column with provenance. This verifies feature execution, not predictive quality.
The live study used three KO RSS items and retained market prices, but had zero
usable historical news rows. Its proposed results/guidance and dividend factors
remained pending; both fits used two price columns, and the latest forecast stayed
unobserved. It did not train on historical market news.

Also on 2026-10-30, an active-news study using saved Codex CLI authentication,
registered KO RSS and retained real daily prices completed 20 stages: two search
requests, six page-read attempts, two reviews, two hypotheses and two real fits.
One page was readable; five returned HTTP 403 and remained unavailable. Both
hypotheses cited the retrieved article and executed `log_return_volatility_10`.
The initial fit used two columns and the revised fit three. Development MAE
worsened from `2.219264` to `2.516885`; Sharpe fell from `3.795628` to `0.231598`.
The report preserved those negative outcomes and did not fabricate company links
from unreadable pages. A separate Codex review of a successfully retrieved official
CCEP relationship page proposed CCEP with that article's ID; it was a review
interface check, not a discovery result from the full study. API discovery was
validated offline, not with a live API key. None of these runs establish predictive gain.

A subsequent live Codex study on 2026-10-30 exercised explicit adoption and usage
reporting. It completed six LLM stages, two searches and two fits in about 165 seconds
of measured research time. MAE changed from `2.219264` to `2.516885`, so the default
rule retained the initial attempt and its latest forecast rather than the revision.
Three pages were attempted (one readable, two HTTP 403); the follow-up returned no
additional sources instead of repeating those denied hosts. All six LLM stages
reported usage: 150,191 input tokens, including 26,112 cached inputs, and 2,513 output
tokens. These are provider-reported counts, not a monetary-cost estimate or evidence
of predictive advantage. A final offline regression also checks that rendering
failure never announces a completed run.

Related-price integration was exercised on 2026-10-30 with retained public KO/KOF
daily collections. CCEP/MNST failed the existing exchange-scope check; their receipts
and reasons remained visible. With simulated news/hypotheses and real LightGBM fits,
the first attempt used `peer.KOF.peer_return_5`, and the revision used
`peer.KOF.relative_return_10`. MAE worsened from `2.3691` to `2.5643`, so the initial
attempt was retained.

A separate real Codex/RSS study reused those daily collections. Codex selected
`peer.KOF.relative_return_5` alongside momentum and volatility in its first fit,
then removed the peer input and selected `research.qlib_ma_v1_10` after feedback.
Both fits used 60 training rows and 20 common development observations. MAE changed
from `2.6801` to `2.5094`, still worse than the unchanged-price baseline (`2.5015`).
Net strategy return fell from `0.31%` to `-0.21%`; Sharpe fell from `0.2667` to
`-0.1911`. The preselected MAE rule adopted the revision despite those worse strategy
diagnostics. Historical news remained unavailable, and the latest forecast remained
unobserved. This demonstrates executable peer selection and deactivation by Codex,
not independent predictive improvement or a benefit from the peer factor.
