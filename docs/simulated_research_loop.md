# Run the simulated stock-research loop

The first product loop has one entry point. It lets you follow a stock from
context gathering to a prediction, observe a simulated result, change the chosen
factors using feedback and produce the next prediction. Start with the complete
interaction and replace individual services as they become useful.

## Start

After the [Linux/Python 3.12 installation](../README.md#installation-and-usage),
run from the repository root:

```bash
quant-rsi research-demo KO
```

The default output is a readable trace. Use JSON when inspecting the step outputs
or passing the result to another program:

```bash
quant-rsi research-demo KO --json
quant-rsi research-demo KO --catalog instruments/shared/equities.json
```

The default command does not collect data, call an external LLM, train a model or create
a research workspace. It reads actual `FactorCatalog` definitions from the given
file. Related companies, headlines, feature values, model outputs and observations
are simulated. A catalog definition describes a factor; its presence does not
mean the demo computed that factor from prices or extracted it from real news.

## Use a real LLM for hypotheses

Keep the same entry point and replace only the two hypothesis calls. The default
is `--llm simulated`; with the default news mode it remains offline and needs no credentials.

### Codex CLI with existing login

Install a Codex CLI version supporting `exec`, `--ephemeral`, `--output-schema`,
`--ignore-user-config` and `--ignore-rules`, and make `codex` available on `PATH`.
Authenticate using `codex login` if you have not already done so:

```bash
quant-rsi research-demo KO --llm codex
quant-rsi research-demo KO --llm codex --llm-model MODEL_ID --llm-timeout 240
```

Replace `MODEL_ID` with a model available to your account, or omit it to use the
Codex built-in default. The adapter reuses saved CLI authentication, but does not
resume this chat or any previous Codex conversation. Each hypothesis uses a fresh
`codex exec --ephemeral` invocation in a temporary directory. The second receives
the previous hypothesis and feedback explicitly. User configuration/rules are
ignored for these calls; shell, apps, multi-agent and web search are disabled.
See the official [Codex non-interactive guide](https://learn.chatgpt.com/docs/non-interactive-mode).

### OpenAI Responses API

Make `OPENAI_API_KEY` available in the process environment, then select a model
your API account can use with Responses API Structured Outputs:

```bash
quant-rsi research-demo KO --llm api --llm-model MODEL_ID --json
```

`--llm-model` is required for API mode. This adapter calls the official
`https://api.openai.com/v1/responses` endpoint directly using Python's standard
library; no additional SDK is required. It requests a strict JSON schema via
`text.format` and sets `store: false`. See the official
[Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
Codex login alone does not supply the API mode's `OPENAI_API_KEY`.

### What changes and what stays simulated

A successful live run makes two external LLM calls: an initial selection and one
revision after feedback. Both receive the stock/news context and actual catalog
definitions. Responses must select existing factor IDs, cite supplied news IDs,
and include a hypothesis and illustrative `event_signal` between `-1` and `1`.
The revision must change the selected factor set. These constraints demonstrate
the interaction; requiring a change does not establish a better research method.

`--llm-timeout` is a positive number of seconds per call, defaulting to `180`.
Missing credentials, unavailable CLI/model, connection/timeout failures, rejected
or invalid output, unknown factor IDs or an unchanged revision stop the command.
The adapter does not retry or fall back to simulated hypotheses.

The trace identifies `codex_cli` or `openai_api` and reports `mode: mixed` with a
simulation notice. Related stocks and price data remain synthetic. Predictions
use illustrative contributions by default, or actual fitting on synthetic prices
with `--model lightgbm`. News is simulated unless `--news issuer-rss` is selected. LLM outputs can
vary, so the exact default numbers below apply only to simulated mode. Neither a
successful response nor a revised factor selection proves predictive quality;
the default model's second prediction has no observed outcome, while LightGBM
reuses development feedback as described below. Live calls use the selected
account's service access and usage allowance.

On 2026-10-29, a live Codex CLI 0.165.0 smoke run using existing ChatGPT login
completed both calls for KO. It selected momentum/mean return, then revised to
mean return after simulated feedback. This verifies the connection, not prediction
accuracy; future LLM selections may differ. API mode was tested with offline
responses only because an API key was unavailable.

## Use real issuer news

The default `--news simulated` preserves the offline loop. To replace just news,
or combine real news with the live hypothesis adapter:

```bash
quant-rsi research-demo KO --news issuer-rss --news-limit 5 --json
quant-rsi research-demo KO --news issuer-rss --llm codex
```

`--news-limit` accepts integers from `1` to `20`, defaulting to `5`. The adapter
reuses the existing registered source, bounded transport and RSS parser in
[`market/news_feed.py`](../src/quant_rsi/market/news_feed.py). Currently only KO's
official issuer feed is registered. It makes one request capped at 256 KiB and
parses at most 128 items, selecting the newest items up to the requested limit.
Responses reaching the byte limit or exceeding the item bound are rejected.
It does not search
the web, fetch article bodies, poll additional pages or collect peer-company news.

Each selected item includes its ID, symbol, URL, title (up to 300 characters) and
summary (up to 2,000 characters), with HTML stripped and truncation flagged,
along with publication date/time and precision and the feed's
`received_at`. Selection/count metadata shows how many items reach the LLM.
Publication precision is preserved; a date-only item is not assigned an invented
publication time. RSS titles and summaries are limited source context, not full
articles or a claim of complete news coverage. Summaries may be empty; those
items supply only a title.

The same selected news context reaches both hypothesis calls. In Codex/API mode,
the model is told whether news is real or simulated through its provider label;
its illustrative `event_signal` still is not an executed or calibrated text factor.
Related companies and prices remain synthetic. The default model and observation
are simulated; optional LightGBM uses the synthetic price path for fitting and
observation. Live news alone also marks the output as a mixed simulation.

Unsupported symbols, an empty feed, transport failures and invalid feed content
stop the run. There is no fallback to invented news or historical backfill. This
entry point creates no research workspace or persistent news observation ledger.
For retained collection and observation, see [bounded news collection](news_collection.md).

On 2026-10-29, a live run with `--news issuer-rss --news-limit 3 --llm codex`
received HTTP 200, selected 3 of 10 feed items and completed both Codex CLI 0.165.0
calls using saved authentication. The real news reached both calls, and feedback
led to a changed factor set. This verifies news-to-hypothesis wiring; the model,
prices and observation remained simulated, so it establishes no forecast value.

## Fit LightGBM on synthetic prices

Replace the prediction stub without adding a data download or service dependency:

```bash
quant-rsi research-demo KO --model lightgbm --json
quant-rsi research-demo KO --model lightgbm --news issuer-rss --llm codex
```

The default remains `--model simulated`. LightGBM mode uses the existing installed
LightGBM dependency, shared model parameters and boosting rounds. It generates a
fixed-seed synthetic path of 128 sessions and uses session index `100` as the
forecast origin. Selected momentum/mean-return definitions become actual numeric
feature columns using the existing feature implementation. Training labels are
five-session log returns; only labels ending at or before the origin enter either
fit. Both fits use the same origin and training mask.

Before either hypothesis call, the catalog is filtered to executable numeric
definitions. The trace records excluded IDs and the reason. Text definitions are
excluded because this model has no historical text feature column. Real issuer
news can inform the LLM's choice of numeric factors, but current news is never
backfilled into historical training rows. `event_signal` does not become a model
feature in this mode.

The observed price is the synthetic close five sessions after the origin, rather
than the default preset `98`. That target is excluded from training. Once its
error informs factor revision, however, the second prediction is a **development
reestimate at the same origin**, not a fresh forecast with an independent outcome.
The summary reports both errors against that reused observation and labels
`second_outcome` as `reused_development_observation`; `improvement` remains
`not_evaluated`. A smaller second error would not establish predictive gain.

The execution is real feature calculation and model fitting; the price history,
outcome and related companies remain synthetic. This is still the product demo,
without a research workspace, market-data experiment or independent holdout.
On 2026-10-29, both the offline model run and a combined run with
`--model lightgbm --news issuer-rss --news-limit 3 --llm codex` completed.
The latter used Codex CLI 0.165.0 with saved authentication, two LLM calls and two
real LightGBM fits (91 training rows, 24 boosting rounds each). Codex revised
momentum to mean return; error against the reused synthetic observation changed
from approximately `-1.1710` to `-1.1936`, a worse absolute error. This verifies
the complete connection, not an independent gain or a real KO price forecast.

## Test an unmerged feature

An existing clone or installed CLI may still use older code. To test this feature
before merge, run the following from your existing repository. The separate
worktree and environment leave that checkout on its current branch:

```bash
git fetch origin feat/model-research-loop
git worktree add --detach ../Quant_RSI-model-demo FETCH_HEAD
cd ../Quant_RSI-model-demo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
quant-rsi research-demo --help
quant-rsi research-demo KO --model lightgbm
```

Use this while the feature branch is available; after merge, a fresh checkout of
updated `main` contains the feature. Check that `--help` lists `--model`, `--news` and `--llm`
before trying live calls. Installing editable code in the dedicated environment
ensures its CLI uses that worktree's source and catalog.

## Follow the whole interaction

1. Start with the requested stock symbol and obtain simulated related stocks and
   the selected news context (simulated by default, optional real issuer RSS).
2. Read the existing factor definitions and ask the selected LLM to choose an
   initial factor set.
3. Pass that selected set to the chosen model to produce the first prediction.
4. Obtain the simulated/synthetic observation and compare it with the first prediction.
5. Pass the feedback back to the selected LLM, which revises the factor set.
6. Run the chosen model again with the revised selection and summarize both
   attempts and the reason for the change.

The connection is real even when all services are simulated: the feedback
reaches the next selection, and the revised selection changes the next model
input. The default model's second prediction has not been observed; LightGBM's
second prediction reuses development feedback. Neither demonstrates that the
research method improved.

With the default catalog and services, the reference price is a simulated `100`.
Momentum contributes `+2`, so the first prediction is `102`. The preset observation
is `98`, giving feedback of `102 - 98 = +4`. The simulated LLM then chooses mean
return and announced events. Their fictional contributions are `+0.5` and `-1`,
so the next prediction is `99.5`. The earlier `98` observation belongs to the
first attempt; it is not a measured outcome for the second prediction.

The trace includes the library, both hypotheses, both selected factor lists,
both predictions, feedback and summary. These intermediate outputs make it
possible to see exactly where a replacement changes the flow.

## Minimal inputs and outputs

The Python entry point is `quant_rsi.research.demo.run_demo`. Services are ordinary
callables with plain dictionary inputs and outputs. `context` starts with `symbol`
and the simulated `reference_price`; an optional `price_context(symbol)` callback
supplies the LightGBM adapter's synthetic origin and reference price. The context
then gains `related` and `news`. Each service
can identify itself with a `provider` value in its output.

| Callable | Inputs | Output used by the loop/default services |
| --- | --- | --- |
| `related` | `symbol` | `stocks`: a nonempty list of related symbols |
| `news` | `context` | `items`: a nonempty list of items with `id`, `symbol`, `text` |
| `catalog` | None | `definitions`: existing factor definition dictionaries with `factor_id` |
| `hypothesize` | `context`, `library`; optional `previous`, `feedback` | `factor_ids`; a reason in `hypothesis`; `event_signal` when using the default model's text-factor contribution |
| `predict` | `context`, selected `factors`, `hypothesis` | `predicted_price`, `factor_ids` |
| `observe` | `context`, first `prediction` | `observed_price` |
| `summarize` | `context`, first and second predictions, `feedback` | Summary dictionary, normally including `text` |

`run_demo` calculates feedback as `predicted_price - observed_price`. It calls
`hypothesize` a second time with that feedback and the previous hypothesis,
resolves the newly selected IDs against the library, and calls `predict` again.
The returned dictionary contains `mode`, `target`, `factor_revisions`, `trace`
and `summary`. There is one revision per invocation.

The default catalog contains three IDs in
[`instruments/shared/equities.json`](../instruments/shared/equities.json).
The model stub supports momentum, mean return and text definitions. A different
catalog must support the chosen IDs, or be paired with a matching hypothesis/model
callable. Catalog text definitions remain metadata; the default `event_signal`
is fictional and does not run the extraction instruction. In LightGBM mode the
simulated LLM selects from the filtered numeric catalog, allowing an offline run.

## Replace one callable

For example, supply a different hypothesis function while retaining every other
simulated service:

```python
from quant_rsi.research.demo import render_demo, run_demo


def my_hypothesis(context, library, *, previous=None, feedback=None):
    selected = ["equity.price_mean_return_5"]
    if feedback is not None and feedback["error"] > 0:
        selected.append("equity.announced_events")
    return {
        "provider": "custom_simulated_hypothesis",
        "hypothesis": "Try mean return, then add an event signal after a high forecast.",
        "factor_ids": selected,
        "event_signal": -1.0,
        "feedback_used": feedback,
    }


print(render_demo(run_demo("KO", hypothesize=my_hypothesis)))
```

This example is still simulated. The CLI options above supply ready-made live
hypothesis adapters. For another provider, replace the body of this function
with your LLM call and return the same small dictionary. Give it the
stock/news context, available definitions, previous hypothesis and feedback.
Use a truthful `provider` label. No LLM SDK, credentials or prompt framework is
required to run the default loop.

A practical replacement order is:

1. Enable a live `hypothesize` adapter with `--llm codex` or `--llm api`. Keep news,
   prices, prediction and outcome simulated so you can inspect its choices and revisions.
2. Enable `--news issuer-rss` for KO, or replace `news`/`related` with another
   data/tool adapter independently. Other services retain their existing inputs and outputs.
3. Enable `--model lightgbm` for actual numeric feature calculation and fitting
   on synthetic data. Replace its price input with suitable real data in a later
   adapter; catalog metadata or live news alone does not supply historical prices.
4. Replace `observe` when actual outcomes become available, and `summarize` when
   a richer explanation is useful.

The CLI exposes the stock symbol, catalog path, output format, LLM, news and model options.
Other custom service injection uses the Python entry point. Built-in live LLM,
issuer-news or LightGBM providers set the aggregate mode to `mixed`; arbitrary injected callables do not
automatically change that label. Inspect provider labels and the callable
implementation to distinguish real parts from remaining simulated ones. Later, existing
[bounded research](research_workflow.md), data and model capabilities can supply
these adapters. They are not prerequisites for running this first product loop.
