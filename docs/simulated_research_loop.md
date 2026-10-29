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

The command does not collect data, call an external LLM, train a model or create
a research workspace. It reads actual `FactorCatalog` definitions from the given
file. Related companies, headlines, feature values, model outputs and observations
are simulated. A catalog definition describes a factor; its presence does not
mean the demo computed that factor from prices or extracted it from real news.

## Follow the whole interaction

1. Start with the requested stock symbol and obtain simulated related-stock/news
   context.
2. Read the existing factor definitions and ask the simulated LLM to choose an
   initial factor set.
3. Pass that selected set to the simulated model to produce the first prediction.
4. Obtain the simulated observation and compare it with the first prediction.
5. Pass the feedback back to the simulated LLM, which revises the factor set.
6. Run the simulated model again with the revised selection and summarize both
   attempts and the reason for the change.

The connection is real even though its services are simulated: the feedback
reaches the next selection, and the revised selection changes the next model
input. The second prediction has not been observed. Its existence or direction
does not demonstrate that the research method improved.

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
and the simulated `reference_price`, then gains `related` and `news`. Each service
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

The default simulated LLM expects the three IDs in
[`instruments/shared/equities.json`](../instruments/shared/equities.json).
The model stub supports momentum, mean return and text definitions. A different
catalog must support the chosen IDs, or be paired with a matching hypothesis/model
callable. Catalog text definitions remain metadata; the default `event_signal`
is fictional and does not run the extraction instruction.

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

This example is still simulated. To use an actual LLM, replace the body of that
function with your LLM call and return the same small dictionary. Give it the
stock/news context, available definitions, previous hypothesis and feedback.
Use a truthful `provider` label. No LLM SDK, credentials or prompt framework is
required to run the default loop.

A practical replacement order is:

1. Replace `hypothesize` with a real LLM. Keep news, prices, prediction and outcome
   simulated so you can inspect whether its choices and revisions are useful.
2. Replace `news` or `related` with a data/tool adapter, independently. The remaining
   services continue to work against their existing inputs and outputs.
3. Reuse or replace `catalog` to retrieve factor definitions. Connect actual
   feature calculation and a model through `predict`; real catalog metadata alone
   does not supply numerical feature values. The default reference price remains
   simulated until the prediction adapter supplies its real price context.
4. Replace `observe` when actual outcomes become available, and `summarize` when
   a richer explanation is useful.

The CLI exposes the stock symbol, catalog path and output format. Custom service
injection currently uses the Python entry point. The aggregate mode remains
`simulation` when injecting individual callables: inspect their provider labels
to distinguish the real parts from the remaining simulated ones. Later, existing
[bounded research](research_workflow.md), data and model capabilities can supply
these adapters. They are not prerequisites for running this first product loop.
