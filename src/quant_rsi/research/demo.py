"""A visible, offline product loop; replace ordinary callables one stage at a time."""

import json
from copy import deepcopy
from pathlib import Path

from pydantic import TypeAdapter

from quant_rsi.factors.contracts import FactorCatalog, InstrumentId


def load_catalog(path=Path("instruments/shared/equities.json")):
    """Reuse existing definitions as metadata, without making them executable."""
    catalog = FactorCatalog.model_validate_json(Path(path).read_bytes())
    return {
        "provider": "local_factor_catalog",
        "definitions": [item.model_dump(mode="json") for item in catalog.definitions],
    }


def simulated_related(symbol):
    return {
        "provider": "simulated_relationships",
        "stocks": ["DEMO_PEER_A", "DEMO_PEER_B"],
        "reason": f"Invented peers of {symbol}; no relationship search was performed.",
    }


def simulated_news(context):
    return {
        "provider": "simulated_news",
        "items": [
            {
                "id": "demo-demand",
                "symbol": context["symbol"],
                "text": "SIMULATED NEWS: demand is weaker than expected.",
            },
            {
                "id": "demo-peer",
                "symbol": context["related"]["stocks"][0],
                "text": "SIMULATED NEWS: a related company announces a price reduction.",
            },
        ],
    }


def simulated_hypothesis(context, library, *, previous=None, feedback=None):
    """A stand-in for an LLM reading news, the catalog and the previous error."""
    item = context["news"]["items"][0]
    event_signal = -1.0 if "weaker" in item["text"].lower() else 1.0
    if feedback is None:
        factors = ["equity.price_momentum_5"]
        reason = "Start with price momentum; retain the news hypothesis for feedback."
    elif feedback["error"] > 0:
        factors = ["equity.price_mean_return_5", "equity.announced_events"]
        reason = "The first prediction was too high; try mean return plus the news signal."
    else:
        factors = ["equity.price_momentum_5", "equity.price_mean_return_5"]
        reason = "The first prediction was not too high; add a second price factor."
    if context.get("model_mode") == "lightgbm":
        available = [item["factor_id"] for item in library["definitions"]]
        factors = available[:1] if previous is None else available[1:]
        reason = "Try a numeric recipe, then change the numeric selection after model feedback."
    return {
        "provider": "simulated_llm",
        "hypothesis": reason,
        "evidence_news_ids": [item["id"]],
        "event_signal": event_signal,
        "factor_ids": factors,
        "previous_factor_ids": previous["factor_ids"] if previous else [],
        "feedback_used": deepcopy(feedback),
    }


def select_factors(library, hypothesis):
    ids = hypothesis["factor_ids"]
    if not isinstance(ids, list) or not ids or not all(isinstance(key, str) for key in ids):
        raise ValueError("hypothesis must select a nonempty list of factor IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("hypothesis selected duplicate factor IDs")
    definitions = {item["factor_id"]: item for item in library["definitions"]}
    unknown = set(ids) - definitions.keys()
    if unknown:
        raise ValueError(f"hypothesis selected unknown factors: {sorted(unknown)}")
    return [deepcopy(definitions[key]) for key in ids]


def simulated_predict(context, factors, hypothesis):
    """Fixed illustrative contributions, not feature calculation or model training."""
    contributions = {}
    for factor in factors:
        if factor["kind"] == "text":
            value = hypothesis["event_signal"]
        else:
            recipe = factor["recipe"]
            coefficient = {"momentum": 2.0, "mean_return": 0.5}.get(recipe["op"])
            if coefficient is None:
                raise ValueError("this model stub only supports momentum, mean_return and text")
            value = coefficient * recipe["window"] / 5
        contributions[factor["factor_id"]] = value
    return {
        "provider": "simulated_model",
        "predicted_price": round(context["reference_price"] + sum(contributions.values()), 4),
        "factor_ids": [factor["factor_id"] for factor in factors],
        "simulated_contributions": contributions,
        "horizon_sessions": 5,
    }


def simulated_observe(context, prediction):
    return {"provider": "simulated_observation", "observed_price": 98.0}


def simulated_summary(context, first, second, feedback):
    return {
        "provider": "simulated_summary",
        "text": (
            f"{context['symbol']}: first simulated price {first['predicted_price']}, "
            f"preset observation {feedback['observed_price']}, error {feedback['error']:+g}. "
            f"After changing factors, simulated price is {second['predicted_price']}. "
            "The second outcome is not observed; improvement has not been evaluated."
        ),
        "factors_changed": first["factor_ids"] != second["factor_ids"],
        "second_outcome": "not_observed",
        "improvement": "not_evaluated",
    }


def run_demo(
    symbol,
    *,
    related=simulated_related,
    news=simulated_news,
    catalog=load_catalog,
    hypothesize=simulated_hypothesis,
    predict=simulated_predict,
    observe=simulated_observe,
    summarize=simulated_summary,
    price_context=None,
):
    """One factor revision, no network, fits, workspace or persistence by default.

    Inputs/outputs are plain dictionaries. Inject a callable to replace one stage;
    keep the rest simulated. Providers identify themselves in their output.
    """
    symbol = TypeAdapter(InstrumentId).validate_python(symbol.strip().upper())
    context = {"symbol": symbol, "reference_price": 100.0}
    if price_context is not None:
        context.update(price_context(symbol))
    trace = []

    def record(stage, value):
        trace.append({"stage": stage, "output": deepcopy(value)})
        return value

    record("target", {"provider": "simulated_price_context", **context})
    context["related"] = record("related_stocks", related(symbol))
    context["news"] = record("news", news(deepcopy(context)))
    library = record("factor_library", catalog())
    initial = record("initial_hypothesis", hypothesize(deepcopy(context), deepcopy(library)))
    factors = record("initial_factors", select_factors(library, initial))
    first = record("first_prediction", predict(deepcopy(context), factors, deepcopy(initial)))
    observed = observe(deepcopy(context), deepcopy(first))
    feedback = record(
        "feedback",
        {
            **observed,
            "predicted_price": first["predicted_price"],
            "error": round(first["predicted_price"] - observed["observed_price"], 4),
            "error_definition": "predicted_price_minus_observed_price",
        },
    )
    revised = record(
        "revised_hypothesis",
        hypothesize(
            deepcopy(context),
            deepcopy(library),
            previous=deepcopy(initial),
            feedback=deepcopy(feedback),
        ),
    )
    factors = record("revised_factors", select_factors(library, revised))
    second = record("second_prediction", predict(deepcopy(context), factors, deepcopy(revised)))
    summary = record(
        "summary",
        summarize(deepcopy(context), deepcopy(first), deepcopy(second), deepcopy(feedback)),
    )
    live_llm = any(
        hypothesis.get("provider") in {"codex_cli", "openai_api"}
        for hypothesis in (initial, revised)
    )
    live_news = context["news"].get("provider") == "issuer_rss"
    trained_model = first.get("provider") == "lightgbm"
    notice = "Product wiring demo. News, LLM, prices and predictions are simulated by default."
    if live_llm or live_news:
        notice = (
            f"Product wiring demo with {'real' if live_llm else 'simulated'} LLM hypotheses "
            f"and {'real issuer RSS' if live_news else 'simulated'} news. "
            "Related stocks, prices, predictions and observations are simulated; "
            "improvement is not evaluated."
        )
    if trained_model:
        notice = (
            f"Real LightGBM training with {'real' if live_llm else 'simulated'} LLM hypotheses "
            f"and {'real issuer RSS' if live_news else 'simulated'} news. Prices, related stocks "
            "and observed outcomes are synthetic. Both estimates use the same development origin; "
            "the revision reuses disclosed feedback. No independent improvement or market forecast."
        )
        if first.get("price_provider") == "public_daily_prices":
            notice = (
                "Real daily prices and LightGBM training with "
                f"{'real' if live_llm else 'simulated'} LLM hypotheses "
                f"and {'real issuer RSS' if live_news else 'simulated'} news. "
                "Related stocks remain simulated. Historical estimates reuse development feedback; "
                "the separate latest forecast has no observed outcome. "
                "No independent gain established."
            )
    return {
        "mode": "mixed" if live_llm or live_news or trained_model else "simulation",
        "notice": notice,
        "target": symbol,
        "factor_revisions": 1,
        "trace": trace,
        "summary": summary,
    }


def render_demo(result):
    label = "MIXED / SIMULATION" if result["mode"] == "mixed" else "SIMULATION"
    lines = [f"# {label}: {result['target']} research loop", "", result["notice"], ""]
    for index, event in enumerate(result["trace"], 1):
        lines.extend(
            [
                f"## {index}. {event['stage']}",
                "",
                "```json",
                json.dumps(event["output"], indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    return "\n".join(lines)
