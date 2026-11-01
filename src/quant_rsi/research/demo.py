"""A visible, offline product loop; replace ordinary callables one stage at a time."""

import json
from copy import deepcopy
from pathlib import Path

from pydantic import TypeAdapter

from quant_rsi.factors.contracts import FactorCatalog, InstrumentId


def research_question(value):
    """One optional user focus, shared by CLI, queued tasks and research context."""
    if not isinstance(value, str) or len(value.strip()) > 1000:
        raise ValueError("Research question must be text of at most 1000 characters")
    return value.strip()


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
    research_ids = context["news"].get("research_evidence_ids", [])
    if research_ids:
        item = next(item for item in context["news"]["items"] if item["id"] == research_ids[0])
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
    if context.get("model_mode") in {"lightgbm", "ridge"}:
        available = [item["factor_id"] for item in library["definitions"]]
        factors = available[:1] if previous is None else available[1:]
        reason = "Try a numeric recipe, then change the numeric selection after model feedback."
    proposal = {}
    if context.get("allow_factor_proposals"):
        op = "qlib_ma_v1" if previous else "momentum"
        if previous and research_ids:
            op = "log_return_volatility"
        factors = [f"research.{op}_10"]
        reason = "Simulated researcher proposes a bounded recipe, then revises after feedback."
        if research_ids:
            reason = (
                "SIMULATED article demand uncertainty motivates a momentum hypothesis, "
                "then a volatility recipe after feedback; not real market evidence."
            )
        proposal = {"proposed_factors": [{"op": op, "window": 10, "description": reason}]}
        if "news_feature_support" in context:
            name, window = ("demand", 3) if previous else ("earnings", 7)
            proposal["proposed_news_factors"] = [
                {
                    "name": name,
                    "keywords": [name],
                    "window_days": window,
                    "description": "Simulated retained-headline count proposal.",
                }
            ]
            if context["news_feature_support"]["runnable"]:
                factors.append(f"news.{name}_{window}")
        guidance = [
            item
            for item in library["definitions"]
            if item.get("recipe", {}).get("op") == "guidance_direction" and item.get("runnable")
        ]
        if guidance and previous is None:
            factors.append(guidance[0]["factor_id"])
        business = [
            d
            for d in library["definitions"]
            if d.get("runnable") and d.get("recipe", {}).get("op") == "financial"
        ]
        if business:
            chosen = business[-1] if previous else business[0]
            factors.append(chosen["factor_id"])
            reason += (
                " SIMULATED business hypothesis tests reported growth, then revises the "
                "financial ratio after feedback."
            )
        peers = [
            item
            for item in library["definitions"]
            if item.get("runnable")
            and item.get("recipe", {}).get("op")
            == ("relative_return" if previous else "peer_return")
            and item["recipe"]["window"] == (10 if previous else 5)
        ]
        if peers:
            factors.append(peers[0]["factor_id"])
            reason += (
                " The simulated researcher includes a peer return, then tests a relative return."
            )
    return {
        **proposal,
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
    if any(definitions[key].get("availability_reason") == "event_recipe_expired" for key in ids):
        raise ValueError(
            "Selected event recipe has expired; its original deadline cannot be renewed"
        )
    if any(definitions[key].get("runnable") is False for key in ids):
        raise ValueError("Selected factor is pending: insufficient usable historical inputs")
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
    on_stage=None,
    on_start=None,
    rounds=2,
    fit_budget=None,
    patience=None,
    select=None,
    evaluate_selected=None,
):
    """One factor revision, no network, fits, workspace or persistence by default.

    Inputs/outputs are plain dictionaries. Inject a callable to replace one stage;
    keep the rest simulated. Providers identify themselves in their output.
    """
    from quant_rsi.research.study_iterations import iterate, validate_limits

    fit_budget = rounds if fit_budget is None else fit_budget
    validate_limits(rounds, fit_budget, patience)
    symbol = TypeAdapter(InstrumentId).validate_python(symbol.strip().upper())
    context = {"symbol": symbol, "reference_price": 100.0}
    if price_context is not None:
        context.update(price_context(symbol))
    trace = []

    def record(stage, value):
        trace.append({"stage": stage, "output": deepcopy(value)})
        if on_stage is not None:
            on_stage(deepcopy(trace[-1]))
        return value

    def call(stage, function, *args, **kwargs):
        if on_start is not None:
            on_start(stage)
        return record(stage, function(*args, **kwargs))

    call("target", lambda: {"provider": "simulated_price_context", **context})
    context["related"] = call("related_stocks", related, symbol)
    context["news"] = call("news", news, deepcopy(context))
    library = call("factor_library", catalog)
    iterations = iterate(
        context,
        library,
        hypothesize=hypothesize,
        predict=predict,
        observe=observe,
        resolve_factors=select_factors,
        call=call,
        rounds=rounds,
        fit_budget=fit_budget,
        patience=patience,
        select=select,
    )
    attempts = iterations["attempts"]
    initial, revised = attempts[0]["hypothesis"], attempts[-1]["hypothesis"]
    first, second = attempts[0]["prediction"], attempts[-1]["prediction"]
    terminal = (
        call("terminal_evaluation", evaluate_selected, deepcopy(iterations))
        if evaluate_selected is not None
        else None
    )
    summary = call(
        "summary",
        summarize,
        deepcopy(context),
        deepcopy(first),
        deepcopy(second),
        deepcopy(iterations["feedback"]),
        **({"iterations": deepcopy(iterations)} if select is not None else {}),
        **({"terminal_evaluation": deepcopy(terminal)} if terminal is not None else {}),
    )
    live_llm = any(
        hypothesis.get("provider") in {"codex_cli", "openai_api"}
        for hypothesis in (initial, revised)
    )
    live_news = context["news"].get("provider") == "issuer_rss"
    configured_related = context["related"].get("provider") == "instrument_relationship_profile"
    trained_model = first.get("provider") in {"lightgbm", "ridge"}
    model_label = "Ridge" if first.get("provider") == "ridge" else "LightGBM"
    notice = "Product wiring demo. News, LLM, prices and predictions are simulated by default."
    if live_llm or live_news:
        notice = (
            f"Product wiring demo with {'real' if live_llm else 'simulated'} LLM hypotheses "
            f"and {'real issuer RSS' if live_news else 'simulated'} news. "
            "Prices, predictions and observations are simulated; "
            "improvement is not evaluated."
        )
    if trained_model:
        notice = (
            f"Real {model_label} training with "
            f"{'real' if live_llm else 'simulated'} LLM hypotheses "
            f"and {'real issuer RSS' if live_news else 'simulated'} news. Prices "
            "and observed outcomes are synthetic. Both estimates use the same development origin; "
            "the revision reuses disclosed feedback. No independent improvement or market forecast."
        )
        if first.get("price_provider") == "public_daily_prices":
            notice = (
                f"Real daily prices and {model_label} training with "
                f"{'real' if live_llm else 'simulated'} LLM hypotheses "
                f"and {'real issuer RSS' if live_news else 'simulated'} news. "
                "Historical estimates reuse development feedback; "
                "the separate latest forecast has no observed outcome. "
                "No independent gain established."
            )
    notice += (
        " Related stocks use source-backed configuration, not automatic discovery or "
        "measured return correlations."
        if configured_related
        else " Related stocks remain simulated."
    )
    if context["news"].get("coverage_status") == "partial":
        notice += " Related-issuer news coverage is incomplete; see the news coverage entries."
    return {
        "mode": "mixed"
        if live_llm or live_news or trained_model or configured_related
        else "simulation",
        "notice": notice,
        "target": symbol,
        "factor_revisions": len(attempts) - 1,
        "iterations": iterations,
        "trace": trace,
        "summary": summary,
        **({"terminal_evaluation": terminal} if terminal is not None else {}),
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
