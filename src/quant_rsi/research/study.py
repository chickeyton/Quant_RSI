"""One connected research run using the demo stages and existing numerical/account engines."""

import json
import math
from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from quant_rsi.factors.contracts import InstrumentId
from quant_rsi.market.market_contracts import SideCosts, decimal_text
from quant_rsi.research.demo import (
    load_catalog,
    run_demo,
    simulated_hypothesis,
    simulated_news,
    simulated_related,
)
from quant_rsi.research.demo_model import LightGBMDemo
from quant_rsi.research.demo_news import issuer_news
from quant_rsi.research.demo_prices import load_daily_prices
from quant_rsi.research.demo_proposals import FactorProposal
from quant_rsi.research.demo_related import profile_related
from quant_rsi.research.demo_strategy import evaluate_strategy
from quant_rsi.research.llm import LLMHypothesizer
from quant_rsi.research.study_news import (
    NewsHistory,
    load_archive,
    retain_snapshot,
    simulated_history,
)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_study(args):
    """Keep a completed stage log even when a later provider fails; never overwrite a run."""
    symbol = TypeAdapter(InstrumentId).validate_python(args.symbol.strip().upper())
    if args.prices == "synthetic" and args.prices_dir is not None:
        raise ValueError("--prices-dir requires --prices daily")
    root = args.output.resolve()
    checkout = Path(__file__).resolve().parents[3]
    if root.is_relative_to(checkout):
        raise ValueError("Keep study artifacts outside the source checkout")
    if args.news_archive and args.news_archive.resolve().is_relative_to(checkout):
        raise ValueError("Keep news snapshots outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    configuration.update(
        symbol=symbol,
        model="lightgbm",
        research_rounds=2,
        generated_recipes="bounded_price_operators_and_retained_headline_counts",
    )
    _write(root / "config.json", configuration)
    last_stage = "initialization"

    def record(event):
        nonlocal last_stage
        with (root / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        last_stage = event["stage"]

    try:
        return _run(args, symbol, root, configuration, record)
    except Exception as exc:
        _write(
            root / "failure.json",
            {
                "status": "failed",
                "last_completed_stage": last_stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": "Completed trace stages are retained. This directory is not resumable.",
            },
        )
        raise


def _run(args, symbol, root, configuration, record):
    # Validate provider options before collection or fitting; no silent fallback.
    hypothesis = (
        simulated_hypothesis
        if args.llm == "simulated"
        else LLMHypothesizer(args.llm, args.llm_model, args.llm_timeout)
    )
    SideCosts(commission_bps=args.commission_bps, slippage_bps=args.slippage_bps, tax_bps="0")
    decimal_text(args.initial_cash, positive=True)
    if not math.isfinite(args.threshold):
        raise ValueError("Strategy threshold must be finite")
    library = load_catalog(args.catalog)
    for op in ("qlib_roc_v1", "qlib_ma_v1"):
        definition = FactorProposal(
            op=op, window=10, description="Source-pinned local Qlib-style price adapter."
        ).definition()
        if definition["factor_id"] not in {x["factor_id"] for x in library["definitions"]}:
            library["definitions"].append(definition)
    prices = (
        load_daily_prices(symbol, args.prices_dir or root / "prices")
        if args.prices == "daily"
        else None
    )
    model = LightGBMDemo(library, prices=prices, evaluation_sessions=args.evaluation_sessions)
    if prices is None:
        model.data = replace(model.data, instruments=(symbol,))
    related = (profile_related if args.related == "profile" else simulated_related)(symbol)
    news_context = {"symbol": symbol, "related": related}
    current_news = (
        issuer_news(news_context, limit=args.news_limit)
        if args.news == "issuer-rss"
        else simulated_news(news_context)
    )
    archive = args.news_archive or root / "news_archive"
    snapshots = load_archive(archive)
    if args.prices == "daily" and any(
        snapshot["provider"] != "issuer_rss" for snapshot in snapshots
    ):
        raise ValueError("Simulated news history cannot enter a daily-price study")
    if args.news == "issuer-rss":
        retain_snapshot(archive, current_news)
        snapshots = load_archive(archive)
    elif args.prices == "synthetic" and not snapshots:
        snapshots = simulated_history(model.data, symbol)
    history = NewsHistory(snapshots)
    cutoffs = (
        model.price_info["session_close_times"]
        if model.price_info
        else [model.data.timestamp(i, "close") for i in range(len(model.data.dates))]
    )
    model.configure_news(history, cutoffs, symbol)
    model.library["news_feature_support"] = model.context(symbol)["news_feature_support"]
    _write(root / "news_history.json", snapshots)
    _write(
        root / "inputs.json",
        {
            "price_provider": model.price_provider,
            "dates": list(model.data.dates),
            "target": symbol,
            "opens": model.data.opens[:, 0].tolist(),
            "closes": model.data.closes[:, 0].tolist(),
            "price_evidence": model.price_info,
            "news_history_file": "news_history.json",
            "news_snapshot_count": len(snapshots),
        },
    )

    def context(target):
        return model.context(target) | {"allow_factor_proposals": True}

    def predict(context, factors, proposal):
        prediction = model.predict(context, factors, proposal)
        prediction["strategy"] = evaluate_strategy(
            model.data,
            symbol,
            prediction["development"]["rows"],
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            initial_cash=args.initial_cash,
            threshold=args.threshold,
        )
        return prediction

    def observe(context, prediction):
        return model.observe(context, prediction) | {
            "strategy_diagnostics": {
                key: prediction["strategy"][key]["metrics"] for key in ("strategy", "buy_and_hold")
            },
        }

    result = run_demo(
        symbol,
        catalog=lambda: model.library,
        price_context=context,
        related=lambda target: related,
        news=lambda context: current_news,
        hypothesize=hypothesis,
        predict=predict,
        observe=observe,
        summarize=model.summarize,
        on_stage=record,
    )
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    first, second = (steps[key] for key in ("first_prediction", "second_prediction"))
    first_ids, second_ids = set(first["factor_ids"]), set(second["factor_ids"])
    metrics = second["development"]["metrics"]
    baseline = second["development"]["baselines"]["unchanged_price"]["metrics"]
    first_mae = first["development"]["metrics"]["mae_price"]
    change = metrics["mae_price"] - first_mae
    first_return = float(first["strategy"]["strategy"]["metrics"]["net_return"])
    revised_return = float(second["strategy"]["strategy"]["metrics"]["net_return"])
    buy_hold_return = float(second["strategy"]["buy_and_hold"]["metrics"]["net_return"])
    result["notice"] = (
        f"Connected research study with {model.price_provider} prices. "
        f"Hypotheses: {steps['initial_hypothesis']['provider']}; "
        f"news: {steps['news']['provider']}; "
        f"relationships: {steps['related_stocks']['provider']}. "
        "Both attempts use the same retrospective development window; no independent gain "
        "is established. Inspect source coverage and the separate unobserved latest forecast."
    )
    result["configuration"] = configuration
    news_definitions = [
        item for item in steps["revised_factor_library"]["definitions"] if item["kind"] == "news"
    ]
    news_used = any(
        item["kind"] == "news"
        for key in ("initial_factors", "revised_factors")
        for item in steps[key]
    )
    result["study"] = {
        "engine": "local_lightgbm_and_existing_finite_account",
        "factor_changes": {
            "activated": sorted(second_ids - first_ids),
            "deactivated": sorted(first_ids - second_ids),
            "retained": sorted(first_ids & second_ids),
            "meaning": "Run-local selection changes; definitions and evidence remain in the trace.",
        },
        "news_use": (
            "Selected retained-headline counts entered the fitted feature matrix. "
            "Counts cover the selected RSS corpus, not all issuer news; no sentiment is inferred."
            if news_used
            else "News informed hypotheses and recipe proposals only; no news column was fitted."
        ),
        "news_factors": {
            "support": model.context(symbol)["news_feature_support"],
            "definitions": news_definitions,
            "snapshot_count": len(snapshots),
            "archive_directory": str(archive.resolve()) if args.news == "issuer-rss" else None,
            "executed_initial": [
                x["factor_id"] for x in steps["initial_factors"] if x["kind"] == "news"
            ],
            "executed_revised": [
                x["factor_id"] for x in steps["revised_factors"] if x["kind"] == "news"
            ],
            "timing_note": "Original receipt times are preserved source records, not independently "
            "authenticated clocks. Real price cutoffs use the retained exchange calendar.",
        },
        "external_engine": "Qlib-style local price adapters; no Qlib or QuantDinger execution.",
        "conclusion": (
            f"{symbol}: completed two fits and one factor revision over "
            f"{metrics['sample_count']} development observations. Price MAE: initial "
            f"{first_mae:.4f}, revised {metrics['mae_price']:.4f}, "
            f"unchanged-price baseline {baseline['mae_price']:.4f}. "
            f"The revision {'increased' if change > 0 else 'decreased' if change < 0 else 'kept'} "
            f"MAE by {abs(change):.4f}. "
            f"Net strategy return: initial {first_return:.2%}, revised {revised_return:.2%}, "
            f"buy-and-hold {buy_hold_return:.2%}. "
            "These are retrospective development results after factor selection and feedback, "
            "not independent accuracy or evidence of improvement. Strategy results are "
            "hypothetical and include explicit costs. The latest forecast, when present, "
            "has no observed outcome."
        ),
    }
    from quant_rsi.research.study_report import render_study

    markdown, html = render_study(result)
    (root / "report.md").write_text(markdown, encoding="utf-8")
    (root / "report.html").write_text(html, encoding="utf-8")
    _write(root / "result.json", result)
    return result
