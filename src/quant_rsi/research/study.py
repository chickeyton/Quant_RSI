"""One connected research run using the demo stages and existing numerical/account engines."""

import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from quant_rsi.factors.contracts import InstrumentId
from quant_rsi.market.market_contracts import SideCosts, decimal_text
from quant_rsi.research.active_news import attach_research, research_news
from quant_rsi.research.demo import (
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
from quant_rsi.research.study_estimators import model_configuration
from quant_rsi.research.study_factor_profiles import load_study_catalog
from quant_rsi.research.study_guidance import (
    NOTICE as GUIDANCE_NOTICE,
)
from quant_rsi.research.study_guidance import (
    GuidanceHistory,
    collect_guidance,
    extract_guidance,
)
from quant_rsi.research.study_history import (
    collection_snapshot,
    history_view,
    mature_feedback,
    observe_prices,
    register_study,
)
from quant_rsi.research.study_iterations import validate_limits
from quant_rsi.research.study_news import (
    NewsHistory,
    load_archive,
    load_snapshot,
    retain_snapshot,
    simulated_history,
)
from quant_rsi.research.study_peers import prepare_related_prices
from quant_rsi.research.study_progress import StudyProgress
from quant_rsi.research.study_resume import StudyCache, run_lock
from quant_rsi.research.study_selection import select_attempt


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def terminal_summary(terminal):
    summary = {
        "status": terminal["status"],
        "selected_round": terminal["selected_round"],
        "sample_count": terminal["sample_count"],
        "limitation": terminal["limitation"],
    }
    if terminal["status"] != "completed":
        return summary | {
            "text": "Terminal evaluation unavailable: selected features are missing. "
            "No rows were dropped and no alternative model was selected."
        }
    evaluation, strategy = terminal["evaluation"], terminal["strategy"]
    metrics = strategy["strategy"]["metrics"]
    sharpe = metrics["sharpe_ratio"]
    sharpe_text = f"{sharpe:.4f}" if sharpe is not None else "N/A"
    summary.update(
        metrics=evaluation["metrics"],
        baselines={name: value["metrics"] for name, value in evaluation["baselines"].items()},
        strategy_metrics=metrics,
        buy_and_hold_metrics=strategy["buy_and_hold"]["metrics"],
        text=f"Reserved terminal evaluation ({terminal['sample_count']} observations), "
        f"same fitted round {terminal['selected_round']}, no refit: "
        f"price MAE {evaluation['metrics']['mae_price']:.4f}, "
        "unchanged-price MAE "
        f"{evaluation['baselines']['unchanged_price']['metrics']['mae_price']:.4f}; "
        f"net return {float(metrics['net_return']):.2%}, Sharpe {sharpe_text}, "
        f"buy-and-hold {float(strategy['buy_and_hold']['metrics']['net_return']):.2%}. "
        + terminal["limitation"],
    )
    return summary


def run_study(args, *, resume=False):
    symbol, root, configuration = _configuration(args)
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    with run_lock(root):
        return _run_study(args, symbol, root, configuration, resume=resume)


def _configuration(args):
    from quant_rsi.research.demo import research_question
    from quant_rsi.research.study_walk_forward import price_history_sessions

    args.question = research_question(getattr(args, "question", ""))
    symbol = TypeAdapter(InstrumentId).validate_python(args.symbol.strip().upper())
    horizon = getattr(args, "horizon_sessions", 5)
    folds = getattr(args, "walk_forward_folds", 0)
    requested_history = getattr(args, "historical_sessions", None)
    price_history_sessions(horizon, args.evaluation_sessions, folds, requested_history)
    if requested_history is not None and args.prices != "daily":
        raise ValueError("--historical-sessions requires --prices daily")
    estimator_configuration = model_configuration(
        args.model, args.ridge_alpha, horizon_sessions=horizon
    )
    compare_external = getattr(args, "compare_qlib", False)
    if compare_external:
        if args.model != "lightgbm":
            raise ValueError("--compare-qlib currently requires --model lightgbm")
        from quant_rsi.research.study_qlib import require_qlib

        require_qlib()
    fit_budget = args.rounds if args.max_fits is None else args.max_fits
    validate_limits(args.rounds, fit_budget, args.patience)
    if args.prices == "synthetic" and args.prices_dir is not None:
        raise ValueError("--prices-dir requires --prices daily")
    if args.related_prices_dir and (args.prices != "daily" or not args.related_prices):
        raise ValueError("--related-prices-dir requires daily prices and enabled related prices")
    root = args.output.resolve()
    checkout = Path(__file__).resolve().parents[3]
    if root.is_relative_to(checkout):
        raise ValueError("Keep study artifacts outside the source checkout")
    if args.news_archive and args.news_archive.resolve().is_relative_to(checkout):
        raise ValueError("Keep news snapshots outside the source checkout")
    if getattr(args, "news_snapshot", None):
        if args.news != "issuer-rss":
            raise ValueError("--news-snapshot requires --news issuer-rss")
        args.news_snapshot = args.news_snapshot.resolve()
        if args.news_snapshot.is_relative_to(checkout):
            raise ValueError("Keep news snapshots outside the source checkout")
    for name in ("external_drivers_dir", "external_drivers_cache_dir"):
        if getattr(args, name, None):
            if not getattr(args, "external_drivers", False):
                raise ValueError(f"--{name.replace('_', '-')} requires --external-drivers")
            value = getattr(args, name).resolve()
            setattr(args, name, value)
            if value.is_relative_to(checkout):
                raise ValueError("Keep external driver snapshots outside the checkout")
    if getattr(args, "financials_dir", None):
        if not getattr(args, "financials", False):
            raise ValueError("--financials-dir requires --financials")
        args.financials_dir = args.financials_dir.resolve()
        if args.financials_dir.is_relative_to(checkout):
            raise ValueError("Keep financial snapshots outside the checkout")
    if args.history_dir:
        args.history_dir = args.history_dir.resolve()
        if args.history_dir.is_relative_to(checkout) or args.history_dir == root:
            raise ValueError("Keep shared study history outside the checkout and run directory")
    if args.related_prices_dir and args.related_prices_dir.resolve().is_relative_to(checkout):
        raise ValueError("Keep related price collections outside the source checkout")
    configuration = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration.update(
        symbol=symbol,
        model_configuration=estimator_configuration,
        research_rounds=args.rounds,
        fit_budget=fit_budget,
        comparison_fit_budget=int(compare_external)
        + int(getattr(args, "financials", False))
        + int(getattr(args, "external_drivers", False)),
        walk_forward_fit_budget=folds,
        generated_recipes="bounded_price_news_and_business_expressions",
    )
    return symbol, root, configuration


def _run_study(args, symbol, root, configuration, *, resume=False):
    cache = StudyCache(root, configuration, resume=resume)
    if not resume:
        _write(root / "config.json", configuration)
    last_stage = "initialization"
    progress = StudyProgress(root, args.llm, quiet=args.quiet, previous_attempts=cache.previous)
    cache.progress = progress
    if getattr(args, "ui", False) and not getattr(args, "check_inputs", False):
        from quant_rsi.research.study_ui import start_view

        start_view(root)

    def record(event):
        nonlocal last_stage
        progress.finish(event["stage"])
        with (root / "trace.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        last_stage = event["stage"]

    try:
        return _run(args, symbol, root, configuration, record, progress, cache)
    except (Exception, KeyboardInterrupt) as exc:
        runtime = progress.fail()
        _write(
            root / "failure.json",
            {
                "status": "failed",
                "last_completed_stage": last_stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "runtime": runtime,
                "note": "Completed operations are retained. "
                "Resume with quant-rsi research-resume PATH.",
            },
        )
        raise
    finally:
        if getattr(args, "ui", False) and not getattr(args, "check_inputs", False):
            from quant_rsi.research.study_ui import save_view

            try:
                save_view(root)
            except OSError as exc:
                print(f"[quant-rsi] Could not save live.html: {exc}", file=sys.stderr)


def _run(args, symbol, root, configuration, record, progress, cache):
    if (cache.root / "final_result.json").exists():
        packet = cache.call("final_result", symbol, lambda: None)
        return _finish_study(args, symbol, root, packet["result"], packet["prices"], progress)
    # Validate provider options before collection or fitting; no silent fallback.
    hypothesis = (
        None
        if getattr(args, "check_inputs", False)
        else simulated_hypothesis
        if args.llm == "simulated"
        else LLMHypothesizer(args.llm, args.llm_model, args.llm_timeout, on_usage=progress.usage)
    )
    SideCosts(commission_bps=args.commission_bps, slippage_bps=args.slippage_bps, tax_bps="0")
    decimal_text(args.initial_cash, positive=True)
    if not math.isfinite(args.threshold):
        raise ValueError("Strategy threshold must be finite")
    library = load_study_catalog(symbol, args.catalog, args.factor_root)
    for op in ("qlib_roc_v1", "qlib_ma_v1", "log_return_volatility"):
        definition = FactorProposal(
            op=op,
            window=10,
            description="Sample standard deviation of 10 daily log returns (ddof=1), unannualized."
            if op == "log_return_volatility"
            else "Source-pinned local Qlib-style price adapter.",
        ).definition()
        if definition["factor_id"] not in {x["factor_id"] for x in library["definitions"]}:
            library["definitions"].append(definition)
    progress.start("load_prices")

    def collect_target():
        directory = args.prices_dir or root / "prices"
        if (
            args.prices_dir is None
            and directory.exists()
            and any(directory.iterdir())
            and (
                not (directory / "collection_receipt.json").exists()
                or json.loads((directory / "collection_receipt.json").read_bytes()).get("status")
                == "failed"
            )
        ):
            # Keep partial collection evidence; retry transport into a fresh directory.
            from uuid import uuid4

            directory = root / "price_retries" / uuid4().hex
        load_daily_prices(symbol, directory, **price_window)
        return str(directory.resolve())

    from quant_rsi.research.study_walk_forward import price_history_sessions

    horizon = getattr(args, "horizon_sessions", 5)
    folds = getattr(args, "walk_forward_folds", 0)
    history_sessions = price_history_sessions(
        horizon, args.evaluation_sessions, folds, getattr(args, "historical_sessions", None)
    )
    price_window = {"horizon_sessions": horizon}
    if folds or getattr(args, "historical_sessions", None) is not None:
        price_window["historical_sessions"] = history_sessions
    prices = (
        load_daily_prices(
            symbol,
            cache.call("target_collection", symbol, collect_target),
            **price_window,
        )
        if args.prices == "daily"
        else None
    )
    if prices:
        cache.call(
            "target_inputs",
            {
                "metadata": prices[1],
                "dates": list(prices[0].dates),
                "opens": prices[0].opens.tolist(),
                "closes": prices[0].closes.tolist(),
            },
            lambda: True,
        )
    model = LightGBMDemo(
        library,
        prices=prices,
        evaluation_sessions=args.evaluation_sessions,
        terminal_sessions=args.terminal_sessions,
        estimator=args.model,
        ridge_alpha=args.ridge_alpha,
        fit_cache=cache,
        horizon_sessions=getattr(args, "horizon_sessions", 5),
        walk_forward_folds=folds,
    )
    _write(root / "walk_forward_plan.json", model.walk_forward_plan)
    if prices is None:
        model.data = replace(model.data, instruments=(symbol,))
    _write(root / "evaluation_plan.json", model.terminal_plan())
    progress.finish(
        "load_prices",
        {"provider": model.price_provider, "symbol": symbol, "sessions": len(model.data.dates)},
    )
    history_prices = (
        collection_snapshot(
            symbol, prices[1]["collection_directory"], horizon_sessions=model.horizon
        )
        if args.history_dir and prices
        else None
    )
    prior_outcomes = None
    if args.history_dir and not getattr(args, "check_inputs", False):
        progress.start("observe_prior_forecasts")
        if history_prices:
            observe_prices(args.history_dir, symbol, history_prices)
        prior_outcomes = cache.call(
            "prior_outcomes",
            symbol,
            lambda: mature_feedback(
                args.history_dir,
                symbol,
                horizon_sessions=model.horizon,
                question=configuration["question"],
            ),
        )
        _write(root / "prior_outcomes.json", prior_outcomes)
        progress.finish(
            "observe_prior_forecasts",
            {
                "mature_forecasts_for_feedback": len(prior_outcomes["entries"]),
                "history_directory": str(args.history_dir),
                "notice": prior_outcomes["notice"],
                "prior_outcomes": prior_outcomes,
            },
        )
    progress.start("collect_news_context")
    related = cache.call(
        "related",
        symbol,
        lambda: (profile_related if args.related == "profile" else simulated_related)(symbol),
    )
    news_context = {"symbol": symbol, "related": related}
    snapshot_path = getattr(args, "news_snapshot", None)

    def collect_current_news():
        if snapshot_path:
            return load_snapshot(snapshot_path, symbol, limit=args.news_limit)
        if args.news == "issuer-rss":
            return issuer_news(news_context, limit=args.news_limit)
        return simulated_news(news_context)

    current_news = cache.call(
        "issuer_news",
        news_context | {"retained_snapshot": str(snapshot_path)} if snapshot_path else news_context,
        collect_current_news,
    )
    archive = args.news_archive or (
        args.history_dir
        / "news_archive"
        / ("real" if args.prices == "daily" else "simulated")
        / symbol
        if args.history_dir
        else root / "news_archive"
    )

    def collect_snapshots():
        snapshots = load_archive(archive)
        if args.prices == "daily" and any(
            snapshot["provider"] != "issuer_rss" for snapshot in snapshots
        ):
            raise ValueError("Simulated news history cannot enter a daily-price study")
        if args.news == "issuer-rss":
            retain_snapshot(archive, current_news)
            snapshots = load_archive(archive)
        elif args.prices == "synthetic" and not snapshots:
            snapshots = simulated_history(model.data, symbol, guidance=args.news_research)
        return snapshots

    snapshots = cache.call("news_history", current_news, collect_snapshots)
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
            "related_prices_file": "related_prices.json",
        },
    )
    progress.finish(
        "collect_news_context",
        {
            "related": related,
            "news": current_news,
            "acquisition": "retained_snapshot" if snapshot_path else "provider",
        },
    )
    business = None
    if getattr(args, "financials", False):
        from quant_rsi.research.study_financials import (
            collect_financials,
            prepare_financials,
            sample_coverage,
            simulated_financials,
        )

        progress.start("load_financials")
        required = [int(i) for i in (*model.train, *model.origins)]

        def financial_inputs():
            if args.prices == "synthetic":
                return simulated_financials(symbol, cutoffs, required)
            directory = args.financials_dir or root / "financials"
            try:
                snapshot = collect_financials(
                    symbol,
                    directory,
                    args.factor_root,
                    getattr(args, "financials_source", "auto"),
                )
                return prepare_financials(snapshot, cutoffs, required)
            except (OSError, ValueError, KeyError) as exc:
                attempts = directory / "source_attempts.json"
                return {
                    "definitions": [],
                    "columns": {},
                    "notice": "Financial data unavailable: " + str(exc),
                    "provider": "unavailable",
                    "source_attempts": json.loads(attempts.read_bytes())
                    if attempts.exists()
                    else [],
                }

        business = cache.call(
            "financial_features",
            {"symbol": symbol, "cutoffs": cutoffs, "required": required},
            financial_inputs,
        )
        business["sample_coverage"] = sample_coverage(business, model.train, model.origins)
        model.configure_financials(business)
        _write(root / "financial_features.json", business)
        progress.finish(
            "load_financials",
            {k: v for k, v in business.items() if k not in {"columns", "evidence"}},
        )

    external = None
    if getattr(args, "external_drivers", False):
        from quant_rsi.research.eia_petroleum import (
            collect_petroleum,
            prepare_petroleum,
            simulated_petroleum,
        )
        from quant_rsi.research.study_financials import sample_coverage

        progress.start("load_external_drivers")
        required = [int(i) for i in (*model.train, *model.origins)]

        def external_inputs():
            if args.prices == "synthetic":
                return simulated_petroleum(cutoffs, required)
            directory = args.external_drivers_dir or root / "external_drivers"
            try:
                snapshot = collect_petroleum(
                    directory,
                    cutoffs[0][:10],
                    cutoffs[-1][:10],
                    cache_directory=getattr(args, "external_drivers_cache_dir", None),
                    history_directory=args.history_dir,
                )
                return prepare_petroleum(snapshot, cutoffs, required)
            except (OSError, ValueError, KeyError) as exc:
                return {
                    "definitions": [],
                    "columns": {},
                    "provider": "unavailable",
                    "notice": "External drivers unavailable: " + str(exc),
                    "failures": [{"error_type": type(exc).__name__, "reason": str(exc)}],
                }

        external = cache.call(
            "external_features", {"cutoffs": cutoffs, "required": required}, external_inputs
        )
        external["sample_coverage"] = sample_coverage(external, model.train, model.origins)
        model.configure_external(external)
        _write(root / "external_features.json", external)
        progress.finish(
            "load_external_drivers",
            {k: v for k, v in external.items() if k not in {"columns", "evidence"}},
        )

    from quant_rsi.research.study_inputs import input_report

    readiness = input_report(root, symbol, model, current_news, business, external)
    progress.start("check_inputs")
    progress.finish("check_inputs", readiness)
    if getattr(args, "check_inputs", False):
        progress.complete()
        return readiness

    prelude = []
    guidance_summary = None
    if args.news_research:

        def record_research(event):
            prelude.append(event)
            record(event)

        progress.start("extract_guidance")
        if snapshots and all(s["provider"] == "simulated_news_archive" for s in snapshots):
            guidance_records = [
                extract_guidance(s, symbol, None, extracted_at=s["received_at"]) for s in snapshots
            ]
            guidance_status = "simulated_historical_extractions"
        elif args.llm != "simulated" and args.news == "issuer-rss":

            def mark_guidance_request():
                progress.stages[-1]["llm_request"] = True

            guidance_records, _ = cache.call(
                "guidance",
                current_news,
                lambda: collect_guidance(
                    archive.with_name(archive.name + "_guidance"),
                    current_news,
                    symbol,
                    hypothesis,
                    on_extract=mark_guidance_request,
                ),
            )
            guidance_status = "retained_real_extractions"
        else:
            guidance_records = []
            guidance_status = "unavailable_real_extractor_or_issuer_source"
        model.configure_guidance(GuidanceHistory(snapshots, guidance_records))
        _write(root / "guidance_extractions.json", guidance_records)
        latest = max(guidance_records, key=lambda r: r["extracted_at"], default=None)
        guidance_summary = {
            "status": guidance_status,
            "record_count": len(guidance_records),
            "notice": GUIDANCE_NOTICE,
            "definitions": deepcopy(model.guidance_support),
            "latest_extraction": {
                key: latest[key]
                for key in ("provider", "extracted_at", "facts", "llm_mode", "requested_model")
            }
            | {"items": [item for item in latest["snapshot"]["items"] if item["symbol"] == symbol]}
            if latest
            else None,
            "evidence_file": "guidance_extractions.json",
        }
        record_research({"stage": "extract_guidance", "output": guidance_summary})
        discovery = research_news(
            symbol,
            related,
            None if args.llm == "simulated" else hypothesis,
            on_stage=record_research,
            on_start=progress.start,
            cache=cache,
            question=configuration["question"],
        )
        _write(root / "news_research.json", discovery)
        current_news, related = attach_research(current_news, related, discovery, symbol)

    progress.start("load_related_prices")

    def collect_peers():
        value = prepare_related_prices(
            model.data,
            cutoffs,
            list(model.train) + list(model.origins),
            related,
            mode=args.prices,
            directory=args.related_prices_dir or root / "related_prices",
            enabled=args.related_prices,
            simulated_suggestions=args.llm == "simulated",
        )
        value["columns"] = {
            key: [float(x) if math.isfinite(x) else None for x in column]
            for key, column in value["columns"].items()
        }
        return value

    peers = cache.call(
        "peers", {"related": related, "dates": list(model.data.dates)}, collect_peers
    )
    import numpy as np

    peers["columns"] = {
        key: np.asarray(column, dtype=float) for key, column in peers["columns"].items()
    }
    if args.terminal_sessions:
        for support in peers["support"]:
            if "aligned_sessions" in support:
                values = peers["inputs"][support["symbol"]]["aligned_closes"]
                support["aligned_sessions"] = sum(
                    values[i] is not None for i in list(model.train) + list(model.origins)
                )
                support["alignment_scope"] = "training_and_development_only"
    model.configure_peers(peers)
    if "instrument_profile" in library:
        model.configure_profile(library["instrument_profile"])
        _write(
            root / "factor_profile.json",
            {
                "specification": library["instrument_profile"],
                "execution": model.profile_features["definitions"],
            },
        )
    related["price_features"] = peers["support"]
    _write(
        root / "related_prices.json",
        {
            "support": peers["support"],
            "inputs": peers["inputs"],
            "definitions": peers["definitions"],
            "alignment": (
                "exact_session_and_peer_close_at_or_before_target_close; no_fill_no_row_drop"
            ),
        },
    )
    progress.finish("load_related_prices", {"related": related})

    from quant_rsi.market.market_contracts import trusted_now
    from quant_rsi.research.study_factor_memory import reuse_recipes

    model.library["recipe_as_of"] = cache.call("recipe_as_of", symbol, trusted_now)
    model.library, factor_memory = reuse_recipes(
        model.library, (prior_outcomes or {}).get("factor_memory", [])
    )
    if factor_memory["entries"]:
        _write(root / "factor_memory.json", factor_memory)
        progress.start("restore_factor_recipes")
        progress.finish("restore_factor_recipes", factor_memory)

    hypothesis_number = 0

    def cached_hypothesis(context, library, **kwargs):
        nonlocal hypothesis_number
        hypothesis_number += 1
        return cache.call(
            f"hypothesis_{hypothesis_number}",
            [context, library, kwargs],
            lambda: hypothesis(context, library, **kwargs),
        )

    def context(target):
        return model.context(target) | {
            "research_question": configuration["question"],
            "prior_outcomes": {k: v for k, v in prior_outcomes.items() if k != "factor_memory"}
            if prior_outcomes
            else None,
            "factor_memory": factor_memory,
            "guidance": guidance_summary,
            "allow_factor_proposals": True,
            "selection_policy": {
                "metric": args.selection_metric,
                "rule": "strict_improvement_else_retain_incumbent; "
                "missing metrics retain incumbent",
                "scope": "development_only",
            },
        }

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
            horizon_sessions=model.horizon,
        )
        return prediction

    def observe(context, prediction):
        return model.observe(context, prediction) | {
            "strategy_diagnostics": {
                key: prediction["strategy"][key]["metrics"] for key in ("strategy", "buy_and_hold")
            },
        }

    def evaluate_selected(iterations):
        chosen = iterations["selected_round"]
        selected = iterations["attempts"][chosen - 1]["prediction"]
        receipt = {
            "selected_round": chosen,
            "factor_ids": selected["factor_ids"],
            "selection_metric": args.selection_metric,
            "development_score": select_attempt(selected, selected, args.selection_metric)[
                "initial_score"
            ],
            "completed_fits": iterations["completed_fits"],
            "terminal_plan": model.terminal_plan(),
        }
        _write(root / "selected_attempt.json", receipt)
        terminal = model.evaluate_terminal(chosen)
        if terminal["status"] == "completed":
            terminal["strategy"] = evaluate_strategy(
                model.data,
                symbol,
                terminal["evaluation"]["rows"],
                commission_bps=args.commission_bps,
                slippage_bps=args.slippage_bps,
                initial_cash=args.initial_cash,
                threshold=args.threshold,
                horizon_sessions=model.horizon,
            )
            terminal["strategy"]["evidence_mode"] = "historical_terminal_hypothetical_account"
        _write(root / "terminal_evaluation.json", terminal)
        return terminal

    def summarize(context, first, second, feedback, *, iterations, terminal_evaluation=None):
        summary = model.summarize(context, first, second, feedback)
        chosen = iterations["selected_round"]
        selected = iterations["attempts"][chosen - 1]["prediction"]
        selection = select_attempt(
            first, selected if chosen != 1 else second, args.selection_metric
        )
        selection["selected_round"] = chosen
        selection["selected_score"] = select_attempt(selected, selected, args.selection_metric)[
            "initial_score"
        ]
        if len(iterations["attempts"]) > 2:
            selection["selected_attempt"] = f"round_{chosen}"
            selection["reason"] = (
                f"Selected round {chosen} by strict {args.selection_metric} improvement on the "
                "same development window; ties and missing metrics retain the incumbent."
            )
        selection["forecast_note"] = "Latest forecast uses the selected fitted model; no refit."
        for key in ("latest_forecast", "latest_forecast_unavailable"):
            summary.pop(key, None)
            if key in selected:
                summary[key] = deepcopy(selected[key])
        summary["selection"] = selection
        summary["research_rounds"] = [
            {
                "round": item["round"],
                "factor_ids": item["factor_ids"],
                "accepted": item["accepted"],
                "mae_price": item["development_metrics"]["mae_price"],
                "net_return": item["prediction"]["strategy"]["strategy"]["metrics"]["net_return"],
                "sharpe_ratio": item["prediction"]["strategy"]["strategy"]["metrics"][
                    "sharpe_ratio"
                ],
            }
            for item in iterations["attempts"]
        ]
        summary["stop_reason"] = iterations["stop_reason"]
        if terminal_evaluation is not None:
            summary["terminal_evaluation"] = terminal_summary(terminal_evaluation)
        summary["text"] = (
            f"{symbol}: {iterations['completed_fits']} real fits. "
            f"Stopped: {iterations['stop_reason']}. "
            f"{selection['reason']} The latest forecast, when available, follows the "
            f"{selection['selected_attempt']} attempt and has no observed outcome. "
            "Selection reuses development feedback; it does not establish independent gain."
        )
        if terminal_evaluation is not None:
            summary["text"] += " " + summary["terminal_evaluation"]["text"]
        return summary

    result = run_demo(
        symbol,
        catalog=lambda: model.library,
        price_context=context,
        related=lambda target: related,
        news=lambda context: current_news,
        hypothesize=cached_hypothesis,
        predict=predict,
        observe=observe,
        summarize=summarize,
        on_stage=record,
        on_start=progress.start,
        rounds=args.rounds,
        fit_budget=configuration["fit_budget"],
        patience=args.patience,
        select=lambda incumbent, trial: select_attempt(incumbent, trial, args.selection_metric),
        evaluate_selected=evaluate_selected if args.terminal_sessions else None,
    )
    result["trace"] = prelude + result["trace"]
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    attempts = result["iterations"]["attempts"]
    first, second = attempts[0]["prediction"], attempts[-1]["prediction"]
    selection = result["summary"]["selection"]
    selected_metrics = result["summary"]["research_rounds"][selection["selected_round"] - 1]
    selected_sharpe = selected_metrics["sharpe_ratio"]
    selected_sharpe_text = f"{selected_sharpe:.4f}" if selected_sharpe is not None else "N/A"
    first_ids, second_ids = set(first["factor_ids"]), set(second["factor_ids"])
    metrics = second["development"]["metrics"]
    baseline = second["development"]["baselines"]["unchanged_price"]["metrics"]
    first_mae = first["development"]["metrics"]["mae_price"]
    change = metrics["mae_price"] - first_mae
    first_return = float(first["strategy"]["strategy"]["metrics"]["net_return"])
    revised_return = float(second["strategy"]["strategy"]["metrics"]["net_return"])
    buy_hold_return = float(second["strategy"]["buy_and_hold"]["metrics"]["net_return"])
    risk_accounts = [
        ("initial", first["strategy"]["strategy"]["metrics"]),
        ("revised", second["strategy"]["strategy"]["metrics"]),
        ("buy-and-hold", second["strategy"]["buy_and_hold"]["metrics"]),
    ]
    sharpe_summary = ", ".join(
        f"{label} "
        + (
            f"{account_metrics['sharpe_ratio']:.4f}"
            if account_metrics["sharpe_ratio"] is not None
            else f"N/A ({account_metrics['risk_statistics']['status']})"
        )
        for label, account_metrics in risk_accounts
    )
    risk_info = risk_accounts[0][1]["risk_statistics"]
    discovery_summary = ""
    if args.news_research:
        discovery_summary = (
            f"Active news research: {len(discovery['searches'])} search requests, "
            f"{len(discovery['evidence_news_ids'])}/{len(discovery['articles'])} articles read, "
            f"{len(discovery['related_candidates'])} suggested related companies "
            f"({discovery['provider']}). "
        )
    result["notice"] = (
        f"Connected research study with {model.price_provider} prices and "
        f"real {model.model_label} training at a {model.horizon}-session horizon. "
        f"Hypotheses: {steps['initial_hypothesis']['provider']}; "
        f"news: {steps['news']['provider']}; "
        f"relationships: {steps['related_stocks']['provider']}. "
        "All attempts use the same retrospective development window; no independent gain "
        "is established. Inspect source coverage and the separate unobserved latest forecast."
    )
    result["configuration"] = configuration
    final_prefix = "revised" if len(attempts) == 2 else f"round_{len(attempts)}"
    news_definitions = [
        item
        for item in steps[f"{final_prefix}_factor_library"]["definitions"]
        if item["kind"] == "news" and item["recipe"]["op"] == "headline_keyword_count"
    ]
    news_used = any(
        item["kind"] == "news" for key in steps if key.endswith("_factors") for item in steps[key]
    )
    result["study"] = {
        "engine": f"local_{args.model}_and_existing_finite_account",
        "guidance": guidance_summary,
        "factor_memory": factor_memory
        | {
            "entries": [
                entry
                | {
                    "selected_in_this_study": entry["status"]
                    in {"restored_candidate", "available_in_current_catalog"}
                    and entry["factor_id"]
                    in attempts[selection["selected_round"] - 1]["prediction"]["factor_ids"]
                }
                for entry in factor_memory["entries"]
            ]
        },
        "factor_profile": deepcopy(model.profile_features["definitions"])
        if model.profile_features is not None
        else [],
        "selection": selection,
        "factor_changes": {
            "activated": sorted(second_ids - first_ids),
            "deactivated": sorted(first_ids - second_ids),
            "retained": sorted(first_ids & second_ids),
            "meaning": "Initial-to-final trial changes, not the adopted factor set. "
            "See selection and round decisions; definitions remain in the trace.",
        },
        "news_use": (
            "Selected issuer-news features entered the fitted feature matrix. "
            "Features cover the selected RSS corpus, not all issuer news; no sentiment is inferred."
            if news_used
            else "News was supplied as hypothesis context; no news column was fitted."
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
                x["factor_id"] for x in steps[f"{final_prefix}_factors"] if x["kind"] == "news"
            ],
            "timing_note": "Original receipt times are preserved source records, not independently "
            "authenticated clocks. Real price cutoffs use the retained exchange calendar.",
        },
        "external_engine": "Qlib-style local price adapters; no Qlib or QuantDinger execution.",
        "conclusion": (
            discovery_summary + f"{symbol}: completed {len(attempts)} fits and "
            f"{len(attempts) - 1} revision attempts at a {model.horizon}-session horizon; "
            f"stopped: {result['iterations']['stop_reason']}. "
            f"Final trial evaluated over "
            f"{metrics['sample_count']} development observations. Price MAE: initial "
            f"{first_mae:.4f}, revised {metrics['mae_price']:.4f}, "
            f"unchanged-price baseline {baseline['mae_price']:.4f}. "
            f"The revision {'increased' if change > 0 else 'decreased' if change < 0 else 'kept'} "
            f"MAE by {abs(change):.4f}. "
            f"{selection['reason']} "
            f"Selected round {selection['selected_round']}: "
            f"price MAE {selected_metrics['mae_price']:.4f}, "
            f"net return {float(selected_metrics['net_return']):.2%}, "
            f"Sharpe {selected_sharpe_text}. "
            f"Net strategy return: initial {first_return:.2%}, revised {revised_return:.2%}, "
            f"buy-and-hold {buy_hold_return:.2%}. "
            f"Annualized Sharpe: {sharpe_summary} "
            f"({risk_info['daily_return_count']} daily returns; "
            "252 sessions/year; risk-free rate 0%). "
            "Short-sample Sharpe is descriptive, not a reliable long-term estimate. "
            "These are retrospective development results after factor selection and feedback, "
            "not independent accuracy or evidence of improvement. Strategy results are "
            "hypothetical and include explicit costs. The latest forecast, when present, "
            "has no observed outcome."
        ),
    }
    if "terminal_evaluation" in result:
        result["study"]["conclusion"] += (
            " " + terminal_summary(result["terminal_evaluation"])["text"]
        )
    if getattr(args, "financials", False):
        from quant_rsi.research.financial_comparison import compare_financials

        chosen = selection["selected_round"]
        progress.start("financial_comparison")
        comparison = cache.call(
            "financial_comparison",
            model.comparison_inputs(chosen),
            lambda: compare_financials(
                model,
                chosen,
                attempts[chosen - 1]["prediction"],
                {
                    "commission_bps": args.commission_bps,
                    "slippage_bps": args.slippage_bps,
                    "initial_cash": args.initial_cash,
                    "threshold": args.threshold,
                    "horizon_sessions": model.horizon,
                },
            ),
        )
        result["study"]["financial_comparison"] = comparison
        result["study"]["conclusion"] += (
            f" Financial-feature diagnostic: {comparison['extra_fits']} additional fits "
            "outside the research fit budget; does not alter selection."
        )
        result["study"]["financial_data"] = {
            k: v for k, v in model.financial_features.items() if k not in {"columns", "evidence"}
        }
        _write(root / "financial_comparison.json", comparison)
        event = {"stage": "financial_comparison", "output": comparison}
        record(event)
        result["trace"].append(event)

    if external is not None:
        result["study"]["external_data"] = {
            k: v for k, v in external.items() if k not in {"columns", "evidence"}
        }
        from quant_rsi.research.financial_comparison import (
            compare_external as compare_external_inputs,
        )

        chosen = selection["selected_round"]
        progress.start("external_comparison")
        comparison = cache.call(
            "external_comparison",
            model.comparison_inputs(chosen),
            lambda: compare_external_inputs(
                model,
                chosen,
                attempts[chosen - 1]["prediction"],
                {
                    "commission_bps": args.commission_bps,
                    "slippage_bps": args.slippage_bps,
                    "initial_cash": args.initial_cash,
                    "threshold": args.threshold,
                    "horizon_sessions": model.horizon,
                },
            ),
        )
        result["study"]["external_comparison"] = comparison
        result["study"]["conclusion"] += (
            f" External-driver diagnostic: {comparison['extra_fits']} additional fits "
            "outside the research fit budget; does not alter selection."
        )
        _write(root / "external_comparison.json", comparison)
        event = {"stage": "external_comparison", "output": comparison}
        record(event)
        result["trace"].append(event)

    if getattr(args, "compare_qlib", False):
        from quant_rsi.research.study_qlib import compare_qlib

        progress.start("qlib_comparison")
        chosen = selection["selected_round"]

        def compare():
            comparison_root = root / "qlib-comparison"
            if comparison_root.exists():
                from uuid import uuid4

                comparison_root = root / ("qlib-comparison-retry-" + uuid4().hex)
            return compare_qlib(
                model.comparison_inputs(chosen),
                model.data,
                attempts[chosen - 1]["prediction"],
                result.get("terminal_evaluation"),
                comparison_root,
                policy={
                    "commission_bps": args.commission_bps,
                    "slippage_bps": args.slippage_bps,
                    "initial_cash": args.initial_cash,
                    "threshold": args.threshold,
                    "horizon_sessions": model.horizon,
                },
            )

        comparison = cache.call("qlib_comparison", model.comparison_inputs(chosen), compare)
        event = {"stage": "qlib_comparison", "output": comparison}
        record(event)
        result["trace"].append(event)
        result["study"]["qlib_comparison"] = comparison
        result["study"]["external_engine"] = comparison["notice"]
        result["study"]["conclusion"] += " Actual Qlib comparison completed; see the engine table."
    if folds:
        from quant_rsi.research.study_walk_forward import evaluate_windows

        progress.start("walk_forward")
        diagnostic = evaluate_windows(
            model,
            selection["selected_round"],
            {
                "commission_bps": args.commission_bps,
                "slippage_bps": args.slippage_bps,
                "initial_cash": args.initial_cash,
                "threshold": args.threshold,
                "horizon_sessions": model.horizon,
            },
        )
        _write(root / "walk_forward.json", diagnostic)
        result["study"]["walk_forward"] = diagnostic
        result["study"]["conclusion"] += (
            f" Post-selection walk-forward diagnostic: {diagnostic['status']}, "
            f"{len(diagnostic['folds'])} windows and {diagnostic['extra_fits']} extra fits. "
            "Historical recipe selection remains exposed; this is not independent validation."
        )
        event = {"stage": "walk_forward", "output": diagnostic}
        record(event)
        result["trace"].append(event)
    # Freeze measured research time for the reports, but do not announce success yet.
    result["runtime"] = progress.snapshot("completed")
    cache.call("final_result", symbol, lambda: {"result": result, "prices": history_prices})
    return _finish_study(args, symbol, root, result, history_prices, progress)


def _finish_study(args, symbol, root, result, history_prices, progress):
    from quant_rsi.research.study_report import render_study

    if args.history_dir:
        register_study(args.history_dir, root, result, history_prices)
        result["study"]["history"] = history_view(args.history_dir, symbol)
    if not (root / "trace.jsonl").exists():
        (root / "trace.jsonl").write_text(
            "".join(json.dumps(event, allow_nan=False) + "\n" for event in result["trace"]),
            encoding="utf-8",
        )
    markdown, html = render_study(result)
    (root / "report.md").write_text(markdown, encoding="utf-8")
    (root / "report.html").write_text(html, encoding="utf-8")
    _write(root / "result.json", result)
    progress.complete(result["runtime"])
    return result
