import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from pydantic import ValidationError

from quant_rsi.core.controller import create_run, inspect_run, resume_run
from quant_rsi.evaluation.comparison import (
    cancel_comparison,
    decide_comparison,
    inspect_comparison,
    register_comparison,
    resume_comparison,
)
from quant_rsi.evaluation.evidence import ComparisonInput, init_evidence, parse_program
from quant_rsi.execution.baseline import infer_at, inspect_baseline, seal_baseline
from quant_rsi.execution.observer import (
    advance_observer,
    inspect_observer,
    register_observer,
    stop_observer,
)
from quant_rsi.forecast.forecast_contracts import parse_development_plan
from quant_rsi.market.market_contracts import DataContract, parse_observation_plan
from quant_rsi.market.market_data import ingest, inspect_data
from quant_rsi.research.package_experiment import create_package, inspect_package, resume_package
from quant_rsi.research.research import create_research, inspect_research, resume_research
from quant_rsi.research.researcher import ResearchPlan
from quant_rsi.research.trajectory import init_trajectory, inspect_trajectory, resume_trajectory
from quant_rsi.research.trajectory_state import TrajectoryPlan


def default_history_dir(prices):
    """Same per-source layout as the workbench; real and synthetic outcomes never mix."""
    root = os.environ.get("QUANT_RSI_HISTORY_DIR")
    return (Path(root) if root else Path.home() / ".quant_rsi" / "history") / prices


def _check_llm_choice(args):
    if args.cursor_dir is not None and args.llm != "cursor":
        raise ValueError("--cursor-dir is only used with --llm cursor")
    if args.command == "research-demo" and args.llm == "cursor" and args.cursor_dir is None:
        raise ValueError("research-demo --llm cursor requires --cursor-dir")
    if getattr(args, "news_research", False) and args.llm in {"deepseek", "cursor"}:
        raise ValueError("News research web search requires --llm codex or --llm api")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded quant research and development observations"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("research-demo", "research-study"):
        study = name == "research-study"
        demo = commands.add_parser(
            name,
            help="connected research report"
            if study
            else "simulate one complete stock research loop",
        )
        demo.add_argument("symbol", nargs="?", default="KO")
        demo.add_argument(
            "--catalog",
            type=Path,
            default=None if study else Path("instruments/shared/equities.json"),
            help="explicit standalone catalog; bypasses study instrument bindings",
        )
        demo.add_argument("--json", action="store_true", help="emit the complete trace as JSON")
        demo.add_argument(
            "--llm",
            choices=("simulated", "codex", "api", "deepseek", "cursor"),
            default="codex" if study else "simulated",
        )
        demo.add_argument(
            "--llm-model",
            help="model ID; required for API, optional for Codex and DeepSeek "
            "(DeepSeek default: deepseek-chat)",
        )
        demo.add_argument("--llm-timeout", type=float, default=180, help="seconds per LLM call")
        demo.add_argument(
            "--cursor-dir",
            type=Path,
            help="directory where --llm cursor reads hypothesis JSON "
            "(research-study default: OUTPUT/cursor-llm)",
        )
        demo.add_argument(
            "--related",
            choices=("simulated", "profile"),
            default="profile" if study else "simulated",
        )
        demo.add_argument(
            "--news",
            choices=("simulated", "issuer-rss"),
            default="issuer-rss" if study else "simulated",
        )
        demo.add_argument("--news-limit", type=int, default=5, help="RSS items per issuer (1-20)")
        demo.add_argument(
            "--prices", choices=("synthetic", "daily"), default="daily" if study else "synthetic"
        )
        demo.add_argument("--prices-dir", type=Path, help="retained daily collection directory")
        if study:
            demo.add_argument(
                "--news-snapshot",
                type=Path,
                help="reuse a retained issuer RSS snapshot without fetching RSS again",
            )
            demo.add_argument(
                "--history-dir",
                type=Path,
                help="shared receipts, outcomes, factor ledger, news and industry publications "
                "(default: $QUANT_RSI_HISTORY_DIR or ~/.quant_rsi/history, per price source)",
            )
            demo.add_argument(
                "--no-history",
                action="store_true",
                help="do not read or record shared study history for this run",
            )
            demo.add_argument(
                "--factor-root",
                type=Path,
                default=Path("instruments"),
                help="root of shared and instrument factor profiles (default: instruments)",
            )
            demo.add_argument(
                "--related-prices",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="collect up to three related symbols and offer aligned price factors",
            )
            demo.add_argument(
                "--related-prices-dir",
                type=Path,
                help="root of retained daily peer collections, one subdirectory per symbol",
            )
            demo.add_argument(
                "--financials",
                action="store_true",
                help="publication-dated financial features with one optional diagnostic fit",
            )
            demo.add_argument(
                "--financials-dir", type=Path, help="retained financial snapshot directory"
            )
            demo.add_argument(
                "--financials-source",
                choices=("auto", "sec", "issuer"),
                default="auto",
                help="auto tries SEC then the configured official issuer archive",
            )
            demo.add_argument(
                "--check-inputs",
                action="store_true",
                help="check target prices, news and financial coverage without LLM calls or fits",
            )
            demo.add_argument(
                "--external-drivers",
                action="store_true",
                help="EIA weekly US inventory factors with up to one extra diagnostic fit",
            )
            demo.add_argument(
                "--external-drivers-dir",
                type=Path,
                help="retained EIA weekly petroleum snapshot directory",
            )
            demo.add_argument(
                "--external-drivers-cache-dir",
                type=Path,
                help="shared EIA resource cache; use a fresh directory for each research cycle",
            )
            demo.add_argument("--model", choices=("lightgbm", "ridge"), default="lightgbm")
            demo.add_argument(
                "--latest-fit",
                choices=("refit", "selected"),
                default="refit",
                help="daily prices: refit on mature labels (one extra fit), or reuse research fit",
            )
            demo.add_argument(
                "--compare-qlib",
                action="store_true",
                help="one additional upstream Qlib LightGBM fit on the selected study inputs",
            )
            demo.add_argument(
                "--ridge-alpha", type=float, help="positive ridge penalty (default: 1; ridge only)"
            )
            demo.add_argument("--output", type=Path, required=True, help="new report directory")
            demo.add_argument(
                "--question",
                default="",
                help="optional research focus (up to 1000 characters)",
            )
            demo.add_argument(
                "--news-research",
                action="store_true",
                help="search/read current articles and suggest sourced company relationships",
            )
            demo.add_argument(
                "--news-archive",
                type=Path,
                help="retain/reuse issuer RSS snapshots with original receipt times",
            )
            demo.add_argument("--evaluation-sessions", type=int, default=20, choices=range(2, 21))
            demo.add_argument(
                "--historical-sessions",
                type=int,
                help="minimum daily-price history (up to 445 sessions); all retained rows are used",
            )
            demo.add_argument(
                "--walk-forward-folds",
                type=int,
                default=0,
                choices=(0, 2, 3),
                help="post-selection chronological diagnostic (one extra fit per fold)",
            )
            demo.add_argument(
                "--horizon-sessions",
                type=int,
                default=5,
                choices=(5, 20, 60),
                help="prediction target in trading sessions (default: 5)",
            )
            demo.add_argument(
                "--terminal-sessions",
                type=int,
                default=10,
                choices=range(21),
                help="reserved final evaluation observations (default: 10; 0: development only)",
            )
            demo.add_argument(
                "--rounds",
                type=int,
                default=2,
                choices=range(2, 11),
                help="maximum research fits, including the initial fit (default: 2)",
            )
            demo.add_argument(
                "--max-fits",
                type=int,
                choices=range(2, 11),
                help="fit allowance; defaults to --rounds",
            )
            demo.add_argument(
                "--patience",
                type=int,
                choices=range(1, 11),
                help="stop after this many consecutive non-improving trials",
            )
            demo.add_argument("--initial-cash", default="10000")
            demo.add_argument("--commission-bps", default="5")
            demo.add_argument("--slippage-bps", default="5")
            demo.add_argument("--threshold", type=float, default=0.0)
            demo.add_argument(
                "--selection-metric",
                choices=("mae", "sharpe"),
                default="mae",
                help="preselect the development metric for adopting a revision (default: mae)",
            )
            demo.add_argument("--quiet", action="store_true", help="hide stderr stage progress")
            demo.add_argument(
                "--ui",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="open the live research view (default: interactive terminal, except --json)",
            )
        else:
            demo.add_argument("--model", choices=("simulated", "lightgbm"), default="simulated")
    workbench = commands.add_parser("research-ui", help="local research launcher and task history")
    workbench.add_argument("--workspace", type=Path, required=True)
    workbench.add_argument("--port", type=int, default=0)
    workbench.add_argument("--open", action=argparse.BooleanOptionalAction, default=True)
    retry = commands.add_parser("research-resume", help="continue an interrupted study")
    retry.add_argument("directory", type=Path)
    retry.add_argument("--ui", action="store_true")
    history = commands.add_parser(
        "research-history", help="inspect or observe saved study forecasts"
    )
    history.add_argument("action", choices=("status", "observe", "factors", "constellation"))
    history.add_argument("symbol")
    history.add_argument(
        "--history-dir",
        type=Path,
        help="default: the daily-price directory under $QUANT_RSI_HISTORY_DIR "
        "or ~/.quant_rsi/history",
    )
    history.add_argument("--json", action="store_true", help="emit all retained history fields")
    history.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="open the latest study view with current history",
    )
    history.add_argument(
        "--prices-dir", type=Path, help="later retained daily collection; required for observe"
    )
    history.add_argument(
        "--output",
        type=Path,
        help="constellation HTML file (default: <history-dir>/constellation/<SYMBOL>.html)",
    )
    workflow = commands.add_parser("research-workflow", help="one frozen research/account profile")
    workflow_commands = workflow.add_subparsers(dest="workflow_command", required=True)
    for name in ("prepare", "run"):
        command = workflow_commands.add_parser(name)
        command.add_argument("profile", type=Path)
        if name == "run":
            command.add_argument("--max-fits", type=int, default=1)
    for name in ("resume", "status", "replay", "report", "observe"):
        command = workflow_commands.add_parser(name)
        command.add_argument("campaign_id")
        if name == "resume":
            command.add_argument("--max-fits", type=int, default=1)
        if name == "observe":
            command.add_argument("--snapshot-ref", required=True)
            command.add_argument("--request-id", required=True)
    observation = commands.add_parser(
        "forecast-observe", help="record outcomes of committed five-session forecasts; no fits"
    )
    observation_commands = observation.add_subparsers(dest="observation_command", required=True)
    for name in ("register", "settle", "status", "replay"):
        command = observation_commands.add_parser(name)
        command.add_argument("experiment_id")
        if name == "settle":
            command.add_argument("--snapshot-ref", required=True)
            command.add_argument("--request-id", required=True)
    collection = commands.add_parser(
        "news-collect", help="bounded prospective issuer RSS observations"
    )
    collection_commands = collection.add_subparsers(dest="collection_command", required=True)
    collection_commands.add_parser("init").add_argument("plan", type=Path)
    for name in ("poll", "process", "status", "replay"):
        command = collection_commands.add_parser(name)
        command.add_argument("campaign_id")
        if name in {"poll", "process"}:
            command.add_argument("--request-id", required=True)
        if name == "process":
            command.add_argument("--news-campaign-id", required=True)
    factors = commands.add_parser("factors", help="validate layered factor specifications; no fits")
    factor_commands = factors.add_subparsers(dest="factor_command", required=True)
    factor_validate = factor_commands.add_parser("validate")
    factor_validate.add_argument("market")
    factor_validate.add_argument("instrument")
    factor_validate.add_argument("--root", type=Path, default=Path("instruments"))
    discovery = commands.add_parser("factor-discovery", help="bounded typed factor discovery")
    discovery_commands = discovery.add_subparsers(dest="discovery_command", required=True)
    discovery_init = discovery_commands.add_parser("init")
    discovery_init.add_argument("plan", type=Path)
    discovery_init.add_argument("--inputs", type=Path, required=True)
    for name in ("resume", "status", "replay"):
        command = discovery_commands.add_parser(name)
        command.add_argument("campaign_id")
        if name == "resume":
            command.add_argument("--max-fits", type=int, default=1)
    forecast = commands.add_parser("forecast", help="fixed five-session daily price forecast")
    forecast_commands = forecast.add_subparsers(dest="forecast_command", required=True)
    forecast_run = forecast_commands.add_parser("run")
    forecast_run.add_argument("ticker")
    forecast_run.add_argument("--request-id", required=True)
    forecast_run.add_argument("--output", type=Path, required=True)
    forecast_run.add_argument("--prepare-only", action="store_true")
    for name in ("status", "replay"):
        forecast_commands.add_parser(name).add_argument("request_id")
    forecast_compare = commands.add_parser(
        "forecast-compare", help="fixed paired forecast evaluation and separate publication fit"
    )
    context_commands = forecast_compare.add_subparsers(dest="context_command", required=True)
    context_prepare = context_commands.add_parser("prepare")
    context_prepare.add_argument("inputs", type=Path)
    context_prepare.add_argument("--experiment-id", required=True)
    for name in ("run", "status", "replay"):
        command = context_commands.add_parser(name)
        command.add_argument("experiment_id")
        if name == "run":
            command.add_argument("--protocol-ref", required=True)
            command.add_argument("--max-fits", type=int, default=1)
    campaigns = commands.add_parser(
        "forecast-campaign", help="shared immutable allowances for registered forecast runs"
    )
    campaign_commands = campaigns.add_subparsers(dest="campaign_command", required=True)
    campaign_commands.add_parser("init").add_argument("plan", type=Path)
    recursive_init = campaign_commands.add_parser("recursive-init")
    recursive_init.add_argument("plan", type=Path)
    recursive_init.add_argument("--market-preparation", type=Path)
    recursive_prepare = campaign_commands.add_parser("recursive-prepare")
    recursive_prepare.add_argument("profile", type=Path)
    recursive_prepare.add_argument("--output", type=Path, required=True)
    recursive_prepare.add_argument("--campaign-id", required=True)
    recursive_prepare.add_argument("--source-campaign-id", required=True)
    for name in ("status", "replay"):
        campaign_commands.add_parser(name).add_argument("campaign_id")
    enhanced = commands.add_parser(
        "forecast-workflow", help="configurable enhanced price forecast with durable recovery"
    )
    enhanced_commands = enhanced.add_subparsers(dest="enhanced_command", required=True)
    enhanced_prepare = enhanced_commands.add_parser("prepare")
    enhanced_prepare.add_argument("workflow_id")
    enhanced_prepare.add_argument("--campaign-id")
    enhanced_prepare.add_argument("--profile", type=Path)
    enhanced_prepare.add_argument("--output", type=Path)
    enhanced_preflight = enhanced_commands.add_parser("preflight")
    enhanced_preflight.add_argument("workflow_id")
    enhanced_preflight.add_argument("--campaign-id")
    enhanced_preflight.add_argument("--profile", type=Path)
    enhanced_execute = enhanced_commands.add_parser("execute")
    enhanced_execute.add_argument("workflow_id")
    enhanced_execute.add_argument("--campaign-id", required=True)
    enhanced_execute.add_argument("--profile", type=Path, required=True)
    enhanced_execute.add_argument("--max-fits", type=int, required=True)
    enhanced_execute.add_argument("--output", type=Path)
    enhanced_report = enhanced_commands.add_parser("report")
    enhanced_report.add_argument("workflow_id")
    enhanced_report.add_argument("--output", type=Path, required=True)
    enhanced_report.add_argument("--replay", action="store_true")
    for name in ("run", "status", "replay"):
        command = enhanced_commands.add_parser(name)
        command.add_argument("workflow_id")
        command.add_argument("--output", type=Path)
        if name == "run":
            command.add_argument("--config-ref")
            command.add_argument("--protocol-ref")
            command.add_argument("--max-fits", type=int, default=1)
    trajectory = commands.add_parser(
        "trajectory", help="finite development trajectory; no promotion"
    )
    trajectory_commands = trajectory.add_subparsers(dest="trajectory_command", required=True)
    trajectory_commands.add_parser("init").add_argument("plan", type=Path)
    for name in ("run", "resume", "status", "replay"):
        command = trajectory_commands.add_parser(name)
        if name in ("run", "resume"):
            command.add_argument("--steps", type=int, default=1)
    observer = commands.add_parser(
        "observer", help="finite durable baseline observation; no broker"
    )
    observer_commands = observer.add_subparsers(dest="observer_command", required=True)
    registering = observer_commands.add_parser("register")
    registering.add_argument("plan", type=Path)
    registering.add_argument("--episode-id", required=True)
    for name in ("run", "resume", "status", "replay", "stop"):
        command = observer_commands.add_parser(name)
        command.add_argument("episode_id")
        if name in ("run", "resume", "stop"):
            command.add_argument("--request-id", required=True)
        if name == "run":
            command.add_argument("--steps", type=int, default=1)
    baseline = commands.add_parser("baseline", help="seal and inspect actual local C2 models")
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_commands.add_parser("seal").add_argument("run_directory", type=Path)
    for name in ("status", "replay"):
        baseline_commands.add_parser(name).add_argument("bundle_ref")
    inference = commands.add_parser(
        "infer", help="independent inference; no fit, proposal or orders"
    )
    inference.add_argument("bundle_ref")
    inference.add_argument("snapshot_ref")
    inference.add_argument("session_id")
    inference.add_argument("--as-of")
    inference.add_argument(
        "--basis",
        choices=(
            "documented_historical_availability",
            "assumed_historical_availability",
            "observed_receipts",
        ),
    )
    data = commands.add_parser("data", help="immutable local event imports")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    importing = data_commands.add_parser("ingest")
    importing.add_argument("contract", type=Path)
    importing.add_argument("input", type=Path)
    importing.add_argument("--request-id", required=True)
    importing.add_argument("--evidence-file", type=Path)
    for name in ("status", "replay"):
        data_commands.add_parser(name).add_argument("snapshot_ref")
    evidence = commands.add_parser(
        "evidence", help="initialize the configured single evidence authority"
    )
    evidence_commands = evidence.add_subparsers(dest="evidence_command", required=True)
    evidence_commands.add_parser("init").add_argument("program", type=Path)
    compare = commands.add_parser(
        "compare", help="protected engineering-only comparison; no promotion"
    )
    compare_commands = compare.add_subparsers(dest="compare_command", required=True)
    register = compare_commands.add_parser("register")
    register.add_argument("plan", type=Path)
    register.add_argument("--output", type=Path, required=True)
    for name in ("run", "resume", "status", "replay", "cancel", "decide"):
        command = compare_commands.add_parser(name)
        command.add_argument("run_directory", type=Path)
        if name in {"run", "resume"}:
            command.add_argument("--steps", type=int, default=1)
        if name == "decide":
            command.add_argument("--request-id", required=True)
    run = commands.add_parser("run", help="create a new synthetic development run")
    run.add_argument("plan", type=Path)
    run.add_argument("--output", required=True, type=Path)
    for name in ("resume", "replay", "status"):
        commands.add_parser(name).add_argument("run_directory", type=Path)
    research = commands.add_parser("research", help="finite-step development research")
    research_commands = research.add_subparsers(dest="research_command", required=True)
    research_run = research_commands.add_parser("run")
    research_run.add_argument("plan", type=Path)
    research_run.add_argument("--output", required=True, type=Path)
    research_run.add_argument("--steps", type=int, default=1)
    for name in ("resume", "replay", "status"):
        command = research_commands.add_parser(name)
        command.add_argument("run_directory", type=Path)
        if name == "resume":
            command.add_argument("--steps", type=int, default=1)
    package = commands.add_parser(
        "package", help="bounded researcher-package development experiments"
    )
    package_commands = package.add_subparsers(dest="package_command", required=True)
    package_run = package_commands.add_parser("run")
    package_run.add_argument("--evidence", type=Path, required=True)
    package_run.add_argument("--output", type=Path, required=True)
    package_run.add_argument("--state-policy", choices=["reset"], required=True)
    package_run.add_argument("--steps", type=int, default=1)
    for name in ("resume", "replay", "status"):
        command = package_commands.add_parser(name)
        command.add_argument("run_directory", type=Path)
        if name == "resume":
            command.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "research-ui":
            from quant_rsi.research.study_workbench import serve

            serve(args.workspace, port=args.port, open_browser=args.open)
            return 0
        if args.command == "research-resume":
            from quant_rsi.research.study import run_study
            from quant_rsi.research.study_resume import resume_arguments

            result = run_study(resume_arguments(args.directory, ui=args.ui), resume=True)
            if result.get("kind") == "input_check":
                print(f"Input check completed: {args.directory.resolve() / 'input_report.html'}")
                return 0
            print(
                f"Research completed: {args.directory.resolve() / 'report.html'}\n"
                + result["study"]["conclusion"]
            )
            return 0
        if args.command == "research-history":
            from quant_rsi.research.study_history import (
                collection_snapshot,
                history_view,
                observe_prices,
            )

            symbol = args.symbol.strip().upper()
            if args.history_dir is None:
                args.history_dir = default_history_dir("daily")
            if args.output is not None and args.action != "constellation":
                raise ValueError("--output is only used by research-history constellation")
            if args.action == "constellation":
                from quant_rsi.core.artifacts import atomic_write
                from quant_rsi.research.study_constellation import constellation, render

                if args.prices_dir is not None:
                    raise ValueError("research-history constellation does not read prices")
                from quant_rsi.research.study_queue import stock_symbol

                symbol = stock_symbol(symbol)
                data = constellation(args.history_dir, symbol)
                output = (
                    args.output or args.history_dir / "constellation" / f"{symbol}.html"
                ).resolve()
                atomic_write(output, render(data).encode())
                print(
                    f"{symbol}: {len(data['stars'])} factor stars and "
                    f"{len(data['relations'])} relations from {len(data['studies'])} studies: "
                    f"{output}"
                )
                if args.ui if args.ui is not None else sys.stderr.isatty():
                    import webbrowser

                    webbrowser.open(output.as_uri())
                return 0
            if args.action == "factors":
                from quant_rsi.research.study_history import factor_ledger

                if args.prices_dir is not None:
                    raise ValueError("research-history factors does not read prices")
                ledger = factor_ledger(args.history_dir, symbol)
                if args.json:
                    print(json.dumps(ledger, indent=2, allow_nan=False))
                else:
                    print(
                        f"{symbol}: {len(ledger['factors'])} remembered factor versions from "
                        f"{ledger['study_count']} studies in {args.history_dir}"
                    )
                    for item in ledger["factors"]:
                        mae = item["best_mae_price"]
                        print(
                            f"  {item['factor_id']} [{item['horizon_sessions']} sessions, "
                            f"definition {item['definition_sha256'][:12]}]: "
                            f"tried {item['rounds_tried']} rounds, "
                            f"selected {item['rounds_selected']}, best development MAE "
                            + ("n/a" if mae is None else f"{mae:.4f}")
                            + f", last seen {item['last_seen']}"
                        )
                return 0
            if args.action == "observe":
                if args.prices_dir is None:
                    raise ValueError("research-history observe requires --prices-dir")
                result = observe_prices(
                    args.history_dir, symbol, collection_snapshot(symbol, args.prices_dir)
                )
            else:
                if args.prices_dir is not None:
                    raise ValueError("research-history status does not read prices")
                result = history_view(args.history_dir, symbol)
            if (args.ui if args.ui is not None else sys.stderr.isatty()) and result["entries"]:
                from quant_rsi.research.study_ui import start_view

                start_view(Path(result["entries"][-1]["run_directory"]))
            if args.json:
                print(json.dumps(result, indent=2, allow_nan=False))
            else:
                entries = result["entries"]
                pending = sum(e["status"] == "pending" for e in entries)
                settled = sum(e["status"] == "settled" for e in entries)
                print(
                    f"{symbol}: {len(entries)} studies shown; "
                    f"{pending} pending, {settled} settled, "
                    f"{len(entries) - pending - settled} unavailable/non-prospective. "
                    "Use --json for full evidence or --ui for the study view."
                )
            return 0
        if args.command == "research-study":
            from quant_rsi.research.study import run_study

            _check_llm_choice(args)
            if args.ui is None:
                args.ui = sys.stderr.isatty() and not args.json
            if args.no_history:
                if args.history_dir is not None:
                    raise ValueError("--no-history cannot be combined with --history-dir")
            elif args.history_dir is None:
                args.history_dir = default_history_dir(args.prices)
            result = run_study(args)
            if args.check_inputs:
                print(
                    json.dumps(result, indent=2, allow_nan=False)
                    if args.json
                    else f"Input check completed: {args.output.resolve() / 'input_report.html'}"
                )
                if args.ui:
                    import webbrowser

                    webbrowser.open((args.output.resolve() / "input_report.html").as_uri())
                return 0
            print(
                json.dumps(result, indent=2, allow_nan=False)
                if args.json
                else f"Research completed: {args.output.resolve() / 'report.html'}\n"
                + result["study"]["conclusion"]
            )
            return 0
        if args.command == "research-demo":
            _check_llm_choice(args)
            from quant_rsi.research.demo import (
                load_catalog,
                render_demo,
                run_demo,
                simulated_hypothesis,
                simulated_news,
            )

            news = simulated_news
            if args.news == "issuer-rss":
                from quant_rsi.research.demo_news import issuer_news

                def news(context):
                    return issuer_news(context, limit=args.news_limit)

            hypothesize = simulated_hypothesis
            providers = {"catalog": lambda: load_catalog(args.catalog)}
            if args.prices == "daily" and (args.model != "lightgbm" or args.prices_dir is None):
                raise ValueError("--prices daily requires --model lightgbm and --prices-dir")
            if args.prices == "synthetic" and args.prices_dir is not None:
                raise ValueError("--prices-dir requires --prices daily")
            if args.model == "lightgbm":
                from quant_rsi.research.demo_model import LightGBMDemo

                prices = None
                if args.prices == "daily":
                    from quant_rsi.research.demo_prices import load_daily_prices

                    prices = load_daily_prices(args.symbol.strip().upper(), args.prices_dir)
                model = LightGBMDemo(load_catalog(args.catalog), prices=prices)
                providers = {
                    "catalog": lambda: model.library,
                    "price_context": model.context,
                    "predict": model.predict,
                    "observe": model.observe,
                    "summarize": model.summarize,
                }
            if args.llm != "simulated":
                from quant_rsi.research.llm import LLMHypothesizer

                hypothesize = LLMHypothesizer(
                    args.llm, args.llm_model, args.llm_timeout, handoff_dir=args.cursor_dir
                )
            if args.related == "profile":
                from quant_rsi.research.demo_related import profile_related

                providers["related"] = profile_related
            result = run_demo(
                args.symbol,
                hypothesize=hypothesize,
                news=news,
                **providers,
            )
            print(json.dumps(result, indent=2) if args.json else render_demo(result))
            return 0
        if args.command == "research-workflow":
            from quant_rsi.research import workflow

            if args.workflow_command in {"prepare", "run"}:
                report = workflow.prepare(json.loads(args.profile.read_bytes()))
                if args.workflow_command == "run":
                    report = workflow.resume(report["campaign_id"], max_fits=args.max_fits)
            elif args.workflow_command == "resume":
                report = workflow.resume(args.campaign_id, max_fits=args.max_fits)
            elif args.workflow_command == "observe":
                report = workflow.observe(
                    args.campaign_id, args.snapshot_ref, request_id=args.request_id
                )
            else:
                report = workflow.inspect(
                    args.campaign_id, replay=args.workflow_command == "replay"
                )
            if args.workflow_command == "report":
                print(workflow.render_report(report))
            else:
                print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "forecast-observe":
            from quant_rsi.forecast import forecast_settlement

            if args.observation_command == "register":
                report = forecast_settlement.register(args.experiment_id)
            elif args.observation_command == "settle":
                report = forecast_settlement.settle(
                    args.experiment_id, args.snapshot_ref, request_id=args.request_id
                )
            else:
                report = forecast_settlement.inspect(
                    args.experiment_id, replay=args.observation_command == "replay"
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "news-collect":
            from quant_rsi.market import news_collection

            if args.collection_command == "init":
                report = news_collection.initialize(
                    news_collection.CollectionPlan.model_validate_json(args.plan.read_bytes())
                )
            elif args.collection_command == "poll":
                report = news_collection.poll(args.campaign_id, args.request_id)
            elif args.collection_command == "process":
                report = news_collection.process(
                    args.campaign_id, args.request_id, args.news_campaign_id
                )
            else:
                report = news_collection.inspect(args.campaign_id)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "factor-discovery":
            from quant_rsi.factors import discovery

            if args.discovery_command == "init":
                plan = discovery.DiscoveryPlan.model_validate_json(args.plan.read_bytes())
                report = discovery.initialize(json.loads(args.inputs.read_bytes()), plan)
            elif args.discovery_command == "resume":
                report = discovery.resume(args.campaign_id, max_fits=args.max_fits)
            else:
                report = discovery.inspect(
                    args.campaign_id, replay=args.discovery_command == "replay"
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "factors":
            from quant_rsi.factors.catalog import resolve_instrument_factors

            report = resolve_instrument_factors(args.root, args.market, args.instrument)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "forecast":
            from quant_rsi.forecast.forecast_requests import inspect_forecast
            from quant_rsi.forecast.forecast_workflow import display_forecast, run_forecast

            if args.forecast_command == "run":
                report = run_forecast(
                    args.ticker,
                    args.output,
                    request_id=args.request_id,
                    prepare_only=args.prepare_only,
                )
            else:
                report = inspect_forecast(args.request_id, replay=args.forecast_command == "replay")
            print(json.dumps(display_forecast(report), indent=2, allow_nan=False))
            return 0
        if args.command == "forecast-campaign":
            from quant_rsi.forecast.forecast_campaign import init_campaign, inspect_campaign

            if args.campaign_command == "init":
                report = init_campaign(json.loads(args.plan.read_text()))
            elif args.campaign_command == "recursive-init":
                from quant_rsi.forecast.recursive.forecast_recursive_campaign import (
                    init_recursive_campaign,
                )

                report = init_recursive_campaign(
                    json.loads(args.plan.read_text()), market_preparation=args.market_preparation
                )
            elif args.campaign_command == "recursive-prepare":
                from quant_rsi.forecast.recursive.forecast_recursive_preparation import (
                    prepare_market_campaign,
                )

                report = prepare_market_campaign(
                    json.loads(args.profile.read_text()),
                    args.output,
                    campaign_id=args.campaign_id,
                    source_campaign_id=args.source_campaign_id,
                )
            else:
                report = inspect_campaign(
                    args.campaign_id, replay=args.campaign_command == "replay"
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "forecast-workflow":
            from quant_rsi.forecast import forecast_usability as usability
            from quant_rsi.forecast.enhanced_forecast_workflow import (
                inspect_workflow,
                prepare_workflow,
                run_workflow,
            )

            action = args.enhanced_command
            stage, parent, files = action, None, None
            output = getattr(args, "output", None)
            try:
                profile = None
                if getattr(args, "profile", None) is not None:
                    stage = "profile"
                    profile = usability.load_profile(args.profile)
                    stage = action
                if action == "preflight":
                    report = usability.preflight(
                        args.workflow_id, campaign_id=args.campaign_id, profile=profile
                    )
                elif action == "execute":
                    report, parent = usability.execute(
                        args.workflow_id,
                        campaign_id=args.campaign_id,
                        profile=profile,
                        max_fits=args.max_fits,
                    )
                    if output is None:
                        output = Path(report["workspace"]) / "reports" / args.workflow_id
                elif action == "report":
                    report, parent = usability.report_context(args.workflow_id, replay=args.replay)
                elif action == "prepare":
                    report = prepare_workflow(
                        args.workflow_id, campaign_id=args.campaign_id, config=profile
                    )
                elif action == "run":
                    report = run_workflow(
                        args.workflow_id,
                        config_ref=args.config_ref,
                        protocol_ref=args.protocol_ref,
                        max_fits=args.max_fits,
                    )
                else:
                    report = inspect_workflow(args.workflow_id, replay=action == "replay")
                if output is not None:
                    stage = "report"
                    if parent is None:
                        _, parent = usability.report_context(args.workflow_id)
                    files = usability.write_reports(report, parent, output)
            except usability.ACTION_ERRORS as exc:
                try:
                    retry = usability.retry_command(argv if argv is not None else sys.argv[1:])
                except usability.HANDLED_ERRORS:
                    retry = None
                report, parent = usability.failure_report(
                    args.workflow_id, exc, stage=stage, retry=retry
                )
                if output is not None:
                    try:
                        files = usability.write_reports(report, parent, output)
                    except usability.HANDLED_ERRORS as report_error:
                        print(f"quant-rsi: report not written: {report_error}", file=sys.stderr)
                if files is not None:
                    report = dict(report, report_files=files)
                print(json.dumps(report, indent=2, allow_nan=False))
                print(f"quant-rsi: {report['failure']['message']}", file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 1
            if files is not None:
                report = dict(report, report_files=files)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "forecast-compare":
            from quant_rsi.forecast.forecast_experiment import (
                inspect_experiment,
                prepare_experiment,
                resume_experiment,
            )

            if args.context_command == "prepare":
                report = prepare_experiment(json.loads(args.inputs.read_text()), args.experiment_id)
            elif args.context_command == "run":
                report = resume_experiment(
                    args.experiment_id, protocol_ref=args.protocol_ref, max_fits=args.max_fits
                )
            else:
                report = inspect_experiment(
                    args.experiment_id, replay=args.context_command == "replay"
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "trajectory":
            if args.trajectory_command == "init":
                report = init_trajectory(TrajectoryPlan.model_validate_json(args.plan.read_bytes()))
            elif args.trajectory_command in ("run", "resume"):
                report = resume_trajectory(steps=args.steps)
            else:
                report = inspect_trajectory()
        elif args.command == "observer":
            if args.observer_command == "register":
                report = register_observer(
                    parse_observation_plan(args.plan.read_bytes()), args.episode_id
                )
            elif args.observer_command in ("run", "resume"):
                report = advance_observer(
                    args.episode_id,
                    request_id=args.request_id,
                    steps=args.steps if args.observer_command == "run" else None,
                )
            elif args.observer_command == "stop":
                report = stop_observer(args.episode_id, request_id=args.request_id)
            else:
                report = inspect_observer(args.episode_id, replay=args.observer_command == "replay")
        elif args.command == "baseline":
            report = (
                seal_baseline(args.run_directory)
                if args.baseline_command == "seal"
                else inspect_baseline(args.bundle_ref)
            )
        elif args.command == "infer":
            report = infer_at(
                args.bundle_ref,
                args.snapshot_ref,
                args.session_id,
                as_of=args.as_of,
                basis=args.basis,
            )
        elif args.command == "data":
            if args.data_command == "ingest":
                report = ingest(
                    DataContract.model_validate_json(args.contract.read_bytes()),
                    args.input,
                    request_id=args.request_id,
                    evidence_file=args.evidence_file,
                )
            else:
                report = inspect_data(args.snapshot_ref, replay=args.data_command == "replay")
        elif args.command == "evidence":
            report = init_evidence(parse_program(args.program.read_bytes()))
        elif args.command == "compare":
            if args.compare_command == "register":
                report = register_comparison(
                    ComparisonInput.model_validate_json(args.plan.read_bytes()), args.output
                )
            elif args.compare_command in {"run", "resume"}:
                report = resume_comparison(args.run_directory, steps=args.steps)
            elif args.compare_command == "cancel":
                report = cancel_comparison(args.run_directory)
            elif args.compare_command == "decide":
                report = decide_comparison(args.run_directory, args.request_id)
            else:
                report = inspect_comparison(
                    args.run_directory, replay=args.compare_command == "replay"
                )
        elif args.command == "package":
            if args.package_command == "run":
                report = create_package(
                    args.evidence, args.output, state_policy=args.state_policy, steps=args.steps
                )
            elif args.package_command == "resume":
                report = resume_package(args.run_directory, steps=args.steps)
            else:
                report = inspect_package(
                    args.run_directory, replay=args.package_command == "replay"
                )
        elif args.command == "research":
            if args.research_command == "run":
                plan = ResearchPlan.model_validate_json(args.plan.read_bytes())
                report = create_research(plan, args.output, steps=args.steps)
            elif args.research_command == "resume":
                report = resume_research(args.run_directory, steps=args.steps)
            else:
                report = inspect_research(
                    args.run_directory, replay=args.research_command == "replay"
                )
        elif args.command == "run":
            plan = parse_development_plan(args.plan.read_bytes())
            report = create_run(plan, args.output)
        elif args.command == "resume":
            report = resume_run(args.run_directory)
        else:
            report = inspect_run(args.run_directory, replay=args.command == "replay")
    except (ValueError, OSError, ValidationError, sqlite3.DatabaseError) as exc:
        print(f"quant-rsi: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0
