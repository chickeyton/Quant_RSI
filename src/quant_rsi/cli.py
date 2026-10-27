import argparse
import json
import sqlite3
import sys
from pathlib import Path

from pydantic import ValidationError

from quant_rsi.baseline import infer_at, inspect_baseline, seal_baseline
from quant_rsi.comparison import (
    cancel_comparison,
    decide_comparison,
    inspect_comparison,
    register_comparison,
    resume_comparison,
)
from quant_rsi.controller import create_run, inspect_run, resume_run
from quant_rsi.evidence import ComparisonInput, init_evidence, parse_program
from quant_rsi.forecast_contracts import parse_development_plan
from quant_rsi.market_contracts import DataContract, parse_observation_plan
from quant_rsi.market_data import ingest, inspect_data
from quant_rsi.observer import advance_observer, inspect_observer, register_observer, stop_observer
from quant_rsi.package_experiment import create_package, inspect_package, resume_package
from quant_rsi.research import create_research, inspect_research, resume_research
from quant_rsi.researcher import ResearchPlan
from quant_rsi.trajectory import init_trajectory, inspect_trajectory, resume_trajectory
from quant_rsi.trajectory_state import TrajectoryPlan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded quant research and development observations"
    )
    commands = parser.add_subparsers(dest="command", required=True)
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
    for name in ("status", "replay"):
        campaign_commands.add_parser(name).add_argument("campaign_id")
    enhanced = commands.add_parser(
        "forecast-workflow", help="configurable enhanced price forecast with durable recovery"
    )
    enhanced_commands = enhanced.add_subparsers(dest="enhanced_command", required=True)
    enhanced_prepare = enhanced_commands.add_parser("prepare")
    enhanced_prepare.add_argument("workflow_id")
    enhanced_prepare.add_argument("--campaign-id")
    for name in ("run", "status", "replay"):
        command = enhanced_commands.add_parser(name)
        command.add_argument("workflow_id")
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
        if args.command == "forecast":
            from quant_rsi.forecast_requests import inspect_forecast
            from quant_rsi.forecast_workflow import display_forecast, run_forecast

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
            from quant_rsi.forecast_campaign import init_campaign, inspect_campaign

            if args.campaign_command == "init":
                report = init_campaign(json.loads(args.plan.read_text()))
            else:
                report = inspect_campaign(
                    args.campaign_id, replay=args.campaign_command == "replay"
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "forecast-workflow":
            from quant_rsi.enhanced_forecast_workflow import (
                inspect_workflow,
                prepare_workflow,
                run_workflow,
            )

            if args.enhanced_command == "prepare":
                report = prepare_workflow(args.workflow_id, campaign_id=args.campaign_id)
            elif args.enhanced_command == "run":
                report = run_workflow(
                    args.workflow_id,
                    config_ref=args.config_ref,
                    protocol_ref=args.protocol_ref,
                    max_fits=args.max_fits,
                )
            else:
                report = inspect_workflow(
                    args.workflow_id, replay=args.enhanced_command == "replay"
                )
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "forecast-compare":
            from quant_rsi.forecast_experiment import (
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
