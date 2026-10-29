"""Standalone intraday/IBKR commands: `python -m quant_rsi.intraday <command>`.

These commands do not use the POSIX-only controller, so they also run on Windows. They are
not yet wired into the C2/C4/C5 controller, sealing registry or `quant-rsi` CLI.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from quant_rsi.brokers.contracts import InstrumentBinding
from quant_rsi.core.artifacts import canonical
from quant_rsi.market.market_contracts import CalendarSpec, MarketSession


def _print(value) -> None:
    sys.stdout.write(json.dumps(value, indent=2, default=str) + "\n")


def _session(config, client_id):
    from quant_rsi.brokers.ibkr.connection import IbkrSession
    from quant_rsi.brokers.ibkr.ibapi_transport import RealIbapiTransport

    session = IbkrSession(
        RealIbapiTransport(),
        host=config.host,
        port=config.port,
        client_id=client_id,
        timeout=config.request_timeout_seconds,
    )
    session.connect()
    return session


def _config(path):
    from quant_rsi.brokers.ibkr.config import load_config

    return load_config(Path(path))


def cmd_ibkr_check(args) -> int:
    from quant_rsi.brokers.ibkr.guard import LiveAccountRefused, check_paper

    config = _config(args.config)
    report = {}
    for role, client in (("data", config.data_client_id), ("orders", config.orders_client_id)):
        session = _session(config, client)
        info = session.info
        report[role] = {
            "server_version": info.server_version,
            "accounts": list(info.accounts),
            "clock_skew_seconds": info.clock_skew_seconds,
            "clock_skew_ok": info.clock_skew_seconds <= config.max_clock_skew_seconds,
        }
        if role == "orders":
            try:
                report["paper_guard"] = check_paper(
                    config, port=session.port, accounts=info.accounts, client_id=client
                )
            except LiveAccountRefused as error:
                report["paper_guard"] = str(error)
        session.disconnect()
    _print(report)
    return 0


def cmd_resolve(args) -> int:
    from quant_rsi.brokers.ibkr.instruments import resolve
    from quant_rsi.intraday.profile import IntradayWorkflowConfig

    config = _config(args.config)
    profile = IntradayWorkflowConfig.model_validate_json(Path(args.profile).read_bytes())
    session = _session(config, config.data_client_id)
    bindings = [resolve(session, spec).model_dump(mode="json") for spec in profile.specs()]
    session.disconnect()
    Path(args.output).write_bytes(canonical(bindings))
    _print(bindings)
    return 0


def cmd_collect(args) -> int:
    from quant_rsi.brokers.ibkr.pacing import PacingLimiter
    from quant_rsi.intraday.collect import collect_intraday
    from quant_rsi.intraday.profile import IntradayWorkflowConfig, build_contract
    from quant_rsi.intraday.storage import SnapshotStore

    config = _config(args.config)
    profile = IntradayWorkflowConfig.model_validate_json(Path(args.profile).read_bytes())
    bindings = tuple(
        InstrumentBinding.model_validate(b) for b in json.loads(Path(args.bindings).read_bytes())
    )
    calendar = CalendarSpec.model_validate_json(Path(args.calendar).read_bytes())
    contract = build_contract(
        profile, bindings, calendar, data_entitlement="delayed" if args.delayed else "realtime"
    )
    store = SnapshotStore(Path(args.store))
    session = _session(config, config.data_client_id)
    pacing = PacingLimiter(Path(args.store) / "pacing.sqlite")
    ref = collect_intraday(session, pacing, store, contract)
    session.disconnect()
    _print({"snapshot_ref": ref})
    return 0


def _plan(path):
    from quant_rsi.intraday.contracts import IntradayDevelopmentPlan

    return IntradayDevelopmentPlan.model_validate_json(Path(path).read_bytes())


def cmd_evaluate(args) -> int:
    from quant_rsi.intraday.evaluation import evaluate_development
    from quant_rsi.intraday.storage import SnapshotStore

    snapshot = SnapshotStore(Path(args.store)).load(args.snapshot)
    _print(evaluate_development(snapshot, _plan(args.plan)))
    return 0


def cmd_fit(args) -> int:
    from quant_rsi.intraday.features import build_features
    from quant_rsi.intraday.model import fit, label_vector
    from quant_rsi.intraday.storage import SnapshotStore

    snapshot = SnapshotStore(Path(args.store)).load(args.snapshot)
    plan = _plan(args.plan)
    table = build_features(snapshot, plan.leaves, context_tolerance=plan.context_missing_tolerance)
    model = fit(
        table.values,
        label_vector(snapshot, table, plan),
        table.names,
        seed=plan.seed,
        min_rows=plan.min_training_rows,
    )
    Path(args.output).write_text(model.model_text, encoding="utf-8", newline="\n")
    _print({"model_sha256": model.model_digest, "training_rows": model.training_rows})
    return 0


def _runner(args, config, order_session):
    from quant_rsi.brokers.ibkr.orders import IbkrPaperBackend
    from quant_rsi.intraday.paper import EpisodeJournal, PaperPlan, PaperRunner
    from quant_rsi.intraday.storage import SnapshotStore

    plan = PaperPlan.model_validate_json(Path(args.paper_plan).read_bytes())
    snapshot = SnapshotStore(Path(args.store)).load(plan.snapshot_ref)
    bindings = {b.instrument: b for b in snapshot.contract.instruments}
    backend = IbkrPaperBackend(order_session, config, bindings)
    model_text = Path(args.model).read_text(encoding="utf-8")
    from quant_rsi.brokers.ibkr.config import in_restart_window
    from quant_rsi.brokers.ibkr.connection import ClockSkewMonitor

    return PaperRunner(
        EpisodeJournal(Path(args.journal)),
        plan,
        model_text,
        snapshot,
        backend,
        restart_window=lambda now: in_restart_window(config, now),
        clock_skew=ClockSkewMonitor(order_session),
        archive=SnapshotStore(Path(args.store)),
    )


def cmd_paper_register(args) -> int:
    config = _config(args.config)
    session = _session(config, config.orders_client_id)
    runner = _runner(args, config, session)
    now = datetime.now(timezone.utc)
    runner.register(now)
    runner.activate(now)
    _print({"episode": runner.episode, "status": runner.journal.status(runner.episode)})
    return 0


def cmd_paper_run(args) -> int:
    from quant_rsi.intraday.live import run_session

    config = _config(args.config)
    orders = _session(config, config.orders_client_id)
    data = _session(config, config.data_client_id)
    runner = _runner(args, config, orders)
    if not runner.journal.claim_request(runner.episode, args.request_id, "run"):
        sys.stderr.write("request ID already used; resuming from committed state\n")
    today = MarketSession.model_validate_json(Path(args.session).read_bytes())
    _print(run_session(runner, data, today, order_session=orders))
    return 0


def cmd_paper_stop(args) -> int:
    from quant_rsi.intraday.paper import EpisodeJournal

    journal = EpisodeJournal(Path(args.journal))
    now = datetime.now(timezone.utc)
    if journal.claim_request(args.episode, args.request_id, "stop") and journal.status(
        args.episode
    ) in {"active", "registered"}:
        journal.set_status(args.episode, "stopping", now, f"stop:{args.request_id}")
    _print({"episode": args.episode, "status": journal.status(args.episode)})
    return 0


def cmd_paper_status(args) -> int:
    from quant_rsi.intraday.paper import EpisodeJournal

    journal = EpisodeJournal(Path(args.journal))
    _print(
        {
            "episode": args.episode,
            "status": journal.status(args.episode),
            "intents": [
                {"intent_id": r["intent"].intent_id, "status": r["status"], "filled": r["filled"]}
                for r in journal.intents(args.episode)
            ],
            "fills": journal.fills(args.episode),
            "sessions": [e["payload"] for e in journal.events(args.episode, "session_closed")],
        }
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m quant_rsi.intraday")
    sub = root.add_subparsers(dest="command", required=True)

    def add(name, func, *options):
        command = sub.add_parser(name)
        for option in options:
            command.add_argument(
                f"--{option}",
                required=option != "delayed",
                action="store_true" if option == "delayed" else None,
            )
        command.set_defaults(func=func)

    add("ibkr-check", cmd_ibkr_check, "config")
    add("resolve", cmd_resolve, "config", "profile", "output")
    add("collect", cmd_collect, "config", "profile", "bindings", "calendar", "store", "delayed")
    add("evaluate", cmd_evaluate, "store", "snapshot", "plan")
    add("fit", cmd_fit, "store", "snapshot", "plan", "output")
    add("paper-register", cmd_paper_register, "config", "store", "paper-plan", "model", "journal")
    add(
        "paper-run",
        cmd_paper_run,
        "config",
        "store",
        "paper-plan",
        "model",
        "journal",
        "session",
        "request-id",
    )
    add("paper-stop", cmd_paper_stop, "journal", "episode", "request-id")
    add("paper-status", cmd_paper_status, "journal", "episode")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
