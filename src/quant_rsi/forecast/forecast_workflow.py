"""One fixed daily-data baseline: collect, register, research, seal, forecast, replay."""

from pathlib import Path

from quant_rsi.core import controller
from quant_rsi.core.artifacts import atomic_write, canonical, decode, digest
from quant_rsi.execution.baseline import seal_forecast
from quant_rsi.forecast.forecast_requests import inspect_forecast, issue_forecast
from quant_rsi.forecast.forecast_research import evaluation_origins
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import DataContract
from quant_rsi.market.market_data import ingest, market_context
from quant_rsi.research import research
from quant_rsi.research.researcher import ResearchPlan


def build_plan(collection, snapshot_ref):
    contract = DataContract.model_validate_json(canonical(collection["contract"]))
    sessions = [session.session_id for session in contract.calendar.sessions]
    count = sessions.index(collection["origin_session_id"]) + 1
    if count < 60:
        raise ValueError("at least 60 completed registered daily sessions are required")
    cutoff = count - 35
    with market_context() as (registry, _):
        snapshot, registered = registry.snapshot(snapshot_ref)
        if registered != contract:
            raise ValueError("collection contract differs from the registered snapshot")
        for ref, path in collection["evidence_paths"].items():
            if registry.store.put(Path(path).read_bytes()) != ref:
                raise ValueError("collection evidence hash mismatch")
        actions = []
        for item in snapshot["events"]:
            event = item["event"]
            if event["kind"] != "CORPORATE_ACTION_NOTICE":
                continue
            interpretation = collection["cash_actions"].get(event["event_id"])
            if interpretation is None:
                continue  # Retained unknown actions cause abstention in the forecast kernel.
            actions.append(
                dict(
                    schema_version=1,
                    notice_ref=item["payload_ref"],
                    action_kind="cash_dividend",
                    **interpretation,
                )
            )
    return ResearchPlan.model_validate_json(
        canonical(
            dict(
                schema_version=1,
                plan_type="development_research",
                development=dict(
                    schema_version=1,
                    plan_type="forecast_development",
                    training_mode="M0",
                    data=dict(
                        source="local_event_snapshot",
                        sessions=count,
                        instruments=len(contract.instruments),
                        train_cutoff=cutoff,
                        seed=7,
                        snapshot_ref=snapshot_ref,
                        decision_delay_seconds=60,
                    ),
                    objective=dict(
                        schema_version=1,
                        kind="five_session_raw_close",
                        horizon_sessions=5,
                        target="log_close_ratio",
                        currency=contract.currency,
                        price_unit="raw_currency_per_share",
                        price_unit_claim="explicit_assumption",
                        price_unit_evidence_ref=collection["price_unit_evidence_ref"],
                        calendar_evidence_ref=collection["calendar_evidence_ref"],
                        notices=actions,
                        unknown_action_policy="affected_or_unknown_interval_abstain",
                        prior_exposure="previously_exposed_historical_exploration",
                        source_qualifications=collection["qualifications"],
                    ),
                    budget=dict(max_attempts=1, worker_timeout_seconds=180, max_worker_seconds=180),
                    primary_metric="negative_mae_pp",
                    evaluation_stride=5,
                    min_development_sessions=3,
                    baseline="momentum_5",
                    proposals=[
                        dict(
                            name="momentum_5",
                            factor=dict(op="momentum", window=5),
                            hypothesis="Fixed initial five-session raw-close prediction baseline.",
                        )
                    ],
                ),
                policy=dict(
                    algorithm="neighbor_search_v1",
                    window_min=5,
                    window_max=5,
                    min_improvement=0.0,
                    max_no_progress=1,
                ),
            )
        )
    )


def recover_collection(output, ticker):
    """Reuse an atomically completed collection after a missing workflow pointer."""
    completed = []
    for path in sorted((output / "collections").glob("*/collection_receipt.json")):
        receipt = decode(path.read_bytes())
        if receipt["status"] != "collected":
            continue
        result = receipt.get("result")
        if result is None:
            raise ValueError("completed collection is missing its recovery manifest")
        if (
            receipt["ticker"] != ticker
            or result["origin_session_id"] != receipt["origin_session_id"]
        ):
            raise ValueError("completed collection request binding differs")
        if Path(result["receipt_path"]) != path:
            raise ValueError("completed collection receipt path differs")
        for ref, evidence_path in result["evidence_paths"].items():
            if digest(Path(evidence_path).read_bytes()) != ref:
                raise ValueError("completed collection evidence hash differs")
        normalized = receipt["normalized_inputs"]
        contract_path = path.parent / "data_contract.json"
        if (
            digest(Path(result["events_path"]).read_bytes()) != normalized["events_file_sha256"]
            or digest(contract_path.read_bytes()) != normalized["contract_file_sha256"]
            or decode(contract_path.read_bytes()) != result["contract"]
        ):
            raise ValueError("completed collection normalized input binding differs")
        completed.append(result)
    if len(completed) > 1:
        raise ValueError("multiple completed collections require an existing frozen pointer")
    return completed[0] if completed else None


def run_forecast(ticker, output, *, request_id, prepare_only=False):
    from quant_rsi.market.daily_data import collect_daily

    ticker = ticker.strip().upper()
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("forecast request ID must contain 1..128 characters")
    output = Path(output).resolve()
    with market_context(initialize=True) as (registry, _):
        connection = registry.connection
        connection.execute(
            "CREATE TABLE IF NOT EXISTS forecast_runs "
            "(request_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, output TEXT NOT NULL)"
        )
        previous = connection.execute(
            "SELECT ticker,output FROM forecast_runs WHERE request_id=?", (request_id,)
        ).fetchone()
        if previous is not None and tuple(previous) != (ticker, str(output)):
            raise ValueError(
                "request ID already belongs to another ticker/output; reuse its directory"
            )
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO forecast_runs VALUES (?,?,?)",
                (request_id, ticker, str(output)),
            )
    output.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(output):
        identity = dict(ticker=ticker, request_id=request_id)
        identity_path = output / "request.json"
        if identity_path.exists() and decode(identity_path.read_bytes()) != identity:
            raise ValueError("output directory already belongs to a different forecast request")
        if not identity_path.exists():
            if any(
                p.name != ".controller.lock" and not p.name.startswith(".pending-")
                for p in output.iterdir()
            ):
                raise ValueError("new forecast output directory must be empty")
            atomic_write(identity_path, canonical(identity))
        try:
            collection_path = output / "collection.json"
            if not collection_path.exists():
                collection = recover_collection(output, ticker)
                if collection is None:
                    attempts = output / "collections"
                    attempts.mkdir(exist_ok=True)
                    attempt = attempts / f"attempt-{len(list(attempts.iterdir())) + 1:04d}"
                    collection = collect_daily(ticker, attempt)
                atomic_write(collection_path, canonical(collection))
            collection = decode(collection_path.read_bytes())
            contract = DataContract.model_validate_json(canonical(collection["contract"]))
            if list(contract.instruments) != [ticker]:
                raise ValueError("collector returned a different instrument")
            imported = ingest(
                contract,
                Path(collection["events_path"]),
                request_id="forecast-input-" + digest(canonical(identity)),
                evidence_file=(
                    Path(collection["evidence_paths"][contract.availability_evidence_ref])
                    if contract.availability_evidence_ref
                    else None
                ),
            )
            plan = build_plan(collection, imported["snapshot_ref"])
            protocol_path = output / "protocol.json"
            protocol = dict(
                research_plan=plan.model_dump(mode="json"),
                evaluation_origin_indices=list(evaluation_origins(plan.development)),
                collection_ref=digest(collection_path.read_bytes()),
                raw_ref=digest(Path(collection["raw_path"]).read_bytes()),
                collection_receipt_ref=digest(Path(collection["receipt_path"]).read_bytes()),
                fit_recipe="fixed_existing_lightgbm_24_rounds_seed7_momentum5",
                benchmark="unchanged_price_and_fixed_training_mature_label_mean",
                actual_fits_reserved=1,
                worker_seconds_reserved=180,
            )
            if protocol_path.exists():
                frozen = decode(protocol_path.read_bytes())
                if frozen["protocol"] != protocol:
                    raise ValueError("forecast inputs differ from the pre-fit frozen protocol")
            else:
                atomic_write(
                    protocol_path,
                    canonical(
                        dict(
                            frozen_at=market_contracts.trusted_now(),
                            protocol=protocol,
                        )
                    ),
                )
            if prepare_only:
                return dict(
                    request_id=request_id,
                    status="prepared",
                    actual_fits=0,
                    origin_session=collection["origin_session_id"],
                    protocol_path=str(protocol_path),
                    protocol_ref=digest(protocol_path.read_bytes()),
                    historical_sessions=plan.development.data.sessions,
                    evaluation_origin_indices=protocol["evaluation_origin_indices"],
                )
            run = output / "research"
            if (run / "research_initialization.json").exists() or not (
                run / "manifest.json"
            ).exists():
                research.create_research(plan, run, initialize_only=True)
            progress = research.inspect_research(run, replay=True)
            if progress["incumbent"] is None:
                progress = research.resume_research(run, steps=1)
            if progress["incumbent"] is None:
                raise ValueError(
                    "fixed baseline did not produce an eligible model; "
                    "inspect research status/logs. "
                    "Its attempt remains charged; this command does not start model search."
                )
            sealed = seal_forecast(run)
            result = issue_forecast(
                sealed["bundle_ref"],
                imported["snapshot_ref"],
                collection["origin_session_id"],
                request_id=request_id,
            )
            atomic_write(output / "forecast.json", canonical(result))
            return result
        except Exception as exc:
            failure = dict(
                failed_at=market_contracts.trusted_now(),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            atomic_write(
                output / "failures" / (digest(canonical(failure)) + ".json"), canonical(failure)
            )
            raise


def replay_forecast(request_id):
    return inspect_forecast(request_id, replay=True)


def display_forecast(report):
    if "forecast" not in report:
        return report
    forecast = report["forecast"]
    return dict(
        request_id=report["request_id"],
        status=report["status"],
        origin_session=forecast["origin_session"],
        data_cutoff=forecast["data_cutoff"],
        issued_at=forecast["issued_at"],
        generated_at=report["generated_at"],
        target_session=forecast["target_session"],
        target_close_at=forecast["target_close_at"],
        result_commit_confirmed_at=(
            report["confirmation"]["result_commit_confirmed_at"] if report["confirmation"] else None
        ),
        model_version=forecast["model_version"],
        currency=forecast["currency"],
        rows=[
            {
                key: value
                for key, value in row.items()
                if key not in {"source_refs", "action_evidence"}
            }
            for row in forecast["rows"]
        ],
        historical_evaluation=forecast["historical_evaluation"],
        historical_benchmarks={
            name: baseline["metrics"]
            for name, baseline in forecast["historical_benchmarks"].items()
        },
        qualifications=forecast["qualifications"],
        result_ref=report["result_ref"],
        replay_verified=report["replay_verified"],
    )
