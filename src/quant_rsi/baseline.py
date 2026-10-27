"""Seal actual C2 incumbents and perform inference without fit or proposal calls."""

from pathlib import Path

from quant_rsi import controller, market_contracts, research
from quant_rsi.artifacts import canonical, digest
from quant_rsi.asof import make_feature_view, predict_view
from quant_rsi.forecast_contracts import parse_development_plan
from quant_rsi.local_research import LocalResearchData
from quant_rsi.market_contracts import timestamp
from quant_rsi.market_data import market_context
from quant_rsi.schemas import FactorSpec


def seal_baseline(root: Path):
    return _seal_model(root, forecast=False)


def seal_forecast(root: Path):
    return _seal_model(root, forecast=True)


def _seal_model(root: Path, *, forecast):
    root = root.resolve()
    with controller.controller_lock(root):
        journal, source, manifest, plan, data = research.open_research(root)
        try:
            is_forecast = plan.development.plan_type == "forecast_development"
            if is_forecast != forecast:
                raise ValueError("forecast models require the forecast seal/consumer interface")
            if plan.development.data.source != "local_event_snapshot":
                raise ValueError("baseline observer requires an actual local-data C2 model")
            trace = research.verify_trace(journal, source, manifest, plan, data, replay=True)
            checkpoint_ref, checkpoint = trace[:2]
            if checkpoint["incumbent"] is None or any(
                row["checkpoint_ref"] is None for row in journal.requests()
            ):
                raise ValueError(
                    "sealing requires a committed, eligible incumbent and no pending research step"
                )
            selected = checkpoint["incumbent"]
            source_run_ref = digest(canonical(manifest))
            with market_context() as (registry, _):
                store = registry.store
                existing = registry.connection.execute(
                    "SELECT bundle_ref FROM bundles WHERE source_run_ref=? AND checkpoint_ref=?",
                    (source_run_ref, checkpoint_ref),
                ).fetchall()
                if existing:
                    if len(existing) != 1:
                        raise ValueError("duplicate baseline registration binding")
                    bundle, _, _, _ = _read_model(registry, existing[0][0], forecast=forecast)
                    return bundle_report(existing[0][0], bundle)
                registered_snapshot, contract = registry.snapshot(data.spec.snapshot_ref)
                if registered_snapshot != data.snapshot or contract != data.contract:
                    raise ValueError("C2 data differs from the actual registered immutable source")
                if forecast:
                    validate_forecast_evidence(registry, plan.development.objective, data.snapshot)
                kernel_result = source.json(selected["result_ref"])
                job = source.json(kernel_result["job_ref"])
                metadata = source.json(kernel_result["worker_metadata_ref"])
                registered_at = market_contracts.trusted_now()
                if timestamp(registered_at) < timestamp(metadata["actual_fit_completed_at"]):
                    raise ValueError("trusted registration clock precedes actual model fit")
                refs = set(kernel_result.values()) | {
                    selected["result_ref"],
                    checkpoint_ref,
                    manifest["data_ref"],
                    manifest["research_plan_ref"],
                    manifest["runtime_ref"],
                    job["binding"]["manifest"]["plan_ref"],
                }
                for ref in refs:
                    store.put(source.get(ref))
                store.put_json(manifest)
                bundle = dict(
                    kind="RegisteredForecastModel" if forecast else "RegisteredBaselineModel",
                    schema_version=1,
                    purpose="five_session_price_forecast"
                    if forecast
                    else "developer_visible_observation",
                    source_run_ref=source_run_ref,
                    source_checkpoint_ref=checkpoint_ref,
                    kernel_result_ref=selected["result_ref"],
                    contract_ref=data.snapshot["contract_ref"],
                    source_snapshot_ref=data.spec.snapshot_ref,
                    data_ref=manifest["data_ref"],
                    runtime_ref=manifest["runtime_ref"],
                    model_ref=kernel_result["model_ref"],
                    factor=selected["factor"],
                    preprocessing="causal_three_features_v1",
                    training_information_cutoff=metadata["training_information_cutoff"],
                    training_view_asof=metadata["training_view_asof"],
                    actual_fit_completed_at=metadata["actual_fit_completed_at"],
                    model_effective_at=registered_at,
                    registered_at=registered_at,
                    qualifications=metadata["qualifications"],
                    promotion_status="none",
                )
                if forecast:
                    objective = plan.development.objective.model_dump(mode="json")
                    bundle.update(
                        objective=objective,
                        objective_ref=digest(canonical(objective)),
                        primary_metric=plan.development.primary_metric,
                    )
                if forecast and plan.schema_version == 3:
                    from quant_rsi.forecast_recursive_output import bind_final_forecast

                    bundle = bind_final_forecast(root, manifest, checkpoint_ref, bundle, store)
                bundle_ref = store.put_json(bundle)
                with registry.connection:
                    registry.connection.execute(
                        "INSERT INTO bundles VALUES (?,?,?,?)",
                        (bundle_ref, source_run_ref, checkpoint_ref, registered_at),
                    )
                return bundle_report(bundle_ref, bundle)
        finally:
            journal.close()


def read_bundle(registry, bundle_ref):
    return _read_model(registry, bundle_ref, forecast=False)


def read_forecast_bundle(registry, bundle_ref):
    return _read_model(registry, bundle_ref, forecast=True)


def validate_forecast_evidence(registry, objective, snapshot):
    registry.store.get(objective.price_unit_evidence_ref)
    registry.store.get(objective.calendar_evidence_ref)
    actions = {
        item["payload_ref"]: item
        for item in snapshot["events"]
        if item["event"]["kind"] == "CORPORATE_ACTION_NOTICE"
    }
    from quant_rsi.forecast_kernel import cash_interpretation_matches

    for notice in objective.notices:
        registry.store.get(notice.evidence_ref)
        if notice.notice_ref not in actions:
            raise ValueError("forecast notice is absent from the registered snapshot")
        if notice.action_kind == "cash_dividend" and not cash_interpretation_matches(
            actions[notice.notice_ref], notice
        ):
            raise ValueError("forecast cash interpretation differs from its source payload")


def _read_model(registry, bundle_ref, *, forecast):
    row = registry.connection.execute(
        "SELECT * FROM bundles WHERE bundle_ref=?", (bundle_ref,)
    ).fetchone()
    if row is None:
        raise ValueError("model bundle is not registered by the local baseline controller")
    store = registry.store
    bundle = store.json(bundle_ref)
    if (
        bundle["kind"] != ("RegisteredForecastModel" if forecast else "RegisteredBaselineModel")
        or bundle["purpose"]
        != ("five_session_price_forecast" if forecast else "developer_visible_observation")
        or (bundle["source_run_ref"], bundle["source_checkpoint_ref"], bundle["registered_at"])
        != (row["source_run_ref"], row["checkpoint_ref"], row["registered_at"])
    ):
        raise ValueError("baseline registration identity mismatch")
    if store.json(bundle["runtime_ref"]) != controller.runtime_identity():
        raise ValueError("baseline source/runtime changed; use the recorded implementation")
    manifest = store.json(bundle["source_run_ref"])
    if forecast:
        source_plan = research.ResearchPlan.model_validate_json(
            store.get(manifest["research_plan_ref"])
        )
        if source_plan.schema_version == 3:
            from quant_rsi.forecast_recursive_output import verify_final_forecast

            verify_final_forecast(store, bundle)
        elif bundle.get("schema_version") != 1 or "recursive_lineage_ref" in bundle:
            raise ValueError("legacy forecast cannot acquire a recursive output identity")
    data = LocalResearchData.from_json(store.json(bundle["data_ref"]))
    if (
        manifest["data_ref"] != bundle["data_ref"]
        or manifest["runtime_ref"] != bundle["runtime_ref"]
        or data.snapshot["contract_ref"] != bundle["contract_ref"]
        or data.spec.snapshot_ref != bundle["source_snapshot_ref"]
    ):
        raise ValueError("baseline source data/preprocessing binding mismatch")
    checkpoint = store.json(bundle["source_checkpoint_ref"])
    if (
        checkpoint["incumbent"]["result_ref"] != bundle["kernel_result_ref"]
        or checkpoint["incumbent"]["factor"] != bundle["factor"]
    ):
        raise ValueError("baseline model is not the actual selected C2 incumbent")
    result = store.json(bundle["kernel_result_ref"])
    job = store.json(result["job_ref"])
    plan = parse_development_plan(store.get(job["binding"]["manifest"]["plan_ref"]))
    if (plan.plan_type == "forecast_development") != forecast:
        raise ValueError("forecast target cannot be consumed by the legacy baseline interface")
    if forecast:
        objective = plan.objective.model_dump(mode="json")
        if (
            bundle.get("objective") != objective
            or bundle.get("objective_ref") != digest(canonical(objective))
            or bundle.get("primary_metric") != plan.primary_metric
        ):
            raise ValueError("forecast bundle target/objective mismatch")
        validate_forecast_evidence(registry, plan.objective, data.snapshot)
    if (
        bundle["model_ref"] != result["model_ref"]
        or job["binding"]["proposal"]["factor"] != bundle["factor"]
    ):
        raise ValueError("baseline model/factor differs from the bound Kernel result")
    metadata = store.json(result["worker_metadata_ref"])
    for key in ("training_information_cutoff", "training_view_asof", "actual_fit_completed_at"):
        if bundle[key] != metadata[key]:
            raise ValueError("baseline fit/information metadata mismatch")
    if bundle["model_effective_at"] != bundle["registered_at"] or timestamp(
        bundle["registered_at"]
    ) < timestamp(bundle["actual_fit_completed_at"]):
        raise ValueError("baseline effective time precedes actual registration/fit")
    controller.replay_result(store, result, job, plan, data)
    return (
        bundle,
        store.get(bundle["model_ref"]).decode(),
        data.contract,
        FactorSpec.model_validate_json(canonical(bundle["factor"])),
    )


def bundle_report(bundle_ref, bundle):
    report = dict(
        bundle_ref=bundle_ref,
        model_ref=bundle["model_ref"],
        contract_ref=bundle["contract_ref"],
        source_run_ref=bundle["source_run_ref"],
        source_checkpoint_ref=bundle["source_checkpoint_ref"],
        training_information_cutoff=bundle["training_information_cutoff"],
        training_view_asof=bundle["training_view_asof"],
        actual_fit_completed_at=bundle["actual_fit_completed_at"],
        model_effective_at=bundle["model_effective_at"],
        purpose=bundle["purpose"],
        promotion_status="none",
    )
    if bundle["kind"] == "RegisteredForecastModel":
        report.update(objective=bundle["objective"], objective_ref=bundle["objective_ref"])
    return report


def inspect_baseline(bundle_ref):
    with market_context() as (registry, _):
        bundle, _, _, _ = read_bundle(registry, bundle_ref)
        return bundle_report(bundle_ref, bundle)


def infer_at(bundle_ref, snapshot_ref, session_id, *, as_of=None, basis=None):
    with market_context() as (registry, _):
        bundle, model_text, contract, factor = read_bundle(registry, bundle_ref)
        snapshot, input_contract = registry.snapshot(snapshot_ref)
        if contract != input_contract or snapshot["contract_ref"] != bundle["contract_ref"]:
            raise ValueError(
                "inference snapshot differs from the model's registered input interface"
            )
        started = market_contracts.trusted_now()
        requested_at = as_of or started
        if timestamp(requested_at) > timestamp(started):
            raise ValueError("inference cannot borrow a future knowledge boundary")
        view = make_feature_view(snapshot, contract, session_id, requested_at, factor, basis=basis)
        prediction = predict_view(model_text, view)
        finished = market_contracts.trusted_now()
        if timestamp(finished) < timestamp(started):
            raise ValueError("trusted prediction clock moved backwards")
        effective = timestamp(requested_at) >= timestamp(bundle["model_effective_at"])
        return dict(
            prediction=prediction,
            model_bundle_ref=bundle_ref,
            snapshot_ref=snapshot_ref,
            purpose="developer_visible_observation",
            actual_prediction_started_at=started,
            prediction_finished_at=finished,
            model_effective_for_requested_time=effective,
            usage="asof_observation" if effective else "post_fit_historical_exploration",
            training_information_age_seconds=(
                timestamp(finished) - timestamp(bundle["training_information_cutoff"])
            ).total_seconds(),
            actual_model_age_seconds=(
                timestamp(finished) - timestamp(bundle["actual_fit_completed_at"])
            ).total_seconds(),
            forward_qualification="requires_preexisting_episode_and_durable_submission_controller",
        )
