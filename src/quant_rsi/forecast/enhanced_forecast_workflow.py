"""Configurable price-context workflow on the existing forecast experiment ledger."""

import re
from datetime import timedelta

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.core.journal import Journal
from quant_rsi.evaluation.evidence import authority_root
from quant_rsi.forecast import forecast_context_data as context
from quant_rsi.forecast import forecast_experiment as experiment
from quant_rsi.forecast.forecast_inputs import ForecastInputs
from quant_rsi.forecast.forecast_workflow_config import ForecastWorkflowConfig
from quant_rsi.market import market_contracts


def workflow_root(workflow_id):
    if not isinstance(workflow_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", workflow_id):
        raise ValueError("invalid forecast workflow ID")
    return authority_root() / "enhanced_forecast_workflows" / workflow_id


def _config(value):
    return ForecastWorkflowConfig.model_validate_json(canonical(value))


def _record(path, value):
    body = canonical(value)
    if path.exists():
        if path.read_bytes() != body:
            raise ValueError(f"workflow content conflict: {path.name}")
    else:
        atomic_write(path, body)


def _initialize(root, config, managed):
    marker = root / "bootstrap.json"
    if (root / "manifest.json").exists():
        return
    if marker.exists():
        manifest = decode(marker.read_bytes())
        if manifest["config"] != config.model_dump(mode="json"):
            raise ValueError("workflow ID already binds another configuration")
    else:
        if any(
            p.name != ".controller.lock" and not p.name.startswith(".pending-")
            for p in root.iterdir()
        ):
            raise ValueError("workflow authority damaged; refusing reinitialization")
        if experiment.experiment_root(root.name).exists():
            raise ValueError("experiment ID is already in use outside this workflow")
        body = config.model_dump(mode="json")
        manifest = dict(
            kind="EnhancedForecastWorkflow",
            schema_version=1,
            workflow_id=root.name,
            workspace=str(authority_root().parent),
            config=body,
            config_ref=digest(canonical(body)),
            runtime=controller.runtime_identity(),
            created_at=market_contracts.trusted_now(),
            managed_run=managed.binding,
        )
        _record(marker, manifest)
    journal = Journal(root, digest(canonical(manifest)))
    journal.close()
    _record(root / "manifest.json", manifest)
    marker.unlink()


def _open(root, *, verify_runtime, managed):
    manifest_raw = (root / "manifest.json").read_bytes()
    manifest = decode(manifest_raw)
    if (
        manifest["kind"] != "EnhancedForecastWorkflow"
        or manifest["schema_version"] != 1
        or workflow_root(manifest["workflow_id"]) != root
        or manifest["workspace"] != str(authority_root().parent)
        or digest(canonical(manifest["config"])) != manifest["config_ref"]
    ):
        raise ValueError("workflow identity/configuration/workspace changed")
    if manifest.get("managed_run") != managed.binding:
        raise ValueError("workflow parent campaign binding differs")
    config = _config(manifest["config"])
    journal = Journal(root)
    try:
        identity = dict(journal.connection.execute("SELECT key,value FROM metadata"))
        if identity != {"manifest_digest": digest(manifest_raw)}:
            raise ValueError("workflow manifest/ledger identity mismatch")
        if journal.attempts():
            raise ValueError("workflow dispatch must use the bound experiment ledger")
        kinds = [event["kind"] for event in journal.events()]
        order = [
            "run_created",
            "collection_started",
            "collection_committed",
            "preparation_started",
            "inputs_committed",
            "experiment_committed",
        ]
        if kinds != order[: len(kinds)]:
            raise ValueError("workflow phase ledger is incomplete or reordered")
        if verify_runtime and manifest["runtime"] != controller.runtime_identity():
            raise ValueError("source/runtime changed; restore the frozen workflow runtime")
        return journal, manifest, config
    except Exception:
        journal.close()
        raise


def _payload(journal, stage):
    return experiment.event_payload(journal, stage)


def _commit(journal, stage, value):
    experiment.commit_event(journal, stage, value)


def _checked(path, ref):
    body = path.read_bytes()
    if digest(body) != ref:
        raise ValueError(f"workflow artifact changed: {path.name}")
    return decode(body)


def _check_collection(root, payload):
    audit = root / "collection"
    manifest = _checked(audit / "manifest.json", payload["manifest_ref"])
    seen = set()
    for item in manifest["files"]:
        member = audit / item["path"]
        if (
            item["path"] in seen
            or member.is_symlink()
            or not member.resolve().is_relative_to(audit.resolve())
        ):
            raise ValueError("invalid collection manifest member")
        seen.add(item["path"])
        body = member.read_bytes()
        if digest(body) != item["sha256"] or len(body) != item["bytes"]:
            raise ValueError("collected source changed")
    return audit


def _check_inputs(root, payload, config):
    for name, ref in payload["preparation_refs"].items():
        _checked(root / "preparation" / name, ref)
    prepared = context.verify_preparation(root / "preparation")
    _checked(root / "preparation" / "preparation.receipt.json", payload["receipt_ref"])
    inputs = ForecastInputs(prepared["inputs"])
    roles = {"target": inputs.target, **inputs.contexts}
    if (
        inputs.ref != payload["input_ref"]
        or {role: market[0].instruments[0] for role, market in roles.items()} != config.roles
        or any(market[0].spec.sessions != config.history_sessions for market in roles.values())
    ):
        raise ValueError("workflow frozen inputs differ from its configuration")
    return inputs


def _freshness(root, inputs, config):
    now = market_contracts.timestamp(market_contracts.trusted_now())
    prepared = decode((root / "preparation" / "prepared.json").read_bytes())
    for item in prepared["collections"].values():
        age = (now - market_contracts.timestamp(item["http_received_at"])).total_seconds()
        if not 0 <= age <= config.max_source_age_seconds:
            raise ValueError("frozen source receipts are stale or future-dated; no recollection")
    sessions = inputs.target[0].contract.calendar.sessions
    complete = [s.session_id for s in sessions if s.close_at + timedelta(minutes=15) <= now]
    if not complete or complete[-1] != prepared["origin_session_id"]:
        raise ValueError("frozen origin is no longer the latest complete session")


def _bound_report(root, journal, config, *, replay=False):
    collected = _payload(journal, "collection_committed")
    if collected:
        _check_collection(root, collected)
    frozen = _payload(journal, "inputs_committed")
    if frozen:
        _check_inputs(root, frozen, config)
    bound = _payload(journal, "experiment_committed")
    if not bound:
        return None
    _checked(experiment.experiment_root(root.name) / "manifest.json", bound["manifest_ref"])
    report = experiment.inspect_experiment(root.name, replay=replay)
    if (
        report["input_ref"] != frozen["input_ref"]
        or report["protocol_ref"] != bound["protocol_ref"]
    ):
        raise ValueError("workflow experiment/input/protocol binding differs")
    return report


def _terminal(report):
    if report["forecast"] is not None:
        return True
    selection = report["selection"]
    if selection is not None and selection["selected_recipe"] is None:
        return True
    return (
        len(report["attempts"]) == report["budget"]["max_attempts"]
        and all(row["status"] not in {"reserved", "running"} for row in report["attempts"])
        and report["attempts"][-1]["status"] != "completed"
    )


def _prepare_workflow(value, workflow_id, managed):
    """Collect and freeze one explicit configuration, with zero model dispatches."""
    config = _config(value)
    root = workflow_root(workflow_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(root.parent):
        root.mkdir(exist_ok=True)
        with controller.controller_lock(root):
            _initialize(root, config, managed)
    with controller.controller_lock(root) as owner_fd:
        journal, manifest, saved = _open(root, verify_runtime=True, managed=managed)
        try:
            if config != saved:
                raise ValueError("workflow ID already binds another configuration")
            from quant_rsi.forecast.forecast_campaign import bind_workflow

            bind_workflow(managed, manifest)
            report = _bound_report(root, journal, config)
            if report is not None:
                return _display(root, journal, manifest, report)
            collected = _payload(journal, "collection_committed")
            if not collected:
                from quant_rsi.forecast.forecast_context_collection import collect_context

                # An initialized collection cannot silently disappear on recovery.
                started = _payload(journal, "collection_started")
                if started and not (root / "collection").is_dir():
                    raise ValueError("initialized collection is missing")
                (root / "collection").mkdir(exist_ok=True)
                _commit(journal, "collection_started", {"config_ref": manifest["config_ref"]})
                audit = collect_context(root / "collection", roles=config.roles)
                if audit.resolve() != (root / "collection").resolve():
                    raise ValueError("collector returned another authority path")
                collected = {"manifest_ref": digest((audit / "manifest.json").read_bytes())}
                _commit(journal, "collection_committed", collected)
            audit = _check_collection(root, collected)
            frozen = _payload(journal, "inputs_committed")
            if not frozen:
                started = _payload(journal, "preparation_started")
                out = root / "preparation"
                if started and not out.is_dir():
                    raise ValueError("initialized preparation is missing")
                out.mkdir(exist_ok=True)
                _commit(journal, "preparation_started", {"manifest_ref": collected["manifest_ref"]})
                result = context.prepare_context_data(
                    audit,
                    out,
                    roles=config.roles,
                    history_sessions=config.history_sessions,
                    resume=True,
                    parent_lock_fd=(owner_fd, managed.owner_fd),
                )
                frozen = dict(
                    input_ref=result["meta"]["inputs_ref"],
                    preparation_refs={
                        name: digest((out / name).read_bytes())
                        for name in (
                            "prepared.json",
                            "intent.json",
                            "registry.identity.json",
                            "prepared.binding.json",
                        )
                    },
                    receipt_ref=digest((out / "preparation.receipt.json").read_bytes()),
                )
                _commit(journal, "inputs_committed", frozen)
            inputs = _check_inputs(root, frozen, config)
            _freshness(root, inputs, config)
            # Existing bootstrap intent binds the freeze clock/runtime across interrupted setup.
            report = experiment.prepare_experiment(inputs.value, root.name)
            _commit(
                journal,
                "experiment_committed",
                dict(
                    manifest_ref=digest(
                        (experiment.experiment_root(root.name) / "manifest.json").read_bytes()
                    ),
                    protocol_ref=report["protocol_ref"],
                ),
            )
            return _display(root, journal, manifest, report)
        finally:
            journal.close()


def _run_workflow(workflow_id, managed, *, config_ref=None, protocol_ref=None, max_fits=1):
    """Resume the same fixed cumulative budget; each call may dispatch at most max_fits."""
    root = workflow_root(workflow_id)
    with controller.controller_lock(root) as owner_fd:
        journal, manifest, config = _open(root, verify_runtime=True, managed=managed)
        try:
            if config_ref is not None and config_ref != manifest["config_ref"]:
                raise ValueError("run requires the exact frozen configuration reference")
            report = _bound_report(root, journal, config)
            if report is None:
                raise ValueError("workflow is not frozen; repeat prepare with its original config")
            if protocol_ref is not None and protocol_ref != report["protocol_ref"]:
                raise ValueError("run requires the exact frozen protocol reference")
            protocol_ref = report["protocol_ref"]
            if _terminal(report):
                return _display(root, journal, manifest, report)
            # Recover an existing dispatch/publication even after sources become stale.
            report = experiment.resume_experiment(
                root.name,
                protocol_ref=protocol_ref,
                max_fits=0,
                parent_lock_fd=owner_fd,
            )
            if _terminal(report):
                return _display(root, journal, manifest, report)
            inputs = _check_inputs(root, _payload(journal, "inputs_committed"), config)
            _freshness(root, inputs, config)
            report = experiment.resume_experiment(
                root.name,
                protocol_ref=protocol_ref,
                max_fits=max_fits,
                parent_lock_fd=owner_fd,
            )
            return _display(root, journal, manifest, report)
        finally:
            journal.close()


def _inspect_workflow(workflow_id, managed, *, replay=False):
    root = workflow_root(workflow_id)
    with controller.controller_lock(root):
        journal, manifest, config = _open(root, verify_runtime=replay, managed=managed)
        try:
            return _display(
                root, journal, manifest, _bound_report(root, journal, config, replay=replay)
            )
        finally:
            journal.close()


def _controls(root, report):
    if not report or not report["selection"] or not report["selection"]["selected_recipe"]:
        return None
    from quant_rsi.forecast.forecast_kernel import forecast_errors

    selection = report["selection"]
    group = selection["selected_recipe"]
    store = ArtifactStore(experiment.experiment_root(root.name) / "artifacts")
    folds = [store.json(ref) for ref in selection["report_refs"].values()]
    folds = sorted((f for f in folds if f["recipe"] == group), key=lambda f: f["fold"])
    actual = [row["actual"] for fold in folds for row in fold["rows"]]
    means = [fold["training_mean_log_return"] for fold in folds for _ in fold["rows"]]
    controls = dict(
        unchanged_price=forecast_errors([0.0] * len(actual), actual),
        training_mature_label_mean=forecast_errors(means, actual),
    )
    selected = selection["candidates"][group]["pooled"]
    return dict(
        selected_recipe=group,
        selected_development_metrics=selected,
        naive_controls=controls,
        selected_beats_each_control={
            name: {metric: selected[metric] < value[metric] for metric in ("mae_pp", "rmse_pp")}
            for name, value in controls.items()
        },
        predictive_validity_established=False,
        qualification="Development comparison only. Candidate selection does not establish "
        "predictive advantage; false control comparisons mean the selected model "
        "did not beat that control. Publication uses a separate model.",
    )


def _display(root, journal, manifest, report):
    phases = [event["kind"] for event in journal.events()]
    phase = phases[-1]
    if report:
        phase = (
            "complete" if _terminal(report) else "frozen" if not report["attempts"] else "running"
        )
        if _terminal(report) and report["forecast"] is None:
            phase = "exhausted_without_publication"
    return dict(
        workflow_id=root.name,
        managed_run=manifest["managed_run"],
        status=phase,
        workspace=manifest["workspace"],
        config=manifest["config"],
        config_ref=manifest["config_ref"],
        protocol_ref=report["protocol_ref"] if report else None,
        experiment=report,
        development_comparison=_controls(root, report),
        coverage=dict(
            inputs=["target_prices", "industry_proxy_prices", "market_proxy_prices"],
            news="not_collected_or_used",
            fundamentals="not_collected_or_used",
            corporate_actions="returned notices retained; completeness unknown",
            history="explicit availability assumption, not certified point-in-time",
        ),
    )


def _run_scope(workflow_id):
    from quant_rsi.forecast.forecast_campaign import run_scope

    path = workflow_root(workflow_id) / "manifest.json"
    manifest = decode(path.read_bytes()) if path.exists() else None
    return run_scope(workflow_id, manifest=manifest, required=True)


def prepare_workflow(workflow_id, *, campaign_id=None, config=None):
    """Prepare an allocated run using the immutable registered configuration."""
    with _run_scope(workflow_id) as managed:
        if campaign_id is not None and campaign_id != managed.binding["campaign_id"]:
            raise ValueError("run is allocated to a different campaign")
        if config is not None and _config(config).model_dump(mode="json") != managed.config:
            raise ValueError("run already binds another configuration")
        return _prepare_workflow(managed.config, workflow_id, managed)


def run_workflow(workflow_id, *, config_ref=None, protocol_ref=None, max_fits=1):
    with _run_scope(workflow_id) as managed:
        return _run_workflow(
            workflow_id,
            managed,
            config_ref=config_ref,
            protocol_ref=protocol_ref,
            max_fits=max_fits,
        )


def inspect_workflow(workflow_id, *, replay=False):
    with _run_scope(workflow_id) as managed:
        return _inspect_workflow(workflow_id, managed, replay=replay)


def validate_managed_dispatch(managed, inputs, plan):
    """Shared-engine gate before each new reservation, including direct CLI calls."""
    # The live parent scope serializes managed workflow writers. This read-only
    # check must not acquire a workflow lock after the caller's experiment lock.
    root = workflow_root(managed.binding["run_id"])
    journal, _, config = _open(root, verify_runtime=True, managed=managed)
    try:
        collected = _payload(journal, "collection_committed")
        frozen = _payload(journal, "inputs_committed")
        bound = _payload(journal, "experiment_committed")
        if collected is None or frozen is None or bound is None:
            raise ValueError("managed workflow is not frozen; repeat its preparation")
        _check_collection(root, collected)
        verified = _check_inputs(root, frozen, config)
        child = experiment.experiment_root(root.name)
        _checked(child / "manifest.json", bound["manifest_ref"])
        if verified.ref != inputs.ref or bound["protocol_ref"] != plan.protocol_ref:
            raise ValueError("managed dispatch differs from workflow-frozen inputs/protocol")
        _freshness(root, verified, config)
    finally:
        journal.close()
