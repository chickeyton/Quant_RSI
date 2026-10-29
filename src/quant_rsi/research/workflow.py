"""One frozen profile joining existing discovery, accounts and forecast observations."""

from typing import Literal, Self

from pydantic import Field, model_validator

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.core.journal import Journal
from quant_rsi.factors import discovery
from quant_rsi.forecast import forecast_settlement
from quant_rsi.forecast.forecast_experiment import (
    commit_event,
    event_payload,
    experiment_root,
    inspect_experiment,
    open_experiment,
)
from quant_rsi.forecast.forecast_inputs import ForecastInputs
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import AccountRules, timestamp, utc_text
from quant_rsi.market.market_data import load_snapshot


class WorkflowProfile(StrictModel):
    schema_version: Literal[1] = 1
    inputs: dict
    discovery: discovery.DiscoveryPlan
    account: AccountRules
    strategy_threshold: float = Field(default=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def minimum_loop(self) -> Self:
        if self.discovery.max_rounds < 2:
            raise ValueError("the workflow requires at least one feedback-dependent descendant")
        if self.account.max_positions != 1 or self.account.termination != "preserve_inventory":
            raise ValueError("the first workflow is single-target with common terminal valuation")
        return self


def workflow_root(campaign_id):
    return discovery.campaign_root(campaign_id).parent.parent / "research_workflows" / campaign_id


def _validate_inputs(profile):
    inputs = ForecastInputs(profile.inputs)
    data, _ = inputs.target
    instrument = profile.discovery.instrument
    if (
        tuple(data.instruments) != (instrument,)
        or profile.account.currency != data.contract.currency
        or set(profile.account.fill_cap_units) != {instrument}
    ):
        raise ValueError("workflow account, instrument and registered source differ")
    return inputs


def _open(campaign_id, *, verify_runtime=True):
    root = workflow_root(campaign_id)
    raw = (root / "manifest.json").read_bytes()
    manifest = decode(raw)
    journal = Journal(root)
    try:
        if dict(journal.connection.execute("SELECT key,value FROM metadata")) != {
            "manifest_digest": digest(raw)
        }:
            raise ValueError("workflow manifest/ledger mismatch")
        if manifest["root"] != str(root.resolve()) or manifest["campaign_id"] != campaign_id:
            raise ValueError("workflow authority moved or identity changed")
        store = ArtifactStore(root / "artifacts")
        profile = WorkflowProfile.model_validate_json(store.get(manifest["profile_ref"]))
        if profile.discovery.campaign_id != campaign_id:
            raise ValueError("workflow must retain its original discovery allowance owner")
        if verify_runtime and store.json(manifest["runtime_ref"]) != controller.runtime_identity():
            raise ValueError("workflow requires its recorded source/runtime")
        inputs = _validate_inputs(profile)
        child_manifest = decode(
            (discovery.campaign_root(campaign_id) / "manifest.json").read_bytes()
        )
        if digest(canonical(child_manifest)) != manifest["discovery_manifest_ref"]:
            raise ValueError("workflow discovery authority changed")
        return journal, store, manifest, profile, inputs
    except Exception:
        journal.close()
        raise


def prepare(value):
    profile = WorkflowProfile.model_validate_json(canonical(value))
    _validate_inputs(profile)
    campaign_id = profile.discovery.campaign_id
    root = workflow_root(campaign_id)
    root.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(root):
        store = ArtifactStore(root / "artifacts")
        profile_ref = store.put_json(profile.model_dump(mode="json"))
        path = root / "manifest.json"
        if path.exists():
            journal, _, manifest, _, _ = _open(campaign_id)
            journal.close()
            if manifest["profile_ref"] != profile_ref:
                raise ValueError("workflow ID already binds another frozen profile")
        else:
            marker = root / "initializing.json"
            if marker.exists():
                intent = decode(marker.read_bytes())
                if intent["profile_ref"] != profile_ref:
                    raise ValueError("unfinished workflow binds another profile")
            else:
                if (discovery.campaign_root(campaign_id) / "manifest.json").exists():
                    raise ValueError(
                        "freeze workflow policy before starting its discovery campaign"
                    )
                intent = dict(
                    profile_ref=profile_ref,
                    runtime_ref=store.put_json(controller.runtime_identity()),
                    registered_at=market_contracts.trusted_now(),
                )
                atomic_write(marker, canonical(intent))
            if store.json(intent["runtime_ref"]) != controller.runtime_identity():
                raise ValueError("workflow initialization runtime changed")
            discovery.initialize(profile.inputs, profile.discovery)
            child = decode((discovery.campaign_root(campaign_id) / "manifest.json").read_bytes())
            manifest = intent | dict(
                schema_version=1,
                kind="MinimalResearchWorkflow",
                root=str(root.resolve()),
                campaign_id=campaign_id,
                discovery_manifest_ref=digest(canonical(child)),
                max_observations=256,
            )
            journal = Journal(root, digest(canonical(manifest)))
            journal.close()
            atomic_write(path, canonical(manifest))
            marker.unlink()
    return inspect(campaign_id)


def _fold_accounts(experiment_id, profile, inputs):
    from quant_rsi.execution.strategy_replay import replay_strategy

    journal, store, _, plan, _ = open_experiment(experiment_root(experiment_id))
    try:
        protocol = store.json(plan.protocol_ref)
        attempts = {row["ordinal"]: row for row in journal.attempts()}
        data, _ = inputs.target
        episodes = []
        for fold in protocol["folds"]:
            origins = fold["evaluation_origins"]
            sessions = data.contract.calendar.sessions
            for group in ("B0", "D1"):
                ordinal = next(
                    i
                    for i, p in enumerate(plan.proposals)
                    if p.purpose == "evaluation" and p.fold == fold["fold"] and p.recipe == group
                )
                attempt = attempts.get(ordinal)
                result = (
                    store.json(attempt["result_ref"])
                    if attempt and attempt["status"] == "completed"
                    else None
                )
                metadata = store.json(result["worker_metadata_ref"]) if result else None
                scores = (
                    {r["origin_session"]: r["score"] for r in store.json(result["predictions_ref"])}
                    if result
                    else {}
                )
                rows = []
                for index in origins:
                    session = sessions[index].session_id
                    if session not in scores:
                        continue  # The full frozen grid still produces explicit missing decisions.
                    decision_at = data.decision_at(index)
                    if metadata and timestamp(metadata["training_information_cutoff"]) > timestamp(
                        decision_at
                    ):
                        raise ValueError(
                            "historical strategy prediction uses future training labels"
                        )
                    rows.append(
                        dict(
                            origin_session=session,
                            score=scores.get(session),
                            reason=None
                            if session in scores
                            else "failed_fit_or_ineligible_prediction",
                            model_ref=result["model_ref"] if result else None,
                            result_ref=attempt["result_ref"] if result else None,
                            feature_as_of=utc_text(decision_at),
                            available_at=utc_text(decision_at),
                            confirmation_at=utc_text(decision_at),
                        )
                    )
                account = replay_strategy(
                    data,
                    rows,
                    profile.account,
                    origin_sessions=[sessions[i].session_id for i in origins],
                    end_session=sessions[origins[-1] + 5].session_id,
                    threshold=str(profile.strategy_threshold),
                )
                episodes.append(
                    dict(
                        fold=fold["fold"],
                        recipe_group=group,
                        account=account,
                        fit_status=attempt["status"] if attempt else "not_attempted",
                        fit_error=attempt.get("error") if attempt else None,
                    )
                )
        return episodes
    finally:
        journal.close()


def _publication_prediction(forecast):
    return dict(
        origin_session=forecast["origin_session"],
        score=forecast["predicted_log_return"],
        reason=None if forecast["forward"] else "publication_not_prospective",
        model_ref=forecast["model_ref"],
        result_ref=forecast["publication_result_ref"],
        forecast_ref=forecast["forecast_ref"],
        feature_as_of=forecast["frozen_input_asof"],
        available_at=forecast["issued_at"],
        confirmation_at=forecast["post_commit_confirmed_at"],
    )


def _initialize_paper(journal, store, profile, inputs, experiment_id):
    from quant_rsi.execution.strategy_replay import advance_paper_strategy

    receipt = forecast_settlement.register(experiment_id)
    forecast = receipt["publication"]["forecast"]
    prediction = _publication_prediction(forecast)
    kind = "paper_prepared_" + experiment_id
    saved = event_payload(journal, kind)
    if saved is None:
        state = advance_paper_strategy(
            inputs.target[0],
            prediction,
            profile.account,
            observed_at=market_contracts.trusted_now(),
            threshold=str(profile.strategy_threshold),
        )
        saved = {"state_ref": store.put_json(state)}
        commit_event(journal, kind, saved)
    kind = "paper_confirmed_" + experiment_id
    confirmed = event_payload(journal, kind)
    if confirmed is None:
        state = advance_paper_strategy(
            inputs.target[0],
            prediction,
            profile.account,
            observed_at=market_contracts.trusted_now(),
            previous=store.json(saved["state_ref"]),
            threshold=str(profile.strategy_threshold),
        )
        confirmed = {"state_ref": store.put_json(state)}
        commit_event(journal, kind, confirmed)
    return dict(experiment_id=experiment_id, forecast=forecast, state_ref=confirmed["state_ref"])


def _build_report(journal, store, manifest, profile, inputs, result):
    trials = []
    for item in result["history"]:
        experiment_id = item["experiment_id"]
        verified = inspect_experiment(experiment_id, replay=True)
        paper = (
            _initialize_paper(journal, store, profile, inputs, experiment_id)
            if verified["forecast"]
            else None
        )
        trials.append(
            dict(
                **item,
                accounts=_fold_accounts(experiment_id, profile, inputs),
                paper=paper,
            )
        )
    return dict(
        kind="MinimalResearchWorkflowReport",
        campaign_id=profile.discovery.campaign_id,
        profile_ref=manifest["profile_ref"],
        discovery=result,
        trials=trials,
        feedback_objective="development_forecast_mae_only",
        strategy_selection="frozen_threshold_not_optimized_by_account_returns",
        episode_policy="independent_fold_accounts_no_concatenated_equity_curve",
        evidence_status="development_only",
        independent_confirmation=False,
        predictive_or_trading_advantage_claim=False,
    )


def resume(campaign_id, *, max_fits=1):
    root = workflow_root(campaign_id)
    with controller.controller_lock(root):
        journal, store, manifest, profile, inputs = _open(campaign_id)
        try:
            if event_payload(journal, "workflow_completed") is None:
                result = discovery.resume(campaign_id, max_fits=max_fits)
                if result["status"] != "ready_or_running":
                    report = _build_report(journal, store, manifest, profile, inputs, result)
                    commit_event(
                        journal, "workflow_completed", {"report_ref": store.put_json(report)}
                    )
        finally:
            journal.close()
    return inspect(campaign_id)


def _observations(journal, store):
    rows = []
    for event in journal.events():
        if event["kind"].startswith("observation_completed_"):
            rows.append(store.json(event["payload"]["result_ref"]))
    return rows


def _validate_snapshot(inputs, snapshot, observed_at, previous=None):
    original = inputs.target[0].snapshot
    if (
        snapshot["registry_id"] != original["registry_id"]
        or snapshot["contract_ref"] != original["contract_ref"]
        or snapshot["receipts"][: len(original["receipts"])] != original["receipts"]
        or snapshot["receipt_watermark"] < original["receipt_watermark"]
    ):
        raise ValueError("workflow observation must extend its frozen target receipt prefix")
    if any(timestamp(r["first_seen_at"]) > timestamp(observed_at) for r in snapshot["receipts"]):
        raise ValueError("workflow receipt is later than its observation clock")
    if previous:
        prior, _ = load_snapshot(previous["snapshot_ref"])
        if snapshot["receipt_watermark"] < prior["receipt_watermark"] or timestamp(
            observed_at
        ) < timestamp(previous["observed_at"]):
            raise ValueError("workflow observation cannot roll back clocks or receipts")


def _paper_state(history, paper, store):
    for observation in reversed(history):
        for row in observation["records"]:
            if row["experiment_id"] == paper["experiment_id"]:
                return row["paper"]
    return store.json(paper["state_ref"])


def observe(campaign_id, snapshot_ref, *, request_id):
    from quant_rsi.execution.strategy_replay import advance_paper_strategy

    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("request ID must contain 1..128 characters")
    key = digest(request_id.encode())
    root = workflow_root(campaign_id)
    with controller.controller_lock(root):
        journal, store, manifest, profile, inputs = _open(campaign_id)
        try:
            terminal = event_payload(journal, "workflow_completed")
            if terminal is None:
                raise ValueError("complete the workflow before observing its publications")
            intent_kind, done_kind = "observation_intent_" + key, "observation_completed_" + key
            intent = event_payload(journal, intent_kind)
            if intent is not None and intent["snapshot_ref"] != snapshot_ref:
                raise ValueError("observation request ID already binds another snapshot")
            completed = event_payload(journal, done_kind)
            if completed:
                saved = store.json(completed["result_ref"])
                if saved["status"] == "rejected":
                    raise ValueError(saved["error"])
                return saved
            pending = [
                e
                for e in journal.events()
                if e["kind"].startswith("observation_intent_")
                and event_payload(
                    journal,
                    "observation_completed_" + e["kind"].removeprefix("observation_intent_"),
                )
                is None
            ]
            if pending and pending[0]["kind"] != intent_kind:
                raise ValueError("resume the pending observation request first")
            history = _observations(journal, store)
            snapshot, _ = load_snapshot(snapshot_ref)
            if intent is None:
                if len(history) >= manifest["max_observations"]:
                    raise ValueError("workflow observation allowance exhausted")
                intent = dict(
                    request_id=request_id,
                    snapshot_ref=snapshot_ref,
                    observed_at=market_contracts.trusted_now(),
                )
                _validate_snapshot(
                    inputs, snapshot, intent["observed_at"], history[-1] if history else None
                )
                commit_event(journal, intent_kind, intent)
            else:
                _validate_snapshot(
                    inputs, snapshot, intent["observed_at"], history[-1] if history else None
                )
            report = store.json(terminal["report_ref"])
            records = []
            error = None
            try:
                for trial in report["trials"]:
                    paper = trial["paper"]
                    if paper is None:
                        continue
                    updated = advance_paper_strategy(
                        inputs.target[0],
                        _publication_prediction(paper["forecast"]),
                        profile.account,
                        observed_at=intent["observed_at"],
                        previous=_paper_state(history, paper, store),
                        threshold=str(profile.strategy_threshold),
                        snapshot=snapshot,
                    )
                    outcome = forecast_settlement.settle(
                        paper["experiment_id"], snapshot_ref, request_id="workflow-" + key
                    )
                    records.append(
                        dict(experiment_id=paper["experiment_id"], outcome=outcome, paper=updated)
                    )
            except ValueError as exc:
                # A deterministic child rejection cannot be fixed by retrying the same intent.
                # Preserve any preceding child outcomes; I/O interruptions remain resumable.
                error = str(exc)
            result = intent | dict(
                sequence=len(history) + 1,
                status="rejected" if error is not None else "completed",
                error=error,
                records=records,
            )
            commit_event(journal, done_kind, {"result_ref": store.put_json(result)})
            if error is not None:
                raise ValueError(error)
            return result
        finally:
            journal.close()


def _replay_paper(journal, store, profile, inputs, report, observations):
    from quant_rsi.execution.strategy_replay import advance_paper_strategy

    states, forecasts, settlements = {}, {}, {}
    for trial in report["trials"]:
        paper = trial["paper"]
        if paper is None:
            continue
        key = paper["experiment_id"]
        receipt = forecast_settlement.inspect(key, replay=True)
        if receipt["publication"]["forecast"] != paper["forecast"]:
            raise ValueError("workflow publication differs from its registered forecast")
        forecasts[key] = _publication_prediction(paper["forecast"])
        state = None
        for prefix in ("paper_prepared_", "paper_confirmed_"):
            event = event_payload(journal, prefix + key)
            saved = store.json(event["state_ref"])
            state = advance_paper_strategy(
                inputs.target[0],
                forecasts[key],
                profile.account,
                observed_at=saved["observed_at"],
                previous=state,
                threshold=str(profile.strategy_threshold),
            )
            if state != saved:
                raise ValueError("paper initialization replay differs")
        if state != store.json(paper["state_ref"]):
            raise ValueError("paper report lost its confirmed initial state")
        states[key] = state
        settlements[key] = {r["request_id"]: r for r in receipt["observations"]}
    previous = None
    seen = set()
    for index, observation in enumerate(observations):
        request_id = observation["request_id"]
        key = digest(request_id.encode())
        intent = event_payload(journal, "observation_intent_" + key)
        if (
            request_id in seen
            or observation["sequence"] != index + 1
            or intent != {k: observation[k] for k in ("request_id", "snapshot_ref", "observed_at")}
        ):
            raise ValueError("paper observation ordering/intent differs")
        snapshot, _ = load_snapshot(observation["snapshot_ref"])
        _validate_snapshot(inputs, snapshot, observation["observed_at"], previous)
        roster = list(states)
        if observation["status"] == "rejected" and observation["error"]:
            roster = roster[: len(observation["records"])]
        elif observation["status"] != "completed" or observation["error"] is not None:
            raise ValueError("invalid paper observation status")
        if [r["experiment_id"] for r in observation["records"]] != roster:
            raise ValueError("paper observation omitted a registered publication")
        for row in observation["records"]:
            experiment_id = row["experiment_id"]
            expected = advance_paper_strategy(
                inputs.target[0],
                forecasts[experiment_id],
                profile.account,
                observed_at=observation["observed_at"],
                previous=states[experiment_id],
                threshold=str(profile.strategy_threshold),
                snapshot=snapshot,
            )
            if (
                expected != row["paper"]
                or settlements[experiment_id].get("workflow-" + key) != row["outcome"]
            ):
                raise ValueError("paper account/outcome replay differs")
            states[experiment_id] = expected
        seen.add(request_id)
        previous = observation


def inspect(campaign_id, *, replay=False):
    root = workflow_root(campaign_id)
    with controller.controller_lock(root):
        journal, store, manifest, profile, inputs = _open(campaign_id, verify_runtime=replay)
        try:
            terminal = event_payload(journal, "workflow_completed")
            result = discovery.inspect(campaign_id, replay=replay)
            report = store.json(terminal["report_ref"]) if terminal else None
            observations = _observations(journal, store)
            if report and report["discovery"] != result:
                raise ValueError("completed workflow discovery changed")
            if replay and report:
                for trial in report["trials"]:
                    if trial["accounts"] != _fold_accounts(trial["experiment_id"], profile, inputs):
                        raise ValueError("workflow account replay differs")
                _replay_paper(journal, store, profile, inputs, report, observations)
            return dict(
                campaign_id=campaign_id,
                status=result["status"],
                profile_ref=manifest["profile_ref"],
                discovery=result,
                report=report,
                observations=observations,
            )
        finally:
            journal.close()


def render_report(status):
    """A concise readable view; complete account/evidence records remain in JSON."""
    discovery_report = status["discovery"]
    lines = [
        f"# Research workflow: {status['campaign_id']}",
        "",
        f"Status: {status['status']}",
        f"Fits: {discovery_report['fits_attempted']} / {discovery_report['allocated_fits']}",
        "Evidence: development-only; no independent predictive or trading advantage claim.",
        "Feedback: forecast MAE drives the next candidate; account returns do not select it.",
        "Each fold starts with its own capital; returns are not a continuous portfolio.",
        "",
    ]
    report = status["report"]
    if report is None:
        return "\n".join(lines + ["Resume the registered workflow to complete its report."])
    lines += [
        "| Round | Fold | Recipe | Arm | Net return % | Max drawdown | Turnover | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    details = []
    for trial in report["trials"]:
        for episode in trial["accounts"]:
            for arm, metrics in episode["account"]["metrics"].items():
                net = metrics["net_return"]
                net = "unresolved" if net is None else f"{float(net) * 100:.4f}"
                lines.append(
                    f"| {trial['round']} | {episode['fold']} | {episode['recipe_group']} | "
                    f"{arm} | {net} | {metrics['max_drawdown']} | {metrics['turnover']} | "
                    f"{metrics['status']} |"
                )
                details.append(
                    f"Costs ({trial['round']}/{episode['fold']}/{episode['recipe_group']}/{arm}): "
                    f"{metrics['fees']}; unresolved sessions: {metrics['unresolved_sessions']}; "
                    f"risk events: {metrics['suspended_reasons']}"
                )
    lines += [""] + [text + "\n" for text in details]
    lines += ["", f"Recorded observation requests: {len(status['observations'])}"]
    if status["observations"]:
        latest = status["observations"][-1]
        lines.append(f"Latest observation: {latest['status']}; error={latest['error']}")
        for row in status["observations"][-1]["records"]:
            lines.append(
                f"{row['experiment_id']}: paper={row['paper']['status']}, "
                f"forecast outcome={row['outcome']['outcome']['status']}"
            )
            for arm, metrics in row["paper"]["metrics"].items():
                lines.append(
                    f"  {arm}: evidence={metrics['status']}, return={metrics['net_return']}, "
                    f"risk/correction reasons={metrics['suspended_reasons']}"
                )
    else:
        lines.append(
            "Future observations pending; historical runs do not create prospective evidence."
        )
    return "\n".join(lines)
