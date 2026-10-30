"""Append-only outcomes for committed five-session publications; no fitting or collection."""

import math

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.core.journal import Journal
from quant_rsi.forecast.forecast_experiment import (
    event_payload,
    experiment_root,
    open_experiment,
)
from quant_rsi.forecast.forecast_kernel import forecast_errors, mature_forecast_label
from quant_rsi.forecast.forecast_publication import confirmed_report, publication_report
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import timestamp
from quant_rsi.market.market_data import inspect_data, load_snapshot

MAX_OBSERVATIONS = 256


def observation_root(experiment_id):
    source = experiment_root(experiment_id)  # Reuse the registered identifier rules.
    return source.parent.parent / "forecast_observations" / experiment_id


def _publication(experiment_id):
    root = experiment_root(experiment_id)
    with controller.controller_lock(root):
        journal, store, manifest, plan, inputs = open_experiment(root, verify_runtime=False)
        try:
            publication = event_payload(journal, "forecast_publication_committed")
            confirmation = event_payload(journal, "forecast_publication_confirmed")
            selection = event_payload(journal, "forecast_selection_committed")
            rows = journal.attempts()
            if not publication or not confirmation or not selection:
                raise ValueError("observation requires a committed and confirmed publication")
            if len(rows) != len(plan.proposals) or rows[-1]["status"] != "completed":
                raise ValueError("publication fit is not completed")
            saved = store.json(publication["forecast_ref"])
            expected = publication_report(
                inputs,
                plan,
                store.json(plan.protocol_ref),
                store,
                rows[-1]["result_ref"],
                selection["selection_ref"],
                saved["issued_at"],
            )
            if expected != saved:
                raise ValueError("publication differs from its committed model/input binding")
            forecast = confirmed_report(saved, confirmation)
            binding = dict(
                experiment_id=experiment_id,
                experiment_manifest_ref=digest(canonical(manifest)),
                experiment_runtime_ref=manifest["runtime_ref"],
                forecast=forecast,
                confirmation=confirmation,
                input_ref=inputs.ref,
            )
            return binding, inputs
        finally:
            journal.close()


def register(experiment_id):
    root = observation_root(experiment_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    with controller.controller_lock(root.parent):
        root.mkdir(exist_ok=True)
        with controller.controller_lock(root):
            binding, _ = _publication(experiment_id)
            path = root / "manifest.json"
            store = ArtifactStore(root / "artifacts")
            if path.exists():
                manifest = decode(path.read_bytes())
                if manifest["publication"] != binding:
                    raise ValueError("observation registration binds another publication")
            else:
                now = market_contracts.trusted_now()
                if timestamp(now) < timestamp(binding["confirmation"]["confirmed_at"]):
                    raise ValueError("registration clock precedes publication confirmation")
                manifest = dict(
                    kind="FiveSessionForecastObservation",
                    schema_version=1,
                    publication=binding,
                    registered_at=now,
                    runtime_ref=store.put_json(controller.runtime_identity()),
                    max_observations=MAX_OBSERVATIONS,
                )
                atomic_write(path, canonical(manifest))
            # Recover a crash after the immutable manifest write, without changing clocks.
            journal = Journal(root, digest(canonical(manifest)))
            journal.close()
    return inspect(experiment_id)


def _open(experiment_id, *, verify_runtime):
    root = observation_root(experiment_id)
    manifest = decode((root / "manifest.json").read_bytes())
    journal = Journal(root)
    try:
        identity = dict(journal.connection.execute("SELECT key,value FROM metadata"))
        if identity != {"manifest_digest": digest(canonical(manifest))}:
            raise ValueError("observation manifest/ledger identity mismatch")
        if (
            manifest["kind"] != "FiveSessionForecastObservation"
            or manifest["schema_version"] != 1
            or manifest["max_observations"] != MAX_OBSERVATIONS
        ):
            raise ValueError("unsupported observation registration")
        binding, inputs = _publication(experiment_id)
        if binding != manifest["publication"]:
            raise ValueError("observation publication binding changed")
        store = ArtifactStore(root / "artifacts")
        if verify_runtime and store.json(manifest["runtime_ref"]) != controller.runtime_identity():
            raise ValueError("observation requires its recorded runtime; preserve the workspace")
        return journal, store, manifest, inputs
    except Exception:
        journal.close()
        raise


def _outcome(manifest, inputs, snapshot_ref, observed_at):
    snapshot, contract = load_snapshot(snapshot_ref)
    original, objective = inputs.target
    if (
        snapshot["registry_id"] != original.snapshot["registry_id"]
        or snapshot["contract_ref"] != original.snapshot["contract_ref"]
        or snapshot["receipt_watermark"] < original.snapshot["receipt_watermark"]
        or snapshot["receipts"][: len(original.snapshot["receipts"])]
        != original.snapshot["receipts"]
    ):
        raise ValueError("outcome snapshot must extend the frozen target source receipt prefix")
    if any(
        timestamp(row["first_seen_at"]) > timestamp(observed_at) for row in snapshot["receipts"]
    ):
        raise ValueError("outcome receipt is later than the trusted observation clock")
    forecast = manifest["publication"]["forecast"]
    label = mature_forecast_label(
        snapshot,
        contract,
        forecast["origin_session"],
        forecast["instrument"],
        observed_at,
        objective,
        basis="observed_receipts",
    )
    metrics = None
    reason = label.get("reason")
    status = "immature" if label["status"] == "immature" else "insufficient_evidence"
    if not forecast["forward"]:
        status, reason = "not_prospective", "publication_retrospective_or_abstained"
    elif label["status"] == "mature":
        if float(label["anchor_price"]) != forecast["origin_price"]:
            # Never change the issued forecast's denominator after seeing a correction.
            status, reason = "insufficient_evidence", "published_origin_price_revised"
        else:
            status, reason = "settled", None
            actual_price = float(label["target_price"])
            actual_return = math.log(actual_price) - math.log(forecast["origin_price"])
            metrics = dict(
                forecast=forecast_errors([forecast["predicted_log_return"]], [actual_return]),
                unchanged_price=forecast_errors([0.0], [actual_return]),
                actual_price=actual_price,
                price_error=forecast["forecast_price"] - actual_price,
            )
    return dict(
        status=status,
        reason=reason,
        label=label,
        metrics=metrics,
        snapshot_ref=snapshot_ref,
        registry_id=snapshot["registry_id"],
        contract_ref=snapshot["contract_ref"],
        receipt_watermark=snapshot["receipt_watermark"],
        observed_at=observed_at,
        knowledge_basis="observed_receipts",
        forecast_ref=forecast["forecast_ref"],
        independent_confirmation=False,
        economic_or_statistical_advantage_claim=False,
    )


def _history(journal, store, manifest, inputs, *, replay):
    observations = []
    requests = set()
    last_time = manifest["registered_at"]
    for event in journal.events():
        if event["kind"] == "run_created":
            continue
        if event["kind"] != "forecast_outcome_observed" or event["ordinal"] is not None:
            raise ValueError("unexpected forecast observation event")
        row = store.json(event["payload"]["observation_ref"])
        previous = observations[-1]["observation_ref"] if observations else None
        if (
            row["request_id"] in requests
            or row["sequence"] != len(observations) + 1
            or row["previous_ref"] != previous
            or timestamp(row["outcome"]["observed_at"]) < timestamp(last_time)
            or row["outcome"]["forecast_ref"] != manifest["publication"]["forecast"]["forecast_ref"]
        ):
            raise ValueError("observation history binding/clock differs")
        if replay:
            inspect_data(row["outcome"]["snapshot_ref"], replay=True)
            expected = _outcome(
                manifest, inputs, row["outcome"]["snapshot_ref"], row["outcome"]["observed_at"]
            )
            if row["outcome"] != expected:
                raise ValueError("replayed forecast outcome differs")
        observations.append(row | event["payload"])
        requests.add(row["request_id"])
        last_time = row["outcome"]["observed_at"]
    if len(observations) > manifest["max_observations"] or journal.attempts():
        raise ValueError("observation budget or no-fit ledger invariant violated")
    return observations


def settle(experiment_id, snapshot_ref, *, request_id):
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("an observation request ID of 1..128 characters is required")
    root = observation_root(experiment_id)
    with controller.controller_lock(root):
        journal, store, manifest, inputs = _open(experiment_id, verify_runtime=True)
        try:
            history = _history(journal, store, manifest, inputs, replay=False)
            for row in history:
                if row["request_id"] == request_id:
                    if row["outcome"]["snapshot_ref"] != snapshot_ref:
                        raise ValueError("observation request ID binds another snapshot")
                    return row
            if len(history) >= manifest["max_observations"]:
                raise ValueError("forecast observation budget exhausted")
            now = market_contracts.trusted_now()
            prior_time = (
                history[-1]["outcome"]["observed_at"] if history else manifest["registered_at"]
            )
            if timestamp(now) < timestamp(prior_time):
                raise ValueError("trusted observation clock moved backwards")
            if (
                history
                and load_snapshot(snapshot_ref)[0]["receipt_watermark"]
                < history[-1]["outcome"]["receipt_watermark"]
            ):
                raise ValueError("observation cannot roll back the source receipt watermark")
            row = dict(
                request_id=request_id,
                sequence=len(history) + 1,
                previous_ref=history[-1]["observation_ref"] if history else None,
                outcome=_outcome(manifest, inputs, snapshot_ref, now),
            )
            payload = dict(observation_ref=store.put_json(row))
            with journal.connection:
                journal.event(None, "forecast_outcome_observed", payload)
            return row | payload
        finally:
            journal.close()


def inspect(experiment_id, *, replay=False):
    with controller.controller_lock(observation_root(experiment_id)):
        journal, store, manifest, inputs = _open(experiment_id, verify_runtime=replay)
        try:
            history = _history(journal, store, manifest, inputs, replay=replay)
            return dict(
                experiment_id=experiment_id,
                publication=manifest["publication"],
                registered_at=manifest["registered_at"],
                max_observations=MAX_OBSERVATIONS,
                observations=history,
                latest=history[-1] if history else None,
                fit_reservations=0,
                independent_confirmation=False,
            )
        finally:
            journal.close()
