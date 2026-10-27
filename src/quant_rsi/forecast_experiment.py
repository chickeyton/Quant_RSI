"""A fixed forecast batch on the existing worker and cumulative Journal ledger."""

import re

from quant_rsi import controller, market_contracts
from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.evidence import authority_root
from quant_rsi.forecast_comparison import select_recipe, verify_comparison_result
from quant_rsi.forecast_inputs import ForecastInputs
from quant_rsi.forecast_protocol import DEFAULT_FOLDS, ForecastComparisonPlan, freeze_protocol
from quant_rsi.forecast_publication import confirmed_report, publication_report
from quant_rsi.journal import Journal


def experiment_root(experiment_id):
    if not isinstance(experiment_id, str) or not re.fullmatch(
        r"[a-z][a-z0-9_-]{0,63}", experiment_id
    ):
        raise ValueError("invalid forecast experiment ID")
    return authority_root() / "forecast_experiments" / experiment_id


def prepare_experiment(value, experiment_id, *, fold_origins=DEFAULT_FOLDS):
    inputs = ForecastInputs(value)
    root = experiment_root(experiment_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    origins = [list(values) for values in fold_origins]
    with controller.controller_lock(root.parent):
        root.mkdir(exist_ok=True)
        with controller.controller_lock(root):
            if (root / "manifest.json").exists():
                journal, store, manifest, plan, _ = open_experiment(root)
                try:
                    protocol = store.json(plan.protocol_ref)
                    if (
                        plan.input_ref != inputs.ref
                        or [f["evaluation_origins"] for f in protocol["folds"]] != origins
                    ):
                        raise ValueError("experiment ID already binds different inputs/protocol")
                    return summarize(journal, store, manifest, plan)
                finally:
                    journal.close()
            store = ArtifactStore(root / "artifacts")
            marker = root / "bootstrap.json"
            if marker.exists():
                intent = decode(marker.read_bytes())
                if intent["input_ref"] != inputs.ref or intent["fold_origins"] != origins:
                    raise ValueError("unfinished initialization binds different inputs")
                if store.json(intent["runtime_ref"]) != controller.runtime_identity():
                    raise ValueError("initialization requires its recorded runtime")
            else:
                if (root / "journal.sqlite").exists():
                    raise ValueError("journal exists without initialization binding")
                intent = dict(
                    experiment_id=experiment_id,
                    input_ref=store.put_json(inputs.value),
                    fold_origins=origins,
                    publication_asof=market_contracts.trusted_now(),
                    runtime_ref=store.put_json(controller.runtime_identity()),
                )
                atomic_write(marker, canonical(intent))
            plan = freeze_protocol(
                inputs, store, publication_asof=intent["publication_asof"], fold_origins=origins
            )
            manifest = dict(
                kind="ForecastComparisonExperiment",
                schema_version=1,
                experiment_id=experiment_id,
                data_ref=inputs.ref,
                plan_ref=store.put_json(plan.model_dump(mode="json")),
                protocol_ref=plan.protocol_ref,
                runtime_ref=intent["runtime_ref"],
            )
            journal = Journal(root, digest(canonical(manifest)))
            try:
                atomic_write(root / "manifest.json", canonical(manifest))
                marker.unlink()
                return summarize(journal, store, manifest, plan)
            finally:
                journal.close()


def open_experiment(root, *, verify_runtime=True):
    manifest_raw = (root / "manifest.json").read_bytes()
    journal = Journal(root)
    try:
        rows = journal.connection.execute(
            "SELECT value FROM metadata WHERE key='manifest_digest'"
        ).fetchall()
        if len(rows) != 1 or rows[0][0] != digest(manifest_raw):
            raise ValueError("forecast experiment manifest/ledger identity mismatch")
        manifest = decode(manifest_raw)
        if manifest["kind"] != "ForecastComparisonExperiment" or manifest["schema_version"] != 1:
            raise ValueError("unsupported forecast experiment")
        if root.resolve() != experiment_root(manifest["experiment_id"]).resolve():
            raise ValueError("forecast experiment was moved outside its registered identity")
        store = ArtifactStore(root / "artifacts")
        plan = ForecastComparisonPlan.model_validate_json(store.get(manifest["plan_ref"]))
        inputs = ForecastInputs(store.json(manifest["data_ref"]))
        if plan.input_ref != inputs.ref or plan.protocol_ref != manifest["protocol_ref"]:
            raise ValueError("forecast experiment inputs/protocol differ")
        if verify_runtime and store.json(manifest["runtime_ref"]) != controller.runtime_identity():
            raise ValueError("source/runtime changed; use the frozen version for resume/replay")
        validate_ledger(journal, store, plan)
        return journal, store, manifest, plan, inputs
    except Exception:
        journal.close()
        raise


def validate_ledger(journal, store, plan):
    rows = journal.attempts()
    if len(rows) > plan.budget.max_attempts or [r["ordinal"] for r in rows] != list(
        range(len(rows))
    ):
        raise ValueError("forecast attempt ledger is not a contiguous bounded history")
    events = journal.events()
    reserved = [e for e in events if e["kind"] == "proposal_reserved"]
    starts = [e for e in events if e["kind"] == "worker_slot_reserved"]
    finishes = [e for e in events if e["kind"] == "attempt_finished"]
    if len(reserved) != len(rows) or len({e["ordinal"] for e in starts}) != len(starts):
        raise ValueError("forecast reservation history is missing or duplicated")
    if len({e["ordinal"] for e in finishes}) != len(finishes):
        raise ValueError("forecast completion history is duplicated")
    for row, reservation in zip(rows, reserved, strict=True):
        ordinal = row["ordinal"]
        proposal = plan.proposals[ordinal]
        if (
            row["name"] != proposal.name
            or store.json(row["proposal_ref"]) != proposal.model_dump(mode="json")
            or reservation["ordinal"] != ordinal
            or reservation["payload"] != {"proposal_ref": row["proposal_ref"]}
        ):
            raise ValueError("forecast proposal differs from its reservation")
        charges = [e for e in starts if e["ordinal"] == ordinal]
        expected_charge = 180 if charges else 0
        if row["worker_seconds"] != expected_charge or (
            charges and charges[0]["payload"] != {"seconds": 180}
        ):
            raise ValueError("forecast fit charge differs from its committed reservation")
        if row["status"] in {"completed", "failed", "interrupted"} and not charges:
            raise ValueError("executed forecast attempt is missing its fit charge")
        finished = [e for e in finishes if e["ordinal"] == ordinal]
        if row["status"] in {"reserved", "running"}:
            if finished or (row["status"] == "running") != bool(charges):
                raise ValueError("forecast pending state differs from its events")
        elif len(finished) != 1 or any(
            finished[0]["payload"][key] != row[column]
            for key, column in (
                ("status", "status"),
                ("result_ref", "result_ref"),
                ("error", "error"),
                ("elapsed_seconds", "elapsed_seconds"),
            )
        ):
            raise ValueError("forecast terminal state differs from its completion event")
    if any(e["ordinal"] not in range(len(rows)) for e in starts + finishes):
        raise ValueError("forecast ledger event has no bound attempt")
    if sum(r["worker_seconds"] for r in rows) > plan.budget.max_worker_seconds:
        raise ValueError("forecast worker budget exceeded")


def event_payload(journal, kind):
    rows = journal.connection.execute(
        "SELECT payload FROM events WHERE kind=? ORDER BY sequence", (kind,)
    ).fetchall()
    if len(rows) > 1:
        raise ValueError("duplicate committed forecast transition")
    return decode(rows[0][0]) if rows else None


def commit_event(journal, kind, payload):
    prior = event_payload(journal, kind)
    if prior is not None:
        if prior != payload:
            raise ValueError("conflicting forecast transition")
        return
    with journal.connection:
        journal.event(None, kind, payload)


def selection_for(journal, store, plan):
    reports = {}
    for row in journal.attempts():
        if row["ordinal"] < len(plan.proposals) - 1 and row["status"] == "completed":
            result = store.json(row["result_ref"])
            report_ref = result["report_ref"]
            reports[row["name"]] = (report_ref, store.json(report_ref))
    return select_recipe(plan, reports)


def bound_manifest(manifest, selection):
    return manifest | {"selection_ref": selection["selection_ref"]} if selection else manifest


def publish(journal, store, manifest, plan, inputs):
    rows = journal.attempts()
    if len(rows) != len(plan.proposals) or rows[-1]["status"] != "completed":
        return
    selection = event_payload(journal, "forecast_selection_committed")
    if selection is None:
        raise ValueError("publication lacks committed selection")
    publication = event_payload(journal, "forecast_publication_committed")
    if publication is None:
        report = publication_report(
            inputs,
            plan,
            store.json(plan.protocol_ref),
            store,
            rows[-1]["result_ref"],
            selection["selection_ref"],
            market_contracts.trusted_now(),
        )
        publication = dict(forecast_ref=store.put_json(report))
        commit_event(journal, "forecast_publication_committed", publication)
    confirmation = event_payload(journal, "forecast_publication_confirmed")
    if confirmation is None:
        confirmation = publication | {"confirmed_at": market_contracts.trusted_now()}
        confirmed_report(store.json(publication["forecast_ref"]), confirmation)
        commit_event(journal, "forecast_publication_confirmed", confirmation)


def resume_experiment(experiment_id, *, protocol_ref, max_fits=1):
    if type(max_fits) is not int or not 1 <= max_fits <= 10:
        raise ValueError("max_fits must be an integer from 1 to 10")
    root = experiment_root(experiment_id)
    with controller.controller_lock(root) as owner_fd:
        journal, store, manifest, plan, inputs = open_experiment(root)
        try:
            if protocol_ref != plan.protocol_ref:
                raise ValueError("run requires the exact frozen protocol reference")
            dispatched = 0
            while True:
                rows = journal.attempts()
                evaluations = len(plan.proposals) - 1
                selection = event_payload(journal, "forecast_selection_committed")
                if len(rows) >= evaluations and all(
                    row["status"] not in {"reserved", "running"} for row in rows[:evaluations]
                ):
                    expected = selection_for(journal, store, plan)
                    ref = store.put_json(expected)
                    commit_event(journal, "forecast_selection_committed", {"selection_ref": ref})
                    selection = {"selection_ref": ref}
                    if expected["selected_recipe"] is None:
                        break
                if rows and rows[-1]["status"] == "running":
                    controller.recover_attempt(
                        root,
                        rows[-1],
                        bound_manifest(
                            manifest, selection if rows[-1]["ordinal"] == evaluations else None
                        ),
                        plan,
                        inputs,
                        store,
                        journal,
                    )
                    continue
                if len(rows) == len(plan.proposals) and rows[-1]["status"] != "reserved":
                    publish(journal, store, manifest, plan, inputs)
                    break
                if dispatched >= max_fits:
                    break
                ordinal = len(rows) - 1 if rows and rows[-1]["status"] == "reserved" else len(rows)
                proposal = plan.proposals[ordinal]
                if ordinal == evaluations and selection is None:
                    raise ValueError("publication cannot be reserved before selection")
                value = proposal.model_dump(mode="json")
                if ordinal == len(rows):
                    journal.reserve(
                        ordinal, proposal.name, store.put_json(value), plan.budget.max_attempts
                    )
                controller.execute_attempt(
                    root,
                    ordinal,
                    value,
                    bound_manifest(manifest, selection if ordinal == evaluations else None),
                    plan,
                    inputs,
                    store,
                    journal,
                    owner_fd,
                )
                dispatched += 1
            return summarize(journal, store, manifest, plan)
        finally:
            journal.close()


def summarize(journal, store, manifest, plan):
    attempts = journal.attempts()
    protocol = store.json(plan.protocol_ref)
    selection = event_payload(journal, "forecast_selection_committed")
    publication = event_payload(journal, "forecast_publication_committed")
    confirmation = event_payload(journal, "forecast_publication_confirmed")
    return dict(
        experiment_id=manifest["experiment_id"],
        input_ref=plan.input_ref,
        protocol_ref=plan.protocol_ref,
        budget=plan.budget.model_dump(mode="json"),
        fit_reservations=sum(row["worker_seconds"] > 0 for row in attempts),
        reserved_worker_seconds=sum(row["worker_seconds"] for row in attempts),
        attempts=attempts,
        folds=protocol["folds"],
        publication_training_rows=protocol["publication"]["training_rows"],
        selection=store.json(selection["selection_ref"]) if selection else None,
        forecast=confirmed_report(store.json(publication["forecast_ref"]), confirmation)
        if publication and confirmation
        else None,
        qualifications=protocol["qualifications"],
    )


def inspect_experiment(experiment_id, *, replay=False):
    root = experiment_root(experiment_id)
    with controller.controller_lock(root):
        journal, store, manifest, plan, inputs = open_experiment(root, verify_runtime=replay)
        try:
            selection = event_payload(journal, "forecast_selection_committed")
            if replay:
                for row in journal.attempts():
                    if row["status"] != "completed":
                        continue
                    proposal = plan.proposals[row["ordinal"]]
                    effective = bound_manifest(
                        manifest, selection if proposal.purpose == "publication" else None
                    )
                    result = store.json(row["result_ref"])
                    expected_job = controller.job_for(
                        row["ordinal"], proposal.model_dump(mode="json"), effective, plan
                    )
                    if store.json(result["job_ref"]) != expected_job:
                        raise ValueError("replay job binding differs")
                    metadata = store.json(result["worker_metadata_ref"])
                    model_text = store.get(result["model_ref"]).decode()
                    if digest(model_text.encode()) != metadata["model_digest"]:
                        raise ValueError("replay model hash mismatch")
                    result_value = metadata | {"predictions": store.json(result["predictions_ref"])}
                    report = verify_comparison_result(
                        inputs, plan, proposal, store, effective, result_value, model_text
                    )
                    if report != store.json(result["report_ref"]):
                        raise ValueError("replayed report differs from the committed result")
                if selection and store.json(selection["selection_ref"]) != selection_for(
                    journal, store, plan
                ):
                    raise ValueError("replayed selection differs")
                publication = event_payload(journal, "forecast_publication_committed")
                if publication:
                    persisted = store.json(publication["forecast_ref"])
                    report = publication_report(
                        inputs,
                        plan,
                        store.json(plan.protocol_ref),
                        store,
                        journal.attempts()[-1]["result_ref"],
                        selection["selection_ref"],
                        persisted["issued_at"],
                    )
                    if report != persisted:
                        raise ValueError("replayed publication differs")
            return summarize(journal, store, manifest, plan)
        finally:
            journal.close()
