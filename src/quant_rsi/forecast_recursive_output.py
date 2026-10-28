"""Seal the actual final M0 forecast with its complete frozen recursive closure."""

from pathlib import Path

from quant_rsi.artifacts import canonical, digest
from quant_rsi.trajectory import summarize
from quant_rsi.trajectory_state import TrajectoryLedger, trajectory_root


def bind_final_forecast(source_root, manifest, checkpoint_ref, bundle, output_store):
    root = trajectory_root()
    with TrajectoryLedger(root, readonly=True) as authority:
        report = summarize(authority)
        state, program, store = authority.state, authority.program, authority.store
        if program.get("schema_version") != 3 or report["status"] != "SEALED":
            raise ValueError("recursive forecast sealing requires its completed trajectory")
        head = state["head"]
        receipt = store.json(head["research_receipt_ref"])
        selection = store.json(head["selection_ref"])
        identity = store.json(program["root_registration_ref"])["identity"]
        expected_source = (
            Path(identity["source_directory"])
            if state["generation"] == 0
            else root / "children" / f"stage_{state['generation'] - 1}" / "arms" / "candidate"
        )
        if (
            source_root.resolve() != expected_source.resolve()
            or receipt["manifest_ref"] != digest(canonical(manifest))
            or head["checkpoint_ref"] != checkpoint_ref
            or selection["model_ref"] != bundle["model_ref"]
            or selection["result_ref"] != bundle["kernel_result_ref"]
            or selection["role"] != "active"
        ):
            raise ValueError("recursive forecast must seal its actual final selected child")
        # Transfer the bounded, verified original closure, including all model/mask ancestors.
        refs = []
        for path in sorted((root / "artifacts").iterdir()):
            if not path.is_file():
                raise ValueError("unexpected directory in trajectory artifact store")
            ref = output_store.put(store.get(path.name))
            refs.append(ref)
        closure = dict(
            kind="SealedForecastRecursiveClosure",
            schema_version=3,
            authority_root=str(root),
            anchor=authority.anchor,
            control_ref=output_store.put(authority.connection.serialize()),
            state_ref=authority.state_ref,
            terminal_ref=state["terminal_ref"],
            head=head,
            artifact_refs=refs,
            report=report,
            qualification="development_selected_M0_model_no_publication_refit",
        )
        result = bundle | dict(
            schema_version=3,
            recursive_lineage_ref=output_store.put_json(closure),
        )
        verify_final_forecast(output_store, result)
        return result


def verify_final_forecast(store, bundle):
    closure = store.json(bundle["recursive_lineage_ref"])
    if (
        closure["kind"] != "SealedForecastRecursiveClosure"
        or closure["schema_version"] != 3
        or bundle["schema_version"] != 3
        or len(closure["artifact_refs"]) != len(set(closure["artifact_refs"]))
    ):
        raise ValueError("invalid frozen recursive forecast closure")
    for ref in closure["artifact_refs"]:
        store.get(ref)
    with TrajectoryLedger.frozen(store, closure) as authority:
        report = summarize(authority)
        head = authority.state["head"]
        receipt = store.json(head["research_receipt_ref"])
        selection = store.json(head["selection_ref"])
        if (
            authority.program.get("schema_version") != 3
            or report["status"] != "SEALED"
            or closure["report"] != report
            or closure["terminal_ref"] != authority.state["terminal_ref"]
            or closure["head"] != head
            or receipt["manifest_ref"] != bundle["source_run_ref"]
            or head["checkpoint_ref"] != bundle["source_checkpoint_ref"]
            or selection["role"] != "active"
            or selection["model_ref"] != bundle["model_ref"]
            or selection["result_ref"] != bundle["kernel_result_ref"]
        ):
            raise ValueError("forecast output differs from its frozen final trajectory head")
    return report
