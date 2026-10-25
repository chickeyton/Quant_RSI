"""Bidirectional trajectory/C4 bindings; stored metadata never grants execution."""

from pathlib import Path

from quant_rsi.artifacts import ArtifactStore, canonical, decode, digest
from quant_rsi.package_journal import PackageJournal, targets_for
from quant_rsi.trajectory_state import TrajectoryLedger, require_owner, trajectory_root


def stored_binding(root):
    manifest_path = root / "manifest.json"
    marker_path = root / "package_initialization.json"
    if manifest_path.is_file():
        manifest = decode(manifest_path.read_bytes())
        binding = verify_delegation(manifest, ArtifactStore(root / "artifacts"))
        return binding
    if marker_path.is_file():
        saved = decode(marker_path.read_bytes())
        context = saved.get("trajectory_context")
        authorization = saved.get("trajectory_authorization")
        if (context is None) != (authorization is None):
            raise ValueError("incomplete delegated C4 initialization binding")
        if context is not None:
            if context["authorization_ref"] != digest(canonical(authorization)):
                raise ValueError("delegated C4 initialization authorization changed")
            if saved["campaign_id"] != authorization["child_campaign_id"]:
                raise ValueError("delegated C4 initialization child identity changed")
            return {"context": context, "authorization": authorization}
    return None


def verify_delegation(manifest, store):
    """Verify the frozen child-side closure without accessing a mutable parent authority."""
    context_ref = manifest.get("trajectory_context_ref")
    authorization_ref = manifest.get("trajectory_authorization_ref")
    if context_ref is None and authorization_ref is None:
        return None
    if context_ref is None or authorization_ref is None:
        raise ValueError("delegated C4 manifest lost its parent binding")
    context = store.json(context_ref)
    authorization = store.json(authorization_ref)
    if (
        context["kind"] != "TrajectoryC4Context"
        or context["authorization_ref"] != authorization_ref
        or context["authority_id"] != authorization["authority_id"]
        or authorization["kind"] != "TrajectoryC4Authorization"
        or authorization["child_campaign_id"] != manifest["campaign_id"]
        or authorization["base_package_ref"] != manifest["base_package_ref"]
        or authorization["research_plan_ref"] != manifest["plan_ref"]
        or authorization["source_files_ref"] != manifest["source_files_ref"]
        or authorization["capacity"]
        != {
            "worker_attempts": manifest["limits"]["research_attempts"],
            "registered_worker_seconds": manifest["limits"]["worker_seconds"],
        }
        or authorization["round_steps"] != 1
    ):
        raise ValueError("delegated C4 immutable binding mismatch")
    return {"context": context, "authorization": authorization}


def authorize_package(root, owner_fd, *, context=None, source=None, steps=1):
    """Authorize only the parent's current fixed operation, even if context is omitted."""
    root = root.resolve()
    persisted = stored_binding(root)
    if persisted is not None:
        if context is not None and context != persisted["context"]:
            raise ValueError("delegated C4 cannot change its trajectory context")
        context = persisted["context"]
    if context is None:
        if owner_fd is not None:
            raise ValueError("trajectory owner without a delegated C4 identity")
        return None
    if not isinstance(context, dict) or set(context) != {
        "kind",
        "authority_root",
        "authority_id",
        "authorization_ref",
    }:
        raise ValueError("invalid trajectory C4 execution context")
    parent = trajectory_root()
    if context["kind"] != "TrajectoryC4Context" or context["authority_root"] != str(parent):
        raise ValueError("delegated C4 belongs to another configured trajectory")
    require_owner(parent, owner_fd)
    with TrajectoryLedger(parent) as authority:
        state, program = authority.state, authority.program
        pending = state["pending"]
        if (
            state["status"] != "ACTIVE"
            or not isinstance(pending, dict)
            or pending.get("kind") != "package_round"
            or context["authority_id"] != authority.anchor["authority_id"]
            or pending["authorization_ref"] != context["authorization_ref"]
            or pending["steps"] != steps
            or steps != 1
        ):
            raise ValueError("delegated C4 requires its active parent operation")
        authorization = authority.store.json(context["authorization_ref"])
        stage_index = pending["stage"]
        if not 0 <= stage_index < len(state["stages"]):
            raise ValueError("delegated C4 has no parent stage reservation")
        stage = state["stages"][stage_index]
        if (
            authorization["kind"] != "TrajectoryC4Authorization"
            or authorization["authority_id"] != authority.anchor["authority_id"]
            or authorization["program_ref"] != authority.anchor["program_ref"]
            or authorization["stage"] != stage_index
            or stage["authorization_ref"] != context["authorization_ref"]
            or stage["capacity"] != authorization["capacity"]
            or stage["request_id"] != authorization["request_id"]
            or pending["child_campaign_id"] != authorization["child_campaign_id"]
            or root != (parent / authorization["child_directory"]).resolve()
            or authorization["base_package_ref"] != program["initial_head"]["package_ref"]
            or authorization["research_plan_ref"] != program["source_plan_ref"]
            or authorization["source_files_ref"] != program["source_files_ref"]
            or authorization["development_data_ref"] != program["source_data_ref"]
            or authorization["source_directory"] != str(parent / "source")
            or authorization["round_steps"] != 1
        ):
            raise ValueError("delegated C4 differs from its irrevocable parent authorization")
        if source is not None and Path(source).resolve() != (parent / "source").resolve():
            raise ValueError("delegated C4 source differs from its frozen parent input")
        if persisted is not None and authorization != persisted["authorization"]:
            raise ValueError("child-side and parent-side authorizations differ")
        if (root / "journal.sqlite").is_file():
            journal = PackageJournal(root)
            try:
                tables = journal.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                if not tables:
                    if (
                        not (root / "package_initialization.json").is_file()
                        or pending["round_index"] != 0
                        or pending["targets"] != {"parent": 1, "candidate": 1}
                    ):
                        raise ValueError(
                            "empty delegated C4 journal lacks its initialization intent"
                        )
                    return {"context": context, "authorization": authorization}
                rounds = journal.rounds()
                index = pending["round_index"]
                if len(rounds) not in {index, index + 1}:
                    raise ValueError("delegated C4 round differs from its parent target")
                if len(rounds) == index + 1:
                    row = rounds[-1]
                    if (
                        row["ordinal"] != index
                        or row["receipt_ref"] is not None
                        or row["steps"] != steps
                        or targets_for(row) != pending["targets"]
                    ):
                        raise ValueError("delegated C4 round already committed or target changed")
                if journal.state()["completion_ref"] is not None:
                    raise ValueError("completed delegated C4 cannot receive another execution")
            finally:
                journal.close()
        elif pending["round_index"] != 0 or pending["targets"] != {"parent": 1, "candidate": 1}:
            raise ValueError("delegated C4 initialization has a noninitial parent target")
        return {"context": context, "authorization": authorization}
