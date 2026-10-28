"""Immutable forecast campaign allocations around the existing child fit journals.

One workspace authority serializes managed work. It reserves each complete 10-fit,
1800-slot-second child ceiling up front and never maintains a second fit ledger.
The caller carries the returned owner descriptor through its existing child locks.
"""

import fcntl
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.core.journal import Journal
from quant_rsi.evaluation.evidence import authority_root
from quant_rsi.forecast.forecast_workflow_config import ForecastWorkflowConfig

ALLOCATION = {"max_attempts": 10, "max_worker_seconds": 1800}
EXPOSURE = {
    "prior_exposure": "previously_exposed_historical_exploration",
    "permitted_evidence": "public_historical_price_context",
    "protected_evidence_access": "none",
    "predictive_validity_established": False,
}
WORKFLOW_PHASES = [
    "collection_started",
    "collection_committed",
    "preparation_started",
    "inputs_committed",
    "experiment_committed",
]
_ACTIVE = ContextVar("forecast_campaign_owner", default=None)


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("invalid forecast campaign/run ID")
    return value


class _Budget(StrictModel):
    max_fit_reservations: int = Field(ge=1)
    max_worker_seconds: int = Field(ge=1)


class _Run(StrictModel):
    instrument_id: str
    run_id: str
    config: ForecastWorkflowConfig

    @model_validator(mode="after")
    def identity(self):
        _id(self.run_id)
        if self.instrument_id != "US/" + self.config.target:
            raise ValueError("instrument_id must be US/<exact configured target>")
        return self


class _Plan(StrictModel):
    kind: Literal["ForecastCampaign"]
    schema_version: Literal[1]
    campaign_id: str
    budget: _Budget
    runs: list[_Run] = Field(min_length=1)

    @model_validator(mode="after")
    def roster(self):
        _id(self.campaign_id)
        if len({row.run_id for row in self.runs}) != len(self.runs):
            raise ValueError("duplicate run ID in campaign roster")
        if (
            len(self.runs) * ALLOCATION["max_attempts"] > self.budget.max_fit_reservations
            or len(self.runs) * ALLOCATION["max_worker_seconds"] > self.budget.max_worker_seconds
        ):
            raise ValueError("campaign grant cannot reserve the complete child roster")
        return self


@dataclass(frozen=True)
class ManagedRun:
    binding: dict
    config: dict
    allocation: dict
    owner_fd: int


@dataclass
class _LegacyAuthority:
    root: Path
    owner_fd: int
    pid: int
    lock_stat: os.stat_result


@dataclass
class _Authority:
    root: Path
    identity: dict
    authority_ref: str
    journal: Journal
    store: ArtifactStore
    owner_fd: int
    pid: int
    lock_stat: os.stat_result
    journal_stat: os.stat_result


def _paths():
    base = authority_root()
    return base, base / "forecast_campaigns", base / "forecast_campaign_identity.json"


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or non-regular campaign evidence: {path.name}")
    return path.read_bytes()


def _put_file(path, value):
    body = canonical(value)
    if path.exists() or path.is_symlink():
        if _read(path) != body:
            raise ValueError("conflicting immutable campaign identity")
    else:
        atomic_write(path, body)


@contextmanager
def _lock(base):
    base.mkdir(exist_ok=True)
    path = base / ".forecast_campaign.lock"
    if path.is_symlink():
        raise ValueError("campaign lock must not be a symbolic link")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    witness = None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another campaign controller or worker owns the authority") from error
        # The private duplicate retains the original OFD even if a caller closes/recycles fd.
        witness = os.dup(fd)
        yield fd
    finally:
        if witness is None:
            os.close(fd)
        else:
            try:
                if os.path.samestat(os.fstat(fd), os.fstat(witness)):
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.close(fd)
            except OSError:
                pass  # Never close an independently recycled descriptor.
            os.close(witness)


def _require_owner(authority):
    base, root, _ = _paths()
    path = base / ".forecast_campaign.lock"
    if root != authority.root or authority.pid != os.getpid():
        raise ValueError("campaign scope workspace or process changed")
    if isinstance(authority, _LegacyAuthority):
        anchor = base / "forecast_campaign_identity.json"
        if root.exists() or root.is_symlink() or anchor.exists() or anchor.is_symlink():
            raise ValueError("legacy scope changed to a managed authority")
    else:
        anchor = decode(_read(base / "forecast_campaign_identity.json"))
        if (
            anchor
            != {
                "kind": "ForecastCampaignAnchor",
                "schema_version": 1,
                "phase": "ready",
                "identity": authority.identity,
            }
            or digest(_read(root / "identity.json")) != authority.authority_ref
            or not (root / "journal.sqlite").is_file()
            or (root / "journal.sqlite").is_symlink()
            or not os.path.samestat((root / "journal.sqlite").stat(), authority.journal_stat)
        ):
            raise ValueError("active campaign parent authority changed or disappeared")
    try:
        current = os.fstat(authority.owner_fd)
        if not os.path.samestat(current, authority.lock_stat) or not os.path.samestat(
            current, path.stat()
        ):
            raise ValueError("campaign owner descriptor identity changed")
        with path.open("r+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("campaign owner lock is not held")
        fcntl.flock(authority.owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, TypeError) as error:
        raise ValueError("campaign requires its actual owner descriptor") from error


def _journal_identity(journal, manifest_ref):
    metadata = dict(journal.connection.execute("SELECT key,value FROM metadata"))
    events = journal.events()
    if (
        metadata != {"manifest_digest": manifest_ref}
        or not events
        or events[0]["kind"] != "run_created"
        or events[0]["ordinal"] is not None
        or events[0]["payload"] != {"manifest_digest": manifest_ref}
        or [e["sequence"] for e in events] != list(range(1, len(events) + 1))
        or sum(e["kind"] == "run_created" for e in events) != 1
    ):
        raise ValueError("campaign/child journal identity or event history changed")
    return events


def _bootstrap(base, root, anchor):
    if anchor.exists():
        value = decode(_read(anchor))
        if value.get("phase") == "ready":
            return
        if value.get("phase") != "initializing":
            raise ValueError("invalid campaign anchor phase")
        identity = value["identity"]
    else:
        if root.exists():
            raise ValueError("campaign authority exists without its outside anchor")
        identity = {
            "kind": "ForecastCampaignAuthority",
            "schema_version": 1,
            "authority_id": str(uuid4()),
            "workspace": str(base.parent),
        }
        value = {
            "kind": "ForecastCampaignAnchor",
            "schema_version": 1,
            "phase": "initializing",
            "identity": identity,
        }
    if identity.get("workspace") != str(base.parent):
        raise ValueError("campaign authority cannot be copied or relocated")
    if any(
        (base / name).exists() for name in ("forecast_experiments", "enhanced_forecast_workflows")
    ):
        raise ValueError("managed bootstrap requires no preexisting forecast child directories")
    if not anchor.exists():
        atomic_write(anchor, canonical(value))
    root.mkdir(exist_ok=True)
    _put_file(root / "identity.json", identity)
    journal = Journal(root, digest(canonical(identity)))
    try:
        events = _journal_identity(journal, digest(canonical(identity)))
        if len(events) != 1 or journal.attempts():
            raise ValueError("unfinished bootstrap has committed campaign activity")
    finally:
        journal.close()
    atomic_write(anchor, canonical(value | {"phase": "ready"}))


def _open_authority(root, anchor, fd):
    value = decode(_read(anchor))
    identity_raw = _read(root / "identity.json")
    identity = decode(identity_raw)
    expected = {
        "kind": "ForecastCampaignAnchor",
        "schema_version": 1,
        "phase": "ready",
        "identity": identity,
    }
    if (
        value != expected
        or set(identity) != {"kind", "schema_version", "authority_id", "workspace"}
        or identity["kind"] != "ForecastCampaignAuthority"
        or identity["schema_version"] != 1
        or not isinstance(identity["authority_id"], str)
        or not identity["authority_id"]
        or identity["workspace"] != str(root.parent.parent)
        or root.is_symlink()
    ):
        raise ValueError("campaign authority/anchor/workspace identity mismatch")
    ref = digest(identity_raw)
    journal = Journal(root)
    try:
        _journal_identity(journal, ref)
        if journal.attempts():
            raise ValueError("campaign authority must not contain a second fit ledger")
    except Exception:
        journal.close()
        raise
    return _Authority(
        root,
        identity,
        ref,
        journal,
        ArtifactStore(root / "artifacts"),
        fd,
        os.getpid(),
        os.fstat(fd),
        (root / "journal.sqlite").stat(),
    )


@contextmanager
def _authority(*, initialize=False):
    active = _ACTIVE.get()
    if active is not None:
        _require_owner(active)
        if isinstance(active, _LegacyAuthority):
            if initialize:
                raise ValueError("cannot initialize campaign inside an active legacy scope")
            yield None
        else:
            yield active
        return
    base, root, anchor = _paths()
    with _lock(base) as fd:
        if initialize:
            _bootstrap(base, root, anchor)
        if not root.exists() and not anchor.exists():
            legacy = _LegacyAuthority(root, fd, os.getpid(), os.fstat(fd))
            token = _ACTIVE.set(legacy)
            try:
                yield None
            finally:
                _ACTIVE.reset(token)
            return
        if not root.is_dir() or not anchor.is_file():
            raise ValueError("initialized campaign authority or outside anchor is missing")
        authority = _open_authority(root, anchor, fd)
        token = _ACTIVE.set(authority)
        try:
            yield authority
        finally:
            _ACTIVE.reset(token)
            authority.journal.close()


def _registration(authority, campaign_ref, run, exposure_ref):
    base = authority.root.parent
    return {
        "authority_ref": authority.authority_ref,
        "campaign_ref": campaign_ref,
        "exposure_ref": exposure_ref,
        "campaign_id": authority.store.json(campaign_ref)["plan"]["campaign_id"],
        "instrument_id": run["instrument_id"],
        "run_id": run["run_id"],
        "workspace": authority.identity["workspace"],
        "config_ref": digest(canonical(run["config"])),
        "config": run["config"],
        "allocation": ALLOCATION,
        "workflow_path": str(base / "enhanced_forecast_workflows" / run["run_id"]),
        "experiment_path": str(base / "forecast_experiments" / run["run_id"]),
    }


def _registry(authority):
    _require_owner(authority)
    events = _journal_identity(authority.journal, authority.authority_ref)
    campaigns, registrations, acknowledgements = {}, {}, {}
    recursive_sources, recursive_directories = set(), set()
    for event in events[1:]:
        if event["ordinal"] is not None:
            raise ValueError("campaign authority event has a child fit ordinal")
        payload = event["payload"]
        if event["kind"] == "campaign_registered":
            if set(payload) != {"campaign_id", "campaign_ref", "allocation_refs"}:
                raise ValueError("invalid campaign registration event")
            grant = authority.store.json(payload["campaign_ref"])
            plan = _Plan.model_validate_json(canonical(grant["plan"])).model_dump(mode="json")
            if (
                grant
                != {
                    "kind": "ForecastCampaignGrant",
                    "schema_version": 1,
                    "authority_ref": authority.authority_ref,
                    "plan": plan,
                    "exposure_ref": digest(canonical(EXPOSURE)),
                }
                or authority.store.json(grant["exposure_ref"]) != EXPOSURE
                or payload["campaign_id"] != plan["campaign_id"]
                or plan["campaign_id"] in campaigns
                or len(payload["allocation_refs"]) != len(plan["runs"])
            ):
                raise ValueError("campaign grant/exposure/roster mismatch")
            campaigns[plan["campaign_id"]] = (payload["campaign_ref"], plan)
            for ref, run in zip(payload["allocation_refs"], plan["runs"], strict=True):
                record = authority.store.json(ref)
                expected = _registration(
                    authority, payload["campaign_ref"], run, grant["exposure_ref"]
                )
                if (
                    record != expected
                    or run["run_id"] in registrations
                    or run["run_id"] in recursive_sources
                ):
                    raise ValueError("duplicate or changed workspace-global run registration")
                registrations[run["run_id"]] = (ref, record)

        elif event["kind"] == "recursive_campaign_registered":
            from quant_rsi.forecast.recursive.forecast_recursive_campaign import (
                validate_registration,
            )

            plan, grant = validate_registration(authority, payload)
            if (
                plan["campaign_id"] in campaigns
                or plan["source_campaign_id"] in registrations
                or plan["source_campaign_id"] in recursive_sources
                or plan["source_directory"] in recursive_directories
            ):
                raise ValueError("duplicate recursive campaign or source reservation")
            campaigns[plan["campaign_id"]] = (payload["grant_ref"], plan)
            recursive_sources.add(grant["source_campaign_id"])
            recursive_directories.add(grant["source_directory"])
        elif event["kind"] in {"workflow_bound", "experiment_bound"}:
            run_id = payload["run_id"]
            key = (run_id, event["kind"])
            if run_id not in registrations or key in acknowledgements:
                raise ValueError("unknown or duplicate campaign child acknowledgement")
            if (
                event["kind"] == "experiment_bound"
                and (run_id, "workflow_bound") not in acknowledgements
            ):
                raise ValueError("experiment was acknowledged before its workflow")
            acknowledgements[key] = payload
        else:
            raise ValueError("unknown campaign authority event")
    return campaigns, registrations, acknowledgements


def _managed(authority, ref, record):
    binding = {
        name: record[name]
        for name in (
            "authority_ref",
            "campaign_ref",
            "exposure_ref",
            "campaign_id",
            "instrument_id",
            "run_id",
            "workspace",
        )
    }
    binding["allocation_ref"] = ref
    return ManagedRun(
        decode(canonical(binding)),
        decode(canonical(record["config"])),
        dict(ALLOCATION),
        authority.owner_fd,
    )


def _scope_record(scope):
    authority = _ACTIVE.get()
    if not isinstance(scope, ManagedRun) or authority is None:
        raise ValueError("managed operation requires its active parent campaign scope")
    _require_owner(authority)
    _, records, acks = _registry(authority)
    row = records.get(scope.binding.get("run_id"))
    if row is None or scope != _managed(authority, *row):
        raise ValueError("managed run binding/configuration/owner differs")
    return authority, row[1], acks


def _binding(scope, manifest):
    if manifest.get("managed_run") != scope.binding:
        raise ValueError("child managed_run binding is missing or differs")


def _workflow_manifest(scope, manifest):
    _binding(scope, manifest)
    if (
        manifest.get("kind") != "EnhancedForecastWorkflow"
        or manifest.get("schema_version") != 1
        or manifest.get("workflow_id") != scope.binding["run_id"]
        or manifest.get("workspace") != scope.binding["workspace"]
        or manifest.get("config") != scope.config
        or manifest.get("config_ref") != digest(canonical(scope.config))
    ):
        raise ValueError("registered workflow manifest/configuration differs")


def _workflow_state(scope, record, acks):
    root = Path(record["workflow_path"])
    ack = acks.get((record["run_id"], "workflow_bound"))
    path = root / "manifest.json"
    if not path.exists():
        if ack is not None:
            raise ValueError("acknowledged workflow authority is missing")
        marker = root / "bootstrap.json"
        if marker.exists():
            _workflow_manifest(scope, decode(_read(marker)))
            if (root / "journal.sqlite").exists():
                journal = Journal(root)
                try:
                    # A matching bootstrap may precede the schema transaction.
                    # Only a completely empty schema is eligible for initializer recovery.
                    if journal.connection.execute(
                        "SELECT 1 FROM sqlite_master LIMIT 1"
                    ).fetchone() and (
                        len(_journal_identity(journal, digest(_read(marker)))) != 1
                        or journal.attempts()
                    ):
                        raise ValueError("unacknowledged workflow has activity")
                finally:
                    journal.close()
            return {"state": "initializing", "events": {}}
        if root.exists() and any(
            p.name != ".controller.lock" and not p.name.startswith(".pending-")
            for p in root.iterdir()
        ):
            raise ValueError("workflow contents lack initialization identity")
        return {"state": "registered", "events": {}}
    raw = _read(path)
    manifest = decode(raw)
    _workflow_manifest(scope, manifest)
    journal = Journal(root)
    try:
        events = _journal_identity(journal, digest(raw))
        kinds = [e["kind"] for e in events[1:]]
        if kinds != WORKFLOW_PHASES[: len(kinds)] or journal.attempts():
            raise ValueError("workflow phase journal is not the original ordered history")
        if any(e["ordinal"] is not None for e in events):
            raise ValueError("workflow journal contains fit ordinals")
        if ack is None:
            if len(events) != 1 or any((root / p).exists() for p in ("collection", "preparation")):
                raise ValueError("unacknowledged workflow has collection/preparation activity")
        elif ack != {"run_id": record["run_id"], "manifest_ref": digest(raw)}:
            raise ValueError("workflow acknowledgement differs from its original manifest")
        else:
            authority = _ACTIVE.get()
            if authority.store.get(ack["manifest_ref"]) != raw:
                raise ValueError("workflow acknowledgement artifact differs")
        return {
            "state": "acknowledged" if ack else "initialized_unacknowledged",
            "manifest": manifest,
            "manifest_ref": digest(raw),
            "events": {e["kind"]: e["payload"] for e in events[1:]},
        }
    finally:
        journal.close()


def _input_contract(scope, value, plan, protocol):
    from quant_rsi.forecast.forecast_contracts import ForecastObjective
    from quant_rsi.forecast.forecast_protocol import DEFAULT_FOLDS

    if value.get("kind") != "ForecastInputs" or value.get("schema_version") != 1:
        raise ValueError("managed experiment requires ForecastInputs")
    if set(value["contexts"]) != {"industry", "market"}:
        raise ValueError("managed experiment requires both registered context roles")
    for role, market in {"target": value["target"], **value["contexts"]}.items():
        objective = ForecastObjective.model_validate_json(canonical(market["objective"]))
        data = market["data"]
        contract = data["contract"]
        if (
            objective.prior_exposure != EXPOSURE["prior_exposure"]
            or objective.currency != scope.config["currency"]
            or objective.price_unit != scope.config["price_unit"]
            or contract["instruments"] != [scope.config[role]]
            or contract["price_basis"] != "raw"
            or contract["currency"] != scope.config["currency"]
            or contract["knowledge_basis"] != "assumed_historical_availability"
            or data["specification"]["sessions"] != scope.config["history_sessions"]
        ):
            raise ValueError("managed input roles/unit/exposure/history differ from registration")
    if value["target"]["data"]["specification"] != plan.data.model_dump(mode="json"):
        raise ValueError("managed plan data differs from its frozen target")
    if (
        protocol.get("kind") != "FrozenForecastComparison"
        or protocol.get("schema_version") != 1
        or protocol.get("input_ref") != plan.input_ref
        or protocol.get("target_instrument") != scope.config["target"]
        or [f["evaluation_origins"] for f in protocol["folds"]] != [list(f) for f in DEFAULT_FOLDS]
        or protocol.get("model_inputs") != ["target_prices", "industry_prices", "market_prices"]
        or protocol.get("sidecar_inputs_used_in_training") != []
        or protocol.get("selection", {}).get("independent_confirmation") is not False
        or "development_comparison_not_independent_statistical_confirmation"
        not in protocol.get("qualifications", [])
    ):
        raise ValueError("managed experiment changed the fixed historical-exploration protocol")


def _experiment_state(scope, record, acks, workflow, *, supplied=None, for_dispatch=False):
    from quant_rsi.forecast.forecast_experiment import validate_ledger
    from quant_rsi.forecast.forecast_protocol import DEFAULT_FOLDS, ForecastComparisonPlan

    root = Path(record["experiment_path"])
    ack = acks.get((record["run_id"], "experiment_bound"))
    path = root / "manifest.json"
    zero = {"attempts": 0, "charged_fits": 0, "worker_seconds": 0}
    if not path.exists():
        if ack is not None or supplied is not None or for_dispatch:
            raise ValueError("acknowledged/required experiment authority is missing")
        marker = root / "bootstrap.json"
        if marker.exists():
            intent = decode(_read(marker))
            frozen = workflow["events"].get("inputs_committed")
            if (
                workflow["state"] != "acknowledged"
                or frozen is None
                or intent.get("experiment_id") != record["run_id"]
                or intent.get("input_ref") != frozen["input_ref"]
                or intent.get("fold_origins") != [list(f) for f in DEFAULT_FOLDS]
                or intent.get("managed_run") != scope.binding
            ):
                raise ValueError("pending experiment is not its registered workflow initialization")
            ArtifactStore(root / "artifacts").get(intent["runtime_ref"])
            if (root / "journal.sqlite").exists():
                journal = Journal(root)
                try:
                    if journal.connection.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone():
                        metadata = dict(
                            journal.connection.execute("SELECT key,value FROM metadata")
                        )
                        events = _journal_identity(journal, metadata["manifest_digest"])
                        if journal.attempts() or len(events) != 1:
                            raise ValueError(
                                "unacknowledged experiment has reservations or charges"
                            )
                finally:
                    journal.close()
            return {"state": "initializing", "costs": zero}
        if root.exists() and any(
            p.name not in {".controller.lock", "artifacts"} and not p.name.startswith(".pending-")
            for p in root.iterdir()
        ):
            raise ValueError("experiment contents lack initialization identity")
        return {"state": "registered", "costs": zero}
    raw = _read(path)
    manifest = decode(raw)
    _binding(scope, manifest)
    if (
        manifest.get("kind") != "ForecastComparisonExperiment"
        or manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != record["run_id"]
        or workflow["state"] != "acknowledged"
    ):
        raise ValueError("experiment lacks its exact acknowledged workflow")
    store = ArtifactStore(root / "artifacts")
    plan = ForecastComparisonPlan.model_validate_json(store.get(manifest["plan_ref"]))
    if supplied is not None and (
        manifest != supplied[0]
        or plan.model_dump(mode="json") != supplied[1].model_dump(mode="json")
    ):
        raise ValueError("supplied experiment manifest/plan differs from persisted original")
    if (
        plan.input_ref != manifest["data_ref"]
        or plan.protocol_ref != manifest["protocol_ref"]
        or plan.budget.model_dump(mode="json") != {**ALLOCATION, "worker_timeout_seconds": 180}
    ):
        raise ValueError("managed experiment input/protocol/ceiling differs from allocation")
    frozen = workflow["events"].get("inputs_committed")
    if frozen is None or frozen["input_ref"] != plan.input_ref:
        raise ValueError("experiment inputs were not frozen by its acknowledged workflow")
    value = store.json(plan.input_ref)
    protocol = store.json(plan.protocol_ref)
    _input_contract(scope, value, plan, protocol)
    runtime = store.json(manifest["runtime_ref"])
    if runtime != workflow["manifest"]["runtime"]:
        raise ValueError("workflow and experiment runtime identities differ")
    if for_dispatch:
        from quant_rsi.core.controller import runtime_identity

        if runtime != runtime_identity():
            raise ValueError("managed dispatch requires its original runtime")
    expected = {
        "run_id": record["run_id"],
        "manifest_ref": digest(raw),
        "input_ref": plan.input_ref,
        "protocol_ref": plan.protocol_ref,
        "runtime_ref": manifest["runtime_ref"],
        "plan_ref": manifest["plan_ref"],
        "workflow_manifest_ref": workflow["manifest_ref"],
        "workflow_inputs_ref": digest(canonical(frozen)),
        "allocation_ref": scope.binding["allocation_ref"],
        "exposure_ref": scope.binding["exposure_ref"],
    }
    journal = Journal(root)
    try:
        events = _journal_identity(journal, digest(raw))
        validate_ledger(journal, store, plan)
        rows = journal.attempts()
        if ack is None:
            if rows or len(events) != 1:
                raise ValueError("unacknowledged experiment has reservations or charges")
            if for_dispatch:
                raise ValueError("managed dispatch requires parent experiment acknowledgement")
        elif ack != expected:
            raise ValueError("experiment acknowledgement differs from original bindings")
        else:
            authority = _ACTIVE.get()
            if (
                authority.store.get(ack["manifest_ref"]) != raw
                or authority.store.json(ack["workflow_inputs_ref"]) != frozen
            ):
                raise ValueError("experiment parent acknowledgement evidence differs")
        costs = {
            "attempts": len(rows),
            "charged_fits": sum(r["worker_seconds"] > 0 for r in rows),
            "worker_seconds": sum(r["worker_seconds"] for r in rows),
        }
        return {
            "state": "acknowledged" if ack else "initialized_unacknowledged",
            "costs": costs,
            "expected_ack": expected,
            "manifest": manifest,
            "frozen_inputs": frozen,
        }
    finally:
        journal.close()


def _children(authority, registrations, acks):
    result = {}
    for run_id, (ref, record) in registrations.items():
        scope = _managed(authority, ref, record)
        workflow = _workflow_state(scope, record, acks)
        experiment = _experiment_state(scope, record, acks, workflow)
        result[run_id] = {"scope": scope, "workflow": workflow, "experiment": experiment}
    return result


def _legacy(run_id, manifest):
    if manifest is not None and "managed_run" in manifest:
        raise ValueError("managed child cannot fall back to a legacy workspace")
    base = authority_root()
    for parent in ("enhanced_forecast_workflows", "forecast_experiments"):
        for name in ("manifest.json", "bootstrap.json"):
            path = base / parent / run_id / name
            if path.exists() and "managed_run" in decode(_read(path)):
                raise ValueError("managed child is missing its original parent authority")


@contextmanager
def run_scope(run_id, *, manifest=None, required=False):
    _id(run_id)
    with _authority() as authority:
        if authority is None:
            _legacy(run_id, manifest)
            if required:
                raise ValueError("this operation requires a registered managed campaign")
            yield None
            return
        _, registrations, acks = _registry(authority)
        if run_id not in registrations:
            raise ValueError("run ID is not registered in the managed workspace")
        children = _children(authority, registrations, acks)
        scope = children[run_id]["scope"]
        if manifest is not None:
            _binding(scope, manifest)
        yield scope


def bind_workflow(scope, manifest):
    authority, record, acks = _scope_record(scope)
    state = _workflow_state(scope, record, acks)
    _workflow_manifest(scope, manifest)
    if state.get("manifest") != manifest:
        raise ValueError("workflow acknowledgement requires the persisted original manifest")
    expected = {"run_id": record["run_id"], "manifest_ref": digest(canonical(manifest))}
    if (record["run_id"], "workflow_bound") in acks:
        return
    authority.store.put_json(manifest)
    with authority.journal.connection:
        authority.journal.event(None, "workflow_bound", expected)


def _legacy_experiment(scope, manifest):
    if scope is not None:
        return False
    with run_scope(manifest["experiment_id"], manifest=manifest) as resolved:
        if resolved is not None:
            raise ValueError("managed experiment requires its active scope argument")
    return True


def verify_experiment(scope, manifest, plan, *, for_dispatch=False):
    if _legacy_experiment(scope, manifest):
        return
    _, record, acks = _scope_record(scope)
    workflow = _workflow_state(scope, record, acks)
    _experiment_state(
        scope, record, acks, workflow, supplied=(manifest, plan), for_dispatch=for_dispatch
    )


def bind_experiment(scope, manifest, plan):
    if _legacy_experiment(scope, manifest):
        return
    authority, record, acks = _scope_record(scope)
    workflow = _workflow_state(scope, record, acks)
    state = _experiment_state(scope, record, acks, workflow, supplied=(manifest, plan))
    if (record["run_id"], "experiment_bound") in acks:
        return
    authority.store.put_json(manifest)
    authority.store.put_json(state["frozen_inputs"])
    with authority.journal.connection:
        authority.journal.event(None, "experiment_bound", state["expected_ack"])


def _report(authority, campaign_id, *, replay):
    campaigns, registrations, acks = _registry(authority)
    if campaign_id not in campaigns:
        raise ValueError("unknown forecast campaign")
    children = _children(authority, registrations, acks)
    campaign_ref, plan = campaigns[campaign_id]
    if plan["kind"] != "ForecastCampaign":
        raise ValueError("recursive grant is not a forecast-comparison campaign")
    runs = []
    for row in plan["runs"]:
        child = children[row["run_id"]]
        scope = child["scope"]
        result = {
            "instrument_id": row["instrument_id"],
            "run_id": row["run_id"],
            "binding": scope.binding,
            "config": scope.config,
            "allocation": scope.allocation,
            "workflow_state": child["workflow"]["state"],
            "experiment_state": child["experiment"]["state"],
            "actual": {
                "fit_reservations": child["experiment"]["costs"]["attempts"],
                "reserved_worker_seconds": child["experiment"]["costs"]["worker_seconds"],
                "charged_fits": child["experiment"]["costs"]["charged_fits"],
            },
        }
        if replay and result["experiment_state"] == "acknowledged":
            from quant_rsi.forecast.forecast_experiment import inspect_experiment

            result["experiment_report"] = inspect_experiment(row["run_id"], replay=True)
        runs.append(result)
    return {
        "kind": "ForecastCampaignReport",
        "schema_version": 1,
        "campaign_id": campaign_id,
        "authority_ref": authority.authority_ref,
        "campaign_ref": campaign_ref,
        "workspace": authority.identity["workspace"],
        "exposure": dict(EXPOSURE),
        "budget": plan["budget"],
        "allocated": {
            "max_attempts": len(runs) * ALLOCATION["max_attempts"],
            "max_worker_seconds": len(runs) * ALLOCATION["max_worker_seconds"],
        },
        "actual": {
            name: sum(r["actual"][name] for r in runs)
            for name in ("fit_reservations", "charged_fits", "reserved_worker_seconds")
        },
        "runs": runs,
    }


def init_campaign(plan):
    """Atomically reserve the complete immutable roster; no collection or fitting."""
    normalized = _Plan.model_validate_json(canonical(plan)).model_dump(mode="json")
    with _authority(initialize=True) as authority:
        campaigns, registrations, acks = _registry(authority)
        _children(authority, registrations, acks)
        campaign_id = normalized["campaign_id"]
        if campaign_id in campaigns:
            if campaigns[campaign_id][1] != normalized:
                raise ValueError("campaign ID already binds another immutable plan")
            return _report(authority, campaign_id, replay=False)
        recursive_sources = {
            value["source_campaign_id"]
            for _, value in campaigns.values()
            if value["kind"] == "ForecastRecursiveCampaign"
        }
        if any(
            row["run_id"] in registrations or row["run_id"] in recursive_sources
            for row in normalized["runs"]
        ):
            raise ValueError("run ID is already allocated in this workspace")
        if any(
            (authority.root.parent / directory / row["run_id"]).exists()
            for row in normalized["runs"]
            for directory in ("enhanced_forecast_workflows", "forecast_experiments")
        ):
            raise ValueError("unregistered child directory already exists")
        exposure_ref = authority.store.put_json(EXPOSURE)
        grant = {
            "kind": "ForecastCampaignGrant",
            "schema_version": 1,
            "authority_ref": authority.authority_ref,
            "plan": normalized,
            "exposure_ref": exposure_ref,
        }
        campaign_ref = authority.store.put_json(grant)
        refs = [
            authority.store.put_json(_registration(authority, campaign_ref, row, exposure_ref))
            for row in normalized["runs"]
        ]
        with authority.journal.connection:
            authority.journal.event(
                None,
                "campaign_registered",
                {"campaign_id": campaign_id, "campaign_ref": campaign_ref, "allocation_refs": refs},
            )
        return _report(authority, campaign_id, replay=False)


def inspect_campaign(id, replay=False):
    """Report original child costs; never create or repair initialized child state."""
    _id(id)
    with _authority() as authority:
        if authority is None:
            raise ValueError("forecast campaign authority is not initialized")
        return _report(authority, id, replay=replay)
