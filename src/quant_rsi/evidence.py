"""One durable evidence authority per explicitly configured trusted workspace.

The controller and scorer are trusted capabilities, not an OS sandbox. Labels and
output directories never locate or reset authority. Private data is read only by
bootstrap and the OPENED scorer broker; development receives a separate view.
"""

import copy
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from quant_rsi import controller, package_experiment
from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.data import MarketData, synthetic_data
from quant_rsi.researcher import ResearchPlan
from quant_rsi.schemas import DataSpec, StrictModel


class ProgramInput(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    package_campaign: str
    task_count: int = Field(ge=1, le=4)


class ComparisonInput(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    mode: Literal["engineering_only", "confirmatory"]


def ref(value):
    return digest(canonical(value))


def authority_root():
    value = os.environ.get("QUANT_RSI_WORKSPACE", "")
    if not value or not Path(value).is_absolute():
        raise ValueError("set QUANT_RSI_WORKSPACE to the absolute trusted workspace path")
    workspace = Path(value).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("configured workspace is not a directory")
    return workspace / ".quant_rsi"


@contextmanager
def locked_authority(*, bootstrap=False):
    root = authority_root()
    if bootstrap:
        root.mkdir(mode=0o700, exist_ok=True)
    elif not root.is_dir():
        raise ValueError("authority is not initialized")
    with controller.controller_lock(root) as fd:
        yield root, fd


def import_packages(path, *, local=False):
    path = Path(path).resolve(strict=True)
    with controller.controller_lock(path):
        journal, store, manifest, plan = package_experiment.open_package(path, replay=True)
        try:
            if plan.development.data.source != ("local_event_snapshot" if local else "synthetic"):
                raise ValueError(
                    "local C5 task program requires a local C4 source"
                    if local
                    else "local C4 requires an explicit local C5 task program"
                )
            decision, reports = package_experiment.verify_package(
                path, journal, store, manifest, plan, replay=True
            )
            if not journal.state()["completion_ref"] or decision["outcome"] != "patch":
                raise ValueError("C5 requires a completed actual C4 package pair")
            grants = {role: store.json(key) for role, key in decision["grants"].items()}
            plans = {role: store.json(g["research_plan_ref"]) for role, g in grants.items()}
            for value in plans.values():
                p = ResearchPlan.model_validate_json(canonical(value))
                d = p.development
                if not local and (d.data.sessions, d.data.instruments, d.data.train_cutoff) != (
                    80,
                    4,
                    40,
                ):
                    raise ValueError("C5 v1 requires the frozen 80/4/40 development shape")
                if d.annualization != 252 or d.budget.model_dump() != {
                    "max_attempts": 8,
                    "worker_timeout_seconds": 20,
                    "max_worker_seconds": 160,
                }:
                    raise ValueError("C5 v1 requires 8 development slots of 20 seconds")
            artifact_refs = [
                manifest[k]
                for k in ("base_package_ref", "proposer_ref", "request_ref", "evidence_ref")
            ] + [decision["patch_ref"]]
            artifact_objects = {key: store.json(key) for key in artifact_refs}
            artifact_objects[ref(manifest)] = manifest
            artifact_objects[journal.state()["decision_ref"]] = decision
            summary = package_experiment.summarize(journal, store, manifest, decision, reports)
            return {
                "campaign_manifest_ref": ref(manifest),
                "decision_ref": journal.state()["decision_ref"],
                "plans": plans,
                "artifacts": artifact_objects,
                "packages": {role: g["package_ref"] for role, g in grants.items()},
                "historical_costs": {
                    "source_development": summary["historical_evidence_cost"],
                    "package_development": summary["research_budget"],
                    "accounting": "already_committed_source_costs_separate_from_new_C5_grants",
                },
            }
        finally:
            journal.close()


def parse_program(raw):
    if decode(raw).get("source") == "local_event_snapshot":
        from quant_rsi.local_evidence import LocalProgramInput

        return LocalProgramInput.model_validate_json(raw)
    return ProgramInput.model_validate_json(raw)


def init_evidence(spec: ProgramInput):
    local = getattr(spec, "source", None) == "local_event_snapshot"
    with locked_authority(bootstrap=True) as (root, _):
        anchor_path = root / "authority.json"
        bootstrap = root / "bootstrap.json"
        inputs = {
            "package_campaign": str(Path(spec.package_campaign).resolve()),
            "task_count": spec.task_count,
        }
        if local:
            inputs["local_tasks"] = [task.model_dump(mode="json") for task in spec.tasks]
        if anchor_path.exists():
            with Authority(root) as authority:
                if authority.program["input_binding"] != inputs:
                    raise ValueError(
                        "authority already has its immutable program; cannot reseed/reset"
                    )
                if bootstrap.exists():
                    saved = decode(bootstrap.read_bytes())
                    if saved["id"] != authority.anchor["authority_id"]:
                        raise ValueError("bootstrap/anchor identity conflict")
                    bootstrap.unlink()
                return authority.summary()
        if bootstrap.exists():
            saved = decode(bootstrap.read_bytes())
            if saved["inputs"] != inputs or saved["runtime"] != controller.runtime_identity():
                raise ValueError("bootstrap belongs to another input/runtime")
        else:
            if local:
                from quant_rsi.local_evidence import validate_existing_market

                validate_existing_market(root)
            if any(
                p.name not in ({".controller.lock", "market"} if local else {".controller.lock"})
                and not p.name.startswith(".pending-")
                for p in root.iterdir()
            ):
                raise ValueError("missing authority anchor; refusing a new identity")
            saved = {
                "id": uuid.uuid4().hex,
                "inputs": inputs,
                "name": spec.name,
                "runtime": controller.runtime_identity(),
                "seed": secrets.randbits(128),
                "imported": import_packages(inputs["package_campaign"], local=local),
            }
            if local:
                from quant_rsi.local_evidence import freeze_sources

                saved["local_sources"] = freeze_sources(spec)
            atomic_write(bootstrap, canonical(saved))
        home = root / "authority"
        store = ArtifactStore(home / "artifacts")
        private = ArtifactStore(home / "protected")
        imported = saved["imported"]
        for key, value in imported["artifacts"].items():
            if store.put_json(value) != key:
                raise ValueError("imported package artifact mismatch")
        local_metadata = {}
        if local:
            from quant_rsi.local_evidence import register_tasks

            source_ref, tasks, local_metadata = register_tasks(
                spec, imported, saved["local_sources"], store, private
            )
        else:
            spec_data = DataSpec(
                source="synthetic",
                sessions=122 * spec.task_count,
                instruments=4,
                train_cutoff=40,
                seed=17,
            )
            data = synthetic_data(spec_data, rng=np.random.default_rng(saved["seed"]))
            source_ref = private.put_json(data.to_json())
            tasks = []
            for i in range(spec.task_count):
                offset = 122 * i
                episode = MarketData(
                    data.dates[offset : offset + 122],
                    data.instruments,
                    data.opens[offset : offset + 122],
                    data.closes[offset : offset + 122],
                )
                view = MarketData(
                    episode.dates[:80], episode.instruments, episode.opens[:80], episode.closes[:80]
                )
                origin = {
                    "source_ref": source_ref,
                    "offset": offset,
                    "sessions": 80,
                    "purpose": "development",
                    "generator_seed_disclosed": False,
                    "data_spec_seed_role": "model_rng_only_for_bound_views",
                }
                view_ref = store.put_json(
                    {
                        "kind": "BoundDevelopmentView",
                        "origin_ref": store.put_json(origin),
                        "data": view.to_json(),
                    }
                )
                tasks.append(
                    {
                        "task": f"task_{i}",
                        "source_ref": source_ref,
                        "offset": offset,
                        "episode_ref": ref(episode.to_json()),
                        "view_ref": view_ref,
                        "sessions": list(episode.dates),
                        "weight": 1 / spec.task_count,
                        "economic_group": "one_market",
                        "dependence_cluster": "one_cluster",
                        "grid": {
                            "development": [0, 79],
                            "training_cutoff": 40,
                            "development_decisions": [40, 77],
                            "decisions": [80, 119],
                            "fills": [81, 120],
                            "valuations": [82, 121],
                            "liquidation": 121,
                            "return_count": 40,
                        },
                    }
                )
        program = {
            "kind": "EvidenceProgram",
            "authority_id": saved["id"],
            "name": saved["name"],
            "input_binding": inputs,
            "objective": "paired_researcher_change_v1",
            "purpose": "single_engineering_exposure",
            "runtime_ref": store.put_json(saved["runtime"]),
            "source_ref": source_ref,
            "source_identity": ref(
                {"source": source_ref, "purpose": "paired_researcher_change_v1"}
            ),
            "tasks": tasks,
            "generation_limit": 1,
            "batch_limit": 1,
            "plans": {role: store.put_json(value) for role, value in imported["plans"].items()},
            "packages": imported["packages"],
            "package_artifacts": sorted(imported["artifacts"]),
            "c4_campaign_manifest_ref": imported["campaign_manifest_ref"],
            "c4_decision_ref": imported["decision_ref"],
            "historical_costs": imported["historical_costs"],
            "allowance": {
                "development_attempts": 16 * spec.task_count,
                "development_seconds": 320 * spec.task_count,
                "scoring_seconds": 80 * spec.task_count,
                "alpha": 0.05,
            },
        }
        program.update(local_metadata)
        program_ref = store.put_json(program)
        anchor = {"kind": "C5Authority", "authority_id": saved["id"], "program_ref": program_ref}
        state = {
            "kind": "C5AuthorityState",
            "authority_id": saved["id"],
            "program_ref": program_ref,
            "batch": None,
            "development_attempts_granted": 0,
            "development_seconds_granted": 0,
            "scoring_seconds_charged": 0,
            "attempts": {},
            "decisions": {},
            "registry": {
                "package_ref": program["packages"]["parent"],
                "revision": 0,
                "last_report_ref": None,
                "last_decision_ref": None,
            },
        }
        db = home / "control.sqlite"
        initialized = False
        if db.exists():
            probe = sqlite3.connect(f"{db.as_uri()}?mode=rw", uri=True)
            try:
                initialized = bool(probe.execute("SELECT name FROM sqlite_master").fetchall())
            finally:
                probe.close()
        if initialized:
            with Authority(root, bootstrap_anchor=anchor) as authority:
                if authority.state != state:
                    raise ValueError("bootstrap ledger conflict")
        else:
            connection = sqlite3.connect(db)
            try:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("CREATE TABLE metadata (anchor_ref TEXT NOT NULL)")
                    connection.execute(
                        "CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), ref TEXT NOT NULL)"
                    )
                    connection.execute(
                        "CREATE TABLE events (sequence INTEGER PRIMARY KEY, "
                        "ref TEXT NOT NULL UNIQUE)"
                    )
                    connection.execute("INSERT INTO metadata VALUES (?)", (ref(anchor),))
                    state_ref = store.put_json(state)
                    connection.execute("INSERT INTO state VALUES (1,?)", (state_ref,))
                    event_ref = store.put_json(
                        {
                            "kind": "bootstrap",
                            "previous_event": None,
                            "before": None,
                            "after": state_ref,
                        }
                    )
                    connection.execute("INSERT INTO events VALUES (0,?)", (event_ref,))
            finally:
                connection.close()
        atomic_write(anchor_path, canonical(anchor))
        bootstrap.unlink()
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        with Authority(root) as authority:
            return authority.summary()


class Authority:
    def __init__(self, root, *, bootstrap_anchor=None, verify_runtime=True):
        self.root = root
        self.home = root / "authority"
        self.store = ArtifactStore(self.home / "artifacts")
        self.anchor = bootstrap_anchor or decode((root / "authority.json").read_bytes())
        self.connection = sqlite3.connect(
            f"{(self.home / 'control.sqlite').as_uri()}?mode=rw", uri=True
        )
        try:
            if self.connection.execute("SELECT anchor_ref FROM metadata").fetchall() != [
                (ref(self.anchor),)
            ]:
                raise ValueError("authority anchor/ledger mismatch")
            self.program = self.store.json(self.anchor["program_ref"])
            if self.program["authority_id"] != self.anchor["authority_id"]:
                raise ValueError("authority/program identity mismatch")
            if (
                verify_runtime
                and self.store.json(self.program["runtime_ref"]) != controller.runtime_identity()
            ):
                raise ValueError("authority requires its recorded source/runtime")
            if not (self.home / "protected" / self.program["source_ref"]).is_file():
                raise ValueError("registered protected source is missing")
            self.verify_chain()
        except Exception:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    @property
    def state_ref(self):
        rows = self.connection.execute("SELECT ref FROM state WHERE id=1").fetchall()
        if len(rows) != 1:
            raise ValueError("authority state missing")
        return rows[0][0]

    @property
    def state(self):
        return self.store.json(self.state_ref)

    def verify_chain(self):
        previous, after = None, None
        for index, (sequence, event_ref) in enumerate(
            self.connection.execute("SELECT sequence,ref FROM events ORDER BY sequence")
        ):
            event = self.store.json(event_ref)
            if sequence != index or event["previous_event"] != previous or event["before"] != after:
                raise ValueError("authority event chain conflict")
            self.store.get(event["after"])
            previous, after = event_ref, event["after"]
        if after != self.state_ref:
            raise ValueError("authority state differs from committed history")
        state = self.state
        if (
            state["authority_id"] != self.anchor["authority_id"]
            or state["program_ref"] != self.anchor["program_ref"]
        ):
            raise ValueError("state belongs to another authority")
        if state["scoring_seconds_charged"] != sum(
            a["seconds"] for a in state["attempts"].values()
        ):
            raise ValueError("scoring debit mismatch")
        if state["scoring_seconds_charged"] > self.program["allowance"]["scoring_seconds"]:
            raise ValueError("program scoring budget exceeded")

    def update(self, kind, change):
        before = self.state_ref
        state = copy.deepcopy(self.state)
        change(state)
        after = self.store.put_json(state)
        with self.connection:
            if self.state_ref != before:
                raise ValueError("authority state compare-and-swap conflict")
            sequence, previous = self.connection.execute(
                "SELECT sequence,ref FROM events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            event_ref = self.store.put_json(
                {"kind": kind, "previous_event": previous, "before": before, "after": after}
            )
            self.connection.execute("UPDATE state SET ref=? WHERE id=1 AND ref=?", (after, before))
            self.connection.execute("INSERT INTO events VALUES (?,?)", (sequence + 1, event_ref))

    def summary(self):
        state = self.state
        return {
            "authority_id": self.anchor["authority_id"],
            "program_ref": self.anchor["program_ref"],
            "batch_state": "AVAILABLE" if state["batch"] is None else state["batch"]["status"],
            "development_attempts_granted": state["development_attempts_granted"],
            "development_seconds_granted": state["development_seconds_granted"],
            "scoring_seconds_charged": state["scoring_seconds_charged"],
            "registry": state["registry"],
        }
