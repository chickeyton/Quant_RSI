"""Stable development-trajectory identity, finite capacity, and atomic state history."""

import copy
import fcntl
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, canonical, decode, digest
from quant_rsi.core.schemas import StrictModel


class TrajectoryBudget(StrictModel):
    worker_attempts: int = Field(strict=True, ge=0)
    registered_worker_seconds: int = Field(strict=True, ge=0)


class TrajectoryPlan(StrictModel):
    schema_version: Literal[1, 3] = 1
    max_generations: int = Field(strict=True, ge=1, le=2, default=1)
    name: str = Field(min_length=1, max_length=160)
    source_directory: Path
    source_checkpoint_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_policy: Literal["reset"]
    max_proposal_stages: int = Field(strict=True, ge=1, le=2)
    budget: TrajectoryBudget

    @model_serializer(mode="wrap")
    def serialize_version(self, handler):
        value = handler(self)
        if self.schema_version == 1:
            value.pop("schema_version", None)
            value.pop("max_generations", None)
        return value

    @model_validator(mode="after")
    def versioned_limits(self):
        if (
            self.schema_version == 1
            and {"schema_version", "max_generations"} & self.model_fields_set
        ):
            raise ValueError("legacy trajectory cannot acquire recursive profile fields")
        if self.schema_version == 3 and "max_generations" not in self.model_fields_set:
            raise ValueError("recursive forecast trajectory requires explicit generations")
        return self

    @field_validator("source_directory")
    @classmethod
    def absolute_source(cls, value):
        if not value.is_absolute():
            raise ValueError("trajectory source_directory must be absolute")
        return value


def trajectory_root():
    value = os.environ.get("QUANT_RSI_WORKSPACE")
    if not value or not Path(value).is_absolute():
        raise ValueError("set an absolute QUANT_RSI_WORKSPACE for the stable trajectory")
    return Path(value).resolve() / ".quant_rsi" / "trajectory"


def anchor_path(root):
    return root.parent / "trajectory_anchor.json"


def lock_path(root):
    return root.parent / ".trajectory.lock"


@contextmanager
def locked_trajectory(*, bootstrap=False):
    root = trajectory_root()
    if bootstrap:
        root.parent.mkdir(parents=True, exist_ok=True)
    elif not anchor_path(root).is_file():
        raise ValueError("trajectory authority is not initialized")
    with lock_path(root).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("trajectory authority is busy") from error
        try:
            yield root, lock.fileno()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def require_owner(root, owner_fd):
    """Require the actual owning open-file description, including inherited owners."""
    path = lock_path(root)
    try:
        if not os.path.samestat(os.fstat(owner_fd), path.stat()):
            raise ValueError("trajectory requires its actual owner lock")
        with path.open("r+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise ValueError("trajectory authority lock is not held")
        fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, TypeError) as error:
        raise ValueError("trajectory requires its actual owner lock") from error


def occupied_capacity(program, state):
    return {
        unit: program.get("initial_capacity", program["initial_costs"])[unit]
        + sum(stage["capacity"][unit] for stage in state["stages"])
        for unit in ("worker_attempts", "registered_worker_seconds")
    }


def validate_state(program, state, anchor):
    if (state["authority_id"], state["program_ref"]) != (
        anchor["authority_id"],
        anchor["program_ref"],
    ):
        raise ValueError("trajectory state identity mismatch")
    plan = TrajectoryPlan.model_validate_json(canonical(program["input"]))
    if len(state["stages"]) > plan.max_proposal_stages:
        raise ValueError("trajectory proposal-stage limit exceeded")
    for index, stage in enumerate(state["stages"]):
        if stage["ordinal"] != index:
            raise ValueError("trajectory stage ordering mismatch")
        for unit in ("worker_attempts", "registered_worker_seconds"):
            capacity, used = stage["capacity"][unit], stage["consumed"][unit]
            if type(capacity) is not int or type(used) is not int or not 0 <= used <= capacity:
                raise ValueError("trajectory child consumption exceeds its commitment")
    for unit, occupied in occupied_capacity(program, state).items():
        historical = program["initial_costs"][unit]
        if (
            type(historical) is not int
            or historical < 0
            or historical > program.get("initial_capacity", program["initial_costs"])[unit]
            or not 0 <= occupied <= getattr(plan.budget, unit)
        ):
            raise ValueError("trajectory cumulative budget exceeded")
    history = state["semantic_history"]
    if (
        type(state["generation"]) is not int
        or type(state["revision"]) is not int
        or not 0 <= state["generation"] <= (plan.max_generations if plan.schema_version == 3 else 1)
        or state["generation"] != state["revision"]
        or len(history) != state["generation"] + 1
        or len(set(history)) != len(history)
        or state["head"]["semantic_ref"] != history[-1]
    ):
        raise ValueError("trajectory semantic head/history mismatch")
    if state["status"] not in {"ACTIVE", "SEALED"}:
        raise ValueError("unknown trajectory status")
    if (state["status"] == "SEALED") != (state["terminal_ref"] is not None):
        raise ValueError("trajectory terminal receipt mismatch")
    if state["status"] == "SEALED" and state["pending"] is not None:
        raise ValueError("sealed trajectory has an active operation")


def validate_transition(before, after):
    if before["status"] == "SEALED":
        raise ValueError("a sealed trajectory cannot resume")
    if len(after["stages"]) < len(before["stages"]):
        raise ValueError("trajectory cannot erase a stage or its commitment")
    for old, new in zip(before["stages"], after["stages"], strict=False):
        if (old["authorization_ref"], old["capacity"]) != (
            new["authorization_ref"],
            new["capacity"],
        ):
            raise ValueError("trajectory authorization or commitment changed")
        if any(new["consumed"][unit] < old["consumed"][unit] for unit in old["consumed"]):
            raise ValueError("trajectory cannot refund a recorded charge")
    if after["revision"] not in {before["revision"], before["revision"] + 1}:
        raise ValueError("trajectory head revision skipped or rewound")
    if after["revision"] == before["revision"] and after["head"] != before["head"]:
        raise ValueError("trajectory head changed without its CAS revision")
    if after["semantic_history"][: len(before["semantic_history"])] != before["semantic_history"]:
        raise ValueError("trajectory semantic history changed")


def validate_initial(program, state):
    if (
        state["status"] != "ACTIVE"
        or state["stages"]
        or state["pending"] is not None
        or state["generation"] != 0
        or state["revision"] != 0
        or state["head"] != program["initial_head"]
        or state["semantic_history"] != [program["initial_head"]["semantic_ref"]]
    ):
        raise ValueError("trajectory bootstrap is not its frozen initial version")


class TrajectoryLedger:
    def __init__(self, root, *, bootstrap_anchor=None, owner_fd=None, readonly=False):
        self.root = root
        self.owner_fd = owner_fd
        self.store = ArtifactStore(root / "artifacts")
        self.anchor = bootstrap_anchor or decode(anchor_path(root).read_bytes())
        mode = "ro" if readonly else "rw"
        self.connection = sqlite3.connect(
            f"{(root / 'control.sqlite').as_uri()}?mode={mode}", uri=True
        )
        self._verify_open()

    @classmethod
    def frozen(cls, store, closure):
        """Replay the original sealed ledger from CAS in memory; no mutable source paths."""
        self = cls.__new__(cls)
        self.root = Path(closure["authority_root"])
        self.owner_fd = None
        self.store = store
        self.anchor = closure["anchor"]
        self.connection = sqlite3.connect(":memory:")
        try:
            self.connection.deserialize(store.get(closure["control_ref"]))
            self.connection.execute("PRAGMA query_only=ON")
            self._verify_open()
        except Exception:
            self.connection.close()
            raise
        if self.state_ref != closure["state_ref"]:
            self.connection.close()
            raise ValueError("frozen trajectory ledger differs from its sealed state")
        return self

    def _verify_open(self):
        try:
            if self.connection.execute("SELECT anchor_ref FROM metadata").fetchall() != [
                (digest(canonical(self.anchor)),)
            ]:
                raise ValueError("trajectory anchor/ledger identity mismatch")
            self.program = self.store.json(self.anchor["program_ref"])
            if (
                self.anchor["kind"] != "TrajectoryAuthority"
                or self.program["kind"] != "DevelopmentTrajectoryProgram"
                or self.program["authority_id"] != self.anchor["authority_id"]
            ):
                raise ValueError("trajectory program/anchor mismatch")
            if self.store.json(self.program["runtime_ref"]) != controller.runtime_identity():
                raise ValueError("trajectory requires its recorded source/runtime")
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
            raise ValueError("trajectory state is missing")
        return rows[0][0]

    @property
    def state(self):
        return self.store.json(self.state_ref)

    def verify_chain(self):
        previous_event, previous_state_ref, previous_state = None, None, None
        for index, (sequence, ref) in enumerate(
            self.connection.execute("SELECT sequence,ref FROM events ORDER BY sequence")
        ):
            event = self.store.json(ref)
            if (
                sequence != index
                or event["previous_event"] != previous_event
                or event["before"] != previous_state_ref
            ):
                raise ValueError("trajectory event chain conflict")
            state = self.store.json(event["after"])
            validate_state(self.program, state, self.anchor)
            if previous_state is not None:
                validate_transition(previous_state, state)
            elif event["kind"] != "trajectory_initialized":
                raise ValueError("trajectory initialization event is missing")
            else:
                validate_initial(self.program, state)
            previous_event, previous_state_ref, previous_state = ref, event["after"], state
        if previous_state_ref != self.state_ref:
            raise ValueError("trajectory state differs from committed history")

    def update(self, kind, change):
        require_owner(self.root, self.owner_fd)
        before_ref, before = self.state_ref, self.state
        state = copy.deepcopy(before)
        change(state)
        validate_state(self.program, state, self.anchor)
        validate_transition(before, state)
        after_ref = self.store.put_json(state)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            if self.state_ref != before_ref:
                raise ValueError("trajectory state CAS conflict")
            sequence, previous = self.connection.execute(
                "SELECT sequence,ref FROM events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            event_ref = self.store.put_json(
                dict(kind=kind, previous_event=previous, before=before_ref, after=after_ref)
            )
            self.connection.execute("UPDATE state SET ref=? WHERE id=1", (after_ref,))
            self.connection.execute("INSERT INTO events VALUES (?,?)", (sequence + 1, event_ref))


def initialize_ledger(root, anchor, state, *, owner_fd):
    require_owner(root, owner_fd)
    store = ArtifactStore(root / "artifacts")
    program = store.json(anchor["program_ref"])
    validate_state(program, state, anchor)
    validate_initial(program, state)
    state_ref = store.put_json(state)
    event_ref = store.put_json(
        dict(kind="trajectory_initialized", previous_event=None, before=None, after=state_ref)
    )
    connection = sqlite3.connect(root / "control.sqlite")
    try:
        with connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            if tables:
                if {row[0] for row in tables} != {"metadata", "state", "events"} or (
                    connection.execute("SELECT anchor_ref FROM metadata").fetchall()
                    != [(digest(canonical(anchor)),)]
                ):
                    raise ValueError("trajectory bootstrap conflicts with an existing ledger")
                return
            connection.execute("CREATE TABLE metadata (anchor_ref TEXT NOT NULL)")
            connection.execute("CREATE TABLE state (id INTEGER PRIMARY KEY, ref TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE events (sequence INTEGER PRIMARY KEY,ref TEXT NOT NULL)"
            )
            connection.execute("INSERT INTO metadata VALUES (?)", (digest(canonical(anchor)),))
            connection.execute("INSERT INTO state VALUES (1,?)", (state_ref,))
            connection.execute("INSERT INTO events VALUES (0,?)", (event_ref,))
    finally:
        connection.close()
