"""Observer identity and atomic checkpoint/event CAS in the existing receipt database."""

import os
import uuid

from quant_rsi.artifacts import atomic_write, canonical, decode, digest

TABLES = {"observer_metadata", "observer_episodes", "observer_events"}
SCHEMA = (
    "CREATE TABLE observer_metadata (anchor_ref TEXT NOT NULL)",
    "CREATE TABLE observer_episodes (episode_id TEXT PRIMARY KEY, plan_ref TEXT UNIQUE NOT NULL, "
    "initial_ref TEXT NOT NULL, head_ref TEXT NOT NULL, sequence INTEGER NOT NULL)",
    "CREATE TABLE observer_events (episode_id TEXT NOT NULL, sequence INTEGER NOT NULL, "
    "event_ref TEXT NOT NULL, PRIMARY KEY(episode_id,sequence), "
    "FOREIGN KEY(episode_id) REFERENCES observer_episodes(episode_id))",
)


def present_tables(registry):
    return {
        row[0]
        for row in registry.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    } & TABLES


def check_schema(registry):
    path = registry.root / "observer_identity.json"
    if not path.exists():
        raise ValueError("observer registry is not initialized")
    anchor = decode(path.read_bytes())
    if (
        set(anchor) != {"kind", "schema_version", "registry_id", "observer_registry_id"}
        or anchor["kind"] != "ObserverRegistry"
        or anchor["schema_version"] != 1
        or anchor["registry_id"] != registry.anchor["registry_id"]
        or present_tables(registry) != TABLES
    ):
        raise ValueError("initialized observer identity/ledger is missing or inconsistent")
    rows = registry.connection.execute("SELECT anchor_ref FROM observer_metadata").fetchall()
    if [row[0] for row in rows] != [digest(canonical(anchor))]:
        raise ValueError("observer identity differs from its existing ledger")
    return anchor


def initialize_schema(registry):
    """Only explicit registration can migrate; interrupted DDL resumes atomically."""
    path, marker = (
        registry.root / "observer_identity.json",
        registry.root / "observer_bootstrap.json",
    )
    if path.exists():
        anchor = check_schema(registry)
        if marker.exists():
            if decode(marker.read_bytes()) != anchor:
                raise ValueError("observer bootstrap identity conflict")
            marker.unlink()
        return anchor
    if marker.exists():
        anchor = decode(marker.read_bytes())
    else:
        if present_tables(registry):
            raise ValueError(
                "observer identity missing; refusing to reset existing episode capital"
            )
        anchor = dict(
            kind="ObserverRegistry",
            schema_version=1,
            registry_id=registry.anchor["registry_id"],
            observer_registry_id=uuid.uuid4().hex,
        )
        atomic_write(marker, canonical(anchor))
    if anchor["registry_id"] != registry.anchor["registry_id"]:
        raise ValueError("observer bootstrap belongs to another receipt registry")
    connection = registry.connection
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        tables = present_tables(registry)
        if not tables:
            for statement in SCHEMA:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO observer_metadata VALUES (?)", (digest(canonical(anchor)),)
            )
        elif tables != TABLES or [
            row[0] for row in connection.execute("SELECT anchor_ref FROM observer_metadata")
        ] != [digest(canonical(anchor))]:
            raise ValueError("interrupted observer initialization conflicts with existing ledger")
    atomic_write(path, canonical(anchor))
    marker.unlink()
    directory = os.open(registry.root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return check_schema(registry)


class ObserverJournal:
    def __init__(self, registry):
        self.registry = registry
        self.anchor = check_schema(registry)
        self.connection, self.store = registry.connection, registry.store

    def episode(self, episode_id):
        row = self.connection.execute(
            "SELECT * FROM observer_episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
        if row is None:
            raise ValueError("observer episode is not registered")
        return dict(row)

    def register(self, episode_id, plan_ref, initial):
        if (
            not isinstance(initial, dict)
            or initial.get("episode_id") != episode_id
            or initial.get("plan_ref") != plan_ref
        ):
            raise ValueError("initial state episode/plan binding mismatch")
        existing = self.connection.execute(
            "SELECT * FROM observer_episodes WHERE episode_id=? OR plan_ref=?",
            (episode_id, plan_ref),
        ).fetchall()
        if existing:
            if (
                len(existing) != 1
                or existing[0]["episode_id"] != episode_id
                or existing[0]["plan_ref"] != plan_ref
            ):
                raise ValueError(
                    "episode/plan already has a stable registration; aliases cannot reset capital"
                )
            return self.episode(episode_id)
        initial_ref = self.store.put_json(initial)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "INSERT INTO observer_episodes VALUES (?,?,?,?,0)",
                (episode_id, plan_ref, initial_ref, initial_ref),
            )
        return self.episode(episode_id)

    def commit(self, episode, next_state, kind, payload, *, objects=None):
        current = self.episode(episode["episode_id"])
        if current != episode:
            raise ValueError("observer checkpoint compare-and-swap conflict")
        if (
            not isinstance(next_state, dict)
            or next_state.get("episode_id") != current["episode_id"]
            or next_state.get("plan_ref") != current["plan_ref"]
        ):
            raise ValueError("next state episode/plan binding mismatch")
        for ref, value in (objects or {}).items():
            if digest(canonical(value)) != ref:
                raise ValueError("derived output artifact hash mismatch")
            self.store.put_json(value)
        state_ref = self.store.put_json(next_state)
        sequence = episode["sequence"] + 1
        event = dict(
            kind=kind,
            episode_id=episode["episode_id"],
            sequence=sequence,
            prior_ref=episode["head_ref"],
            state_ref=state_ref,
            payload=payload,
        )
        event_ref = self.store.put_json(event)
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            update = self.connection.execute(
                "UPDATE observer_episodes SET head_ref=?,sequence=? "
                "WHERE episode_id=? AND head_ref=? AND sequence=?",
                (
                    state_ref,
                    sequence,
                    episode["episode_id"],
                    episode["head_ref"],
                    episode["sequence"],
                ),
            )
            if update.rowcount != 1:
                raise ValueError("observer checkpoint compare-and-swap conflict")
            self.connection.execute(
                "INSERT INTO observer_events VALUES (?,?,?)",
                (episode["episode_id"], sequence, event_ref),
            )
        return self.episode(episode["episode_id"])

    def trace(self, episode_id):
        episode = self.episode(episode_id)
        state = self.store.json(episode["initial_ref"])
        if state["episode_id"] != episode_id or state["plan_ref"] != episode["plan_ref"]:
            raise ValueError("observer initial capital/plan identity mismatch")
        prior, sequence = episode["initial_ref"], 0
        events = []
        for row in self.connection.execute(
            "SELECT * FROM observer_events WHERE episode_id=? ORDER BY sequence", (episode_id,)
        ):
            event = self.store.json(row["event_ref"])
            sequence += 1
            if (
                row["sequence"] != sequence
                or event["sequence"] != sequence
                or event["episode_id"] != episode_id
                or event["prior_ref"] != prior
            ):
                raise ValueError("observer event/checkpoint chain mismatch")
            state = self.store.json(event["state_ref"])
            if state["episode_id"] != episode_id or state["plan_ref"] != episode["plan_ref"]:
                raise ValueError("observer state changed registered identity")
            prior = event["state_ref"]
            events.append(event)
        if episode["head_ref"] != prior or episode["sequence"] != sequence:
            raise ValueError("observer durable head does not match its complete event chain")
        return episode, self.store.json(episode["initial_ref"]), state, events
