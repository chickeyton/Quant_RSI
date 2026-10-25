"""One writer, transactional lifecycle events, and conservative worker-slot accounting."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from quant_rsi.artifacts import canonical

TERMINAL = {"completed", "rejected", "failed", "interrupted", "budget_exhausted"}


class BudgetExhausted(ValueError):
    pass


class Journal:
    extra_schema: dict[str, str] = {}

    def __init__(self, root: Path, manifest_digest: str | None = None):
        path = root / "journal.sqlite"
        if manifest_digest is None and not path.is_file():
            raise ValueError("run has no initialized journal")
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA synchronous=FULL")
        if manifest_digest is not None:
            try:
                self.initialize(manifest_digest)
            except Exception:
                self.connection.close()
                raise

    def initialize(self, manifest_digest: str) -> None:
        # Explicit BEGIN includes DDL; executescript would implicitly commit it first.
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            tables = {
                row[0]
                for row in self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if tables:
                if tables != {"metadata", "attempts", "events"} | self.extra_schema.keys():
                    raise ValueError("ambiguous or damaged initialization schema")
                identity = dict(self.connection.execute("SELECT key,value FROM metadata"))
                created = self.connection.execute(
                    "SELECT payload FROM events WHERE kind='run_created'"
                ).fetchall()
                if (
                    identity != {"manifest_digest": manifest_digest}
                    or len(created) != 1
                    or created[0][0] != canonical({"manifest_digest": manifest_digest}).decode()
                ):
                    raise ValueError("committed journal identity missing or conflicting")
                return  # Preserve every existing attempt, event, and charge.
            self.connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self.connection.execute("""CREATE TABLE attempts (
                        ordinal INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                        proposal_ref TEXT NOT NULL, status TEXT NOT NULL,
                        worker_seconds INTEGER NOT NULL DEFAULT 0,
                        elapsed_seconds REAL, result_ref TEXT, error TEXT
                    )""")
            self.connection.execute("""CREATE TABLE events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL, ordinal INTEGER,
                        kind TEXT NOT NULL, payload TEXT NOT NULL
                    )""")
            for statement in self.extra_schema.values():
                self.connection.execute(statement)
            self.initialize_extra()
            self.connection.execute(
                "INSERT INTO metadata VALUES ('manifest_digest', ?)", (manifest_digest,)
            )
            self.event(None, "run_created", {"manifest_digest": manifest_digest})

    def initialize_extra(self):
        pass

    def close(self):
        self.connection.close()

    def event(self, ordinal: int | None, kind: str, payload: dict):
        self.connection.execute(
            "INSERT INTO events(timestamp, ordinal, kind, payload) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), ordinal, kind, canonical(payload).decode()),
        )

    def attempts(self) -> list[dict]:
        return [
            dict(row) for row in self.connection.execute("SELECT * FROM attempts ORDER BY ordinal")
        ]

    def reserve(self, ordinal: int, name: str, proposal_ref: str, max_attempts: int):
        with self.connection:
            self.reserve_in_transaction(ordinal, name, proposal_ref, max_attempts)

    def reserve_in_transaction(self, ordinal: int, name: str, proposal_ref: str, max_attempts: int):
        if self.connection.execute("SELECT count(*) FROM attempts").fetchone()[0] >= max_attempts:
            raise BudgetExhausted("proposal attempt budget exhausted")
        self.connection.execute(
            "INSERT INTO attempts(ordinal,name,proposal_ref,status) VALUES (?,?,?,'reserved')",
            (ordinal, name, proposal_ref),
        )
        self.event(ordinal, "proposal_reserved", {"proposal_ref": proposal_ref})

    def start(self, ordinal: int, seconds: int, limit: int):
        with self.connection:
            used = self.connection.execute(
                "SELECT coalesce(sum(worker_seconds),0) FROM attempts"
            ).fetchone()[0]
            if used + seconds > limit:
                raise BudgetExhausted("worker-slot budget exhausted")
            changed = self.connection.execute(
                "UPDATE attempts SET status='running',worker_seconds=? "
                "WHERE ordinal=? AND status='reserved'",
                (seconds, ordinal),
            ).rowcount
            if changed != 1:
                raise ValueError("attempt is not reserved")
            self.event(ordinal, "worker_slot_reserved", {"seconds": seconds})

    def finish(
        self,
        ordinal: int,
        status: str,
        *,
        result_ref: str | None = None,
        error: str | None = None,
        elapsed: float | None = None,
    ):
        if status not in TERMINAL:
            raise ValueError("invalid terminal status")
        if (status == "completed") != (result_ref is not None):
            raise ValueError("only completed attempts may reference results")
        with self.connection:
            row = self.connection.execute(
                "SELECT * FROM attempts WHERE ordinal=?", (ordinal,)
            ).fetchone()
            if row is None:
                raise ValueError("attempt does not exist")
            if row["status"] in TERMINAL:
                if (
                    row["status"] == status
                    and row["result_ref"] == result_ref
                    and row["error"] == error
                ):
                    return
                raise ValueError("conflicting completion")
            required = "reserved" if status in {"rejected", "budget_exhausted"} else "running"
            if row["status"] != required:
                raise ValueError("invalid attempt state transition")
            self.connection.execute(
                "UPDATE attempts SET status=?, result_ref=?, error=?, elapsed_seconds=? "
                "WHERE ordinal=?",
                (status, result_ref, error, elapsed, ordinal),
            )
            self.event(
                ordinal,
                "attempt_finished",
                {
                    "status": status,
                    "result_ref": result_ref,
                    "error": error,
                    "elapsed_seconds": elapsed,
                },
            )

    def events(self) -> list[dict]:
        return [
            dict(row) | {"payload": json.loads(row["payload"])}
            for row in self.connection.execute("SELECT * FROM events ORDER BY sequence")
        ]
