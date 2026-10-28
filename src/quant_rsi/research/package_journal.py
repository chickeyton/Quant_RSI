"""One change request and irrevocable paired research grants per C4 campaign."""

from quant_rsi.core.artifacts import canonical, decode
from quant_rsi.core.journal import Journal


class PackageJournal(Journal):
    extra_schema = {
        "package_state": """CREATE TABLE package_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1), request_ref TEXT,
            decision_ref TEXT, completion_ref TEXT)""",
        "package_grants": """CREATE TABLE package_grants (
            arm TEXT PRIMARY KEY CHECK(arm IN ('parent','candidate')),
            grant_ref TEXT NOT NULL UNIQUE, initialized INTEGER NOT NULL DEFAULT 0)""",
        "package_rounds": """CREATE TABLE package_rounds (
            ordinal INTEGER PRIMARY KEY, steps INTEGER NOT NULL,
            targets TEXT NOT NULL, receipt_ref TEXT)""",
    }

    def initialize_extra(self):
        self.connection.execute("INSERT INTO package_state VALUES (1,NULL,NULL,NULL)")

    def state(self):
        return dict(self.connection.execute("SELECT * FROM package_state").fetchone())

    def grants(self):
        return [
            dict(r) for r in self.connection.execute("SELECT * FROM package_grants ORDER BY arm")
        ]

    def rounds(self):
        return [
            dict(r)
            for r in self.connection.execute("SELECT * FROM package_rounds ORDER BY ordinal")
        ]

    def reserve(self, request_ref):
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            previous = self.state()["request_ref"]
            if previous is not None:
                if previous != request_ref:
                    raise ValueError("only one immutable change request is permitted")
                return
            self.connection.execute(
                "UPDATE package_state SET request_ref=? WHERE singleton=1", (request_ref,)
            )
            self.event(0, "change_reserved", {"request_ref": request_ref})

    def decide(self, decision_ref, grants):
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            state = self.state()
            if state["request_ref"] is None:
                raise ValueError("change decision has no request reservation")
            if state["decision_ref"] is not None:
                if (
                    state["decision_ref"] != decision_ref
                    or {r["arm"]: r["grant_ref"] for r in self.grants()} != grants
                ):
                    raise ValueError("change decision or allocations conflict")
                return
            if grants and set(grants) != {"parent", "candidate"}:
                raise ValueError("paired grants must be committed together")
            for arm, grant_ref in sorted(grants.items()):
                self.connection.execute(
                    "INSERT INTO package_grants(arm,grant_ref) VALUES (?,?)", (arm, grant_ref)
                )
            self.connection.execute(
                "UPDATE package_state SET decision_ref=? WHERE singleton=1", (decision_ref,)
            )
            self.event(0, "change_decided", {"decision_ref": decision_ref, "grants": grants})

    def mark_initialized(self, arm):
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = next(r for r in self.grants() if r["arm"] == arm)
            if row["initialized"]:
                return
            self.connection.execute("UPDATE package_grants SET initialized=1 WHERE arm=?", (arm,))
            self.event(
                0 if arm == "parent" else 1,
                "arm_initialized",
                {"arm": arm, "grant_ref": row["grant_ref"]},
            )

    def begin_round(self, steps, targets):
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            rounds = self.rounds()
            if rounds and rounds[-1]["receipt_ref"] is None:
                if rounds[-1]["steps"] != steps:
                    raise ValueError("pending package round must resume with its original --steps")
                return rounds[-1]
            ordinal = len(rounds)
            self.connection.execute(
                "INSERT INTO package_rounds VALUES (?,?,?,NULL)",
                (ordinal, steps, canonical(targets).decode()),
            )
            self.event(ordinal, "package_round_started", {"steps": steps, "targets": targets})
            return self.rounds()[-1]

    def finish_round(self, ordinal, receipt_ref):
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.rounds()[ordinal]
            if row["receipt_ref"] is not None:
                if row["receipt_ref"] != receipt_ref:
                    raise ValueError("package round receipt conflict")
                return
            self.connection.execute(
                "UPDATE package_rounds SET receipt_ref=? WHERE ordinal=?", (receipt_ref, ordinal)
            )
            self.event(ordinal, "package_round_committed", {"receipt_ref": receipt_ref})

    def complete(self, completion_ref):
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            previous = self.state()["completion_ref"]
            if previous is not None:
                if previous != completion_ref:
                    raise ValueError("package completion conflict")
                return
            self.connection.execute(
                "UPDATE package_state SET completion_ref=? WHERE singleton=1", (completion_ref,)
            )
            self.event(None, "package_completed", {"completion_ref": completion_ref})


def targets_for(row):
    return decode(row["targets"].encode())
