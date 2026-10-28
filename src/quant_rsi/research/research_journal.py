"""Campaign reservations, Kernel attempts and checkpoint CAS share one SQLite transaction domain."""

from quant_rsi.core.journal import BudgetExhausted, Journal


class ResearchJournal(Journal):
    extra_schema = {
        "research_requests": """CREATE TABLE research_requests (
            ordinal INTEGER PRIMARY KEY, request_ref TEXT NOT NULL UNIQUE,
            parent_ref TEXT NOT NULL, request_event_seq INTEGER NOT NULL DEFAULT 0,
            decision_ref TEXT, proposal_ref TEXT, experiment_ref TEXT,
            observation_ref TEXT UNIQUE, checkpoint_ref TEXT UNIQUE,
            observation_event_seq INTEGER
        )""",
        "research_state": """CREATE TABLE research_state (
            id INTEGER PRIMARY KEY CHECK(id=1), head_ref TEXT,
            execution_state TEXT NOT NULL, stop_reason TEXT NOT NULL
        )""",
    }

    def initialize_extra(self):
        self.connection.execute("INSERT INTO research_state VALUES (1,NULL,'paused','not_started')")

    def state(self):
        row = self.connection.execute("SELECT * FROM research_state WHERE id=1").fetchone()
        if row is None:
            raise ValueError(
                "research state is missing; refusing to reconstruct a committed ledger"
            )
        return dict(row)

    def requests(self):
        return [
            dict(r)
            for r in self.connection.execute("SELECT * FROM research_requests ORDER BY ordinal")
        ]

    def reserve_request(self, request_ref, parent_ref, initial_ref, max_attempts):
        with self.connection:
            rows = self.requests()
            if len(rows) >= max_attempts:
                raise BudgetExhausted("campaign attempt budget exhausted")
            if any(r["observation_ref"] is None for r in rows):
                raise ValueError("a prior research request is still pending")
            if (self.state()["head_ref"] or initial_ref) != parent_ref:
                raise ValueError("research checkpoint compare-and-swap conflict")
            ordinal = len(rows)
            self.connection.execute(
                "INSERT INTO research_requests(ordinal,request_ref,parent_ref) VALUES (?,?,?)",
                (ordinal, request_ref, parent_ref),
            )
            self.event(
                ordinal,
                "research_request_reserved",
                {
                    "request_ref": request_ref,
                    "parent_ref": parent_ref,
                    "attempt_units": 1,
                },
            )
            sequence = self.connection.execute("SELECT last_insert_rowid()").fetchone()[0]
            self.connection.execute(
                "UPDATE research_requests SET request_event_seq=? WHERE ordinal=?",
                (sequence, ordinal),
            )
            self.connection.execute(
                "UPDATE research_state SET execution_state='running',stop_reason='in_progress'"
            )
        return ordinal

    def commit_decision(self, ordinal, decision_ref, proposal_ref, experiment_ref, name, limit):
        with self.connection:
            row = self.requests()[ordinal]
            if row["decision_ref"] is not None:
                if (row["decision_ref"], row["proposal_ref"], row["experiment_ref"]) != (
                    decision_ref,
                    proposal_ref,
                    experiment_ref,
                ):
                    raise ValueError("conflicting research decision")
                return
            if proposal_ref is not None:
                self.reserve_in_transaction(ordinal, name, proposal_ref, limit)
            self.connection.execute(
                "UPDATE research_requests SET decision_ref=?,proposal_ref=?,experiment_ref=? "
                "WHERE ordinal=?",
                (decision_ref, proposal_ref, experiment_ref, ordinal),
            )
            self.event(
                ordinal,
                "research_decision_committed",
                {
                    "decision_ref": decision_ref,
                    "proposal_ref": proposal_ref,
                    "experiment_ref": experiment_ref,
                },
            )

    def commit_observation(self, ordinal, parent_ref, initial_ref, observation_ref, checkpoint_ref):
        with self.connection:
            row = self.requests()[ordinal]
            if row["observation_ref"] is not None:
                if (row["observation_ref"], row["checkpoint_ref"]) != (
                    observation_ref,
                    checkpoint_ref,
                ):
                    raise ValueError("conflicting observation application")
                return
            if (
                row["decision_ref"] is None
                or row["parent_ref"] != parent_ref
                or (self.state()["head_ref"] or initial_ref) != parent_ref
            ):
                raise ValueError("research checkpoint compare-and-swap conflict")
            self.event(
                ordinal,
                "research_step_committed",
                {
                    "observation_ref": observation_ref,
                    "checkpoint_ref": checkpoint_ref,
                    "parent_ref": parent_ref,
                },
            )
            sequence = self.connection.execute("SELECT last_insert_rowid()").fetchone()[0]
            self.connection.execute(
                "UPDATE research_requests SET observation_ref=?,checkpoint_ref=?,"
                "observation_event_seq=? WHERE ordinal=?",
                (observation_ref, checkpoint_ref, sequence, ordinal),
            )
            self.connection.execute(
                "UPDATE research_state SET head_ref=?,execution_state='running',"
                "stop_reason='in_progress' WHERE id=1",
                (checkpoint_ref,),
            )

    def stop(self, state: str, reason: str):
        with self.connection:
            previous = self.state()
            if (previous["execution_state"], previous["stop_reason"]) == (state, reason):
                return
            self.connection.execute(
                "UPDATE research_state SET execution_state=?,stop_reason=? WHERE id=1",
                (state, reason),
            )
            self.event(None, "research_boundary", {"execution_state": state, "stop_reason": reason})
