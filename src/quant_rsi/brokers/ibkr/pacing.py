"""Historical-data pacing with a persisted request log, so restarts respect the window.

IBKR documents hard limits for bars of 30 seconds or less; they are applied conservatively to
every bar size here. Failed (paced) requests still count toward the window.
"""

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path


class PacingLimiter:
    def __init__(
        self,
        path: Path | None = None,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        window_seconds: float = 600.0,
        max_requests: int = 50,
        identical_seconds: float = 15.0,
        per_contract_seconds: float = 3.0,
        backoff_seconds: float = 60.0,
    ):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(":memory:" if path is None else str(path))
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS requests (at REAL NOT NULL, request_key TEXT NOT NULL,"
            " contract TEXT NOT NULL, outcome TEXT NOT NULL)"
        )
        self.db.execute("CREATE TABLE IF NOT EXISTS backoff (until REAL NOT NULL)")
        self.db.commit()
        self.clock, self.sleep = clock, sleep
        self.window, self.max_requests = window_seconds, max_requests
        self.identical, self.per_contract = identical_seconds, per_contract_seconds
        self.backoff_seconds = backoff_seconds

    def _wait_seconds(self, request_key: str, contract: str) -> float:
        now = self.clock()
        waits = [0.0]
        (until,) = self.db.execute("SELECT COALESCE(MAX(until), 0) FROM backoff").fetchone()
        waits.append(until - now)
        (last_same,) = self.db.execute(
            "SELECT MAX(at) FROM requests WHERE request_key = ?", (request_key,)
        ).fetchone()
        if last_same is not None:
            waits.append(last_same + self.identical - now)
        (last_contract,) = self.db.execute(
            "SELECT MAX(at) FROM requests WHERE contract = ?", (contract,)
        ).fetchone()
        if last_contract is not None:
            waits.append(last_contract + self.per_contract - now)
        recent = [
            at
            for (at,) in self.db.execute(
                "SELECT at FROM requests WHERE at > ? ORDER BY at", (now - self.window,)
            )
        ]
        if len(recent) >= self.max_requests:
            waits.append(recent[len(recent) - self.max_requests] + self.window - now)
        return max(waits)

    def acquire(self, request_key: str, contract: str) -> float:
        """Block until a request is allowed, record it, and return the total wait."""
        waited = 0.0
        while (wait := self._wait_seconds(request_key, contract)) > 0:
            self.sleep(wait)
            waited += wait
        self.db.execute(
            "INSERT INTO requests VALUES (?, ?, ?, 'sent')", (self.clock(), request_key, contract)
        )
        self.db.commit()
        return waited

    def record_violation(self) -> None:
        """IBKR error 162: back off; the failed request remains counted."""
        self.db.execute("INSERT INTO backoff VALUES (?)", (self.clock() + self.backoff_seconds,))
        self.db.commit()

    def close(self) -> None:
        self.db.close()
