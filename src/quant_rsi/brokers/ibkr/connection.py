"""Synchronous facade over an IBKR transport: request IDs, timeouts and typed failures.

Messages that belong to no pending call (order status, executions, fees, order errors) are
buffered and returned by `drain`; order-related messages returned by a call are also
buffered, because duplicate facts are idempotent downstream (fills are keyed by exec ID).
"""

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from quant_rsi.brokers.ibkr.transport import Message, Request, Transport

ORDER_KINDS = frozenset(
    {"open_order", "order_status", "exec_details", "commission_report", "completed_order"}
)
CONNECTION_LOST = frozenset({504, 1100, 1300})
CONNECTION_RESTORED = frozenset({1101, 1102})
PACING_VIOLATION = 162
AUTO_ID = object()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_notice(code: int) -> bool:
    """Informational TWS codes (market-data farm status and similar) are not failures."""
    return 2100 <= code < 2200 or code in {2000, 10167}


class IbkrError(RuntimeError):
    def __init__(self, code: int, message: str, req_id: int | None = None):
        super().__init__(f"IBKR error {code}: {message}")
        self.code = code
        self.message = message
        self.req_id = req_id


class IbkrTimeout(IbkrError):
    def __init__(self, what: str):
        super().__init__(-1, f"timed out waiting for {what}")


class IbkrDisconnected(IbkrError):
    pass


def _server_time(epoch) -> datetime:
    """TWS reports whole seconds, so its clock lies in [epoch, epoch + 1); use the middle.

    Comparing with the truncated second would overstate skew by up to one second.
    """
    return datetime.fromtimestamp(int(epoch) + 0.5, timezone.utc)


@dataclass(frozen=True)
class SessionInfo:
    host: str
    port: int
    client_id: int
    server_version: int
    next_order_id: int
    accounts: tuple[str, ...]
    server_time: datetime
    local_time: datetime

    @property
    def clock_skew_seconds(self) -> float:
        return abs((self.local_time - self.server_time).total_seconds())


@dataclass
class ConnectionEvent:
    kind: str
    code: int
    message: str
    at: datetime


@dataclass
class _State:
    next_req_id: int = 1000
    next_order_id: int | None = None
    accounts: tuple[str, ...] = ()
    server_version: int | None = None
    buffer: list[Message] = field(default_factory=list)
    events: list[ConnectionEvent] = field(default_factory=list)
    restorations: list[int] = field(default_factory=list)
    connectivity_lost: bool = False


class IbkrSession:
    def __init__(
        self,
        transport: Transport,
        *,
        host: str,
        port: int,
        client_id: int,
        clock: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        timeout: float = 30.0,
    ):
        self.transport = transport
        self.host, self.port, self.client_id = host, port, client_id
        self.clock = clock
        self.monotonic = monotonic
        self.timeout = timeout
        self.info: SessionInfo | None = None
        self._state = _State()

    # Connection lifecycle -------------------------------------------------------------
    def connect(self) -> SessionInfo:
        self.transport.connect(self.host, self.port, self.client_id)
        deadline = self.monotonic() + self.timeout
        while self._state.next_order_id is None or not self._state.accounts:
            message = self.transport.receive(max(0.0, deadline - self.monotonic()))
            if message is None:
                if self.monotonic() >= deadline:
                    self.transport.disconnect()
                    raise IbkrTimeout("connection handshake (nextValidId/managedAccounts)")
                continue
            self._absorb(message, in_call=False, waiting=None)
        server = self.call("current_time", {}, end_kinds={"current_time"}, req_id=None)
        server_time = _server_time(server[-1].payload["epoch"])
        self.info = SessionInfo(
            host=self.host,
            port=self.port,
            client_id=self.client_id,
            server_version=int(self._state.server_version or 0),
            next_order_id=int(self._state.next_order_id),
            accounts=self._state.accounts,
            server_time=server_time,
            local_time=self.clock(),
        )
        return self.info

    def disconnect(self) -> None:
        self.transport.disconnect()

    def reconnect(self) -> SessionInfo:
        """Open a fresh connection. Request IDs keep increasing and buffered order messages
        are kept; TWS supplies a new next valid order ID. Callers must reconcile afterwards."""
        self.transport.disconnect()
        state = self._state
        self._state = _State(
            next_req_id=state.next_req_id, buffer=state.buffer, events=state.events
        )
        info = self.connect()
        self._record("reconnected", 0, "")
        return info

    def measure_clock_skew(self) -> float:
        server = self.call("current_time", {}, end_kinds={"current_time"}, req_id=None)
        server_time = _server_time(server[-1].payload["epoch"])
        return abs((self.clock() - server_time).total_seconds())

    @property
    def connection_events(self) -> tuple[ConnectionEvent, ...]:
        return tuple(self._state.events)

    @property
    def connectivity_lost(self) -> bool:
        """TWS reported lost connectivity to IBKR (1100) and has not yet restored it."""
        return self._state.connectivity_lost

    def poll_connectivity(self) -> list[int]:
        """Absorb queued session messages and return restoration codes seen since last call.

        1101 means market-data subscriptions were lost and must be renewed; 1102 means they
        were kept. Either way orders must be reconciled.
        """
        while True:
            message = self.transport.receive(0.0)
            if message is None:
                break
            if self._absorb(message, in_call=False, waiting=None):
                self._state.buffer.append(message)
        codes, self._state.restorations = self._state.restorations, []
        return codes

    # Identifiers ---------------------------------------------------------------------------
    def next_request_id(self) -> int:
        self._state.next_req_id += 1
        return self._state.next_req_id

    def next_order_id(self) -> int:
        if self._state.next_order_id is None:
            raise IbkrDisconnected(-1, "no valid order ID; connect first")
        value = self._state.next_order_id
        self._state.next_order_id += 1
        return value

    # Requests ------------------------------------------------------------------------------
    def send(self, kind: str, params: Mapping[str, object], *, req_id: int | None = None) -> None:
        if not self.transport.is_connected():
            raise IbkrDisconnected(504, "not connected")
        self.transport.send(Request(kind, req_id, params))

    def call(
        self,
        kind: str,
        params: Mapping[str, object],
        *,
        end_kinds: Iterable[str],
        collect_kinds: Iterable[str] = (),
        req_id: object = AUTO_ID,
        timeout: float | None = None,
    ) -> list[Message]:
        """Send one request and collect its messages until an end message arrives.

        The default allocates a request ID; `req_id=None` sends an ID-less request and
        collects by message kind instead.
        """
        if req_id is AUTO_ID:
            req_id = self.next_request_id()
        end_kinds, collect_kinds = frozenset(end_kinds), frozenset(collect_kinds)
        self.send(kind, params, req_id=req_id)
        deadline = self.monotonic() + (self.timeout if timeout is None else timeout)
        collected: list[Message] = []
        while True:
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise IbkrTimeout(f"{kind} (request {req_id})")
            message = self.transport.receive(remaining)
            if message is None:
                if not self.transport.is_connected():
                    raise IbkrDisconnected(504, f"connection closed during {kind}")
                continue
            if not self._absorb(message, in_call=True, waiting=req_id):
                continue
            mine = message.req_id == req_id if req_id is not None else True
            if mine and message.kind in end_kinds:
                collected.append(message)
                return collected
            if mine and message.kind in collect_kinds:
                collected.append(message)
                if message.kind in ORDER_KINDS:
                    self._state.buffer.append(message)
                continue
            self._state.buffer.append(message)

    def drain(self) -> list[Message]:
        """Buffered messages plus anything already queued, without waiting."""
        while True:
            message = self.transport.receive(0.0)
            if message is None:
                break
            if self._absorb(message, in_call=False, waiting=None):
                self._state.buffer.append(message)
        drained, self._state.buffer = self._state.buffer, []
        return drained

    # Message classification ------------------------------------------------------------------
    def _absorb(self, message: Message, *, in_call: bool, waiting: int | None) -> bool:
        """Apply session-level messages; return True when the message needs routing."""
        payload = message.payload
        if message.kind == "server_version":
            self._state.server_version = int(payload["version"])
            return False
        if message.kind == "next_valid_id":
            value = int(payload["order_id"])
            current = self._state.next_order_id
            self._state.next_order_id = value if current is None else max(current, value)
            return False
        if message.kind == "managed_accounts":
            self._state.accounts = tuple(str(x) for x in payload["accounts"] if str(x))
            return False
        if message.kind == "connection_closed":
            self._record("disconnected", 504, "connection closed")
            raise IbkrDisconnected(504, "connection closed")
        if message.kind != "error":
            return True
        code, text = int(payload["code"]), str(payload.get("message", ""))
        if code in CONNECTION_LOST:
            self._record("connection_lost", code, text)
            self._state.connectivity_lost = True
            if in_call:
                raise IbkrDisconnected(code, text, message.req_id)
            return False
        if code in CONNECTION_RESTORED:
            self._record("connection_restored", code, text)
            self._state.connectivity_lost = False
            self._state.restorations.append(code)
            return False
        if is_notice(code):
            self._record("notice", code, text)
            return False
        if waiting is not None and message.req_id == waiting:
            raise IbkrError(code, text, message.req_id)
        return True

    def _record(self, kind: str, code: int, text: str) -> None:
        self._state.events.append(ConnectionEvent(kind, code, text, self.clock()))


class ClockSkewMonitor:
    """Re-measures TWS clock skew at an interval; an unmeasurable skew blocks entries."""

    def __init__(
        self,
        session: IbkrSession,
        *,
        interval_seconds: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.session, self.interval, self.monotonic = session, interval_seconds, monotonic
        self.value = session.info.clock_skew_seconds if session.info else float("inf")
        self.measured_at = self.monotonic()

    def __call__(self) -> float:
        if self.monotonic() - self.measured_at >= self.interval:
            self.measured_at = self.monotonic()
            try:
                self.value = self.session.measure_clock_skew()
            except IbkrError:
                self.value = float("inf")
        return self.value
