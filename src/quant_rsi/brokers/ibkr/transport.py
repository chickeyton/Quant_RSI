"""Project-owned message protocol between the IBKR session and a transport.

`RealIbapiTransport` (in `ibapi_transport`) translates `ibapi` callbacks into these messages.
`FakeIbkrTransport` replays scripted responses so every behaviour is testable offline.

Message payloads use plain JSON-compatible values. Decimal quantities from `ibapi` are passed
as decimal strings so no precision is lost before the trusted ledger parses them.
"""

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Request:
    kind: str
    req_id: int | None
    params: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    kind: str
    req_id: int | None = None
    payload: Mapping[str, object] = field(default_factory=dict)


class Transport(Protocol):
    def connect(self, host: str, port: int, client_id: int) -> None: ...

    def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def send(self, request: Request) -> None: ...

    def receive(self, timeout: float) -> Message | None: ...


Responder = Callable[[Request], Iterable[Message]]


class FakeIbkrTransport:
    """Scripted transport: each sent request is answered by `responder`; tests may push more."""

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        handshake: Iterable[Message] = (),
        refuse_connect: bool = False,
    ):
        self.responder = responder or (lambda request: ())
        self.handshake = tuple(handshake)
        self.refuse_connect = refuse_connect
        self.sent: list[Request] = []
        self.connections: list[tuple[str, int, int]] = []
        self.queue: deque[Message] = deque()
        self.connected = False

    def connect(self, host: str, port: int, client_id: int) -> None:
        if self.refuse_connect:
            raise ConnectionRefusedError("scripted connection refusal")
        self.connections.append((host, port, client_id))
        self.connected = True
        self.queue.extend(self.handshake)

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def send(self, request: Request) -> None:
        if not self.connected:
            raise ConnectionError("not connected")
        self.sent.append(request)
        self.queue.extend(self.responder(request))

    def push(self, *messages: Message) -> None:
        self.queue.extend(messages)

    def receive(self, timeout: float) -> Message | None:
        return self.queue.popleft() if self.queue else None


def handshake_messages(
    *, order_id: int = 1, accounts: Iterable[str] = ("DU1234567",), server_version: int = 200
) -> tuple[Message, ...]:
    return (
        Message("server_version", payload={"version": server_version}),
        Message("next_valid_id", payload={"order_id": order_id}),
        Message("managed_accounts", payload={"accounts": list(accounts)}),
    )
