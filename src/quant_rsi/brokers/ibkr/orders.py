"""IBKR paper order backend. The paper guard is re-checked before every submission.

Orders carry `orderRef = intent_id`, so broker records can always be matched to persisted
intents after a reconnect or restart. Fills are keyed by IBKR's execution ID.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from quant_rsi.brokers.contracts import BrokerEvent, InstrumentBinding, OrderIntent
from quant_rsi.brokers.ibkr.config import IbkrConfig
from quant_rsi.brokers.ibkr.connection import IbkrSession, utc_now
from quant_rsi.brokers.ibkr.guard import check_paper
from quant_rsi.brokers.ibkr.instruments import contract_params
from quant_rsi.brokers.ibkr.transport import Message

CANCELLED = frozenset({"Cancelled", "ApiCancelled"})
REJECTED = frozenset({"Inactive"})
ORDER_CANCELLED = 202
# Order-ID errors that mean the order was refused and will not work. Anything else on an
# order ID (warnings such as 399, "cannot cancel" 161/10148) is a notice only.
ORDER_REJECTION_CODES = frozenset({103, 110, 200, 201, 203, 320, 321, 322})


@dataclass
class Reconciliation:
    open_orders: list[dict] = field(default_factory=list)
    completed_orders: list[dict] = field(default_factory=list)
    executions: list[dict] = field(default_factory=list)
    positions: dict[int, Decimal] = field(default_factory=dict)


def _quantity(value) -> int:
    number = Decimal(str(value))
    if number != number.to_integral_value():
        raise ValueError("fractional share quantity is not supported")
    return int(number)


class IbkrPaperBackend:
    def __init__(
        self,
        session: IbkrSession,
        config: IbkrConfig,
        bindings: Mapping[str, InstrumentBinding],
        *,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.session, self.config, self.clock = session, config, clock
        self.bindings = dict(bindings)
        self.by_con_id = {b.con_id: name for name, b in self.bindings.items()}
        self.intent_by_order: dict[int, str] = {}
        self.account = self.verify()

    def verify(self) -> str:
        if self.session.info is None:
            raise RuntimeError("connect the order session before using the paper backend")
        return check_paper(
            self.config,
            port=self.session.port,
            accounts=self.session.info.accounts,
            client_id=self.session.client_id,
        )

    def submit(self, intent: OrderIntent) -> BrokerEvent:
        self.verify()
        order_id = self.session.next_order_id()
        params = {
            "order_id": order_id,
            "contract": contract_params(self.bindings[intent.instrument]),
            "order": {
                "action": "BUY" if intent.side == "buy" else "SELL",
                "quantity": str(intent.quantity),
                "order_type": "LMT",
                "limit_price": intent.limit_price,
                "tif": "DAY",
                "outside_rth": False,
                "order_ref": intent.intent_id,
                "account": self.account,
                "transmit": True,
            },
        }
        self.session.send("place_order", params)
        self.intent_by_order[order_id] = intent.intent_id
        return self._event("submitted", intent.intent_id, order_id=order_id)

    def cancel(self, order_id: int) -> None:
        self.session.send("cancel_order", {"order_id": order_id})

    def poll(self) -> list[BrokerEvent]:
        return [e for m in self.session.drain() if (e := self.translate(m)) is not None]

    def translate(self, message: Message) -> BrokerEvent | None:
        p = message.payload
        if message.kind in {"open_order", "completed_order"}:
            if p.get("order_ref") and p.get("order_id"):
                self.intent_by_order[int(p["order_id"])] = str(p["order_ref"])
            return None
        if message.kind == "order_status":
            order_id = int(p["order_id"])
            status = str(p["status"])
            kind = (
                "cancelled"
                if status in CANCELLED
                else "rejected"
                if status in REJECTED
                else ("status")
            )
            return self._event(
                kind, self.intent_by_order.get(order_id), order_id=order_id, status=status
            )
        if message.kind == "exec_details":
            if p.get("account") != self.account:
                return None
            order_id = int(p["order_id"])
            intent = str(p.get("order_ref") or "") or self.intent_by_order.get(order_id)
            return self._event(
                "fill",
                intent,
                order_id=order_id,
                exec_id=str(p["exec_id"]),
                quantity=_quantity(p["shares"]),
                price=str(Decimal(str(p["price"]))),
                detail=str(p.get("side", "")),
            )
        if message.kind == "commission_report":
            return self._event(
                "fee", None, exec_id=str(p["exec_id"]), amount=str(Decimal(str(p["amount"])))
            )
        if message.kind == "error" and message.req_id in self.intent_by_order:
            code = int(p.get("code", 0))
            detail = f"{code}: {p.get('message', '')}"
            intent = self.intent_by_order[message.req_id]
            if code == ORDER_CANCELLED:
                return self._event("cancelled", intent, order_id=message.req_id, detail=detail)
            if code in ORDER_REJECTION_CODES:
                return self._event("rejected", intent, order_id=message.req_id, detail=detail)
            # Warnings (399), cancel-not-possible (161, 10148) and other informational codes
            # never change order state; orderStatus remains the authority.
            return self._event("notice", intent, order_id=message.req_id, detail=detail)
        if message.kind == "error":
            return self._event("notice", None, detail=f"{p.get('code')}: {p.get('message', '')}")
        return None

    def translate_execution(self, payload: Mapping[str, object]) -> BrokerEvent | None:
        return self.translate(Message("exec_details", None, payload))

    def _event(self, kind, intent_id, **values) -> BrokerEvent:
        base = dict(
            order_id=None,
            exec_id=None,
            status=None,
            quantity=None,
            price=None,
            amount=None,
            detail="",
        )
        base.update(values)
        return BrokerEvent(kind=kind, intent_id=intent_id, at=self.clock(), **base)

    # Reconciliation queries ------------------------------------------------------------------
    def reconcile(self, since: datetime) -> Reconciliation:
        result = Reconciliation()
        for message in self.session.call(
            "open_orders",
            {},
            end_kinds={"open_order_end"},
            collect_kinds={"open_order", "order_status"},
            req_id=None,
        ):
            if message.kind == "open_order" and message.payload.get("account") == self.account:
                self.translate(message)
                result.open_orders.append(dict(message.payload))
        for message in self.session.call(
            "completed_orders",
            {"api_only": True},
            end_kinds={"completed_orders_end"},
            collect_kinds={"completed_order"},
            req_id=None,
        ):
            if message.payload.get("account") == self.account:
                self.translate(message)
                result.completed_orders.append(dict(message.payload))
        utc = (since - timedelta(seconds=1)).astimezone(timezone.utc)
        since_text = utc.strftime("%Y%m%d-%H:%M:%S")  # IBKR reads yyyymmdd-hh:mm:ss as UTC
        for message in self.session.call(
            "executions",
            {"client_id": self.session.client_id, "account": self.account, "time": since_text},
            end_kinds={"exec_details_end"},
            collect_kinds={"exec_details"},
        ):
            if message.kind == "exec_details" and message.payload.get("account") == self.account:
                result.executions.append(dict(message.payload))
        for message in self.session.call(
            "positions", {}, end_kinds={"position_end"}, collect_kinds={"position"}, req_id=None
        ):
            if message.kind == "position" and message.payload.get("account") == self.account:
                con_id = int(message.payload["con_id"])
                result.positions[con_id] = Decimal(str(message.payload["quantity"]))
        return result

    def available_funds(self) -> Decimal:
        messages = self.session.call(
            "account_summary",
            {"group": "All", "tags": "AvailableFunds"},
            end_kinds={"account_summary_end"},
            collect_kinds={"account_summary"},
        )
        for message in messages:
            p = message.payload
            if p.get("account") == self.account and p.get("tag") == "AvailableFunds":
                return Decimal(str(p["value"]))
        raise RuntimeError("AvailableFunds was not reported for the paper account")
