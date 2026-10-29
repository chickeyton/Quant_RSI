"""Scripted offline stand-in for TWS/IB Gateway; exercises the project message protocol only."""

from datetime import datetime, timezone
from decimal import Decimal

from quant_rsi.brokers.ibkr.config import IbkrConfig
from quant_rsi.brokers.ibkr.connection import IbkrSession
from quant_rsi.brokers.ibkr.transport import FakeIbkrTransport, Message, handshake_messages

ACCOUNT = "DU1234567"


def config(**overrides) -> IbkrConfig:
    value = {
        "schema_version": 1,
        "host": "127.0.0.1",
        "port": 7497,
        "data_client_id": 11,
        "orders_client_id": 21,
        "account_mode": "paper",
        "expected_account": ACCOUNT,
    }
    value.update(overrides)
    return IbkrConfig.model_validate(value)


class FakeBroker:
    """Answers requests; optionally fills submitted orders at their limit price."""

    def __init__(
        self,
        *,
        accounts=(ACCOUNT,),
        fill_sides=("BUY", "SELL"),
        now=None,
        cancel_ack_error=False,
        warn_on_place=False,
        race_cancel_fill=False,
    ):
        """Behaviour switches mirror real TWS traffic the default fake omits:

        - cancel_ack_error: every API cancel is also acknowledged with error 202.
        - warn_on_place: every order gets an informational 399 warning on its order ID.
        - race_cancel_fill: a cancel loses the race; the order fills and 10148 is sent.
        """
        self.accounts = tuple(accounts)
        self.fill_sides = set(fill_sides)
        self.cancel_ack_error = cancel_ack_error
        self.warn_on_place = warn_on_place
        self.race_cancel_fill = race_cancel_fill
        self.now = now or (lambda: datetime(2026, 2, 9, 15, 0, tzinfo=timezone.utc))
        self.orders: dict[int, dict] = {}
        self.executions: list[dict] = []
        self.positions: dict[int, Decimal] = {}
        self.extra_positions: dict[int, Decimal] = {}
        self.bars: dict[tuple, list[dict]] = {}
        self.contract_details: dict[str, list[dict]] = {}
        self.funds = "1000000"
        self.exec_counter = 0
        self.transport = FakeIbkrTransport(
            self.respond, handshake=handshake_messages(order_id=100, accounts=self.accounts)
        )

    def session(self, client_id=21, port=7497, clock=None) -> IbkrSession:
        session = IbkrSession(
            self.transport,
            host="127.0.0.1",
            port=port,
            client_id=client_id,
            clock=clock or self.now,
            timeout=5,
        )
        session.connect()
        return session

    def respond(self, request):
        kind, rid, p = request.kind, request.req_id, request.params
        if kind == "current_time":
            return [Message("current_time", payload={"epoch": int(self.now().timestamp())})]
        if kind == "contract_details":
            found = self.contract_details.get(p["contract"]["symbol"], [])
            return [Message("contract_details", rid, d) for d in found] + [
                Message("contract_details_end", rid)
            ]
        if kind == "historical_data":
            key = (p["contract"]["con_id"], p["bar_size"], p["what_to_show"], p["end"])
            rows = self.bars.get(key, [])
            return [Message("historical_bar", rid, r) for r in rows] + [
                Message("historical_end", rid, {"start": "", "end": p["end"]})
            ]
        if kind == "place_order":
            return self.place(p)
        if kind == "cancel_order":
            order = self.orders.get(p["order_id"])
            if not order or order["status"] in {"Filled", "Cancelled"}:
                return []
            if self.race_cancel_fill:
                cannot = {
                    "code": 10148,
                    "message": "OrderId that needs to be cancelled cannot be"
                    " cancelled, state: Filled",
                }
                return self.fill(p["order_id"]) + [Message("error", p["order_id"], cannot)]
            order["status"] = "Cancelled"
            messages = [Message("order_status", payload=self.status(order))]
            if self.cancel_ack_error:
                ack = {"code": 202, "message": "Order Canceled - reason:"}
                messages.append(Message("error", p["order_id"], ack))
            return messages
        if kind == "open_orders":
            live = [o for o in self.orders.values() if o["status"] in {"Submitted", "PreSubmitted"}]
            return [Message("open_order", payload=self.open(o)) for o in live] + [
                Message("open_order_end")
            ]
        if kind == "completed_orders":
            done = [o for o in self.orders.values() if o["status"] in {"Filled", "Cancelled"}]
            return [Message("completed_order", payload=self.open(o)) for o in done] + [
                Message("completed_orders_end")
            ]
        if kind == "executions":
            return [Message("exec_details", rid, e) for e in self.executions] + [
                Message("exec_details_end", rid)
            ]
        if kind == "positions":
            merged = dict(self.positions)
            for con_id, qty in self.extra_positions.items():
                merged[con_id] = merged.get(con_id, Decimal(0)) + qty
            return [
                Message(
                    "position",
                    payload={
                        "account": ACCOUNT,
                        "con_id": c,
                        "symbol": "",
                        "quantity": str(q),
                        "avg_cost": 0.0,
                    },
                )
                for c, q in merged.items()
            ] + [Message("position_end")]
        if kind == "account_summary":
            return [
                Message(
                    "account_summary",
                    rid,
                    {
                        "account": ACCOUNT,
                        "tag": "AvailableFunds",
                        "value": self.funds,
                        "currency": "USD",
                    },
                ),
                Message("account_summary_end", rid),
            ]
        return []

    def open(self, order):
        return {
            "order_id": order["order_id"],
            "order_ref": order["order_ref"],
            "account": order["account"],
            "con_id": order["con_id"],
            "action": order["action"],
            "quantity": order["quantity"],
            "status": order["status"],
        }

    def status(self, order):
        return {
            "order_id": order["order_id"],
            "status": order["status"],
            "filled": order["filled"],
            "remaining": "0",
            "avg_fill_price": 0.0,
            "perm_id": 1,
            "client_id": 21,
        }

    def place(self, p):
        o = p["order"]
        order = {
            "order_id": p["order_id"],
            "order_ref": o["order_ref"],
            "account": o["account"],
            "con_id": p["contract"]["con_id"],
            "action": o["action"],
            "quantity": o["quantity"],
            "limit": o["limit_price"],
            "status": "Submitted",
            "filled": "0",
        }
        self.orders[p["order_id"]] = order
        messages = [
            Message("open_order", payload=self.open(order)),
            Message("order_status", payload=self.status(order)),
        ]
        if self.warn_on_place:
            warning = {"code": 399, "message": "Order Message: warning"}
            messages.append(Message("error", p["order_id"], warning))
        if o["action"] in self.fill_sides:
            messages += self.fill(p["order_id"])
        return messages

    def fill(self, order_id, price=None, duplicate=False):
        order = self.orders[order_id]
        self.exec_counter += 0 if duplicate else 1
        execution = {
            "exec_id": f"exec-{self.exec_counter}",
            "order_id": order_id,
            "order_ref": order["order_ref"],
            "account": order["account"],
            "con_id": order["con_id"],
            "side": "BOT" if order["action"] == "BUY" else "SLD",
            "shares": order["quantity"],
            "price": float(price or order["limit"]),
            "time": "20260209 10:00:00",
        }
        if not duplicate:
            self.executions.append(execution)
            order["status"], order["filled"] = "Filled", order["quantity"]
            sign = Decimal(1) if order["action"] == "BUY" else Decimal(-1)
            con = order["con_id"]
            self.positions[con] = self.positions.get(con, Decimal(0)) + sign * Decimal(
                order["quantity"]
            )
        return [
            Message("exec_details", None, execution),
            Message(
                "commission_report",
                payload={"exec_id": execution["exec_id"], "amount": 1.0, "currency": "USD"},
            ),
            Message("order_status", payload=self.status(order)),
        ]
