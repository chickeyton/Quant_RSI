"""Real transport over the locally installed, GPL-licensed `ibapi` package.

`ibapi` is imported only when this transport is constructed. It is never vendored into this
repository; install it from the TWS API distribution (see docs/intraday_ibkr_design.md).
This module cannot be exercised in CI; the fake transport covers the protocol behaviour.
"""

import queue
import threading
from decimal import Decimal

from quant_rsi.brokers.ibkr.transport import Message, Request


def _text(value) -> str:
    return str(Decimal(str(value))) if value is not None else "0"


def build(ibapi):
    """Create the EWrapper/EClient pair lazily so importing this module needs no ibapi."""
    client_mod, wrapper_mod = ibapi.client, ibapi.wrapper

    class Bridge(wrapper_mod.EWrapper, client_mod.EClient):
        def __init__(self, inbox: queue.Queue):
            wrapper_mod.EWrapper.__init__(self)
            client_mod.EClient.__init__(self, wrapper=self)
            self.inbox = inbox

        def put(self, kind, req_id=None, **payload):
            self.inbox.put(Message(kind, req_id, payload))

        # Session ---------------------------------------------------------------------------
        def nextValidId(self, orderId: int):
            self.put("next_valid_id", order_id=orderId)

        def managedAccounts(self, accountsList: str):
            self.put("managed_accounts", accounts=[a for a in accountsList.split(",") if a])

        def currentTime(self, time: int):
            self.put("current_time", epoch=time)

        def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
            self.put(
                "error", None if reqId in (-1, None) else reqId, code=errorCode, message=errorString
            )

        def connectionClosed(self):
            self.put("connection_closed")

        # Contracts and history -------------------------------------------------------------
        def contractDetails(self, reqId, contractDetails):
            c = contractDetails.contract
            self.put(
                "contract_details",
                reqId,
                con_id=c.conId,
                symbol=c.symbol,
                sec_type=c.secType,
                currency=c.currency,
                exchange=c.exchange,
                primary_exchange=c.primaryExchange,
                time_zone_id=contractDetails.timeZoneId,
                trading_hours=contractDetails.tradingHours,
                liquid_hours=contractDetails.liquidHours,
            )

        def contractDetailsEnd(self, reqId):
            self.put("contract_details_end", reqId)

        def _bar(self, kind, reqId, bar):
            self.put(
                kind,
                reqId,
                date=bar.date,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=_text(bar.volume),
                wap=_text(bar.wap),
                bar_count=bar.barCount,
            )

        def historicalData(self, reqId, bar):
            self._bar("historical_bar", reqId, bar)

        def historicalDataUpdate(self, reqId, bar):
            self._bar("historical_update", reqId, bar)

        def historicalDataEnd(self, reqId, start, end):
            self.put("historical_end", reqId, start=start, end=end)

        def headTimestamp(self, reqId, headTimestamp):
            self.put("head_timestamp", reqId, value=headTimestamp)

        # Orders ------------------------------------------------------------------------------
        def openOrder(self, orderId, contract, order, orderState):
            self.put(
                "open_order",
                order_id=orderId,
                order_ref=order.orderRef,
                account=order.account,
                con_id=contract.conId,
                action=order.action,
                quantity=_text(order.totalQuantity),
                status=orderState.status,
            )

        def openOrderEnd(self):
            self.put("open_order_end")

        def orderStatus(
            self,
            orderId,
            status,
            filled,
            remaining,
            avgFillPrice,
            permId,
            parentId,
            lastFillPrice,
            clientId,
            whyHeld,
            mktCapPrice,
        ):
            self.put(
                "order_status",
                order_id=orderId,
                status=status,
                filled=_text(filled),
                remaining=_text(remaining),
                avg_fill_price=avgFillPrice,
                perm_id=permId,
                client_id=clientId,
            )

        def completedOrder(self, contract, order, orderState):
            self.put(
                "completed_order",
                order_id=order.orderId,
                order_ref=order.orderRef,
                account=order.account,
                con_id=contract.conId,
                action=order.action,
                quantity=_text(order.totalQuantity),
                status=orderState.status,
                filled=_text(order.filledQuantity),
            )

        def completedOrdersEnd(self):
            self.put("completed_orders_end")

        def execDetails(self, reqId, contract, execution):
            self.put(
                "exec_details",
                None if reqId == -1 else reqId,
                exec_id=execution.execId,
                order_id=execution.orderId,
                order_ref=execution.orderRef,
                account=execution.acctNumber,
                con_id=contract.conId,
                side=execution.side,
                shares=_text(execution.shares),
                price=execution.price,
                time=execution.time,
            )

        def execDetailsEnd(self, reqId):
            self.put("exec_details_end", reqId)

        def commissionAndFeesReport(self, commissionAndFeesReport):
            r = commissionAndFeesReport
            self.put(
                "commission_report",
                exec_id=r.execId,
                amount=r.commissionAndFees,
                currency=r.currency,
            )

        def position(self, account, contract, position, avgCost):
            self.put(
                "position",
                account=account,
                con_id=contract.conId,
                symbol=contract.symbol,
                quantity=_text(position),
                avg_cost=avgCost,
            )

        def positionEnd(self):
            self.put("position_end")

        def accountSummary(self, reqId, account, tag, value, currency):
            self.put(
                "account_summary", reqId, account=account, tag=tag, value=value, currency=currency
            )

        def accountSummaryEnd(self, reqId):
            self.put("account_summary_end", reqId)

    return Bridge


class RealIbapiTransport:
    def __init__(self):
        import ibapi.client
        import ibapi.contract
        import ibapi.execution
        import ibapi.order
        import ibapi.order_cancel
        import ibapi.wrapper

        self.ibapi = ibapi
        self.inbox: queue.Queue = queue.Queue()
        self.bridge = build(ibapi)(self.inbox)
        self.thread: threading.Thread | None = None

    def connect(self, host: str, port: int, client_id: int) -> None:
        self.bridge.connect(host, port, client_id)
        if not self.bridge.isConnected():
            raise ConnectionError(f"could not connect to TWS/IB Gateway at {host}:{port}")
        self.inbox.put(Message("server_version", payload={"version": self.bridge.serverVersion()}))
        self.thread = threading.Thread(target=self.bridge.run, name="ibapi-reader", daemon=True)
        self.thread.start()

    def disconnect(self) -> None:
        self.bridge.disconnect()

    def is_connected(self) -> bool:
        return bool(self.bridge.isConnected())

    def receive(self, timeout: float) -> Message | None:
        try:
            return self.inbox.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def _contract(self, params):
        contract = self.ibapi.contract.Contract()
        contract.symbol = params["symbol"]
        contract.secType = params["sec_type"]
        contract.currency = params["currency"]
        contract.exchange = params["exchange"]
        contract.primaryExchange = params["primary_exchange"]
        if params.get("con_id"):
            contract.conId = int(params["con_id"])
        return contract

    def send(self, request: Request) -> None:
        p, client, rid = request.params, self.bridge, request.req_id
        if request.kind == "current_time":
            client.reqCurrentTime()
        elif request.kind == "contract_details":
            client.reqContractDetails(rid, self._contract(p["contract"]))
        elif request.kind == "historical_data":
            client.reqHistoricalData(
                rid,
                self._contract(p["contract"]),
                "" if p["keep_up_to_date"] else p["end"],
                p["duration"],
                p["bar_size"],
                p["what_to_show"],
                int(p["use_rth"]),
                int(p["format_date"]),
                bool(p["keep_up_to_date"]),
                [],
            )
        elif request.kind == "cancel_historical_data":
            client.cancelHistoricalData(rid)
        elif request.kind == "place_order":
            o = p["order"]
            order = self.ibapi.order.Order()
            order.action = o["action"]
            order.totalQuantity = Decimal(o["quantity"])
            order.orderType = o["order_type"]
            order.lmtPrice = float(o["limit_price"])
            order.tif = o["tif"]
            order.outsideRth = bool(o["outside_rth"])
            order.orderRef = o["order_ref"]
            order.account = o["account"]
            order.transmit = bool(o["transmit"])
            client.placeOrder(int(p["order_id"]), self._contract(p["contract"]), order)
        elif request.kind == "cancel_order":
            client.cancelOrder(int(p["order_id"]), self.ibapi.order_cancel.OrderCancel())
        elif request.kind == "open_orders":
            client.reqOpenOrders()
        elif request.kind == "completed_orders":
            client.reqCompletedOrders(bool(p.get("api_only", True)))
        elif request.kind == "executions":
            flt = self.ibapi.execution.ExecutionFilter()
            flt.clientId, flt.acctCode, flt.time = int(p["client_id"]), p["account"], p["time"]
            client.reqExecutions(rid, flt)
        elif request.kind == "positions":
            client.reqPositions()
        elif request.kind == "account_summary":
            client.reqAccountSummary(rid, p["group"], p["tags"])
        else:
            raise ValueError(f"unsupported IBKR request {request.kind}")
