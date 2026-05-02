"""IBKR broker adapter — ib_insync-backed implementation.

Provides a synchronous API over ib_insync's internally-async client.
IBKRAdapter and IBKRConnector share a single IB() instance passed on
construction to avoid exceeding IB's per-client-ID connection limit.

All credentials and connection parameters come from environment variables.
No secrets are ever hardcoded here.

Environment variables:
    IB_HOST        IB Gateway host (default: 127.0.0.1)
    IB_PORT        IB Gateway port (default: 7497 — paper trading)
    IB_CLIENT_ID   Client identifier (default: 1)
    DRY_RUN        If "true", orders are logged but never submitted (default: true)
"""

import logging
import os
from datetime import datetime, timezone
from threading import Lock

from quant_trading.execution.base_adapter import (
    BrokerAdapter,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)

logger = logging.getLogger(__name__)


def _build_ib_contract(symbol: str):
    """Build a basic US equity contract for ib_insync.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").

    Returns:
        ib_insync.Stock contract.
    """
    # Import here to keep ib_insync an optional import at module level
    from ib_insync import Stock  # type: ignore[import]

    return Stock(symbol, "SMART", "USD")


def _build_ib_order(order: Order):
    """Convert an internal Order to an ib_insync order object.

    Args:
        order: Internal Order dataclass.

    Returns:
        ib_insync MarketOrder, LimitOrder, StopOrder, or StopLimitOrder.
    """
    from ib_insync import LimitOrder, MarketOrder, StopLimitOrder, StopOrder  # type: ignore[import]

    action = order.side.value  # "BUY" or "SELL"
    qty = order.quantity

    if order.order_type == OrderType.MARKET:
        return MarketOrder(action, qty)
    elif order.order_type == OrderType.LIMIT:
        return LimitOrder(action, qty, order.limit_price)
    elif order.order_type == OrderType.STOP:
        return StopOrder(action, qty, order.stop_price)
    elif order.order_type == OrderType.STOP_LIMIT:
        return StopLimitOrder(action, qty, order.limit_price, order.stop_price)
    else:
        raise ValueError(f"Unsupported order type: {order.order_type}")


class IBKRAdapter(BrokerAdapter):
    """ib_insync-backed broker adapter for IB Gateway / TWS.

    Designed to be shared across multiple StrategyRunner instances.
    Thread-safe: a Lock guards IB API calls.

    Args:
        ib: An ib_insync.IB() instance. If None, a new one is created.
            Pass an existing instance to share a connection with IBKRConnector.

    Example:
        from ib_insync import IB
        ib = IB()
        adapter = IBKRAdapter(ib=ib)
        adapter.connect()
        account_value = adapter.get_account_value()
        adapter.disconnect()
    """

    def __init__(self, ib=None) -> None:
        self._host = os.getenv("IB_HOST", "127.0.0.1")
        self._port = int(os.getenv("IB_PORT", "7497"))
        self._client_id = int(os.getenv("IB_CLIENT_ID", "1"))
        self._dry_run = os.getenv("DRY_RUN", "true").strip().lower() == "true"

        # Accept a shared IB instance or create a new one
        if ib is None:
            from ib_insync import IB  # type: ignore[import]
            self._ib = IB()
        else:
            self._ib = ib

        self._lock = Lock()
        self._connected = False

        logger.info(
            "IBKRAdapter configured: host=%s port=%d client_id=%d dry_run=%s",
            self._host,
            self._port,
            self._client_id,
            self._dry_run,
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to IB Gateway / TWS.

        Raises:
            ConnectionError: If the connection attempt fails.
        """
        if self._connected:
            logger.debug("IBKRAdapter already connected — skipping.")
            return
        try:
            self._ib.connect(self._host, self._port, clientId=self._client_id)
            self._connected = True
            logger.info(
                "IBKRAdapter connected: host=%s port=%d client_id=%d",
                self._host,
                self._port,
                self._client_id,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Could not connect to IB Gateway at {self._host}:{self._port} "
                f"(client_id={self._client_id}): {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Disconnect from IB Gateway / TWS."""
        if self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("IBKRAdapter disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib.isConnected()

    # ------------------------------------------------------------------
    # Order operations
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Submit a single order.

        In dry-run mode, logs the order and returns immediately with
        status DRY_RUN without contacting IB.

        Args:
            order: Internal Order to submit.

        Returns:
            Updated Order with broker_order_id and status set.

        Raises:
            RuntimeError: If the IB API call fails.
        """
        # Re-check dry-run at submission time (env var may have changed)
        if os.getenv("DRY_RUN", "true").strip().lower() == "true":
            order.status = OrderStatus.DRY_RUN
            order.broker_order_id = f"DRY_RUN-{id(order)}"
            logger.info(
                "[DRY_RUN] Order not submitted: symbol=%s side=%s qty=%d",
                order.symbol,
                order.side.value,
                order.quantity,
            )
            return order

        try:
            with self._lock:
                contract = _build_ib_contract(order.symbol)
                ib_order = _build_ib_order(order)
                trade = self._ib.placeOrder(contract, ib_order)
                self._ib.sleep(0)   # flush event loop

            order.broker_order_id = str(trade.order.orderId)
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.now(tz=timezone.utc)
            logger.info(
                "Order placed: id=%s symbol=%s side=%s qty=%d type=%s",
                order.broker_order_id,
                order.symbol,
                order.side.value,
                order.quantity,
                order.order_type.value,
            )
            return order
        except Exception as exc:
            raise RuntimeError(f"IB order placement failed for {order.symbol}: {exc}") from exc

    def cancel_order(self, order: Order) -> None:
        """Cancel an open order.

        Args:
            order: Order with a valid broker_order_id.

        Raises:
            ValueError: If broker_order_id is not set.
            RuntimeError: If the cancellation API call fails.
        """
        if order.broker_order_id is None:
            raise ValueError("Cannot cancel order without broker_order_id.")
        try:
            with self._lock:
                # Retrieve the ib_insync Trade object by order ID
                open_trades = self._ib.openTrades()
                ib_trade = next(
                    (t for t in open_trades if str(t.order.orderId) == order.broker_order_id),
                    None,
                )
                if ib_trade is None:
                    logger.warning(
                        "Cancel requested for order %s but it was not found in open trades.",
                        order.broker_order_id,
                    )
                    return
                self._ib.cancelOrder(ib_trade.order)
                self._ib.sleep(0)
            order.status = OrderStatus.CANCELLED
            logger.info("Order cancelled: id=%s", order.broker_order_id)
        except Exception as exc:
            raise RuntimeError(
                f"IB order cancellation failed for id={order.broker_order_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, float]:
        """Return current open positions.

        Returns:
            Dict mapping symbol → net quantity (positive = long).
        """
        with self._lock:
            positions = self._ib.positions()
        return {
            pos.contract.symbol: pos.position
            for pos in positions
        }

    def get_account_value(self) -> float:
        """Return the account's net liquidation value.

        Returns:
            Net liquidation value in account base currency.
        """
        with self._lock:
            account_values = self._ib.accountValues()
        for av in account_values:
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        # Fallback: total cash value if NetLiquidation not found
        for av in account_values:
            if av.tag == "TotalCashValue" and av.currency == "USD":
                return float(av.value)
        logger.warning("Could not find NetLiquidation in account values.")
        return 0.0

    def get_open_orders(self) -> list[Order]:
        """Return all currently open IB orders as internal Order objects.

        Returns:
            List of Order objects with SUBMITTED status.
        """
        with self._lock:
            ib_trades = self._ib.openTrades()

        orders = []
        for trade in ib_trades:
            side = OrderSide.BUY if trade.order.action == "BUY" else OrderSide.SELL
            try:
                order_type = OrderType(trade.order.orderType)
            except ValueError:
                order_type = OrderType.MARKET
            o = Order(
                symbol=trade.contract.symbol,
                side=side,
                quantity=int(trade.order.totalQuantity),
                order_type=order_type,
                limit_price=trade.order.lmtPrice or None,
                stop_price=trade.order.auxPrice or None,
            )
            o.broker_order_id = str(trade.order.orderId)
            o.status = OrderStatus.SUBMITTED
            orders.append(o)
        return orders
