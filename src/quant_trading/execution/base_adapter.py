"""Broker adapter interface and shared order data types.

Defines the contract that every broker adapter must implement.
Concrete adapters (IBKRAdapter, SimulatedAdapter) live in separate modules.
The private repository never imports ib_insync directly — it only imports
from this ABC.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Direction of a trade."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order execution type."""

    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderStatus(str, Enum):
    """Lifecycle state of an order."""

    PENDING = "PENDING"        # created locally, not yet submitted
    SUBMITTED = "SUBMITTED"    # sent to broker
    FILLED = "FILLED"          # fully executed
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    DRY_RUN = "DRY_RUN"        # dry-run mode — never submitted


@dataclass
class Order:
    """Represents a single order to be submitted to a broker.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        side: BUY or SELL.
        quantity: Number of shares/contracts.
        order_type: MKT, LMT, STP, etc.
        limit_price: Required for LMT and STP LMT orders.
        stop_price: Required for STP and STP LMT orders.
        strategy_id: Identifier of the originating strategy (for logging).
        metadata: Arbitrary key-value pairs for strategy-specific context.
    """

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    strategy_id: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Set by the broker adapter after submission
    broker_order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    fill_price: float | None = None

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}.")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type} order requires a limit_price.")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type} order requires a stop_price.")


class BrokerAdapter(ABC):
    """Abstract interface for all broker adapters.

    Implementations must be safe to share across multiple StrategyRunner
    instances running in separate threads (i.e. use locking where necessary).
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker.

        Raises:
            ConnectionError: If the connection cannot be established.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully close the broker connection."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the adapter currently has an active connection."""

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker.

        Mutates ``order`` in place (sets broker_order_id, status, submitted_at)
        and returns it.

        Args:
            order: The order to submit.

        Returns:
            The same Order object with updated status fields.

        Raises:
            RuntimeError: On broker-side rejection or connectivity failure.
        """

    @abstractmethod
    def cancel_order(self, order: Order) -> None:
        """Request cancellation of an open order.

        Args:
            order: An order with a valid broker_order_id.

        Raises:
            ValueError: If the order has no broker_order_id.
            RuntimeError: If the cancellation request fails.
        """

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        """Return current open positions.

        Returns:
            Dict mapping symbol → net quantity (positive = long, negative = short).
        """

    @abstractmethod
    def get_account_value(self) -> float:
        """Return total account value in account currency.

        Returns:
            Net liquidation value of the account.
        """

    @abstractmethod
    def get_open_orders(self) -> list[Order]:
        """Return all currently open (unfinished) orders.

        Returns:
            List of Order objects with SUBMITTED or PARTIALLY_FILLED status.
        """
