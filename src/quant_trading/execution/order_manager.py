"""Order lifecycle manager — submission, retry, and state tracking.

Sits between StrategyRunner and BrokerAdapter. Handles:
- Pre-trade safety checks via SafetyChecker
- Retry with exponential backoff on transient failures
- Order state tracking in memory (not persisted — Phase 5 can add a DB)
- Rate-limiting guard (configurable minimum gap between submissions)
"""

import logging
import time
from datetime import datetime, timezone
from threading import Lock

from quant_trading.execution.base_adapter import BrokerAdapter, Order, OrderStatus
from quant_trading.execution.safety import SafetyChecker, SafetyError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF = 2.0   # seconds; doubles each attempt
_DEFAULT_MIN_ORDER_GAP = 1.0   # minimum seconds between order submissions


class OrderManager:
    """Manages the full lifecycle of orders from creation to fill/cancel.

    Thread-safe: can be shared by multiple StrategyRunner instances.

    Args:
        adapter: The broker adapter to submit orders through.
        safety_checker: Pre-trade safety guard. Defaults to SafetyChecker().
        max_retries: Number of retry attempts on transient failures.
        retry_backoff: Initial backoff in seconds (doubles on each retry).
        min_order_gap: Minimum seconds between consecutive order submissions.

    Example:
        manager = OrderManager(adapter=IBKRAdapter())
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        manager.submit(order, positions={}, account_value=100_000)
    """

    def __init__(
        self,
        adapter: BrokerAdapter,
        safety_checker: SafetyChecker | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
        min_order_gap: float = _DEFAULT_MIN_ORDER_GAP,
    ) -> None:
        self._adapter = adapter
        self._safety = safety_checker or SafetyChecker()
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._min_order_gap = min_order_gap

        self._orders: dict[str, Order] = {}   # broker_order_id → Order
        self._lock = Lock()
        self._last_submission_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        order: Order,
        positions: dict[str, float],
        account_value: float,
        current_price: float | None = None,
    ) -> Order:
        """Run safety checks then submit an order with retry logic.

        Args:
            order: The order to submit. Modified in place.
            positions: Current open positions for safety checks.
            account_value: Account net liquidation value for safety checks.
            current_price: Latest price for notional safety check.

        Returns:
            The order with updated status/broker_order_id fields.

        Raises:
            SafetyError: If any pre-trade safety check fails.
            RuntimeError: If all retry attempts are exhausted.
        """
        # Safety checks — raises SafetyError on failure (including dry-run)
        self._safety.run_checks(order, positions, account_value, current_price)

        # Rate limiting
        self._enforce_rate_limit()

        # Submit with retry
        backoff = self._retry_backoff
        for attempt in range(1, self._max_retries + 1):
            try:
                submitted = self._adapter.submit_order(order)
                submitted.submitted_at = datetime.now(tz=timezone.utc)
                with self._lock:
                    if submitted.broker_order_id:
                        self._orders[submitted.broker_order_id] = submitted
                logger.info(
                    "Order submitted: id=%s symbol=%s side=%s qty=%d",
                    submitted.broker_order_id,
                    submitted.symbol,
                    submitted.side.value,
                    submitted.quantity,
                )
                self._last_submission_time = time.monotonic()
                return submitted
            except RuntimeError as exc:
                if attempt == self._max_retries:
                    logger.error(
                        "Order submission failed after %d attempts: %s", attempt, exc
                    )
                    raise
                logger.warning(
                    "Order submission attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= 2

        # Should never be reached
        raise RuntimeError("Unreachable: retry loop exhausted without return or raise.")

    def cancel(self, order: Order) -> None:
        """Cancel an open order.

        Args:
            order: An order with a valid broker_order_id.

        Raises:
            ValueError: If the order has no broker_order_id.
        """
        if order.broker_order_id is None:
            raise ValueError("Cannot cancel order without a broker_order_id.")
        self._adapter.cancel_order(order)
        order.status = OrderStatus.CANCELLED
        logger.info("Order cancelled: id=%s symbol=%s", order.broker_order_id, order.symbol)

    def get_order(self, broker_order_id: str) -> Order | None:
        """Look up a previously submitted order by its broker ID.

        Args:
            broker_order_id: The ID returned by the broker on submission.

        Returns:
            The Order if found, else None.
        """
        with self._lock:
            return self._orders.get(broker_order_id)

    def all_orders(self) -> list[Order]:
        """Return all orders submitted in this session (any status).

        Returns:
            List of Order objects, ordered by submission time.
        """
        with self._lock:
            return list(self._orders.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_rate_limit(self) -> None:
        """Block until the minimum inter-order gap has elapsed."""
        elapsed = time.monotonic() - self._last_submission_time
        gap = self._min_order_gap - elapsed
        if gap > 0:
            logger.debug("Rate limit: sleeping %.2fs before next submission.", gap)
            time.sleep(gap)
