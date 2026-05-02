"""Safety checker — pre-trade kill-switch and limit enforcement.

All limits are read from environment variables so they can be changed
without touching code. The only code change needed to go live is setting
DRY_RUN=false in the environment.

Environment variables:
    DRY_RUN                  "true"/"false" — blocks all order submission (default: true)
    MAX_ORDER_NOTIONAL       Max USD notional per single order (default: no limit)
    MAX_POSITIONS            Max number of simultaneous open positions (default: no limit)
    MAX_POSITION_SIZE        Max shares held in any one symbol (default: no limit)
"""

import logging
import os

from quant_trading.execution.base_adapter import Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)

_UNSET = object()


def _env_float(key: str, default: float | None = None) -> float | None:
    """Read an optional positive-float environment variable."""
    val = os.getenv(key, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("Invalid value for %s=%r — ignoring limit.", key, val)
        return default


def _env_int(key: str, default: int | None = None) -> int | None:
    """Read an optional positive-int environment variable."""
    val = os.getenv(key, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Invalid value for %s=%r — ignoring limit.", key, val)
        return default


class SafetyError(RuntimeError):
    """Raised when a pre-trade safety check fails."""


class SafetyChecker:
    """Enforces safety limits before any order reaches the broker.

    Instantiate once and reuse across strategies. All limits are re-read
    from environment variables on each call to :meth:`run_checks` so
    changes take effect without restarting the process.

    Example:
        checker = SafetyChecker()
        checker.run_checks(order, positions={"AAPL": 50}, account_value=100_000)
    """

    def run_checks(
        self,
        order: Order,
        positions: dict[str, float],
        account_value: float,
        current_price: float | None = None,
    ) -> None:
        """Run all pre-trade safety checks.

        Args:
            order: The order about to be submitted.
            positions: Current open positions (symbol → qty).
            account_value: Current account net liquidation value.
            current_price: Last price for notional calculation.
                           Required only when MAX_ORDER_NOTIONAL is set.

        Raises:
            SafetyError: If any check fails. The error message describes
                         which check failed and why.
        """
        self._check_dry_run(order)
        self._check_max_order_notional(order, current_price)
        self._check_max_positions(order, positions)
        self._check_max_position_size(order, positions)

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_dry_run(self, order: Order) -> None:
        dry_run = os.getenv("DRY_RUN", "true").strip().lower()
        if dry_run == "true":
            order.status = OrderStatus.DRY_RUN
            logger.info(
                "[DRY_RUN] Order blocked — symbol=%s side=%s qty=%d",
                order.symbol,
                order.side.value,
                order.quantity,
            )
            raise SafetyError(
                f"DRY_RUN=true: order for {order.symbol} was not submitted. "
                "Set DRY_RUN=false to enable live order submission."
            )

    def _check_max_order_notional(self, order: Order, current_price: float | None) -> None:
        limit = _env_float("MAX_ORDER_NOTIONAL")
        if limit is None:
            return
        if current_price is None:
            logger.warning(
                "MAX_ORDER_NOTIONAL=%s set but current_price not provided — skipping notional check.",
                limit,
            )
            return
        notional = current_price * order.quantity
        if notional > limit:
            raise SafetyError(
                f"Order notional {notional:.2f} exceeds MAX_ORDER_NOTIONAL={limit:.2f} "
                f"for {order.symbol} qty={order.quantity} @ {current_price:.4f}."
            )

    def _check_max_positions(self, order: Order, positions: dict[str, float]) -> None:
        limit = _env_int("MAX_POSITIONS")
        if limit is None:
            return
        if order.side == OrderSide.BUY:
            open_count = sum(1 for qty in positions.values() if qty != 0)
            already_held = order.symbol in positions and positions[order.symbol] != 0
            if not already_held and open_count >= limit:
                raise SafetyError(
                    f"Opening {order.symbol} would exceed MAX_POSITIONS={limit} "
                    f"(currently {open_count} open positions)."
                )

    def _check_max_position_size(self, order: Order, positions: dict[str, float]) -> None:
        limit = _env_int("MAX_POSITION_SIZE")
        if limit is None:
            return
        current_qty = positions.get(order.symbol, 0.0)
        if order.side == OrderSide.BUY:
            projected = current_qty + order.quantity
            if projected > limit:
                raise SafetyError(
                    f"Buying {order.quantity} of {order.symbol} would result in position "
                    f"{projected} which exceeds MAX_POSITION_SIZE={limit}."
                )
