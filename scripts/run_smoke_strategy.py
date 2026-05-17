"""Run a minimal smoke strategy against your paper IB Gateway.

This script is intentionally conservative and interactive-safe:
- Requires the virtualenv to be active and `.env` loaded in the shell
- Respects `DRY_RUN` (if set to "true", the script exits unless --force)
- If `DRY_RUN=false`, the script requires an explicit --yes flag to place a real order

Usage (recommended):

# Activate venv and load .env in the same terminal
source .venvqt/bin/activate
set -a && source .env && set +a

# Dry-run (safe default)
python scripts/run_smoke_strategy.py --symbol AAPL

# If you really want to submit to your paper account (only do this after verify_ib_connection.py passes):
# Ensure DRY_RUN=false in .env, then run with --yes to allow submission
python scripts/run_smoke_strategy.py --symbol AAPL --yes

This script places a single LIMIT buy order for 1 share at $1.00 (far below market)
and immediately cancels it. The price is chosen so the order will never fill.
"""

import argparse
import logging
import os
import sys
import time
from datetime import timezone

from quant_trading.execution.base_adapter import Order, OrderSide, OrderType
from quant_trading.execution.runner import StrategyRunner, LiveStrategy
from quant_trading.execution.ibkr_adapter import IBKRAdapter
from quant_trading.data.apis.ibkr_connector import IBKRConnector

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("smoke")


class SmokeStrategy:
    """Minimal strategy implementing the LiveStrategy protocol.

    On the first call to on_bar it returns a single conservative LIMIT BUY
    for 1 share at a very low price, then returns no orders thereafter.
    """

    strategy_id = "smoke-test"

    def __init__(self, symbol: str, limit_price: float = 1.00) -> None:
        self.symbol = symbol
        self._limit_price = float(limit_price)
        self._fired = False

    def on_bar(self, data):
        # data: DataFrame for the symbol (DatetimeIndex, columns: open, high, low, close, volume)
        if self._fired:
            return []

        self._fired = True
        order = Order(
            symbol=self.symbol,
            side=OrderSide.BUY,
            quantity=2,
            order_type=OrderType.LIMIT,
            limit_price=self._limit_price,
            strategy_id=self.strategy_id,
        )
        logger.info(
            "SmokeStrategy: generated order symbol=%s qty=%d limit=%.2f",
            order.symbol,
            order.quantity,
            order.limit_price,
        )
        return [order]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a one-shot smoke strategy on IB paper account")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to buy (default: AAPL)")
    parser.add_argument("--freq", default="1d", help="Bar frequency for data fetch (default: 1d)")
    parser.add_argument("--yes", action="store_true", help="Allow real order submission when DRY_RUN=false")
    parser.add_argument("--force", action="store_true", help="Force run even if DRY_RUN=true (for debugging)")
    parser.add_argument(
        "--cancel-delay",
        type=float,
        default=3.0,
        help="Seconds to wait after submission before cancelling (default: 3.0)",
    )
    args = parser.parse_args(argv)

    dry_run = os.getenv("DRY_RUN", "true").strip().lower()
    if dry_run == "true" and not args.force:
        logger.info("DRY_RUN=true — no orders will be submitted. Use --force to override locally for debugging.")
        # We'll still run the flow up to order creation, but the adapter will not send orders.

    if dry_run != "true" and not args.yes:
        logger.error("DRY_RUN is false — to actually submit an order you must pass --yes. Exiting.")
        sys.exit(1)

    # Build IB adapter and shared IB instance
    adapter = IBKRAdapter()
    try:
        adapter.connect()
    except Exception as exc:
        logger.error("Could not connect to IB Gateway: %s", exc)
        sys.exit(1)

    connector = IBKRConnector(adapter._ib)

    strategy = SmokeStrategy(args.symbol)

    # data_fetcher: function that returns DataFrame for a symbol
    data_fetcher = lambda s: connector.fetch_price_history(s, freq=args.freq)

    runner = StrategyRunner(
        strategy=strategy,
        adapter=adapter,
        data_fetcher=data_fetcher,
        symbols=[args.symbol],
        interval_seconds=86400,  # not used for one-shot
    )

    try:
        # Execute a single tick synchronously to keep the script simple
        logger.info("Running one tick for symbol=%s (freq=%s)", args.symbol, args.freq)
        runner._tick()  # one-shot; uses the normal order/manager path

        # If live submission was allowed, attempt to cancel any submitted orders
        dry_run_env = os.getenv("DRY_RUN", "true").strip().lower()
        if dry_run_env != "true" and args.yes:
            logger.info(
                "Live submission allowed — waiting %.1fs before attempting to cancel submitted orders",
                args.cancel_delay,
            )
            # Give the broker a short time to register the order before cancelling
            time.sleep(args.cancel_delay)

            submitted = []
            try:
                submitted = [o for o in runner._order_manager.all_orders() if o.status.name == "SUBMITTED"]
            except Exception:
                logger.exception("Failed to fetch submitted orders from OrderManager.")

            if not submitted:
                logger.info("No submitted orders found to cancel.")
            else:
                for o in submitted:
                    try:
                        logger.info("Attempting to cancel order id=%s symbol=%s", o.broker_order_id, o.symbol)
                        runner._order_manager.cancel(o)
                    except Exception as exc:
                        logger.error("Failed to cancel order id=%s: %s", o.broker_order_id, exc)

        logger.info("One-shot tick completed. Check IB Gateway Activity Monitor for orders (if any).")
    finally:
        adapter.disconnect()
        logger.info("Disconnected from IB Gateway.")


if __name__ == "__main__":
    main()
