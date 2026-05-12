"""StrategyRunner — polling execution loop for live/paper trading.

Drives a single strategy forward bar-by-bar against a live broker.
Multiple runners can share one IBKRAdapter (and therefore one IB() instance).

Design principles:
- One runner per strategy (isolated failure domains, independent intervals)
- Sync API — ib_async's event loop is managed internally
- Graceful shutdown via stop() or KeyboardInterrupt
- DRY_RUN safety is enforced by SafetyChecker inside OrderManager

The strategy is expected to implement `on_bar()` in the private repository.
This module only defines the runner shell and the interface it calls.
"""

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import pandas as pd

from quant_trading.execution.base_adapter import BrokerAdapter, Order
from quant_trading.execution.order_manager import OrderManager
from quant_trading.execution.safety import SafetyChecker

logger = logging.getLogger(__name__)


@runtime_checkable
class LiveStrategy(Protocol):
    """Protocol that live strategies must implement.

    Implementations live in the private repository.
    This protocol is defined here so the runner can type-check strategies
    without depending on any private-repo classes.
    """

    strategy_id: str

    def on_bar(self, data: pd.DataFrame) -> list[Order]:
        """Called by StrategyRunner on each new bar.

        Args:
            data: Latest OHLCV bars for all symbols in this strategy's universe.
                  DatetimeIndex (UTC), columns: open, high, low, close, volume.

        Returns:
            List of Orders to submit. Return [] for no action.
        """
        ...


class StrategyRunner:
    """Polling execution loop for a single live strategy.

    Fetches the latest bar(s) at a configurable interval, calls the
    strategy's on_bar() hook, and submits any resulting orders via
    OrderManager.

    Args:
        strategy: A live strategy instance (implements LiveStrategy protocol).
        adapter: BrokerAdapter connected to the broker.
        data_fetcher: Callable that returns the latest OHLCV DataFrame.
                      Signature: (symbol: str) -> pd.DataFrame.
                      Typically wraps IBKRConnector.fetch_price_history().
        symbols: List of ticker symbols to fetch on each bar.
        interval_seconds: Sleep time between bar polls. Use 86400 for daily,
                          3600 for hourly, 60 for minute-by-minute.
        order_manager: Optional pre-built OrderManager. If None, a new one
                       is created wrapping the provided adapter.

    Example:
        # In private repo:
        from ib_async import IB
        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        from quant_trading.data.apis.ibkr_connector import IBKRConnector
        from quant_trading.execution.runner import StrategyRunner

        ib = IB()
        adapter = IBKRAdapter(ib=ib)
        connector = IBKRConnector(ib=ib)
        adapter.connect()

        runner = StrategyRunner(
            strategy=MyStrategy(),
            adapter=adapter,
            data_fetcher=lambda sym: connector.fetch_price_history(sym, freq="1d"),
            symbols=["AAPL", "MSFT"],
            interval_seconds=86400,
        )
        runner.start()   # blocks until runner.stop() is called
    """

    def __init__(
        self,
        strategy: LiveStrategy,
        adapter: BrokerAdapter,
        data_fetcher,
        symbols: list[str],
        interval_seconds: int = 86400,
        order_manager: OrderManager | None = None,
    ) -> None:
        if not isinstance(strategy, LiveStrategy):
            raise TypeError(
                f"{strategy!r} does not implement the LiveStrategy protocol "
                "(requires: strategy_id attribute and on_bar() method)."
            )
        self._strategy = strategy
        self._adapter = adapter
        self._data_fetcher = data_fetcher
        self._symbols = symbols
        self._interval = interval_seconds
        self._order_manager = order_manager or OrderManager(adapter)
        self._stop_event = threading.Event()
        self._running = False

        logger.info(
            "StrategyRunner created: strategy=%s symbols=%s interval=%ds",
            strategy.strategy_id,
            symbols,
            interval_seconds,
        )

    def start(self) -> None:
        """Start the polling loop. Blocks until stop() is called.

        Catches and logs all exceptions from the strategy so a single bad
        bar does not terminate the runner.
        """
        self._running = True
        self._stop_event.clear()
        logger.info(
            "StrategyRunner starting: strategy=%s", self._strategy.strategy_id
        )

        try:
            while not self._stop_event.is_set():
                loop_start = time.monotonic()
                self._tick()
                elapsed = time.monotonic() - loop_start
                sleep_time = max(0.0, self._interval - elapsed)
                logger.debug(
                    "StrategyRunner sleeping %.1fs (strategy=%s)",
                    sleep_time,
                    self._strategy.strategy_id,
                )
                self._stop_event.wait(timeout=sleep_time)
        except KeyboardInterrupt:
            logger.info("StrategyRunner interrupted by user.")
        finally:
            self._running = False
            logger.info("StrategyRunner stopped: strategy=%s", self._strategy.strategy_id)

    def stop(self) -> None:
        """Signal the runner to stop after the current tick completes."""
        logger.info("StrategyRunner stop requested: strategy=%s", self._strategy.strategy_id)
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """True if the runner loop is currently active."""
        return self._running

    # ------------------------------------------------------------------
    # Internal tick
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Fetch latest data, call strategy, submit any resulting orders."""
        try:
            data = self._fetch_all_symbols()
            if data.empty:
                logger.warning(
                    "StrategyRunner: no data returned for symbols=%s — skipping tick.",
                    self._symbols,
                )
                return

            positions = self._adapter.get_positions()
            account_value = self._adapter.get_account_value()

            orders = self._strategy.on_bar(data)

            if not orders:
                logger.debug(
                    "StrategyRunner: no orders from strategy=%s at %s",
                    self._strategy.strategy_id,
                    datetime.now(tz=timezone.utc).isoformat(),
                )
                return

            for order in orders:
                # Best-effort current price for notional checks
                current_price: float | None = None
                if order.symbol in data.columns:
                    current_price = float(data["close"].iloc[-1])
                elif "close" in data.columns:
                    current_price = float(data["close"].iloc[-1])

                try:
                    self._order_manager.submit(
                        order,
                        positions=positions,
                        account_value=account_value,
                        current_price=current_price,
                    )
                except Exception as exc:
                    # Log but do not crash the runner on a single order failure
                    logger.error(
                        "StrategyRunner: order submission failed for %s: %s",
                        order.symbol,
                        exc,
                    )

        except Exception as exc:
            logger.error(
                "StrategyRunner: unhandled exception in tick (strategy=%s): %s",
                self._strategy.strategy_id,
                exc,
                exc_info=True,
            )

    def _fetch_all_symbols(self) -> pd.DataFrame:
        """Fetch latest bars for all symbols and stack into one DataFrame.

        Returns a stacked DataFrame with a MultiIndex (datetime, symbol)
        if multiple symbols are present, or a plain DataFrame for one symbol.
        """
        frames = {}
        for symbol in self._symbols:
            try:
                df = self._data_fetcher(symbol)
                frames[symbol] = df
            except Exception as exc:
                logger.error("StrategyRunner: failed to fetch %s: %s", symbol, exc)

        if not frames:
            return pd.DataFrame()

        if len(frames) == 1:
            return next(iter(frames.values()))

        # Multi-symbol: return stacked DataFrame with symbol level
        return pd.concat(frames, axis=0, names=["symbol", "datetime"])


def start_all(runners: list[StrategyRunner]) -> list[Future]:
    """Start multiple StrategyRunners in parallel using a thread pool.

    Each runner blocks in its own thread. Returns the list of Future objects
    so the caller can wait on them or handle exceptions.

    Args:
        runners: List of configured StrategyRunner instances.

    Returns:
        List of concurrent.futures.Future objects, one per runner.

    Example:
        # In private repo:
        futures = start_all([momentum_runner, mean_reversion_runner])
        # Block until all runners finish (e.g. after stop() is called)
        for f in futures:
            f.result()
    """
    executor = ThreadPoolExecutor(
        max_workers=len(runners),
        thread_name_prefix="strategy_runner",
    )
    futures = [executor.submit(runner.start) for runner in runners]
    logger.info("start_all: launched %d strategy runner(s).", len(runners))
    return futures
