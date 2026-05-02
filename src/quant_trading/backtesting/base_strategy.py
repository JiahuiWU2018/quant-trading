"""Abstract base class for backtesting strategies.

This is a thin wrapper around backtrader.Strategy that defines the interface
concrete strategies (in the private repo) must implement.
"""

import logging
from abc import abstractmethod

import backtrader as bt

logger = logging.getLogger(__name__)


class BaseStrategy(bt.Strategy):
    """Abstract strategy interface for Backtrader.

    Concrete strategies must implement `next()` and may override lifecycle hooks.
    The private repository supplies actual strategy logic; this ABC defines
    the contract.

    Attributes:
        params: Backtrader params tuple. Override in subclasses.

    Example (in private repo):
        class MyStrategy(BaseStrategy):
            params = (("fast_ma", 10), ("slow_ma", 50))

            def __init__(self):
                self.ma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast_ma)
                self.ma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow_ma)

            def next(self):
                if self.ma_fast[0] > self.ma_slow[0]:
                    if not self.position:
                        self.buy()
                elif self.ma_fast[0] < self.ma_slow[0]:
                    if self.position:
                        self.close()
    """

    @abstractmethod
    def next(self) -> None:
        """Core strategy logic called on each bar.

        Must be implemented by subclasses. This is the main Backtrader hook
        where order decisions are made.

        Typical pattern:
            - Read current bar data via self.data.close[0], etc.
            - Check indicators and signals
            - Call self.buy(), self.sell(), self.close() as needed

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        ...

    def log(self, txt: str, dt=None) -> None:
        """Utility logger with date prefix.

        Args:
            txt: Message to log.
            dt: Optional datetime; defaults to current bar datetime.
        """
        dt = dt or self.datas[0].datetime.date(0)
        logger.info(f"[{dt.isoformat()}] {txt}")

    def notify_order(self, order: bt.Order) -> None:
        """Called when an order status changes.

        Override in subclass if you need custom order tracking.
        Default implementation logs order completion/rejection.

        Args:
            order: Backtrader order object.
        """
        if order.status in [order.Submitted, order.Accepted]:
            return  # order is in flight

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED: price={order.executed.price:.2f}, "
                    f"cost={order.executed.value:.2f}, comm={order.executed.comm:.2f}"
                )
            elif order.issell():
                self.log(
                    f"SELL EXECUTED: price={order.executed.price:.2f}, "
                    f"cost={order.executed.value:.2f}, comm={order.executed.comm:.2f}"
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER FAILED: status={order.getstatusname()}")

    def notify_trade(self, trade: bt.Trade) -> None:
        """Called when a trade (round-trip position) is closed.

        Override in subclass if you need custom trade tracking.
        Default implementation logs PnL.

        Args:
            trade: Backtrader trade object.
        """
        if not trade.isclosed:
            return

        self.log(f"TRADE CLOSED: PnL (gross)={trade.pnl:.2f}, PnL (net)={trade.pnlcomm:.2f}")
