"""Backtesting engine — wraps Backtrader Cerebro.

Provides a high-level interface for running backtests with sensible defaults.
The private repository supplies the strategy; this module wires everything together.
"""

import logging
from datetime import datetime
from typing import Type

import backtrader as bt
import pandas as pd

from quant_trading.backtesting.base_strategy import BaseStrategy
from quant_trading.backtesting.commission import set_commission, set_slippage
from quant_trading.backtesting.performance import PerformanceAnalyzer, extract_results
from quant_trading.backtesting.sizing import FixedFractionalSizer
from quant_trading.risk.metrics import compute_metrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """High-level backtesting engine wrapping Backtrader Cerebro.

    Args:
        initial_cash: Starting portfolio value.
        commission: Commission as a fraction (e.g., 0.001 = 0.1%).
        slippage_perc: Slippage as a percentage.
        slippage_fixed: Fixed slippage in price units.

    Example:
        engine = BacktestEngine(initial_cash=100_000, commission=0.001)
        engine.add_data(df, name="AAPL")
        engine.add_strategy(MyStrategy, fast_ma=10, slow_ma=50)
        results = engine.run()
        print(results["metrics"])
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,
        slippage_perc: float = 0.0,
        slippage_fixed: float = 0.0,
    ):
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.set_cash(initial_cash)
        set_commission(self.cerebro, commission=commission)
        set_slippage(self.cerebro, slippage_perc=slippage_perc, slippage_fixed=slippage_fixed)
        self.cerebro.addanalyzer(PerformanceAnalyzer, _name="performance")
        self.initial_cash = initial_cash
        logger.info(f"BacktestEngine initialized: cash={initial_cash}, comm={commission}")

    def set_sizer(self, sizer_class: type = FixedFractionalSizer, **kwargs) -> None:
        """Set the position sizer.

        Replaces Backtrader's default fixed-share sizer with a custom one.
        Call this before :meth:`run`. If not called, Backtrader's default
        sizer (1 share per order) is used.

        Args:
            sizer_class: A :class:`bt.Sizer` subclass, such as
                :class:`~quant_trading.backtesting.sizing.FixedFractionalSizer`
                or :class:`~quant_trading.backtesting.sizing.VolatilityTargetedSizer`.
            **kwargs: Parameters forwarded to the sizer constructor
                (e.g., ``fraction=0.05``).

        Example:
            from quant_trading.backtesting.sizing import FixedFractionalSizer

            engine = BacktestEngine()
            engine.set_sizer(FixedFractionalSizer, fraction=0.05)
        """
        if not (isinstance(sizer_class, type) and issubclass(sizer_class, bt.Sizer)):
            raise TypeError(f"{sizer_class} must be a subclass of bt.Sizer.")
        self.cerebro.addsizer(sizer_class, **kwargs)
        logger.info("Sizer set: %s %s", sizer_class.__name__, kwargs)

    def add_data(
        self,
        df: pd.DataFrame,
        name: str = "data",
        fromdate: datetime | None = None,
        todate: datetime | None = None,
    ) -> None:
        """Add a price DataFrame to the backtest.

        Args:
            df: OHLCV DataFrame with DatetimeIndex and columns
                [open, high, low, close, volume].
            name: Name for this data feed.
            fromdate: Optional start date filter.
            todate: Optional end date filter.

        Raises:
            ValueError: If df does not meet requirements.
        """
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")

        data_feed = bt.feeds.PandasData(
            dataname=df,
            fromdate=fromdate,
            todate=todate,
            openinterest=None,  # not used
        )
        self.cerebro.adddata(data_feed, name=name)
        logger.info(f"Data feed '{name}' added: {len(df)} rows")

    def add_strategy(self, strategy_class: Type[BaseStrategy], **kwargs) -> None:
        """Add a strategy to the backtest.

        Args:
            strategy_class: A subclass of BaseStrategy.
            **kwargs: Strategy parameters passed to the strategy constructor.

        Raises:
            TypeError: If strategy_class is not a BaseStrategy subclass.
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError(f"{strategy_class} must be a subclass of BaseStrategy.")
        self.cerebro.addstrategy(strategy_class, **kwargs)
        logger.info(f"Strategy {strategy_class.__name__} added with params: {kwargs}")

    def run(self) -> dict:
        """Run the backtest and return results.

        Returns:
            Dict with keys:
                - initial_value: Starting portfolio value
                - final_value: Ending portfolio value
                - equity_curve: DataFrame [date, value]
                - trades: DataFrame [open_date, close_date, pnl_gross, pnl_net, size]
                - metrics: Dict of performance metrics (Sharpe, drawdown, etc.)

        Raises:
            RuntimeError: If no strategy or data was added.
        """
        if len(self.cerebro.datas) == 0:
            raise RuntimeError("No data feeds added. Call add_data() before run().")
        if len(self.cerebro.strats) == 0:
            raise RuntimeError("No strategy added. Call add_strategy() before run().")

        logger.info("Starting backtest...")
        strats = self.cerebro.run()
        strat = strats[0]

        results = extract_results(self.cerebro, strat)
        results["initial_value"] = self.initial_cash

        # Compute returns and metrics
        equity_curve = results["equity_curve"]["value"]
        returns = equity_curve.pct_change().dropna()
        metrics = compute_metrics(returns, equity_curve)
        results["metrics"] = metrics

        logger.info(f"Backtest complete: final value={results['final_value']:.2f}")
        return results

    def plot(self) -> None:
        """Plot the backtest results using Backtrader's built-in plotter.

        Note: Requires matplotlib. The plot may be large and slow for long backtests.
        """
        self.cerebro.plot()
