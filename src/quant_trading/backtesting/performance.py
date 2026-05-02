"""Backtest performance reporting and trade log export.

Extracts results from a Backtrader run and produces DataFrames for further analysis.
"""

import logging
from typing import List

import backtrader as bt
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceAnalyzer(bt.Analyzer):
    """Custom Backtrader analyzer to capture equity curve and trade log.

    Stores portfolio value at each bar and trade-level details.
    """

    def __init__(self):
        self.equity_curve: List[tuple] = []
        self.trades: List[dict] = []

    def notify_cashvalue(self, cash: float, value: float) -> None:
        """Called on each bar with current cash and portfolio value."""
        dt = self.strategy.datetime.date(0)
        self.equity_curve.append((dt, value))

    def notify_trade(self, trade: bt.Trade) -> None:
        """Called when a trade is closed."""
        if not trade.isclosed:
            return
        self.trades.append(
            {
                "open_date": bt.num2date(trade.dtopen).date(),
                "close_date": bt.num2date(trade.dtclose).date(),
                "pnl_gross": trade.pnl,
                "pnl_net": trade.pnlcomm,
                "size": trade.size,
            }
        )

    def get_analysis(self) -> dict:
        """Return analysis results as a dict of DataFrames."""
        equity_df = pd.DataFrame(self.equity_curve, columns=["date", "value"]).set_index("date")
        trades_df = pd.DataFrame(self.trades)
        return {"equity_curve": equity_df, "trades": trades_df}


def extract_results(cerebro: bt.Cerebro, strat_instance) -> dict:
    """Extract equity curve and trade log from a Backtrader run.

    Args:
        cerebro: The Cerebro instance after run().
        strat_instance: The strategy instance returned by cerebro.run()[0].

    Returns:
        Dict with keys:
            - equity_curve: DataFrame [date, value]
            - trades: DataFrame [open_date, close_date, pnl_gross, pnl_net, size]
            - final_value: Final portfolio value

    Raises:
        AttributeError: If PerformanceAnalyzer was not added to cerebro.
    """
    if not hasattr(strat_instance, "analyzers") or not hasattr(
        strat_instance.analyzers, "performance"
    ):
        raise AttributeError(
            "PerformanceAnalyzer not found. Did you add it with cerebro.addanalyzer()?"
        )

    analysis = strat_instance.analyzers.performance.get_analysis()
    final_value = cerebro.broker.getvalue()

    return {
        "equity_curve": analysis["equity_curve"],
        "trades": analysis["trades"],
        "final_value": final_value,
    }
