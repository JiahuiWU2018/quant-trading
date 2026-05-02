"""Commission and slippage models for backtesting.

Backtrader has built-in commission schemes; this module provides convenience
wrappers and custom models.
"""

import logging

import backtrader as bt

logger = logging.getLogger(__name__)


def set_commission(
    cerebro: bt.Cerebro,
    commission: float = 0.001,
    commission_type: str = "percent",
) -> None:
    """Set a simple commission scheme on a Cerebro instance.

    Args:
        cerebro: Backtrader Cerebro instance.
        commission: Commission value (e.g., 0.001 = 0.1% for percent mode).
        commission_type: One of "percent" or "fixed".

    Raises:
        ValueError: If commission_type is invalid.
    """
    if commission_type == "percent":
        cerebro.broker.setcommission(commission=commission)
    elif commission_type == "fixed":
        cerebro.broker.setcommission(commission=commission, commtype=bt.CommInfoBase.COMM_FIXED)
    else:
        raise ValueError(f"Invalid commission_type: {commission_type}. Use 'percent' or 'fixed'.")
    logger.info(f"Commission set: {commission_type} = {commission}")


def set_slippage(
    cerebro: bt.Cerebro,
    slippage_perc: float = 0.0,
    slippage_fixed: float = 0.0,
) -> None:
    """Set slippage on a Cerebro instance.

    Note: Backtrader's slippage model is basic. For more sophisticated models,
    consider custom order execution logic in the strategy.

    Args:
        cerebro: Backtrader Cerebro instance.
        slippage_perc: Percentage slippage (e.g., 0.001 = 0.1%).
        slippage_fixed: Fixed slippage in price units.
    """
    cerebro.broker.set_slippage_perc(slippage_perc, slip_open=True, slip_limit=True)
    cerebro.broker.set_slippage_fixed(slippage_fixed, slip_open=True, slip_limit=True)
    logger.info(f"Slippage set: perc={slippage_perc}, fixed={slippage_fixed}")
