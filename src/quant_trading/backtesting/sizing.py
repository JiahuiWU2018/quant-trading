"""Position sizing utilities.

Provides helpers for computing position sizes based on various rules:
fixed fractional, volatility targeting, equal weight, etc.

Pure functions (fixed_fractional_size, volatility_targeted_size, equal_weight_size)
are suitable for pre-trade analysis and signal generation.

Backtrader-compatible Sizer classes (FixedFractionalSizer, VolatilityTargetedSizer)
can be passed directly to BacktestEngine.set_sizer() or cerebro.addsizer().
"""

import logging

import backtrader as bt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fixed_fractional_size(
    portfolio_value: float,
    price: float,
    fraction: float = 0.1,
) -> int:
    """Compute position size as a fixed fraction of portfolio value.

    Args:
        portfolio_value: Current portfolio value.
        price: Current price of the asset.
        fraction: Fraction of portfolio to allocate (e.g., 0.1 = 10%).

    Returns:
        Number of shares to buy (integer).

    Raises:
        ValueError: If price is zero or negative.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    notional = portfolio_value * fraction
    return int(notional / price)


def volatility_targeted_size(
    portfolio_value: float,
    price: float,
    returns: pd.Series,
    target_vol: float = 0.10,
    periods_per_year: int = 252,
) -> int:
    """Compute position size to target a specific portfolio volatility.

    Args:
        portfolio_value: Current portfolio value.
        price: Current price of the asset.
        returns: Recent return series for volatility estimation.
        target_vol: Target annualized volatility (e.g., 0.10 = 10%).
        periods_per_year: Number of periods in a year.

    Returns:
        Number of shares to buy (integer).

    Raises:
        ValueError: If price is zero, negative, or returns is empty.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    if returns.empty:
        raise ValueError("returns Series is empty; cannot estimate volatility.")

    realized_vol = returns.std() * np.sqrt(periods_per_year)
    if realized_vol == 0:
        return 0

    # Scale position so that position_vol * position_fraction = target_vol
    fraction = target_vol / realized_vol
    fraction = min(fraction, 1.0)  # cap at 100%

    return fixed_fractional_size(portfolio_value, price, fraction)


def equal_weight_size(
    portfolio_value: float,
    prices: dict[str, float],
) -> dict[str, int]:
    """Compute equal-weight position sizes across multiple assets.

    Args:
        portfolio_value: Total portfolio value.
        prices: Dict mapping symbol → current price.

    Returns:
        Dict mapping symbol → number of shares.

    Raises:
        ValueError: If any price is zero or negative.
    """
    n = len(prices)
    if n == 0:
        return {}

    per_asset = portfolio_value / n
    sizes = {}
    for symbol, price in prices.items():
        if price <= 0:
            raise ValueError(f"price for {symbol} must be positive, got {price}")
        sizes[symbol] = int(per_asset / price)

    return sizes


# ---------------------------------------------------------------------------
# Backtrader-compatible Sizer classes
# ---------------------------------------------------------------------------


class FixedFractionalSizer(bt.Sizer):
    """Backtrader sizer that allocates a fixed fraction of portfolio value.

    Args:
        fraction: Fraction of portfolio to allocate per trade (e.g., 0.1 = 10%).

    Example:
        engine.set_sizer(FixedFractionalSizer, fraction=0.10)
    """

    params = (("fraction", 0.10),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        """Compute share count for the current bar.

        Args:
            comminfo: Commission information object.
            cash: Current available cash.
            data: The data feed for this trade.
            isbuy: True for a buy order, False for a sell order.

        Returns:
            Integer number of shares to trade.
        """
        portfolio_value = self.broker.getvalue()
        price = data.close[0]
        if price <= 0:
            return 0
        size = fixed_fractional_size(portfolio_value, price, self.p.fraction)
        logger.debug(
            "FixedFractionalSizer: value=%.2f price=%.4f fraction=%.3f → size=%d",
            portfolio_value,
            price,
            self.p.fraction,
            size,
        )
        return size


class VolatilityTargetedSizer(bt.Sizer):
    """Backtrader sizer that targets a specific annualized portfolio volatility.

    Uses the most recent `lookback` bars to estimate realized volatility,
    then scales position size so the expected contribution matches `target_vol`.

    Args:
        target_vol: Target annualized volatility (e.g., 0.10 = 10%).
        lookback: Number of recent bars used to estimate volatility.
        periods_per_year: Trading periods per year for annualization.

    Example:
        engine.set_sizer(VolatilityTargetedSizer, target_vol=0.15, lookback=20)
    """

    params = (
        ("target_vol", 0.10),
        ("lookback", 20),
        ("periods_per_year", 252),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        """Compute share count targeting the configured volatility level.

        Args:
            comminfo: Commission information object.
            cash: Current available cash.
            data: The data feed for this trade.
            isbuy: True for a buy order, False for a sell order.

        Returns:
            Integer number of shares to trade.
        """
        portfolio_value = self.broker.getvalue()
        price = data.close[0]
        if price <= 0:
            return 0

        closes = pd.Series(data.close.get(size=self.p.lookback + 1))
        if len(closes) < 2:
            return 0
        returns = closes.pct_change().dropna()

        size = volatility_targeted_size(
            portfolio_value,
            price,
            returns,
            target_vol=self.p.target_vol,
            periods_per_year=self.p.periods_per_year,
        )
        logger.debug(
            "VolatilityTargetedSizer: value=%.2f price=%.4f target_vol=%.3f → size=%d",
            portfolio_value,
            price,
            self.p.target_vol,
            size,
        )
        return size
