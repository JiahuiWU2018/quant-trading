"""Core risk and performance metrics.

Provides functions to compute annualized return, volatility, Sharpe ratio,
Sortino ratio, maximum drawdown, and drawdown duration from a return series
or equity curve.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized return from a return series.

    Args:
        returns: Series of period returns (e.g., daily log or simple returns).
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly).

    Returns:
        Annualized return as a decimal (e.g., 0.15 = 15%).

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compute annualized volatility (standard deviation of returns).

    Args:
        returns: Series of period returns.
        periods_per_year: Number of periods in a year.

    Returns:
        Annualized volatility as a decimal (e.g., 0.20 = 20%).

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Series of period returns.
        risk_free_rate: Annual risk-free rate as a decimal (e.g., 0.02 = 2%).
        periods_per_year: Number of periods in a year.

    Returns:
        Sharpe ratio (dimensionless).

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    excess = returns - (risk_free_rate / periods_per_year)
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute annualized Sortino ratio (downside deviation only).

    Args:
        returns: Series of period returns.
        risk_free_rate: Annual risk-free rate as a decimal.
        periods_per_year: Number of periods in a year.

    Returns:
        Sortino ratio (dimensionless).

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    excess = returns - (risk_free_rate / periods_per_year)
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / downside.std()


def max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """Compute maximum drawdown from an equity curve.

    Args:
        equity_curve: Series of portfolio values indexed by date.

    Returns:
        Tuple of (max_dd, peak_date, trough_date):
            max_dd: Maximum drawdown as a decimal (e.g., -0.25 = -25%).
            peak_date: Date of the peak before the drawdown.
            trough_date: Date of the trough.

    Raises:
        ValueError: If equity_curve is empty.
    """
    if equity_curve.empty:
        raise ValueError("equity_curve Series is empty.")

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax

    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = equity_curve[:trough_date].idxmax()

    return max_dd, peak_date, trough_date


def drawdown_duration(equity_curve: pd.Series) -> pd.Timedelta:
    """Compute the longest drawdown duration (peak to recovery).

    Args:
        equity_curve: Series of portfolio values indexed by date.

    Returns:
        Longest drawdown duration as a Timedelta.

    Raises:
        ValueError: If equity_curve is empty.
    """
    if equity_curve.empty:
        raise ValueError("equity_curve Series is empty.")

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax

    # Find all underwater periods
    underwater = drawdown < 0
    if not underwater.any():
        return pd.Timedelta(0)

    # Group consecutive underwater periods
    groups = (underwater != underwater.shift()).cumsum()
    durations = equity_curve.groupby(groups).apply(
        lambda x: x.index[-1] - x.index[0] if len(x) > 1 else pd.Timedelta(0)
    )
    return durations.max()


def compute_metrics(
    returns: pd.Series,
    equity_curve: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Compute a full suite of performance metrics.

    Args:
        returns: Series of period returns.
        equity_curve: Optional equity curve; if None, constructed from returns.
        risk_free_rate: Annual risk-free rate as a decimal.
        periods_per_year: Number of periods in a year.

    Returns:
        Dictionary with keys:
            - ann_return: Annualized return
            - ann_volatility: Annualized volatility
            - sharpe: Sharpe ratio
            - sortino: Sortino ratio
            - max_dd: Maximum drawdown
            - max_dd_duration: Longest drawdown duration (days)

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")

    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()

    max_dd_val, _, _ = max_drawdown(equity_curve)
    max_dd_dur = drawdown_duration(equity_curve)

    return {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_dd": max_dd_val,
        "max_dd_duration_days": max_dd_dur.days,
    }
