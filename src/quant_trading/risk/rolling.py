"""Rolling risk metrics.

All functions accept a pd.Series or pd.DataFrame of returns and return rolling
statistics as a pd.Series, using a fixed lookback window in number of periods.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rolling_volatility(
    returns: pd.Series,
    window: int = 63,
    periods_per_year: int = 252,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute rolling annualized volatility.

    Args:
        returns: Series of period returns.
        window: Lookback window in number of periods (default 63 approx 1 quarter).
        periods_per_year: Number of periods per year for annualization.
        min_periods: Minimum observations required; defaults to window.

    Returns:
        Series of rolling annualized volatility, NaN where insufficient data.

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    mp = min_periods if min_periods is not None else window
    return returns.rolling(window, min_periods=mp).std() * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute rolling annualized Sharpe ratio.

    Args:
        returns: Series of period returns.
        window: Lookback window in number of periods.
        risk_free_rate: Annual risk-free rate as a decimal.
        periods_per_year: Number of periods per year.
        min_periods: Minimum observations required; defaults to window.

    Returns:
        Series of rolling Sharpe ratio, NaN where insufficient data.

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    mp = min_periods if min_periods is not None else window
    excess = returns - (risk_free_rate / periods_per_year)

    def _sharpe(x: np.ndarray) -> float:
        std = x.std()
        return float(np.sqrt(periods_per_year) * x.mean() / std) if std > 0 else np.nan

    return excess.rolling(window, min_periods=mp).apply(_sharpe, raw=True)


def rolling_sortino(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute rolling annualized Sortino ratio.

    Args:
        returns: Series of period returns.
        window: Lookback window in number of periods.
        risk_free_rate: Annual risk-free rate as a decimal.
        periods_per_year: Number of periods per year.
        min_periods: Minimum observations required; defaults to window.

    Returns:
        Series of rolling Sortino ratio, NaN where insufficient data.

    Raises:
        ValueError: If returns is empty.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    mp = min_periods if min_periods is not None else window
    excess = returns - (risk_free_rate / periods_per_year)

    def _sortino(x: np.ndarray) -> float:
        downside = x[x < 0]
        if len(downside) == 0:
            return np.nan
        std_down = downside.std()
        return (
            float(np.sqrt(periods_per_year) * x.mean() / std_down)
            if std_down > 0
            else np.nan
        )

    return excess.rolling(window, min_periods=mp).apply(_sortino, raw=True)


def rolling_beta(
    returns: pd.Series,
    benchmark: pd.Series,
    window: int = 63,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute rolling beta relative to a benchmark.

    Beta = cov(returns, benchmark) / var(benchmark).

    Args:
        returns: Series of asset/strategy returns.
        benchmark: Series of benchmark returns (must share index with returns).
        window: Lookback window in number of periods.
        min_periods: Minimum observations required; defaults to window.

    Returns:
        Series of rolling beta indexed by date, NaN where insufficient data.

    Raises:
        ValueError: If returns or benchmark is empty, or they share no dates.
    """
    if returns.empty or benchmark.empty:
        raise ValueError("returns and benchmark must not be empty.")
    mp = min_periods if min_periods is not None else window
    aligned = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    if aligned.empty:
        raise ValueError("returns and benchmark have no overlapping observations.")

    r_arr = aligned["r"].values
    b_arr = aligned["b"].values
    betas: list[float] = []
    for i in range(len(aligned)):
        start = max(0, i - window + 1)
        if (i - start + 1) < mp:
            betas.append(np.nan)
        else:
            r_chunk = r_arr[start : i + 1]
            b_chunk = b_arr[start : i + 1]
            var_b = float(b_chunk.var())
            if var_b == 0:
                betas.append(np.nan)
            else:
                cov = float(np.cov(r_chunk, b_chunk)[0, 1])
                betas.append(cov / var_b)
    return pd.Series(betas, index=aligned.index, name="beta")


def rolling_var(
    returns: pd.Series,
    window: int = 63,
    confidence: float = 0.95,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute rolling historical VaR (expressed as a positive loss magnitude).

    Args:
        returns: Series of period returns.
        window: Lookback window in number of periods.
        confidence: Confidence level (e.g., 0.95).
        min_periods: Minimum observations required; defaults to window.

    Returns:
        Series of rolling VaR (positive), NaN where insufficient data.

    Raises:
        ValueError: If returns is empty or confidence is out of (0, 1).
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    mp = min_periods if min_periods is not None else window
    alpha = 1 - confidence

    def _var(x: np.ndarray) -> float:
        return float(-np.quantile(x, alpha))

    return returns.rolling(window, min_periods=mp).apply(_var, raw=True)


def rolling_mean_correlation(
    returns: pd.DataFrame,
    window: int = 63,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Compute rolling mean pairwise correlation across all asset pairs.

    Useful for monitoring portfolio diversification through time.

    Args:
        returns: DataFrame of asset returns (index = dates, columns = assets).
        window: Lookback window in number of periods.
        min_periods: Minimum observations required; defaults to window.

    Returns:
        Series of rolling mean pairwise correlation indexed by date.

    Raises:
        ValueError: If fewer than 2 asset columns are provided, or returns is empty.
    """
    if returns.empty:
        raise ValueError("returns DataFrame is empty.")
    if returns.shape[1] < 2:
        raise ValueError("returns must have at least 2 asset columns.")
    mp = min_periods if min_periods is not None else window

    clean = returns.dropna(how="all")
    result: list[float] = []
    for i in range(len(clean)):
        start = max(0, i - window + 1)
        chunk = clean.iloc[start : i + 1].dropna()
        if len(chunk) < mp:
            result.append(np.nan)
        else:
            corr_matrix = chunk.corr().values
            n = corr_matrix.shape[0]
            upper = corr_matrix[np.triu_indices(n, k=1)]
            result.append(float(np.nanmean(upper)))
    return pd.Series(result, index=clean.index, name="mean_pairwise_correlation")
