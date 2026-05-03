"""Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations.

Supports three estimation methods:
- Historical simulation (non-parametric)
- Parametric normal
- Parametric Student-t (fat tails)

All results are expressed as positive loss magnitudes (e.g., 0.05 = 5% loss).
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class VaRResult:
    """Container for VaR/CVaR results.

    Attributes:
        confidence: Confidence level used (e.g., 0.95).
        method: Estimation method ('historical', 'parametric_normal', 'parametric_t').
        var: Value at Risk (positive loss magnitude).
        cvar: Conditional Value at Risk / Expected Shortfall (positive loss magnitude).
        n_observations: Number of return observations used.
    """

    confidence: float
    method: str
    var: float
    cvar: float
    n_observations: int


def var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
) -> VaRResult:
    """Compute VaR and CVaR via historical simulation.

    Args:
        returns: Series of period returns (e.g., daily simple returns).
        confidence: Confidence level, e.g., 0.95 for 95% VaR.

    Returns:
        VaRResult with method='historical'.

    Raises:
        ValueError: If returns is empty or confidence is out of (0, 1).
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    sorted_returns = np.sort(returns.dropna().values)
    alpha = 1 - confidence
    # VaR: the alpha-quantile loss (negated so it is a positive number)
    var_val = float(-np.quantile(sorted_returns, alpha))
    # CVaR: mean of returns below the VaR threshold
    tail = sorted_returns[sorted_returns <= -var_val]
    cvar_val = float(-tail.mean()) if len(tail) > 0 else var_val

    return VaRResult(
        confidence=confidence,
        method="historical",
        var=var_val,
        cvar=cvar_val,
        n_observations=len(sorted_returns),
    )


def var_parametric_normal(
    returns: pd.Series,
    confidence: float = 0.95,
) -> VaRResult:
    """Compute VaR and CVaR assuming normally distributed returns.

    Args:
        returns: Series of period returns.
        confidence: Confidence level, e.g., 0.95 for 95% VaR.

    Returns:
        VaRResult with method='parametric_normal'.

    Raises:
        ValueError: If returns is empty or confidence is out of (0, 1).
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    clean = returns.dropna()
    mu = float(clean.mean())
    sigma = float(clean.std())
    alpha = 1 - confidence

    z = stats.norm.ppf(alpha)
    var_val = float(-(mu + z * sigma))
    # CVaR for normal: mu + sigma * phi(z_alpha) / alpha
    cvar_val = float(-(mu - sigma * stats.norm.pdf(z) / alpha))

    return VaRResult(
        confidence=confidence,
        method="parametric_normal",
        var=var_val,
        cvar=cvar_val,
        n_observations=len(clean),
    )


def var_parametric_t(
    returns: pd.Series,
    confidence: float = 0.95,
) -> VaRResult:
    """Compute VaR and CVaR by fitting a Student-t distribution to returns.

    Fits degrees-of-freedom, location, and scale via MLE, which better captures
    fat tails common in financial return series.

    Args:
        returns: Series of period returns.
        confidence: Confidence level, e.g., 0.95 for 95% VaR.

    Returns:
        VaRResult with method='parametric_t'.

    Raises:
        ValueError: If returns is empty, confidence is out of (0, 1), or MLE fails.
    """
    if returns.empty:
        raise ValueError("returns Series is empty.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    clean = returns.dropna().values
    # Fit t-distribution: returns (df, loc, scale)
    df, loc, scale = stats.t.fit(clean)
    alpha = 1 - confidence

    t_quantile = stats.t.ppf(alpha, df=df, loc=loc, scale=scale)
    var_val = float(-t_quantile)

    # CVaR for t: E[X | X <= -VaR]
    # = loc - scale * t.pdf(t_alpha, df) / alpha * (df + t_alpha**2) / (df - 1)
    # More robust: use numerical integration via ppf/pdf
    z = stats.t.ppf(alpha, df=df)
    if df > 1:
        cvar_val = float(
            -(loc - scale / alpha * stats.t.pdf(z, df) * (df + z**2) / (df - 1))
        )
    else:
        # df <= 1: CVaR undefined; fall back to VaR
        logger.warning("Fitted df <= 1; CVaR undefined for t-distribution, using VaR.")
        cvar_val = var_val

    return VaRResult(
        confidence=confidence,
        method="parametric_t",
        var=var_val,
        cvar=cvar_val,
        n_observations=len(clean),
    )


def var_summary(
    returns: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute VaR and CVaR using all three methods and return a comparison table.

    Args:
        returns: Series of period returns.
        confidence: Confidence level, e.g., 0.95.

    Returns:
        DataFrame with columns ['method', 'var', 'cvar'] and one row per method.
    """
    results = [
        var_historical(returns, confidence),
        var_parametric_normal(returns, confidence),
        var_parametric_t(returns, confidence),
    ]
    rows = [{"method": r.method, "var": r.var, "cvar": r.cvar} for r in results]
    return pd.DataFrame(rows).set_index("method")
