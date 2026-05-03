"""Covariance matrix estimators.

All estimators accept a DataFrame of returns (n_periods × n_assets) and
return a positive-semi-definite (n_assets × n_assets) numpy array.

Available estimators:
    sample_covariance         — plain sample covariance (noisy for small T)
    ledoit_wolf               — Oracle Approximating Shrinkage (recommended default)
    ewm_covariance            — exponentially weighted (emphasises recent data)
    constant_correlation      — Elton-Gruber constant-correlation shrinkage
    nearest_positive_definite — repair a non-PD matrix (Higham 1988)
"""

import logging

import numpy as np
import pandas as pd
from scipy.linalg import eigh

logger = logging.getLogger(__name__)


def sample_covariance(returns: pd.DataFrame) -> np.ndarray:
    """Plain sample covariance matrix.

    Args:
        returns: DataFrame of returns, shape (n_periods, n_assets).

    Returns:
        (n_assets, n_assets) covariance matrix.

    Raises:
        ValueError: If returns has fewer than 2 rows.
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 periods to compute covariance.")
    cov = returns.cov().values
    return nearest_positive_definite(cov)


def ledoit_wolf(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (Oracle Approximating Shrinkage).

    Uses sklearn's analytical Ledoit-Wolf estimator — no cross-validation
    required. Recommended default for most use cases.

    Args:
        returns: DataFrame of returns, shape (n_periods, n_assets).

    Returns:
        (n_assets, n_assets) shrunk covariance matrix.

    Raises:
        ValueError: If returns has fewer than 2 rows.
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 periods to compute covariance.")
    try:
        from sklearn.covariance import LedoitWolf  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for Ledoit-Wolf estimation. "
            "Install it with: pip install scikit-learn"
        ) from exc

    lw = LedoitWolf()
    lw.fit(returns.values)
    return lw.covariance_


def ewm_covariance(returns: pd.DataFrame, halflife: int = 60) -> np.ndarray:
    """Exponentially weighted covariance matrix.

    Assigns more weight to recent observations. Useful for strategies
    sensitive to regime changes or recent vol spikes.

    Args:
        returns: DataFrame of returns, shape (n_periods, n_assets).
        halflife: EWM halflife in number of periods.

    Returns:
        (n_assets, n_assets) covariance matrix.

    Raises:
        ValueError: If returns has fewer than 2 rows.
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 periods to compute covariance.")
    cov = returns.ewm(halflife=halflife).cov().iloc[-len(returns.columns):].values
    return nearest_positive_definite(cov)


def constant_correlation(returns: pd.DataFrame) -> np.ndarray:
    """Elton-Gruber constant-correlation shrinkage estimator.

    Shrinks the sample correlation matrix toward a matrix where all
    off-diagonal correlations equal the cross-sectional mean correlation.
    Combines with sample standard deviations to produce covariance.

    Args:
        returns: DataFrame of returns, shape (n_periods, n_assets).

    Returns:
        (n_assets, n_assets) shrunk covariance matrix.

    Raises:
        ValueError: If returns has fewer than 2 rows.
    """
    if len(returns) < 2:
        raise ValueError("Need at least 2 periods to compute covariance.")
    n = returns.shape[1]
    sample_cov = returns.cov().values
    std = np.sqrt(np.diag(sample_cov))
    corr = returns.corr().values

    # Mean of all off-diagonal correlations
    mask = ~np.eye(n, dtype=bool)
    mean_corr = corr[mask].mean()

    # Shrinkage target: constant correlation matrix
    target_corr = np.full((n, n), mean_corr)
    np.fill_diagonal(target_corr, 1.0)

    # Convert back to covariance
    shrunk_cov = np.outer(std, std) * target_corr
    return nearest_positive_definite(shrunk_cov)


def nearest_positive_definite(cov: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the nearest positive-definite cone.

    Uses the Higham (1988) algorithm: clip negative eigenvalues to a small
    positive value. Applied automatically by all estimators above as a
    safety pass.

    Args:
        cov: Symmetric (n, n) matrix that may not be positive definite.

    Returns:
        Nearest positive-definite (n, n) matrix.
    """
    cov = (cov + cov.T) / 2  # enforce symmetry
    eigvals, eigvecs = eigh(cov)
    # Clip eigenvalues to a small positive floor
    min_eig = max(1e-8, eigvals.max() * 1e-9)
    eigvals = np.maximum(eigvals, min_eig)
    repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return (repaired + repaired.T) / 2  # ensure numerical symmetry


# Map estimator name strings → callables (used by allocators)
ESTIMATORS: dict[str, callable] = {
    "sample": sample_covariance,
    "ledoit_wolf": ledoit_wolf,
    "ewm": ewm_covariance,
    "constant_correlation": constant_correlation,
}
