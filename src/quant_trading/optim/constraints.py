"""CVXPY constraint builders for portfolio optimisation.

All functions accept a CVXPY variable ``w`` (portfolio weights vector) and
return a list of CVXPY constraints. Constraints are composable — pass any
combination to mean_variance_optimize().

Usage:
    import cvxpy as cp
    from quant_trading.optim.constraints import long_only, fully_invested, max_weight

    w = cp.Variable(n)
    constraints = [
        *long_only(w),
        *fully_invested(w),
        *max_weight(w, 0.20),
    ]
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def long_only(w) -> list:
    """All weights non-negative (no short selling).

    Args:
        w: CVXPY Variable, shape (n,).

    Returns:
        List of CVXPY constraints.
    """
    return [w >= 0]


def fully_invested(w) -> list:
    """Weights sum to 1 (fully invested, no cash drag).

    Args:
        w: CVXPY Variable, shape (n,).

    Returns:
        List of CVXPY constraints.
    """
    import cvxpy as cp  # type: ignore[import]
    return [cp.sum(w) == 1]


def weight_bounds(w, lower: float, upper: float) -> list:
    """Per-asset weight bounds.

    Args:
        w: CVXPY Variable, shape (n,).
        lower: Minimum weight per asset (e.g. 0.0 for long-only).
        upper: Maximum weight per asset (e.g. 0.10 for 10% cap).

    Returns:
        List of CVXPY constraints.
    """
    return [w >= lower, w <= upper]


def max_weight(w, limit: float) -> list:
    """Concentration limit — no single asset may exceed ``limit``.

    Equivalent to weight_bounds(w, 0, limit) but more explicit.

    Args:
        w: CVXPY Variable, shape (n,).
        limit: Maximum weight for any single asset (e.g. 0.20 = 20%).

    Returns:
        List of CVXPY constraints.
    """
    if not 0 < limit <= 1:
        raise ValueError(f"limit must be in (0, 1], got {limit}")
    return [w <= limit]


def max_turnover(w, w_prev: np.ndarray, limit: float) -> list:
    """Limit portfolio turnover relative to previous weights.

    Turnover = sum(|w - w_prev|) / 2, bounded by ``limit``.
    Uses L1 norm linearisation via auxiliary variable.

    Args:
        w: CVXPY Variable, shape (n,).
        w_prev: Previous portfolio weights as numpy array, shape (n,).
        limit: Maximum one-way turnover (e.g. 0.30 = 30% per rebalance).

    Returns:
        List of CVXPY constraints.
    """
    import cvxpy as cp  # type: ignore[import]
    if not 0 < limit <= 1:
        raise ValueError(f"limit must be in (0, 1], got {limit}")
    delta = w - w_prev
    # L1 norm via auxiliary variables
    abs_delta = cp.Variable(w.shape[0])
    return [
        abs_delta >= delta,
        abs_delta >= -delta,
        cp.sum(abs_delta) <= 2 * limit,   # two-way turnover
    ]


def sector_bounds(
    w,
    sector_map: dict[str, list[int]],
    lower: dict[str, float],
    upper: dict[str, float],
) -> list:
    """Bound total weight allocated to each sector or group.

    Args:
        w: CVXPY Variable, shape (n,).
        sector_map: Dict mapping sector name → list of asset indices.
        lower: Dict mapping sector name → minimum sector weight.
        upper: Dict mapping sector name → maximum sector weight.

    Returns:
        List of CVXPY constraints.

    Example:
        sector_map = {"tech": [0, 1, 2], "finance": [3, 4]}
        lower = {"tech": 0.10, "finance": 0.05}
        upper = {"tech": 0.40, "finance": 0.30}
    """
    import cvxpy as cp  # type: ignore[import]
    constraints = []
    for sector, indices in sector_map.items():
        sector_weight = cp.sum(w[indices])
        lo = lower.get(sector, 0.0)
        hi = upper.get(sector, 1.0)
        constraints += [sector_weight >= lo, sector_weight <= hi]
    return constraints


def factor_exposure_bounds(
    w,
    factor_loadings: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> list:
    """Bound portfolio-level factor exposures.

    Portfolio factor exposure = factor_loadings.T @ w
    Each factor exposure must lie within [lower_i, upper_i].

    Args:
        w: CVXPY Variable, shape (n_assets,).
        factor_loadings: Factor loading matrix, shape (n_assets, n_factors).
            Each column is one factor's loadings across all assets.
        lower: Lower bounds on factor exposures, shape (n_factors,).
        upper: Upper bounds on factor exposures, shape (n_factors,).

    Returns:
        List of CVXPY constraints.

    Example:
        # Factor-neutral: all factor exposures between -0.1 and +0.1
        B = factor_model.loadings.values    # (n_assets, n_factors)
        constraints = factor_exposure_bounds(w, B,
            lower=np.full(n_factors, -0.1),
            upper=np.full(n_factors,  0.1))
    """
    factor_loadings = np.atleast_2d(factor_loadings)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if factor_loadings.shape[1] != len(lower) or factor_loadings.shape[1] != len(upper):
        raise ValueError(
            f"factor_loadings has {factor_loadings.shape[1]} factors but "
            f"lower/upper have {len(lower)}/{len(upper)} elements."
        )
    exposure = factor_loadings.T @ w   # (n_factors,)
    return [exposure >= lower, exposure <= upper]
