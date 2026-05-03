"""Portfolio diagnostics: efficient frontier and portfolio analytics.

Intended for research and reporting — not used in live execution paths.

Functions:
    efficient_frontier() — traces the mean-variance efficient frontier.
    portfolio_analytics() — computes annualised statistics for a given set of
        weights, including factor decomposition if a FactorModel is supplied.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .factor import FactorModel, compute_factor_attribution, compute_factor_exposures
from .mean_variance import mean_variance_optimize

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


def efficient_frontier(
    expected_returns: np.ndarray | pd.Series,
    cov: np.ndarray,
    *,
    n_points: int = 50,
    extra_constraints: list | None = None,
) -> pd.DataFrame:
    """Trace the mean-variance efficient frontier.

    Parameterises the frontier via target_return from the global minimum
    variance portfolio return up to the maximum expected return asset.

    Args:
        expected_returns: Annualised expected returns, shape (n_assets,).
            If a Series, its index is used as asset names.
        cov: Covariance matrix, shape (n_assets, n_assets).
            Should be annualised to match expected_returns.
        n_points: Number of frontier points to compute.
        extra_constraints: Additional CVXPY constraints (e.g. long_only).
            Applied at every frontier point.

    Returns:
        DataFrame with columns [vol, ret, sharpe], one row per frontier point,
        indexed 0..n_points-1.  vol and ret are annualised (not percentages).

    Note:
        Points where the solver fails are dropped silently. If no points
        succeed, an empty DataFrame is returned.
    """
    if isinstance(expected_returns, pd.Series):
        mu = expected_returns.values
    else:
        mu = np.asarray(expected_returns)

    # GMV return ~ mean of min-variance weights' expected return
    gmv = mean_variance_optimize(
        expected_returns=mu,
        cov=cov,
        risk_aversion=1e6,  # very high aversion ≈ min-variance
        constraints=extra_constraints or [],
    )
    ret_min = float(gmv.weights @ mu) if gmv.is_optimal else mu.min()
    ret_max = mu.max()

    target_returns = np.linspace(ret_min, ret_max, n_points)

    rows: list[dict] = []
    for target in target_returns:
        result = mean_variance_optimize(
            expected_returns=mu,
            cov=cov,
            target_return=target,
            constraints=extra_constraints or [],
        )
        if not result.is_optimal:
            logger.debug(
                "efficient_frontier: solver failed at target_return=%.4f (%s)",
                target,
                result.status,
            )
            continue
        rows.append(
            {
                "ret": result.expected_return,
                "vol": result.expected_vol,
                "sharpe": result.sharpe,
            }
        )

    if not rows:
        logger.warning("efficient_frontier: no feasible points found.")
        return pd.DataFrame(columns=["ret", "vol", "sharpe"])

    df = pd.DataFrame(rows)
    # Sort by vol for clean plotting
    df = df.sort_values("vol").reset_index(drop=True)
    return df


def portfolio_analytics(
    weights: pd.Series,
    returns: pd.DataFrame,
    *,
    factor_model: FactorModel | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_DAYS,
) -> dict[str, Any]:
    """Compute comprehensive portfolio statistics for a fixed weight vector.

    Args:
        weights: Portfolio weights, index = asset names matching returns columns.
        returns: Asset return history, shape (n_periods, n_assets).
        factor_model: Optional FactorModel for factor exposure/attribution.
        risk_free_rate: Annualised risk-free rate for Sharpe computation.
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        Dictionary with keys:
            annualised_return (float), annualised_vol (float), sharpe (float),
            max_drawdown (float, negative), calmar_ratio (float),
            sortino_ratio (float), skewness (float), kurtosis (float),
            var_95 (float, negative, 1-day 95% historical VaR),
            cvar_95 (float, negative, 1-day 95% historical CVaR),
            hit_rate (float), weight_summary (DataFrame),
            factor_exposures (Series or None),
            factor_attribution (dict or None).
    """
    common_assets = weights.index.intersection(returns.columns)
    if len(common_assets) == 0:
        raise ValueError("weights and returns share no common asset names.")
    if len(common_assets) < len(weights):
        logger.warning(
            "portfolio_analytics: %d assets in weights have no return data — excluded.",
            len(weights) - len(common_assets),
        )

    w = weights.loc[common_assets]
    # Normalise in case weights don't sum to 1 exactly
    w = w / w.sum()

    port_ret: pd.Series = returns[common_assets].dot(w)

    # --- Core stats ---
    ann_ret = float(port_ret.mean() * periods_per_year)
    ann_vol = float(port_ret.std() * np.sqrt(periods_per_year))
    rf_daily = risk_free_rate / periods_per_year
    excess = port_ret - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(periods_per_year)) if excess.std() > 0 else 0.0

    # --- Drawdown ---
    cum = (1 + port_ret).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # --- Sortino ---
    downside = port_ret[port_ret < rf_daily]
    downside_std = float(downside.std() * np.sqrt(periods_per_year)) if len(downside) > 1 else 1.0
    sortino = (ann_ret - risk_free_rate) / downside_std if downside_std > 0 else 0.0

    # --- Higher moments ---
    from scipy import stats as scipy_stats
    skewness = float(scipy_stats.skew(port_ret.dropna()))
    kurtosis = float(scipy_stats.kurtosis(port_ret.dropna()))  # excess kurtosis

    # --- VaR / CVaR (historical, 95%) ---
    sorted_rets = np.sort(port_ret.dropna().values)
    var_idx = int(np.floor(0.05 * len(sorted_rets)))
    var_95 = float(sorted_rets[max(var_idx, 0)])
    cvar_95 = float(sorted_rets[: max(var_idx, 1)].mean()) if var_idx > 0 else var_95

    # --- Hit rate ---
    hit_rate = float((port_ret > 0).mean())

    # --- Weight summary ---
    weight_summary = w.sort_values(ascending=False).rename("weight").to_frame()
    weight_summary["contribution_to_return"] = np.nan
    if len(common_assets) == len(returns.columns):
        mean_rets = returns[common_assets].mean() * periods_per_year
        weight_summary["contribution_to_return"] = w * mean_rets

    # --- Factor diagnostics ---
    factor_exposures = None
    factor_attribution = None
    if factor_model is not None:
        try:
            factor_exposures = compute_factor_exposures(w, factor_model)
            if (
                factor_model.factor_cov is not None
                and factor_model.residual_var is not None
            ):
                factor_attribution = compute_factor_attribution(w, factor_model)
        except Exception as exc:
            logger.warning("portfolio_analytics: factor diagnostics failed: %s", exc)

    return {
        "annualised_return": ann_ret,
        "annualised_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "sortino_ratio": sortino,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "hit_rate": hit_rate,
        "weight_summary": weight_summary,
        "factor_exposures": factor_exposures,
        "factor_attribution": factor_attribution,
    }
