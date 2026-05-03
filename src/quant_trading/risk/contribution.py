"""Risk contribution analysis: marginal and component risk per asset.

Provides tools to decompose portfolio risk (volatility, VaR) into per-asset
contributions, enabling identification of risk concentrations.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskContributionResult:
    """Risk decomposition for a portfolio.

    Attributes:
        total_risk: Portfolio-level risk (annualized volatility).
        marginal_contribution: Marginal risk contribution per asset (∂σ/∂w_i).
        component_contribution: Absolute risk contribution per asset (w_i * ∂σ/∂w_i).
        percent_contribution: Percentage share of total risk per asset.
        diversification_ratio: Weighted-avg individual vol / portfolio vol.
    """

    total_risk: float
    marginal_contribution: pd.Series
    component_contribution: pd.Series
    percent_contribution: pd.Series
    diversification_ratio: float


def risk_contribution(
    weights: pd.Series,
    cov: pd.DataFrame,
    periods_per_year: int = 252,
) -> RiskContributionResult:
    """Compute per-asset risk contributions from weights and covariance matrix.

    Uses the Euler decomposition: for portfolio volatility σ_p,
        MRC_i = (Σ w)_i / σ_p
        CRC_i = w_i * MRC_i
        ∑ CRC_i = σ_p  (Euler's theorem)

    Args:
        weights: Series of portfolio weights (asset → weight).
        cov: Annualized covariance matrix (assets × assets). Must share labels
            with weights.
        periods_per_year: Used to annualize if cov is per-period. Set to 1 if
            cov is already annualized.

    Returns:
        RiskContributionResult with per-asset breakdown.

    Raises:
        ValueError: If weights or cov is empty, or labels do not align.
    """
    if weights.empty:
        raise ValueError("weights Series is empty.")
    if cov.empty:
        raise ValueError("cov DataFrame is empty.")

    assets = weights.index.tolist()
    missing = set(assets) - set(cov.index)
    if missing:
        raise ValueError(f"Assets in weights missing from cov: {missing}")

    w = weights[assets].values.astype(float)
    C = cov.loc[assets, assets].values.astype(float) * periods_per_year

    port_var = float(w @ C @ w)
    port_vol = float(np.sqrt(port_var))

    if port_vol == 0:
        zero = pd.Series(0.0, index=assets)
        return RiskContributionResult(
            total_risk=0.0,
            marginal_contribution=zero,
            component_contribution=zero,
            percent_contribution=zero,
            diversification_ratio=1.0,
        )

    marginal = (C @ w) / port_vol  # ∂σ/∂w
    component = w * marginal  # w_i * ∂σ/∂w_i
    pct = component / port_vol

    # Diversification ratio: weighted average individual vol / portfolio vol
    individual_vols = np.sqrt(np.diag(C))
    weighted_avg_vol = float(w @ individual_vols)
    div_ratio = weighted_avg_vol / port_vol

    return RiskContributionResult(
        total_risk=port_vol,
        marginal_contribution=pd.Series(marginal, index=assets),
        component_contribution=pd.Series(component, index=assets),
        percent_contribution=pd.Series(pct, index=assets),
        diversification_ratio=div_ratio,
    )


def risk_contribution_from_returns(
    weights: pd.Series,
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> RiskContributionResult:
    """Compute per-asset risk contributions from returns data.

    Convenience wrapper that estimates the covariance matrix from returns
    before delegating to :func:`risk_contribution`.

    Args:
        weights: Series of portfolio weights (asset → weight).
        returns: DataFrame of asset returns (index = dates, columns = assets).
        periods_per_year: Number of periods per year (252 for daily).

    Returns:
        RiskContributionResult with per-asset breakdown.

    Raises:
        ValueError: If weights or returns is empty.
    """
    if returns.empty:
        raise ValueError("returns DataFrame is empty.")
    assets = weights.index.tolist()
    missing = set(assets) - set(returns.columns)
    if missing:
        raise ValueError(f"Assets in weights missing from returns: {missing}")

    cov = returns[assets].cov()  # per-period covariance
    return risk_contribution(weights, cov, periods_per_year=periods_per_year)


def component_var(
    weights: pd.Series,
    returns: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.Series:
    """Compute per-asset component VaR via the delta-normal approximation.

    Component VaR_i = ρ(r_i, r_p) * VaR_i_standalone * w_i / σ_p * σ_p
                    = w_i * cov(r_i, r_p) / σ_p * z_alpha

    More specifically, this uses the linear approximation:
        CVaR_i = w_i * β_i * portfolio_var_quantile

    where β_i = cov(r_i, r_p) / var(r_p).

    Note: This is an approximation valid for elliptical return distributions.

    Args:
        weights: Series of portfolio weights (asset → weight).
        returns: DataFrame of asset returns (index = dates, columns = assets).
        confidence: Confidence level for VaR (e.g., 0.95).

    Returns:
        Series of component VaR per asset (positive loss magnitudes).

    Raises:
        ValueError: If weights or returns is empty, or confidence out of (0, 1).
    """
    if returns.empty:
        raise ValueError("returns DataFrame is empty.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    from scipy import stats

    assets = weights.index.tolist()
    w = weights[assets].values.astype(float)
    R = returns[assets].dropna()

    port_returns = R.values @ w
    port_vol = float(port_returns.std())

    alpha = 1 - confidence
    z = float(stats.norm.ppf(alpha))  # negative number

    cov_matrix = R.cov().values
    cov_with_portfolio = cov_matrix @ w  # cov(r_i, r_p)

    # Component VaR_i = -w_i * cov(r_i, r_p) / σ_p * z (positive loss)
    component_vars = -w * cov_with_portfolio / port_vol * z
    return pd.Series(component_vars, index=assets)
