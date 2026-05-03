"""Factor model infrastructure — loading matrix, exposures, and attribution.

Public-repo responsibilities:
  - FactorModel dataclass (generic container — private repo populates betas)
  - FACTOR_FAMILIES enum (documents available Fama-French factor sets)
  - compute_factor_loadings() — OLS regression of returns onto factor returns
  - compute_factor_exposures() — portfolio-level exposure vector
  - compute_factor_attribution() — variance decomposed into systematic vs idiosyncratic

Private-repo responsibilities:
  - Choosing which factor family to use per strategy
  - Running compute_factor_loadings() on private strategy return streams
  - Storing the resulting FactorModel instances

The public repo also provides FamaFrenchConnector (in data/apis/) to source
the factor return data that the private repo regresses against.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class FACTOR_FAMILIES(str, Enum):
    """Supported Fama-French factor families.

    These map to dataset names in pandas_datareader's Fama-French source.
    Use CUSTOM to supply your own factor return DataFrame from the private repo.
    """

    FF3 = "FF3"           # Mkt-RF, SMB, HML (3-factor)
    FF5 = "FF5"           # Mkt-RF, SMB, HML, RMW, CMA (5-factor)
    CARHART4 = "CARHART4" # Mkt-RF, SMB, HML, MOM (4-factor)
    CUSTOM = "CUSTOM"     # Private repo supplies its own factor returns


# Maps FACTOR_FAMILIES → pandas_datareader dataset name(s)
# MOM is a separate dataset that must be joined with FF3 for Carhart4
_FF_DATASET_MAP: dict[str, list[str]] = {
    FACTOR_FAMILIES.FF3: ["F-F_Research_Data_Factors_daily"],
    FACTOR_FAMILIES.FF5: ["F-F_Research_Data_5_Factors_2x3_daily"],
    FACTOR_FAMILIES.CARHART4: [
        "F-F_Research_Data_Factors_daily",
        "F-F_Momentum_Factor_daily",
    ],
}

# Canonical factor names per family
FACTOR_NAMES: dict[str, list[str]] = {
    FACTOR_FAMILIES.FF3: ["Mkt-RF", "SMB", "HML"],
    FACTOR_FAMILIES.FF5: ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    FACTOR_FAMILIES.CARHART4: ["Mkt-RF", "SMB", "HML", "MOM"],
}


@dataclass
class FactorModel:
    """Container for factor loadings (betas) and associated metadata.

    The private repository computes loadings by calling
    ``compute_factor_loadings()`` with strategy/asset return streams and
    Fama-French factor returns, then stores the resulting FactorModel.

    This class is a generic container — it never holds raw strategy returns
    or signal logic.

    Args:
        factor_names: Names of the factors (columns of loadings).
        loadings: Factor loading matrix, shape (n_assets, n_factors).
            Index should be asset/strategy identifiers.
        factor_cov: (n_factors, n_factors) factor return covariance matrix.
            Optional — required for full variance decomposition.
        residual_var: Idiosyncratic variance for each asset, shape (n_assets,).
            Optional — required for full variance decomposition.
        r_squared: R² of each asset's factor regression, shape (n_assets,).
            Diagnostic only.
        family: Which FACTOR_FAMILIES was used (for documentation).
    """

    factor_names: list[str]
    loadings: pd.DataFrame          # (n_assets × n_factors)
    factor_cov: np.ndarray | None = None
    residual_var: np.ndarray | None = None
    r_squared: pd.Series | None = None
    family: FACTOR_FAMILIES | None = None

    def __post_init__(self) -> None:
        if list(self.loadings.columns) != self.factor_names:
            raise ValueError(
                f"loadings columns {list(self.loadings.columns)} do not match "
                f"factor_names {self.factor_names}."
            )


def compute_factor_loadings(
    returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    min_periods: int = 60,
) -> FactorModel:
    """Estimate factor loadings via OLS regression.

    Regresses each column of ``returns`` against ``factor_returns`` using
    ordinary least squares. Returns and factor_returns are aligned on their
    shared date index before regression.

    Args:
        returns: Asset/strategy returns, shape (n_periods, n_assets).
            Index must be a DatetimeIndex.
        factor_returns: Factor return columns to regress against.
            Shape (n_periods, n_factors). Index must be a DatetimeIndex.
            Obtain from FamaFrenchConnector.fetch_factor_returns().
        min_periods: Minimum overlapping periods required. Raises ValueError
            if fewer periods are available after alignment.

    Returns:
        FactorModel populated with loadings, factor_cov, residual_var, r_squared.

    Raises:
        ValueError: If fewer than min_periods overlapping dates exist.

    Example:
        # In private repo:
        from quant_trading.data.apis.fama_french_connector import FamaFrenchConnector
        from quant_trading.optim.factor import compute_factor_loadings, FACTOR_FAMILIES

        connector = FamaFrenchConnector()
        factor_rets = connector.fetch_factor_returns(
            family=FACTOR_FAMILIES.FF3, start="2020-01-01"
        )
        model = compute_factor_loadings(strategy_returns, factor_rets)
    """
    # Align on common dates
    common_idx = returns.index.intersection(factor_returns.index)
    if len(common_idx) < min_periods:
        raise ValueError(
            f"Only {len(common_idx)} overlapping periods between returns and "
            f"factor_returns (minimum required: {min_periods})."
        )

    R = returns.loc[common_idx]
    F = factor_returns.loc[common_idx]

    factor_names = list(F.columns)
    asset_names = list(R.columns)
    n_assets = len(asset_names)
    n_factors = len(factor_names)

    loadings_arr = np.zeros((n_assets, n_factors))
    residual_var_arr = np.zeros(n_assets)
    r_squared_arr = np.zeros(n_assets)

    F_vals = F.values
    # Add intercept column
    F_with_const = np.column_stack([np.ones(len(F_vals)), F_vals])

    for i, asset in enumerate(asset_names):
        y = R[asset].values
        result = stats.linregress  # use numpy lstsq for multi-factor
        coeff, resid, rank, sv = np.linalg.lstsq(F_with_const, y, rcond=None)
        # coeff[0] = intercept (alpha), coeff[1:] = betas
        betas = coeff[1:]
        y_hat = F_with_const @ coeff
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        loadings_arr[i] = betas
        residual_var_arr[i] = ss_res / max(len(y) - n_factors - 1, 1)
        r_squared_arr[i] = r2

    loadings_df = pd.DataFrame(loadings_arr, index=asset_names, columns=factor_names)
    factor_cov = np.cov(F_vals.T) if n_factors > 1 else np.array([[np.var(F_vals.ravel())]])

    logger.info(
        "compute_factor_loadings: %d assets × %d factors, "
        "mean R²=%.3f, periods=%d",
        n_assets,
        n_factors,
        r_squared_arr.mean(),
        len(common_idx),
    )

    return FactorModel(
        factor_names=factor_names,
        loadings=loadings_df,
        factor_cov=factor_cov,
        residual_var=residual_var_arr,
        r_squared=pd.Series(r_squared_arr, index=asset_names, name="r_squared"),
    )


def compute_factor_exposures(
    weights: pd.Series,
    factor_model: FactorModel,
) -> pd.Series:
    """Compute portfolio-level factor exposures.

    Portfolio factor exposure = loadings.T @ weights
    i.e. the weighted average of each asset's factor beta.

    Args:
        weights: Portfolio weights, index = asset names.
        factor_model: FactorModel with loadings aligned to weights index.

    Returns:
        Series of factor exposures, index = factor names.

    Raises:
        ValueError: If weights and loadings indices do not match.
    """
    common = weights.index.intersection(factor_model.loadings.index)
    if len(common) == 0:
        raise ValueError("weights and factor_model.loadings share no common assets.")
    if len(common) < len(weights):
        logger.warning(
            "%d assets in weights have no factor loadings — they will be excluded.",
            len(weights) - len(common),
        )
    w = weights.loc[common].values
    B = factor_model.loadings.loc[common].values   # (n_assets, n_factors)
    exposures = B.T @ w                            # (n_factors,)
    return pd.Series(exposures, index=factor_model.factor_names, name="factor_exposure")


def compute_factor_attribution(
    weights: pd.Series,
    factor_model: FactorModel,
) -> dict:
    """Decompose portfolio variance into systematic and idiosyncratic components.

    Requires factor_model.factor_cov and factor_model.residual_var.

    Total variance = systematic variance + idiosyncratic variance
    Systematic  = exposureᵀ · Σ_f · exposure
    Idiosyncratic = wᵀ · diag(residual_var) · w

    Args:
        weights: Portfolio weights, index = asset names.
        factor_model: FactorModel with factor_cov and residual_var populated.

    Returns:
        Dict with keys:
            total_variance, systematic_variance, idiosyncratic_variance,
            systematic_pct, idiosyncratic_pct, factor_exposures (Series).

    Raises:
        ValueError: If factor_model.factor_cov or residual_var is None.
    """
    if factor_model.factor_cov is None:
        raise ValueError("factor_model.factor_cov is required for variance attribution.")
    if factor_model.residual_var is None:
        raise ValueError("factor_model.residual_var is required for variance attribution.")

    exposures = compute_factor_exposures(weights, factor_model)
    e = exposures.values
    Sigma_f = factor_model.factor_cov

    systematic_var = float(e @ Sigma_f @ e)

    common = weights.index.intersection(factor_model.loadings.index)
    w = weights.loc[common].values
    res_var = factor_model.residual_var[
        [list(factor_model.loadings.index).index(a) for a in common]
    ]
    idiosyncratic_var = float(w @ (res_var * w))

    total_var = systematic_var + idiosyncratic_var
    denom = total_var if total_var > 0 else 1.0

    return {
        "total_variance": total_var,
        "systematic_variance": systematic_var,
        "idiosyncratic_variance": idiosyncratic_var,
        "systematic_pct": systematic_var / denom * 100,
        "idiosyncratic_pct": idiosyncratic_var / denom * 100,
        "factor_exposures": exposures,
    }
