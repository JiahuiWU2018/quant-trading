"""High-level allocation interfaces: AssetAllocator and StrategyAllocator.

AssetAllocator
  Wraps the low-level mean_variance_optimize / risk_parity_optimize calls
  with a clean API: supply returns + optional expected returns → get weights.

StrategyAllocator
  Designed for strategy-of-strategies use: each column of the input DataFrame
  represents a strategy's return stream. Produces capital weights across
  strategies. Accepts explicit expected_returns; if not supplied, falls back
  to historical mean. The private repository is responsible for computing and
  supplying expected_returns.

AllocationResult
  Unified output container for both allocators.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .covariance import ESTIMATORS, sample_covariance
from .factor import FactorModel, compute_factor_attribution, compute_factor_exposures
from .mean_variance import OptimResult, mean_variance_optimize
from .risk_parity import risk_parity_optimize

logger = logging.getLogger(__name__)

# Annualisation factor (daily data)
_TRADING_DAYS = 252


class OptimMethod(str, Enum):
    """Supported optimisation methods."""

    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"


@dataclass
class AllocationResult:
    """Output container for AssetAllocator and StrategyAllocator.

    Args:
        weights: Final portfolio weights, index = asset/strategy names.
        optim_result: Raw OptimResult from the underlying solver.
        expected_returns: Expected returns vector used (annualised).
        factor_exposures: Portfolio factor exposures if a FactorModel was supplied,
            else None.
        factor_attribution: Factor variance attribution dict if a FactorModel was
            supplied and factor_cov/residual_var are available, else None.
        cov_estimator_used: Name of the covariance estimator that was used.
        method: Optimisation method used.
    """

    weights: pd.Series
    optim_result: OptimResult
    expected_returns: pd.Series | None = None
    factor_exposures: pd.Series | None = None
    factor_attribution: dict | None = None
    cov_estimator_used: str = "sample"
    method: OptimMethod = OptimMethod.MEAN_VARIANCE

    @property
    def is_optimal(self) -> bool:
        """True if the underlying solver reported an optimal solution."""
        return self.optim_result.is_optimal

    def summary(self) -> pd.DataFrame:
        """Return a human-readable weight summary DataFrame.

        Returns:
            DataFrame with columns: weight, expected_return (if available),
            factor_exposures (if available).
        """
        df = self.weights.rename("weight").to_frame()
        if self.expected_returns is not None:
            df["expected_return"] = self.expected_returns.reindex(df.index)
        return df


class AssetAllocator:
    """Allocator for a universe of individual assets (equities, ETFs, etc.).

    Intended to be instantiated once and called repeatedly (e.g. monthly
    rebalancing).  The private repository subclasses or wraps this class
    to supply expected returns and a FactorModel.

    Args:
        method: OptimMethod to use (MEAN_VARIANCE or RISK_PARITY).
        cov_estimator: Name of covariance estimator from covariance.ESTIMATORS.
            Defaults to "sample".
        constraints: List of CVXPY constraints as returned by constraints.py
            helper functions. Applied on top of fully_invested + long_only defaults
            if include_default_constraints=True.
        include_default_constraints: If True (default), always add long_only +
            fully_invested. Set to False for long-short strategies.
        factor_model: Optional FactorModel for exposure/attribution reporting.
        risk_aversion: Lambda for mean-variance risk_aversion formulation.
        min_periods: Minimum number of return periods required.
    """

    def __init__(
        self,
        method: OptimMethod | str = OptimMethod.MEAN_VARIANCE,
        cov_estimator: str = "sample",
        constraints: list | None = None,
        include_default_constraints: bool = True,
        factor_model: FactorModel | None = None,
        risk_aversion: float = 1.0,
        min_periods: int = 20,
    ) -> None:
        self.method = OptimMethod(method)
        if cov_estimator not in ESTIMATORS:
            raise ValueError(
                f"Unknown cov_estimator '{cov_estimator}'. "
                f"Valid options: {list(ESTIMATORS)}"
            )
        self.cov_estimator = cov_estimator
        self.extra_constraints = constraints or []
        self.include_default_constraints = include_default_constraints
        self.factor_model = factor_model
        self.risk_aversion = risk_aversion
        self.min_periods = min_periods

    def optimise(
        self,
        returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        weights_prev: pd.Series | None = None,
    ) -> AllocationResult:
        """Compute optimal weights for the asset universe.

        Args:
            returns: Historical return DataFrame, shape (n_periods, n_assets).
                Used for covariance estimation (and expected returns if
                expected_returns is None and method is MEAN_VARIANCE).
            expected_returns: Explicit expected returns vector (annualised),
                index = asset names. Takes priority over historical mean when
                provided. Ignored for RISK_PARITY.
            weights_prev: Previous portfolio weights for max_turnover constraint
                support (passed via extra_constraints). Not used internally
                unless a max_turnover constraint is in extra_constraints.

        Returns:
            AllocationResult with weights and diagnostics.

        Raises:
            ValueError: If returns has fewer rows than min_periods.
        """
        if len(returns) < self.min_periods:
            raise ValueError(
                f"returns has {len(returns)} rows; min_periods={self.min_periods}."
            )

        assets = list(returns.columns)

        # --- Covariance estimation ---
        cov_fn = ESTIMATORS[self.cov_estimator]
        cov = cov_fn(returns)

        # --- Build constraint list for mean_variance_optimize ---
        # Use constraint factories (callables w → list) so they are applied
        # with mean_variance_optimize's own CVXPY variable.
        from .constraints import fully_invested, long_only

        constraint_factories: list = []
        if self.include_default_constraints:
            constraint_factories += [long_only, fully_invested]
        constraint_factories += self.extra_constraints

        # --- Optimise ---
        if self.method == OptimMethod.RISK_PARITY:
            optim_result = risk_parity_optimize(cov)
        else:
            # Expected returns: explicit > historical mean
            if expected_returns is not None:
                mu_ann = expected_returns.reindex(assets).values
            else:
                mu_ann = returns.mean().reindex(assets).values * _TRADING_DAYS
            optim_result = mean_variance_optimize(
                expected_returns=mu_ann,
                cov=cov,
                risk_aversion=self.risk_aversion,
                constraints=constraint_factories,
            )

        weights = pd.Series(optim_result.weights, index=assets, name="weight")

        # --- Factor diagnostics ---
        factor_exposures = None
        factor_attribution = None
        if self.factor_model is not None:
            try:
                factor_exposures = compute_factor_exposures(weights, self.factor_model)
                if (
                    self.factor_model.factor_cov is not None
                    and self.factor_model.residual_var is not None
                ):
                    factor_attribution = compute_factor_attribution(
                        weights, self.factor_model
                    )
            except Exception as exc:
                logger.warning("Factor diagnostics failed: %s", exc)

        mu_series = None
        if expected_returns is not None:
            mu_series = expected_returns.reindex(assets)
        elif self.method != OptimMethod.RISK_PARITY:
            mu_series = pd.Series(
                returns.mean().reindex(assets).values * _TRADING_DAYS,
                index=assets,
                name="expected_return",
            )

        return AllocationResult(
            weights=weights,
            optim_result=optim_result,
            expected_returns=mu_series,
            factor_exposures=factor_exposures,
            factor_attribution=factor_attribution,
            cov_estimator_used=self.cov_estimator,
            method=self.method,
        )


class StrategyAllocator(AssetAllocator):
    """Capital allocator for a portfolio-of-strategies.

    Treats each strategy's return stream as a synthetic "asset" and allocates
    capital across strategies.  Identical to AssetAllocator in mechanics but
    provides clearer intent via its name and a strategy-focused API.

    Explicit expected_returns supplied by the private repository always take
    priority over historical mean estimation.

    Usage:
        allocator = StrategyAllocator(method="risk_parity", cov_estimator="ledoit_wolf")
        result = allocator.optimise(strategy_returns_df)
        print(result.weights)

    Args:
        Same as AssetAllocator.
    """

    def optimise(
        self,
        strategy_returns: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        weights_prev: pd.Series | None = None,
    ) -> AllocationResult:
        """Compute capital weights across strategies.

        Args:
            strategy_returns: Strategy return DataFrame, shape
                (n_periods, n_strategies).  Each column is one strategy.
            expected_returns: Explicit per-strategy expected returns (annualised).
                Index = strategy names matching strategy_returns columns.
                If provided, used instead of historical mean for MEAN_VARIANCE.
            weights_prev: Previous strategy weights for turnover tracking.

        Returns:
            AllocationResult with strategy weights and diagnostics.
        """
        logger.info(
            "StrategyAllocator.optimise: %d strategies, %d periods, method=%s",
            strategy_returns.shape[1],
            strategy_returns.shape[0],
            self.method.value,
        )
        result = super().optimise(
            returns=strategy_returns,
            expected_returns=expected_returns,
            weights_prev=weights_prev,
        )
        return result
