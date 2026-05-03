"""Phase 4 — Portfolio optimisation layer.

Public interface:
    AssetAllocator      — optimise weights across assets within a strategy
    StrategyAllocator   — optimise capital allocation across strategies
    AllocationResult    — output dataclass from both allocators
    FactorModel         — container for factor loadings (private repo populates)
    FACTOR_FAMILIES     — enum of supported Fama-French factor sets
    compute_factor_loadings — OLS regression of returns onto factor returns
"""

from quant_trading.optim.allocator import AllocationResult, AssetAllocator, OptimMethod, StrategyAllocator
from quant_trading.optim.diagnostics import efficient_frontier, portfolio_analytics
from quant_trading.optim.factor import (
    FACTOR_FAMILIES,
    FactorModel,
    compute_factor_attribution,
    compute_factor_exposures,
    compute_factor_loadings,
)
from quant_trading.optim.mean_variance import OptimResult, mean_variance_optimize
from quant_trading.optim.risk_parity import risk_parity_optimize

__all__ = [
    "AssetAllocator",
    "StrategyAllocator",
    "AllocationResult",
    "OptimMethod",
    "FactorModel",
    "FACTOR_FAMILIES",
    "compute_factor_loadings",
    "compute_factor_exposures",
    "compute_factor_attribution",
    "OptimResult",
    "mean_variance_optimize",
    "risk_parity_optimize",
    "efficient_frontier",
    "portfolio_analytics",
]
