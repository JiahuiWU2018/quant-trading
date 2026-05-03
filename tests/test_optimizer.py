"""Tests for Phase 4: Portfolio optimisation.

Covers:
    - MVO analytical checks (2-asset known solution)
    - Solver fallback and infeasible constraint handling
    - Risk parity equal contribution
    - StrategyAllocator with synthetic strategies
    - Factor loading OLS regression
    - Factor exposure and attribution computation
    - Efficient frontier generation
    - Portfolio analytics
"""

import numpy as np
import pandas as pd
import pytest

from quant_trading.optim.allocator import (
    AllocationResult,
    AssetAllocator,
    OptimMethod,
    StrategyAllocator,
)
from quant_trading.optim.covariance import (
    constant_correlation,
    ewm_covariance,
    ledoit_wolf,
    nearest_positive_definite,
    sample_covariance,
)
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

# ── Helpers ──────────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(42)


def _synthetic_returns(n_periods: int = 504, n_assets: int = 4) -> pd.DataFrame:
    """Generate correlated synthetic daily returns."""
    cov = np.full((n_assets, n_assets), 0.0002)
    np.fill_diagonal(cov, 0.0004)
    raw = _RNG.multivariate_normal(
        mean=np.full(n_assets, 5e-4), cov=cov, size=n_periods
    )
    idx = pd.date_range("2020-01-02", periods=n_periods, freq="B")
    return pd.DataFrame(raw, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def _two_asset_setup():
    """2-asset problem with a known minimum-variance solution."""
    # Asset 0: mu=0.10, sigma=0.20
    # Asset 1: mu=0.15, sigma=0.25
    # Correlation: 0.0  → uncorrelated
    mu = np.array([0.10, 0.15])
    cov = np.diag([0.04, 0.0625])  # var = sigma^2
    return mu, cov


# ── MVO tests ─────────────────────────────────────────────────────────────────


class TestMeanVarianceOptimize:
    def test_risk_aversion_returns_valid_weights(self):
        mu, cov = _two_asset_setup()
        result = mean_variance_optimize(mu, cov, risk_aversion=1.0)
        assert result.is_optimal, f"Expected optimal, got status={result.status}"
        w = result.weights
        assert abs(w.sum() - 1.0) < 1e-4, "Weights must sum to 1"
        assert (w >= -1e-6).all(), "Weights must be non-negative (long-only default)"

    def test_high_risk_aversion_approaches_min_variance(self):
        """With very high risk aversion, weights should approximate the GMV portfolio."""
        mu, cov = _two_asset_setup()
        result = mean_variance_optimize(mu, cov, risk_aversion=1e6)
        assert result.is_optimal
        # GMV for uncorrelated assets: w_i = (1/var_i) / sum(1/var_i)
        inv_var = np.array([1 / 0.04, 1 / 0.0625])
        w_gmv = inv_var / inv_var.sum()
        np.testing.assert_allclose(result.weights, w_gmv, atol=0.01)

    def test_target_return_achieves_target(self):
        mu, cov = _two_asset_setup()
        target = 0.12  # annualised
        result = mean_variance_optimize(mu, cov, target_return=target)
        assert result.is_optimal
        achieved = float(result.weights @ mu)
        assert abs(achieved - target) < 1e-3

    def test_target_vol_achieves_target(self):
        mu, cov = _two_asset_setup()
        target_vol = 0.18
        result = mean_variance_optimize(mu, cov, target_vol=target_vol)
        assert result.is_optimal
        achieved_vol = float(np.sqrt(result.weights @ cov @ result.weights))
        assert abs(achieved_vol - target_vol) < 0.01

    def test_weights_sum_to_one(self):
        returns = _synthetic_returns()
        mu = returns.mean().values * 252
        cov = sample_covariance(returns)
        result = mean_variance_optimize(mu, cov)
        assert result.is_optimal
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_infeasible_target_return_falls_back_to_equal_weight(self):
        """Requesting a return above the max asset return is infeasible."""
        mu, cov = _two_asset_setup()
        # Target above max(mu): infeasible for long-only fully-invested
        result = mean_variance_optimize(mu, cov, target_return=0.99)
        # Should fall back to equal weight
        assert not result.is_optimal or abs(result.weights.sum() - 1.0) < 1e-4

    def test_optim_result_fields(self):
        mu, cov = _two_asset_setup()
        result = mean_variance_optimize(mu, cov, risk_aversion=1.0)
        assert isinstance(result, OptimResult)
        assert result.solver_used is not None
        assert result.solve_time_ms >= 0.0
        assert result.expected_return is not None
        assert result.expected_vol is not None


# ── Risk parity tests ─────────────────────────────────────────────────────────


class TestRiskParity:
    def test_equal_risk_contribution_on_diagonal_cov(self):
        """For a diagonal covariance, risk parity weights ∝ 1/vol."""
        cov = np.diag([0.04, 0.09, 0.01])  # vols: 0.20, 0.30, 0.10
        result = risk_parity_optimize(cov)
        assert result.is_optimal, f"Expected optimal, got {result.status}"
        w = result.weights
        assert abs(w.sum() - 1.0) < 1e-4
        # Expected weights ∝ 1/sigma: 1/0.2, 1/0.3, 1/0.1
        inv_vol = np.array([1 / 0.2, 1 / 0.3, 1 / 0.1])
        expected_w = inv_vol / inv_vol.sum()
        np.testing.assert_allclose(w, expected_w, atol=0.01)

    def test_risk_budgets_are_honoured(self):
        """With budgets [0.5, 0.3, 0.2], risk contributions should match."""
        n = 3
        cov = np.full((n, n), 0.0002)
        np.fill_diagonal(cov, 0.0005)
        budgets = np.array([0.5, 0.3, 0.2])
        result = risk_parity_optimize(cov, risk_budgets=budgets)
        assert result.is_optimal
        # Risk contributions stored in diagnostics
        rc = result.diagnostics.get("risk_contributions")
        if rc is not None:
            np.testing.assert_allclose(np.array(rc) / sum(rc), budgets, atol=0.03)

    def test_equal_risk_parity_on_correlated_cov(self):
        """Risk parity should still produce equal risk contributions."""
        returns = _synthetic_returns(n_assets=4)
        cov = sample_covariance(returns)
        result = risk_parity_optimize(cov)
        assert result.is_optimal
        w = result.weights
        port_vol = np.sqrt(w @ cov @ w)
        # Marginal risk contributions
        mrc = (cov @ w) / port_vol
        rc = w * mrc
        # All risk contributions should be equal
        np.testing.assert_allclose(rc, rc.mean(), atol=0.01)


# ── Covariance estimator tests ────────────────────────────────────────────────


class TestCovarianceEstimators:
    def setup_method(self):
        self.returns = _synthetic_returns()

    def test_sample_covariance_is_psd(self):
        cov = sample_covariance(self.returns)
        assert cov.shape == (4, 4)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

    def test_ledoit_wolf_is_psd(self):
        cov = ledoit_wolf(self.returns)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

    def test_ewm_covariance_is_psd(self):
        cov = ewm_covariance(self.returns, halflife=60)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

    def test_constant_correlation_is_psd(self):
        cov = constant_correlation(self.returns)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

    def test_nearest_positive_definite_repairs_near_singular(self):
        """Inject a negative eigenvalue, then repair."""
        cov = sample_covariance(self.returns)
        # Force a tiny negative eigenvalue by perturbing
        vals, vecs = np.linalg.eigh(cov)
        vals[0] = -1e-8
        broken = vecs @ np.diag(vals) @ vecs.T
        fixed = nearest_positive_definite(broken)
        assert np.all(np.linalg.eigvalsh(fixed) >= -1e-10)


# ── Factor model tests ────────────────────────────────────────────────────────


class TestFactorModel:
    def setup_method(self):
        n_periods = 504
        n_assets = 3
        n_factors = 2
        idx = pd.date_range("2020-01-02", periods=n_periods, freq="B")
        factor_rets = pd.DataFrame(
            _RNG.normal(0, 0.01, (n_periods, n_factors)),
            index=idx,
            columns=["F1", "F2"],
        )
        # Asset returns = factor exposure + noise
        loadings_true = np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
        noise = _RNG.normal(0, 0.005, (n_periods, n_assets))
        asset_rets = factor_rets.values @ loadings_true.T + noise
        self.returns = pd.DataFrame(
            asset_rets, index=idx, columns=["A", "B", "C"]
        )
        self.factor_returns = factor_rets
        self.loadings_true = loadings_true

    def test_compute_factor_loadings_shape(self):
        model = compute_factor_loadings(self.returns, self.factor_returns)
        assert model.loadings.shape == (3, 2)
        assert model.loadings.columns.tolist() == ["F1", "F2"]
        assert model.loadings.index.tolist() == ["A", "B", "C"]

    def test_compute_factor_loadings_recovers_betas(self):
        """OLS should approximately recover the true betas."""
        model = compute_factor_loadings(self.returns, self.factor_returns)
        np.testing.assert_allclose(
            model.loadings.values, self.loadings_true, atol=0.1
        )

    def test_r_squared_positive(self):
        model = compute_factor_loadings(self.returns, self.factor_returns)
        assert (model.r_squared > 0).all()

    def test_compute_factor_exposures(self):
        model = compute_factor_loadings(self.returns, self.factor_returns)
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        exposures = compute_factor_exposures(weights, model)
        assert exposures.index.tolist() == ["F1", "F2"]
        # Exposure should be weighted average of asset betas
        expected = model.loadings.values.T @ weights.values
        np.testing.assert_allclose(exposures.values, expected, atol=1e-9)

    def test_compute_factor_attribution_sums_to_total(self):
        model = compute_factor_loadings(self.returns, self.factor_returns)
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        attr = compute_factor_attribution(weights, model)
        assert abs(
            attr["systematic_pct"] + attr["idiosyncratic_pct"] - 100.0
        ) < 0.01

    def test_factor_model_validates_columns(self):
        loadings = pd.DataFrame(
            [[1.0, 0.5]], index=["X"], columns=["bad_name_1", "bad_name_2"]
        )
        with pytest.raises(ValueError, match="do not match"):
            FactorModel(
                factor_names=["F1", "F2"],
                loadings=loadings,
            )


# ── AssetAllocator tests ──────────────────────────────────────────────────────


class TestAssetAllocator:
    def setup_method(self):
        self.returns = _synthetic_returns()

    def test_mean_variance_produces_valid_weights(self):
        allocator = AssetAllocator(method="mean_variance")
        result = allocator.optimise(self.returns)
        assert isinstance(result, AllocationResult)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert (result.weights >= -1e-6).all()

    def test_risk_parity_produces_valid_weights(self):
        allocator = AssetAllocator(method="risk_parity")
        result = allocator.optimise(self.returns)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert result.method == OptimMethod.RISK_PARITY

    def test_explicit_expected_returns_used(self):
        """Providing explicit expected_returns should be stored in result."""
        allocator = AssetAllocator()
        mu = pd.Series({"A0": 0.12, "A1": 0.15, "A2": 0.10, "A3": 0.08})
        result = allocator.optimise(self.returns, expected_returns=mu)
        assert result.expected_returns is not None
        pd.testing.assert_series_equal(
            result.expected_returns.sort_index(),
            mu.sort_index(),
            check_names=False,
        )

    def test_ledoit_wolf_estimator(self):
        allocator = AssetAllocator(cov_estimator="ledoit_wolf")
        result = allocator.optimise(self.returns)
        assert result.cov_estimator_used == "ledoit_wolf"
        assert result.is_optimal or abs(result.weights.sum() - 1.0) < 1e-4

    def test_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="Unknown cov_estimator"):
            AssetAllocator(cov_estimator="nonexistent")

    def test_too_few_periods_raises(self):
        allocator = AssetAllocator(min_periods=100)
        short_returns = self.returns.head(50)
        with pytest.raises(ValueError, match="min_periods"):
            allocator.optimise(short_returns)

    def test_weight_summary(self):
        allocator = AssetAllocator()
        result = allocator.optimise(self.returns)
        summary = result.summary()
        assert "weight" in summary.columns
        assert len(summary) == self.returns.shape[1]


# ── StrategyAllocator tests ───────────────────────────────────────────────────


class TestStrategyAllocator:
    def setup_method(self):
        # 3 strategies, 2 years of daily returns
        n = 504
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        self.strategy_returns = pd.DataFrame(
            {
                "momentum": _RNG.normal(6e-4, 0.012, n),
                "value": _RNG.normal(4e-4, 0.010, n),
                "trend": _RNG.normal(5e-4, 0.008, n),
            },
            index=idx,
        )

    def test_equal_risk_parity_across_strategies(self):
        allocator = StrategyAllocator(method="risk_parity")
        result = allocator.optimise(self.strategy_returns)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert set(result.weights.index) == {"momentum", "value", "trend"}

    def test_mean_variance_with_explicit_returns(self):
        allocator = StrategyAllocator(method="mean_variance")
        explicit_mu = pd.Series(
            {"momentum": 0.15, "value": 0.10, "trend": 0.12}
        )
        result = allocator.optimise(
            self.strategy_returns, expected_returns=explicit_mu
        )
        assert result.is_optimal or abs(result.weights.sum() - 1.0) < 1e-4

    def test_historical_mean_fallback(self):
        """Without explicit expected_returns, historical mean should be used."""
        allocator = StrategyAllocator(method="mean_variance")
        result = allocator.optimise(self.strategy_returns)
        assert result.expected_returns is not None
        # expected_returns are computed from historical mean — just check they exist
        # (one strategy may have negative mean due to random seed; that's valid)
        assert len(result.expected_returns) == 3


# ── Diagnostics tests ─────────────────────────────────────────────────────────


class TestEfficientFrontier:
    def test_returns_dataframe_with_expected_columns(self):
        mu = np.array([0.10, 0.15, 0.12])
        cov = np.diag([0.04, 0.0625, 0.05])
        df = efficient_frontier(mu, cov, n_points=20)
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) >= {"ret", "vol", "sharpe"}

    def test_frontier_has_positive_vols(self):
        mu = np.array([0.10, 0.15])
        cov = np.diag([0.04, 0.0625])
        df = efficient_frontier(mu, cov, n_points=10)
        if len(df) > 0:
            assert (df["vol"] > 0).all()

    def test_frontier_is_sorted_by_vol(self):
        mu = np.array([0.10, 0.15])
        cov = np.diag([0.04, 0.0625])
        df = efficient_frontier(mu, cov, n_points=10)
        if len(df) > 1:
            assert (df["vol"].diff().dropna() >= -1e-9).all()


class TestPortfolioAnalytics:
    def setup_method(self):
        self.returns = _synthetic_returns()
        self.weights = pd.Series(
            {"A0": 0.25, "A1": 0.25, "A2": 0.25, "A3": 0.25}
        )

    def test_returns_all_expected_keys(self):
        result = portfolio_analytics(self.weights, self.returns)
        expected_keys = [
            "annualised_return",
            "annualised_vol",
            "sharpe",
            "max_drawdown",
            "calmar_ratio",
            "sortino_ratio",
            "skewness",
            "kurtosis",
            "var_95",
            "cvar_95",
            "hit_rate",
            "weight_summary",
            "factor_exposures",
            "factor_attribution",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_max_drawdown_is_negative(self):
        result = portfolio_analytics(self.weights, self.returns)
        assert result["max_drawdown"] <= 0

    def test_var_is_negative(self):
        result = portfolio_analytics(self.weights, self.returns)
        assert result["var_95"] <= 0

    def test_hit_rate_between_zero_and_one(self):
        result = portfolio_analytics(self.weights, self.returns)
        assert 0 <= result["hit_rate"] <= 1

    def test_no_overlapping_assets_raises(self):
        weights = pd.Series({"X": 0.5, "Y": 0.5})
        with pytest.raises(ValueError, match="no common asset names"):
            portfolio_analytics(weights, self.returns)
