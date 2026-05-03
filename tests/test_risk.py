"""Tests for Phase 5 risk utilities.

Covers: VaR/CVaR, stress testing, risk contribution, drawdown analytics,
and rolling metrics.
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 500  # trading days


@pytest.fixture
def daily_returns() -> pd.Series:
    """Synthetic daily return series (~0% drift, 20% annual vol)."""
    dates = pd.date_range("2021-01-01", periods=N, freq="B")
    data = RNG.normal(loc=0.0, scale=0.01, size=N)
    return pd.Series(data, index=dates, name="portfolio")


@pytest.fixture
def multi_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """Synthetic multi-asset returns DataFrame (3 assets)."""
    dates = daily_returns.index
    data = RNG.normal(loc=0.0, scale=0.012, size=(N, 3))
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


@pytest.fixture
def equity_curve(daily_returns: pd.Series) -> pd.Series:
    """Cumulative equity curve from daily_returns."""
    return (1 + daily_returns).cumprod()


@pytest.fixture
def portfolio_weights() -> pd.Series:
    return pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


# ===========================================================================
# VaR / CVaR
# ===========================================================================


class TestVaRHistorical:
    def test_returns_var_result(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_historical, VaRResult

        result = var_historical(daily_returns)
        assert isinstance(result, VaRResult)
        assert result.method == "historical"
        assert result.confidence == 0.95

    def test_var_positive(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_historical

        result = var_historical(daily_returns)
        assert result.var > 0
        assert result.cvar > 0

    def test_cvar_ge_var(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_historical

        result = var_historical(daily_returns)
        assert result.cvar >= result.var

    def test_higher_confidence_higher_var(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_historical

        r95 = var_historical(daily_returns, confidence=0.95)
        r99 = var_historical(daily_returns, confidence=0.99)
        assert r99.var >= r95.var

    def test_empty_raises(self) -> None:
        from quant_trading.risk.var import var_historical

        with pytest.raises(ValueError, match="empty"):
            var_historical(pd.Series(dtype=float))

    def test_bad_confidence_raises(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_historical

        with pytest.raises(ValueError, match="confidence"):
            var_historical(daily_returns, confidence=1.5)


class TestVaRParametricNormal:
    def test_returns_var_result(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_parametric_normal, VaRResult

        result = var_parametric_normal(daily_returns)
        assert isinstance(result, VaRResult)
        assert result.method == "parametric_normal"

    def test_cvar_ge_var(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_parametric_normal

        result = var_parametric_normal(daily_returns)
        assert result.cvar >= result.var


class TestVaRParametricT:
    def test_returns_var_result(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_parametric_t, VaRResult

        result = var_parametric_t(daily_returns)
        assert isinstance(result, VaRResult)
        assert result.method == "parametric_t"

    def test_cvar_ge_var(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_parametric_t

        result = var_parametric_t(daily_returns)
        assert result.cvar >= result.var


class TestVaRSummary:
    def test_summary_shape(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.var import var_summary

        df = var_summary(daily_returns)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["var", "cvar"]
        assert set(df.index) == {
            "historical",
            "parametric_normal",
            "parametric_t",
        }


# ===========================================================================
# Stress testing
# ===========================================================================


class TestStressTesting:
    def test_run_stress_test_returns_dataframe(self) -> None:
        from quant_trading.risk.stress import run_stress_test, HISTORICAL_SCENARIOS

        weights = pd.Series({"equities": 0.6, "bonds": 0.4})
        df = run_stress_test(weights, scenarios=HISTORICAL_SCENARIOS)
        assert isinstance(df, pd.DataFrame)
        assert "portfolio_return" in df.columns
        assert len(df) == len(HISTORICAL_SCENARIOS)

    def test_portfolio_return_is_weighted_sum(self) -> None:
        from quant_trading.risk.stress import StressScenario, run_stress_test

        scenario = StressScenario(
            name="test", shocks={"A": -0.10, "B": -0.20}
        )
        weights = pd.Series({"A": 0.6, "B": 0.4})
        df = run_stress_test(weights, scenarios=[scenario])
        expected = 0.6 * (-0.10) + 0.4 * (-0.20)
        assert abs(df.loc["test", "portfolio_return"] - expected) < 1e-10

    def test_asset_map_remapping(self) -> None:
        from quant_trading.risk.stress import GFC_2008, run_stress_test

        weights = pd.Series({"SPY": 0.7, "TLT": 0.3})
        asset_map = {"SPY": "equities", "TLT": "bonds"}
        df = run_stress_test(weights, scenarios=[GFC_2008], asset_map=asset_map)
        expected = 0.7 * GFC_2008.shocks["equities"] + 0.3 * GFC_2008.shocks["bonds"]
        assert abs(df.loc["GFC 2008", "portfolio_return"] - expected) < 1e-10

    def test_missing_asset_defaults_zero(self) -> None:
        from quant_trading.risk.stress import StressScenario, run_stress_test

        scenario = StressScenario(name="shock", shocks={"A": -0.50})
        weights = pd.Series({"A": 0.5, "UNKNOWN": 0.5})
        df = run_stress_test(weights, scenarios=[scenario])
        # UNKNOWN has no shock → 0; portfolio = 0.5 * -0.50 + 0.5 * 0.0
        assert abs(df.loc["shock", "portfolio_return"] - (-0.25)) < 1e-10

    def test_historical_scenario_from_returns(
        self, multi_returns: pd.DataFrame
    ) -> None:
        from quant_trading.risk.stress import historical_scenario_from_returns

        start = str(multi_returns.index[0].date())
        end = str(multi_returns.index[50].date())
        scenario = historical_scenario_from_returns(
            multi_returns, start=start, end=end, name="custom"
        )
        assert scenario.name == "custom"
        assert set(scenario.shocks.keys()) == {"A", "B", "C"}

    def test_historical_scenario_empty_window_raises(
        self, multi_returns: pd.DataFrame
    ) -> None:
        from quant_trading.risk.stress import historical_scenario_from_returns

        with pytest.raises(ValueError, match="No return data"):
            historical_scenario_from_returns(
                multi_returns,
                start="1900-01-01",
                end="1900-12-31",
                name="bad",
            )

    def test_empty_weights_raises(self) -> None:
        from quant_trading.risk.stress import run_stress_test

        with pytest.raises(ValueError, match="empty"):
            run_stress_test(pd.Series(dtype=float))


# ===========================================================================
# Risk contribution
# ===========================================================================


class TestRiskContribution:
    def test_components_sum_to_total_risk(
        self,
        portfolio_weights: pd.Series,
        multi_returns: pd.DataFrame,
    ) -> None:
        from quant_trading.risk.contribution import risk_contribution_from_returns

        result = risk_contribution_from_returns(portfolio_weights, multi_returns)
        total = result.component_contribution.sum()
        assert abs(total - result.total_risk) < 1e-8

    def test_percent_sums_to_one(
        self,
        portfolio_weights: pd.Series,
        multi_returns: pd.DataFrame,
    ) -> None:
        from quant_trading.risk.contribution import risk_contribution_from_returns

        result = risk_contribution_from_returns(portfolio_weights, multi_returns)
        assert abs(result.percent_contribution.sum() - 1.0) < 1e-8

    def test_diversification_ratio_gte_one(
        self,
        portfolio_weights: pd.Series,
        multi_returns: pd.DataFrame,
    ) -> None:
        from quant_trading.risk.contribution import risk_contribution_from_returns

        result = risk_contribution_from_returns(portfolio_weights, multi_returns)
        assert result.diversification_ratio >= 1.0

    def test_missing_asset_in_cov_raises(
        self, portfolio_weights: pd.Series
    ) -> None:
        from quant_trading.risk.contribution import risk_contribution

        bad_cov = pd.DataFrame(
            [[1.0]], index=["A"], columns=["A"]
        )  # missing B, C
        with pytest.raises(ValueError, match="missing from cov"):
            risk_contribution(portfolio_weights, bad_cov)

    def test_component_var_positive(
        self,
        portfolio_weights: pd.Series,
        multi_returns: pd.DataFrame,
    ) -> None:
        from quant_trading.risk.contribution import component_var

        cvar_series = component_var(portfolio_weights, multi_returns)
        # Diversified portfolio with positive weights → most component VaRs positive
        assert cvar_series.sum() > 0

    def test_component_var_sums_approx_portfolio_var(
        self,
        portfolio_weights: pd.Series,
        multi_returns: pd.DataFrame,
    ) -> None:
        from quant_trading.risk.contribution import component_var
        from quant_trading.risk.var import var_parametric_normal

        assets = portfolio_weights.index.tolist()
        w = portfolio_weights.values
        port_ret = (multi_returns[assets].dropna().values @ w)
        port_series = pd.Series(port_ret)
        port_var = var_parametric_normal(port_series).var

        cvar_sum = component_var(portfolio_weights, multi_returns).sum()
        # Euler decomposition: sum of component VaRs ≈ portfolio VaR (delta-normal)
        assert abs(cvar_sum - port_var) / port_var < 0.10  # within 10%

    def test_empty_returns_raises(self, portfolio_weights: pd.Series) -> None:
        from quant_trading.risk.contribution import risk_contribution_from_returns

        with pytest.raises(ValueError, match="empty"):
            risk_contribution_from_returns(
                portfolio_weights, pd.DataFrame(dtype=float)
            )


# ===========================================================================
# Drawdown analytics
# ===========================================================================


class TestDrawdownAnalysis:
    def test_returns_drawdown_analysis(self, equity_curve: pd.Series) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns, DrawdownAnalysis

        result = compute_drawdowns(equity_curve)
        assert isinstance(result, DrawdownAnalysis)

    def test_underwater_le_zero(self, equity_curve: pd.Series) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns

        result = compute_drawdowns(equity_curve)
        assert (result.underwater <= 0).all()

    def test_max_drawdown_negative(self, equity_curve: pd.Series) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns

        result = compute_drawdowns(equity_curve)
        assert result.max_drawdown <= 0

    def test_no_episodes_on_monotonic_curve(self) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns

        dates = pd.date_range("2021-01-01", periods=100, freq="B")
        curve = pd.Series(np.linspace(1.0, 2.0, 100), index=dates)
        result = compute_drawdowns(curve)
        assert result.max_drawdown == 0.0
        assert len(result.episodes) == 0

    def test_episode_trough_le_peak(self, equity_curve: pd.Series) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns

        result = compute_drawdowns(equity_curve)
        for ep in result.episodes:
            assert ep.trough_date >= ep.peak_date
            assert ep.max_drawdown <= 0

    def test_episodes_to_dataframe_shape(self, equity_curve: pd.Series) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns, episodes_to_dataframe

        result = compute_drawdowns(equity_curve)
        df = episodes_to_dataframe(result)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "peak_date",
            "trough_date",
            "recovery_date",
            "max_drawdown",
            "duration_days",
            "recovery_days",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_calmar_sign(self, equity_curve: pd.Series) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns

        result = compute_drawdowns(equity_curve)
        if result.calmar_ratio is not None and result.max_drawdown < 0:
            # calmar = ann_return / |max_dd|; sign depends on ann_return
            assert isinstance(result.calmar_ratio, float)

    def test_empty_equity_curve_raises(self) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns

        with pytest.raises(ValueError, match="empty"):
            compute_drawdowns(pd.Series(dtype=float))

    def test_non_positive_equity_curve_raises(self) -> None:
        from quant_trading.risk.drawdown import compute_drawdowns

        dates = pd.date_range("2021-01-01", periods=5, freq="B")
        curve = pd.Series([1.0, 0.9, -0.1, 0.8, 0.7], index=dates)
        with pytest.raises(ValueError, match="positive"):
            compute_drawdowns(curve)


# ===========================================================================
# Rolling metrics
# ===========================================================================


class TestRollingMetrics:
    def test_rolling_volatility_length(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.rolling import rolling_volatility

        result = rolling_volatility(daily_returns, window=21)
        assert len(result) == len(daily_returns)

    def test_rolling_volatility_nan_prefix(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.rolling import rolling_volatility

        window = 21
        result = rolling_volatility(daily_returns, window=window)
        assert result.iloc[: window - 1].isna().all()
        assert result.iloc[window - 1 :].notna().all()

    def test_rolling_volatility_positive(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.rolling import rolling_volatility

        result = rolling_volatility(daily_returns, window=21)
        assert (result.dropna() > 0).all()

    def test_rolling_sharpe_length(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.rolling import rolling_sharpe

        result = rolling_sharpe(daily_returns, window=21)
        assert len(result) == len(daily_returns)

    def test_rolling_sortino_length(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.rolling import rolling_sortino

        result = rolling_sortino(daily_returns, window=21)
        assert len(result) == len(daily_returns)

    def test_rolling_beta_length(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.rolling import rolling_beta

        benchmark = daily_returns * 0.8 + RNG.normal(0, 0.005, len(daily_returns))
        benchmark.index = daily_returns.index
        result = rolling_beta(daily_returns, benchmark, window=21)
        assert len(result) == len(daily_returns)

    def test_rolling_beta_empty_raises(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.rolling import rolling_beta

        with pytest.raises(ValueError):
            rolling_beta(daily_returns, pd.Series(dtype=float), window=21)

    def test_rolling_var_positive(self, daily_returns: pd.Series) -> None:
        from quant_trading.risk.rolling import rolling_var

        result = rolling_var(daily_returns, window=21)
        assert (result.dropna() > 0).all()

    def test_rolling_var_bad_confidence_raises(
        self, daily_returns: pd.Series
    ) -> None:
        from quant_trading.risk.rolling import rolling_var

        with pytest.raises(ValueError, match="confidence"):
            rolling_var(daily_returns, confidence=2.0)

    def test_rolling_mean_correlation_length(
        self, multi_returns: pd.DataFrame
    ) -> None:
        from quant_trading.risk.rolling import rolling_mean_correlation

        result = rolling_mean_correlation(multi_returns, window=21)
        assert len(result) == len(multi_returns.dropna(how="all"))

    def test_rolling_mean_correlation_range(
        self, multi_returns: pd.DataFrame
    ) -> None:
        from quant_trading.risk.rolling import rolling_mean_correlation

        result = rolling_mean_correlation(multi_returns, window=21)
        valid = result.dropna()
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_rolling_mean_correlation_single_asset_raises(
        self, daily_returns: pd.Series
    ) -> None:
        from quant_trading.risk.rolling import rolling_mean_correlation

        with pytest.raises(ValueError, match="2 asset"):
            rolling_mean_correlation(daily_returns.to_frame(), window=21)

    def test_rolling_empty_raises(self) -> None:
        from quant_trading.risk.rolling import rolling_volatility

        with pytest.raises(ValueError, match="empty"):
            rolling_volatility(pd.Series(dtype=float))


# ===========================================================================
# Integration: __init__ re-exports everything
# ===========================================================================


class TestRiskPackageExports:
    def test_all_symbols_importable(self) -> None:
        from quant_trading.risk import (
            VaRResult,
            var_historical,
            var_parametric_normal,
            var_parametric_t,
            var_summary,
            StressScenario,
            run_stress_test,
            RiskContributionResult,
            risk_contribution,
            risk_contribution_from_returns,
            component_var,
            DrawdownEpisode,
            DrawdownAnalysis,
            compute_drawdowns,
            episodes_to_dataframe,
            rolling_volatility,
            rolling_sharpe,
            rolling_sortino,
            rolling_beta,
            rolling_var,
            rolling_mean_correlation,
        )
        # If we reach here without ImportError, all symbols are accessible
        assert True
