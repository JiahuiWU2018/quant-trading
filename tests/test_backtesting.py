"""Unit tests for Phase 2 backtesting engine and risk metrics."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Risk metrics tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_returns():
    """Daily returns fixture for testing metrics."""
    np.random.seed(42)
    return pd.Series(np.random.randn(252) * 0.01, name="returns")  # ~1% daily vol


@pytest.fixture
def sample_equity_curve():
    """Equity curve fixture."""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.01)
    equity = (1 + returns).cumprod() * 100_000
    equity.index = pd.date_range("2024-01-01", periods=252, freq="D")
    return equity


class TestRiskMetrics:
    def test_annualized_return_positive(self, sample_returns):
        from quant_trading.risk.metrics import annualized_return
        ann_ret = annualized_return(sample_returns)
        assert isinstance(ann_ret, float)

    def test_annualized_volatility(self, sample_returns):
        from quant_trading.risk.metrics import annualized_volatility
        ann_vol = annualized_volatility(sample_returns)
        assert ann_vol > 0

    def test_sharpe_ratio(self, sample_returns):
        from quant_trading.risk.metrics import sharpe_ratio
        sharpe = sharpe_ratio(sample_returns, risk_free_rate=0.02)
        assert isinstance(sharpe, float)

    def test_sortino_ratio(self, sample_returns):
        from quant_trading.risk.metrics import sortino_ratio
        sortino = sortino_ratio(sample_returns, risk_free_rate=0.02)
        assert isinstance(sortino, float)

    def test_max_drawdown(self, sample_equity_curve):
        from quant_trading.risk.metrics import max_drawdown
        max_dd, peak, trough = max_drawdown(sample_equity_curve)
        assert max_dd <= 0
        assert peak <= trough

    def test_drawdown_duration(self, sample_equity_curve):
        from quant_trading.risk.metrics import drawdown_duration
        duration = drawdown_duration(sample_equity_curve)
        assert duration >= pd.Timedelta(0)

    def test_compute_metrics(self, sample_returns, sample_equity_curve):
        from quant_trading.risk.metrics import compute_metrics
        metrics = compute_metrics(sample_returns, sample_equity_curve)
        required_keys = {
            "ann_return",
            "ann_volatility",
            "sharpe",
            "sortino",
            "max_dd",
            "max_dd_duration_days",
        }
        assert required_keys.issubset(metrics.keys())

    def test_metrics_raise_on_empty(self):
        from quant_trading.risk.metrics import annualized_return
        empty = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="empty"):
            annualized_return(empty)


# ---------------------------------------------------------------------------
# Sizing tests
# ---------------------------------------------------------------------------

class TestSizing:
    def test_fixed_fractional_size(self):
        from quant_trading.backtesting.sizing import fixed_fractional_size
        size = fixed_fractional_size(portfolio_value=100_000, price=100, fraction=0.1)
        assert size == 100

    def test_fixed_fractional_size_zero_price_raises(self):
        from quant_trading.backtesting.sizing import fixed_fractional_size
        with pytest.raises(ValueError, match="price must be positive"):
            fixed_fractional_size(100_000, 0, 0.1)

    def test_volatility_targeted_size(self, sample_returns):
        from quant_trading.backtesting.sizing import volatility_targeted_size
        size = volatility_targeted_size(
            portfolio_value=100_000,
            price=100,
            returns=sample_returns,
            target_vol=0.10,
        )
        assert isinstance(size, int)
        assert size >= 0

    def test_equal_weight_size(self):
        from quant_trading.backtesting.sizing import equal_weight_size
        prices = {"AAPL": 150, "MSFT": 300}
        sizes = equal_weight_size(portfolio_value=100_000, prices=prices)
        assert "AAPL" in sizes
        assert "MSFT" in sizes
        assert sizes["AAPL"] > 0


# ---------------------------------------------------------------------------
# BaseSignal tests
# ---------------------------------------------------------------------------

class TestBaseSignal:
    def test_base_signal_is_abstract(self):
        from quant_trading.signals.base import BaseSignal
        with pytest.raises(TypeError):
            BaseSignal()  # cannot instantiate ABC

    def test_validate_prices(self):
        from quant_trading.signals.base import BaseSignal
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [105, 106],
                "low": [95, 96],
                "close": [101, 102],
                "volume": [1000, 1100],
            }
        )
        # Should not raise
        BaseSignal.validate_prices(None, df, min_rows=2)

    def test_validate_prices_empty_raises(self):
        from quant_trading.signals.base import BaseSignal
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            BaseSignal.validate_prices(None, empty)

    def test_validate_prices_missing_column_raises(self):
        from quant_trading.signals.base import BaseSignal
        df = pd.DataFrame({"close": [100, 101]})
        with pytest.raises(ValueError, match="missing required columns"):
            BaseSignal.validate_prices(None, df)


# ---------------------------------------------------------------------------
# BacktestEngine integration test (uses a trivial strategy)
# ---------------------------------------------------------------------------

class TrivialBuyHoldStrategy:
    """Minimal concrete strategy for testing (not using BaseStrategy to avoid BT complexity in unit tests)."""
    pass


class TestBacktestEngine:
    def test_engine_initialization(self):
        from quant_trading.backtesting.engine import BacktestEngine
        engine = BacktestEngine(initial_cash=50_000, commission=0.0005)
        assert engine.initial_cash == 50_000

    def test_add_data_validates_columns(self):
        from quant_trading.backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        bad_df = pd.DataFrame({"close": [100, 101]})
        with pytest.raises(ValueError, match="missing required columns"):
            engine.add_data(bad_df, name="bad")

    def test_add_data_validates_index(self):
        from quant_trading.backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [105, 106],
                "low": [95, 96],
                "close": [101, 102],
                "volume": [1000, 1100],
            }
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            engine.add_data(df, name="test")

    def test_set_sizer_fixed_fractional(self):
        from quant_trading.backtesting.engine import BacktestEngine
        from quant_trading.backtesting.sizing import FixedFractionalSizer
        engine = BacktestEngine()
        # Should not raise
        engine.set_sizer(FixedFractionalSizer, fraction=0.05)

    def test_set_sizer_volatility_targeted(self):
        from quant_trading.backtesting.engine import BacktestEngine
        from quant_trading.backtesting.sizing import VolatilityTargetedSizer
        engine = BacktestEngine()
        engine.set_sizer(VolatilityTargetedSizer, target_vol=0.15, lookback=20)

    def test_set_sizer_rejects_non_sizer(self):
        from quant_trading.backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        with pytest.raises(TypeError, match="must be a subclass of bt.Sizer"):
            engine.set_sizer(object)  # type: ignore


# Full backtest integration test is in a separate notebook for simplicity
