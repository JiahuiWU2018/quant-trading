"""Unit tests for Phase 1 data pipeline.

These tests are designed to run in CI without any external API calls or
credentials. yfinance-dependent tests are integration tests and are
skipped unless the INTEGRATION_TESTS env var is set to "1".
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_ohlcv() -> pd.DataFrame:
    """A minimal synthetic OHLCV DataFrame with UTC DatetimeIndex."""
    idx = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open":   [100.0, 101.0, 102.0, 103.0, 104.0],
            "high":   [105.0, 106.0, 107.0, 108.0, 109.0],
            "low":    [95.0,  96.0,  97.0,  98.0,  99.0],
            "close":  [101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1000,  1100,  1200,  1300,  1400],
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# BaseDataConnector
# ---------------------------------------------------------------------------

class TestBaseConnector:
    def test_validate_freq_valid(self):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        # Should not raise
        BaseDataConnector.validate_freq("1d")
        BaseDataConnector.validate_freq("1h")

    def test_validate_freq_invalid(self):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        with pytest.raises(ValueError, match="Unsupported freq"):
            BaseDataConnector.validate_freq("3d")

    def test_normalize_ohlcv_lowercases_columns(self, sample_ohlcv):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        upper_df = sample_ohlcv.rename(columns=str.upper)
        result = BaseDataConnector.normalize_ohlcv(upper_df)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_normalize_ohlcv_sets_utc(self, sample_ohlcv):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        naive_df = sample_ohlcv.copy()
        naive_df.index = naive_df.index.tz_localize(None)
        result = BaseDataConnector.normalize_ohlcv(naive_df)
        assert str(result.index.tzinfo) == "UTC"

    def test_normalize_ohlcv_missing_column_raises(self, sample_ohlcv):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        df = sample_ohlcv.drop(columns=["volume"])
        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            BaseDataConnector.normalize_ohlcv(df)

    def test_normalize_ohlcv_coerces_non_numeric(self, sample_ohlcv):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        import numpy as np
        dirty = sample_ohlcv.copy()
        # cast column to object so pandas accepts the string assignment
        dirty["close"] = dirty["close"].astype(object)
        dirty.loc[dirty.index[0], "close"] = "n/a"   # non-numeric value
        result = BaseDataConnector.normalize_ohlcv(dirty)
        # row is kept (other columns are still numeric), close on that row is NaN
        assert np.isnan(result["close"].iloc[0])

    def test_normalize_ohlcv_drops_all_nan_rows(self, sample_ohlcv):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        import numpy as np
        all_nan = sample_ohlcv.copy().astype(object)
        all_nan.iloc[1] = np.nan    # entire row NaN
        result = BaseDataConnector.normalize_ohlcv(all_nan)
        assert len(result) == len(sample_ohlcv) - 1

    def test_normalize_ohlcv_sorts_and_deduplicates_index(self, sample_ohlcv):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        # Reverse order + duplicate first timestamp
        df = pd.concat([sample_ohlcv.iloc[::-1], sample_ohlcv.iloc[[0]]])
        result = BaseDataConnector.normalize_ohlcv(df)
        assert result.index.is_monotonic_increasing
        assert not result.index.duplicated().any()
        assert len(result) == len(sample_ohlcv)

    def test_normalize_ohlcv_empty_after_cleaning_raises(self, sample_ohlcv):
        from quant_trading.data.apis.base_connector import BaseDataConnector
        import numpy as np
        all_nan = sample_ohlcv.copy().astype(float)
        all_nan[:] = np.nan
        with pytest.raises(ValueError, match="empty after normalization"):
            BaseDataConnector.normalize_ohlcv(all_nan)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_miss_calls_fetch_fn(self, tmp_path, sample_ohlcv):
        from quant_trading.data.cache import cached_fetch
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 5, tzinfo=timezone.utc)
        fetch_fn = MagicMock(return_value=sample_ohlcv)

        result = cached_fetch(fetch_fn, "TEST", start, end, "1d", "mock", cache_dir=tmp_path)
        fetch_fn.assert_called_once()
        assert result.equals(sample_ohlcv)

    def test_cache_hit_does_not_call_fetch_fn(self, tmp_path, sample_ohlcv):
        from quant_trading.data.cache import cached_fetch
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 5, tzinfo=timezone.utc)
        fetch_fn = MagicMock(return_value=sample_ohlcv)

        # First call populates the cache
        cached_fetch(fetch_fn, "TEST", start, end, "1d", "mock", cache_dir=tmp_path)
        # Second call should use cache
        result = cached_fetch(fetch_fn, "TEST", start, end, "1d", "mock", cache_dir=tmp_path)
        assert fetch_fn.call_count == 1
        assert "close" in result.columns

    def test_cache_bypass(self, tmp_path, sample_ohlcv):
        from quant_trading.data.cache import cached_fetch
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 5, tzinfo=timezone.utc)
        fetch_fn = MagicMock(return_value=sample_ohlcv)

        cached_fetch(fetch_fn, "TEST", start, end, "1d", "mock", cache=False, cache_dir=tmp_path)
        cached_fetch(fetch_fn, "TEST", start, end, "1d", "mock", cache=False, cache_dir=tmp_path)
        assert fetch_fn.call_count == 2

    def test_invalidate_cache(self, tmp_path, sample_ohlcv):
        from quant_trading.data.cache import cached_fetch, invalidate_cache
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 5, tzinfo=timezone.utc)
        fetch_fn = MagicMock(return_value=sample_ohlcv)

        cached_fetch(fetch_fn, "TEST", start, end, "1d", "mock", cache_dir=tmp_path)
        deleted = invalidate_cache("TEST", start, end, "1d", "mock", cache_dir=tmp_path)
        assert deleted is True

        # After invalidation, a second call should re-fetch
        cached_fetch(fetch_fn, "TEST", start, end, "1d", "mock", cache_dir=tmp_path)
        assert fetch_fn.call_count == 2

    def test_invalidate_nonexistent_returns_false(self, tmp_path):
        from quant_trading.data.cache import invalidate_cache
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 1, 5, tzinfo=timezone.utc)
        result = invalidate_cache("NONE", start, end, "1d", "mock", cache_dir=tmp_path)
        assert result is False


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

class TestUniverse:
    def test_deduplication(self):
        from quant_trading.data.universe import Universe
        u = Universe(name="test", symbols=["AAPL", "MSFT", "AAPL"])
        assert len(u) == 2

    def test_contains(self):
        from quant_trading.data.universe import Universe
        u = Universe(name="test", symbols=["AAPL", "MSFT"])
        assert "AAPL" in u
        assert "GOOG" not in u

    def test_filter(self):
        from quant_trading.data.universe import Universe
        u = Universe(name="test", symbols=["AAPL", "MSFT", "GOOG"])
        filtered = u.filter(["AAPL", "GOOG"])
        assert list(filtered) == ["AAPL", "GOOG"]

    def test_exclude(self):
        from quant_trading.data.universe import Universe
        u = Universe(name="test", symbols=["AAPL", "MSFT", "GOOG"])
        result = u.exclude(["MSFT"])
        assert "MSFT" not in result
        assert len(result) == 2

    def test_union(self):
        from quant_trading.data.universe import Universe
        u1 = Universe(name="a", symbols=["AAPL", "MSFT"])
        u2 = Universe(name="b", symbols=["MSFT", "GOOG"])
        combined = u1.union(u2)
        assert len(combined) == 3
        assert "AAPL" in combined
        assert "GOOG" in combined

    def test_iter(self):
        from quant_trading.data.universe import Universe
        symbols = ["AAPL", "MSFT"]
        u = Universe(name="test", symbols=symbols)
        assert list(u) == symbols


# ---------------------------------------------------------------------------
# Loaders (mocked — no real network calls)
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_fetch_price_history_unsupported_source(self):
        from quant_trading.data.loaders import fetch_price_history
        with pytest.raises(ValueError, match="Unknown data source"):
            fetch_price_history(
                "AAPL",
                datetime(2023, 1, 1),
                datetime(2023, 12, 31),
                source="nonexistent_source",
            )

    def test_fetch_multi_logs_and_skips_failed_symbols(self, tmp_path, sample_ohlcv):
        from quant_trading.data.loaders import fetch_multi

        def mock_fetch(symbol, start, end, freq="1d", source="yfinance", cache=True):
            if symbol == "FAIL":
                raise RuntimeError("simulated failure")
            return sample_ohlcv

        with patch("quant_trading.data.loaders.fetch_price_history", side_effect=mock_fetch):
            result = fetch_multi(
                ["AAPL", "FAIL"],
                datetime(2023, 1, 1),
                datetime(2023, 1, 5),
            )
        assert "AAPL" in result.columns
        assert "FAIL" not in result.columns

    def test_fetch_multi_all_fail_raises(self):
        from quant_trading.data.loaders import fetch_multi
        with patch(
            "quant_trading.data.loaders.fetch_price_history",
            side_effect=RuntimeError("fail"),
        ):
            with pytest.raises(RuntimeError, match="No data retrieved"):
                fetch_multi(["FAIL"], datetime(2023, 1, 1), datetime(2023, 1, 5))


# ---------------------------------------------------------------------------
# Integration tests (skipped unless INTEGRATION_TESTS=1)
# ---------------------------------------------------------------------------

INTEGRATION = os.getenv("INTEGRATION_TESTS") == "1"

@pytest.mark.skipif(not INTEGRATION, reason="Set INTEGRATION_TESTS=1 to run")
class TestYFinanceIntegration:
    def test_fetch_real_data(self):
        from quant_trading.data.loaders import fetch_price_history
        df = fetch_price_history(
            "AAPL",
            datetime(2023, 1, 1),
            datetime(2023, 1, 31),
            cache=False,
        )
        assert not df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert str(df.index.tzinfo) == "UTC"
