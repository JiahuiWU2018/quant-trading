"""Unit tests for Phase 3 execution layer.

Tests are fully mocked — no IB Gateway connection required.
Integration tests (requiring IB Gateway) live in tests/integration/.
"""

import os
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_trading.execution.base_adapter import Order, OrderSide, OrderStatus, OrderType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_order():
    return Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        order_type=OrderType.MARKET,
        strategy_id="test_strategy",
    )


@pytest.fixture
def sample_positions():
    return {"AAPL": 50, "MSFT": 100}


@pytest.fixture
def mock_adapter():
    """A mock BrokerAdapter that simulates successful order submission."""
    adapter = MagicMock()
    adapter.is_connected = True
    adapter.get_positions.return_value = {}
    adapter.get_account_value.return_value = 100_000.0
    adapter.get_open_orders.return_value = []

    def fake_submit(order):
        order.broker_order_id = f"MOCK-{id(order)}"
        order.status = OrderStatus.SUBMITTED
        return order

    adapter.submit_order.side_effect = fake_submit
    return adapter


# ---------------------------------------------------------------------------
# Order dataclass validation
# ---------------------------------------------------------------------------


class TestOrderDataclass:
    def test_valid_market_order(self, sample_order):
        assert sample_order.symbol == "AAPL"
        assert sample_order.side == OrderSide.BUY
        assert sample_order.quantity == 10
        assert sample_order.status == OrderStatus.PENDING

    def test_zero_quantity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=0)

    def test_limit_order_without_price_raises(self):
        with pytest.raises(ValueError, match="limit_price"):
            Order(symbol="AAPL", side=OrderSide.BUY, quantity=5, order_type=OrderType.LIMIT)

    def test_limit_order_with_price_ok(self):
        o = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5,
                  order_type=OrderType.LIMIT, limit_price=150.0)
        assert o.limit_price == 150.0

    def test_stop_order_without_price_raises(self):
        with pytest.raises(ValueError, match="stop_price"):
            Order(symbol="AAPL", side=OrderSide.SELL, quantity=5, order_type=OrderType.STOP)


# ---------------------------------------------------------------------------
# SafetyChecker
# ---------------------------------------------------------------------------


class TestSafetyChecker:
    def test_dry_run_blocks_order(self, sample_order, monkeypatch):
        from quant_trading.execution.safety import SafetyChecker, SafetyError
        monkeypatch.setenv("DRY_RUN", "true")
        checker = SafetyChecker()
        with pytest.raises(SafetyError, match="DRY_RUN"):
            checker.run_checks(sample_order, positions={}, account_value=100_000)
        assert sample_order.status == OrderStatus.DRY_RUN

    def test_dry_run_false_passes_with_no_other_limits(self, sample_order, monkeypatch):
        from quant_trading.execution.safety import SafetyChecker
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.delenv("MAX_ORDER_NOTIONAL", raising=False)
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        checker = SafetyChecker()
        # Should not raise
        checker.run_checks(sample_order, positions={}, account_value=100_000)

    def test_max_notional_exceeded_raises(self, monkeypatch):
        from quant_trading.execution.safety import SafetyChecker, SafetyError
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("MAX_ORDER_NOTIONAL", "500")
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        checker = SafetyChecker()
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        with pytest.raises(SafetyError, match="MAX_ORDER_NOTIONAL"):
            checker.run_checks(order, positions={}, account_value=100_000, current_price=100.0)

    def test_max_notional_within_limit_passes(self, monkeypatch):
        from quant_trading.execution.safety import SafetyChecker
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("MAX_ORDER_NOTIONAL", "2000")
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        checker = SafetyChecker()
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        checker.run_checks(order, positions={}, account_value=100_000, current_price=100.0)

    def test_max_positions_exceeded_raises(self, monkeypatch):
        from quant_trading.execution.safety import SafetyChecker, SafetyError
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("MAX_POSITIONS", "2")
        monkeypatch.delenv("MAX_ORDER_NOTIONAL", raising=False)
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        checker = SafetyChecker()
        order = Order(symbol="GOOG", side=OrderSide.BUY, quantity=5)
        # 2 existing positions, buying a new symbol → should fail
        with pytest.raises(SafetyError, match="MAX_POSITIONS"):
            checker.run_checks(order, positions={"AAPL": 10, "MSFT": 20}, account_value=100_000)

    def test_max_position_size_exceeded_raises(self, monkeypatch):
        from quant_trading.execution.safety import SafetyChecker, SafetyError
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("MAX_POSITION_SIZE", "50")
        monkeypatch.delenv("MAX_ORDER_NOTIONAL", raising=False)
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        checker = SafetyChecker()
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        with pytest.raises(SafetyError, match="MAX_POSITION_SIZE"):
            checker.run_checks(order, positions={"AAPL": 45}, account_value=100_000)

    def test_sell_ignores_max_position_size(self, monkeypatch):
        from quant_trading.execution.safety import SafetyChecker
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("MAX_POSITION_SIZE", "5")
        monkeypatch.delenv("MAX_ORDER_NOTIONAL", raising=False)
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        checker = SafetyChecker()
        order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=100)
        # Sells are not blocked by MAX_POSITION_SIZE (closing a position)
        checker.run_checks(order, positions={"AAPL": 100}, account_value=100_000)


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------


class TestOrderManager:
    def test_submit_calls_adapter(self, mock_adapter, monkeypatch):
        from quant_trading.execution.order_manager import OrderManager
        from quant_trading.execution.safety import SafetyChecker
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.delenv("MAX_ORDER_NOTIONAL", raising=False)
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        manager = OrderManager(mock_adapter, min_order_gap=0)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        result = manager.submit(order, positions={}, account_value=100_000)
        mock_adapter.submit_order.assert_called_once_with(order)
        assert result.status == OrderStatus.SUBMITTED

    def test_submit_dry_run_does_not_call_adapter(self, mock_adapter, monkeypatch):
        from quant_trading.execution.order_manager import OrderManager
        from quant_trading.execution.safety import SafetyError
        monkeypatch.setenv("DRY_RUN", "true")
        manager = OrderManager(mock_adapter, min_order_gap=0)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        with pytest.raises(SafetyError):
            manager.submit(order, positions={}, account_value=100_000)
        mock_adapter.submit_order.assert_not_called()

    def test_submit_retries_on_failure(self, monkeypatch):
        from quant_trading.execution.order_manager import OrderManager
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.delenv("MAX_ORDER_NOTIONAL", raising=False)
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)

        adapter = MagicMock()
        adapter.get_positions.return_value = {}
        adapter.get_account_value.return_value = 100_000.0
        # Fail twice, succeed on third attempt
        call_count = {"n": 0}

        def flaky_submit(order):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise RuntimeError("transient error")
            order.broker_order_id = "OK-123"
            order.status = OrderStatus.SUBMITTED
            return order

        adapter.submit_order.side_effect = flaky_submit
        manager = OrderManager(adapter, max_retries=3, retry_backoff=0.0, min_order_gap=0)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        result = manager.submit(order, positions={}, account_value=100_000)
        assert result.status == OrderStatus.SUBMITTED
        assert adapter.submit_order.call_count == 3

    def test_cancel_requires_broker_id(self, mock_adapter):
        from quant_trading.execution.order_manager import OrderManager
        manager = OrderManager(mock_adapter, min_order_gap=0)
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        with pytest.raises(ValueError, match="broker_order_id"):
            manager.cancel(order)

    def test_all_orders_returns_submitted(self, mock_adapter, monkeypatch):
        from quant_trading.execution.order_manager import OrderManager
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.delenv("MAX_ORDER_NOTIONAL", raising=False)
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)
        manager = OrderManager(mock_adapter, min_order_gap=0)
        o1 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
        o2 = Order(symbol="MSFT", side=OrderSide.BUY, quantity=3)
        manager.submit(o1, positions={}, account_value=100_000)
        manager.submit(o2, positions={}, account_value=100_000)
        assert len(manager.all_orders()) == 2


# ---------------------------------------------------------------------------
# IBKRAdapter (mocked — no IB Gateway)
# ---------------------------------------------------------------------------


class TestIBKRAdapterMocked:
    def test_connect_calls_ib(self, monkeypatch):
        monkeypatch.setenv("IB_HOST", "127.0.0.1")
        monkeypatch.setenv("IB_PORT", "7497")
        monkeypatch.setenv("IB_CLIENT_ID", "1")
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True

        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(ib=mock_ib)
        adapter.connect()
        mock_ib.connect.assert_called_once_with("127.0.0.1", 7497, clientId=1)
        assert adapter.is_connected

    def test_dry_run_submit_does_not_call_place_order(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True

        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(ib=mock_ib)
        adapter._connected = True
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10)
        result = adapter.submit_order(order)
        mock_ib.placeOrder.assert_not_called()
        assert result.status == OrderStatus.DRY_RUN
        assert result.broker_order_id.startswith("DRY_RUN")

    def test_get_account_value_reads_net_liquidation(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        mock_ib = MagicMock()
        mock_av = MagicMock()
        mock_av.tag = "NetLiquidation"
        mock_av.currency = "USD"
        mock_av.value = "123456.78"
        mock_ib.accountValues.return_value = [mock_av]
        mock_ib.isConnected.return_value = True

        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(ib=mock_ib)
        adapter._connected = True
        assert adapter.get_account_value() == pytest.approx(123456.78)

    def test_get_positions(self, monkeypatch):
        monkeypatch.setenv("DRY_RUN", "true")
        mock_ib = MagicMock()
        mock_pos = MagicMock()
        mock_pos.contract.symbol = "AAPL"
        mock_pos.position = 100
        mock_ib.positions.return_value = [mock_pos]
        mock_ib.isConnected.return_value = True

        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(ib=mock_ib)
        adapter._connected = True
        positions = adapter.get_positions()
        assert positions == {"AAPL": 100}

    def test_disconnect(self, monkeypatch):
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = False

        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(ib=mock_ib)
        adapter._connected = True
        adapter.disconnect()
        mock_ib.disconnect.assert_called_once()
        assert not adapter._connected


# ---------------------------------------------------------------------------
# StrategyRunner
# ---------------------------------------------------------------------------


class TestStrategyRunner:
    def _make_strategy(self, orders=None):
        """Return a minimal LiveStrategy-compatible stub.

        MagicMock fails @runtime_checkable Protocol isinstance checks because
        attributes are created lazily. Use a concrete stub class instead.
        """
        _orders = orders or []

        class _StubStrategy:
            strategy_id = "test_strategy"

            def on_bar(self, data):
                return _orders

        stub = _StubStrategy()
        # Wrap on_bar so we can assert call counts
        stub.on_bar = MagicMock(side_effect=stub.on_bar)
        return stub

    def _make_df(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        return pd.DataFrame(
            {"open": [100]*5, "high": [105]*5, "low": [95]*5,
             "close": [101]*5, "volume": [1000]*5},
            index=dates,
        )

    def test_runner_creation_valid_strategy(self, mock_adapter):
        from quant_trading.execution.runner import StrategyRunner
        strategy = self._make_strategy()
        runner = StrategyRunner(
            strategy=strategy,
            adapter=mock_adapter,
            data_fetcher=lambda sym: self._make_df(),
            symbols=["AAPL"],
            interval_seconds=60,
        )
        assert not runner.is_running

    def test_runner_rejects_invalid_strategy(self, mock_adapter):
        from quant_trading.execution.runner import StrategyRunner
        with pytest.raises(TypeError, match="LiveStrategy protocol"):
            StrategyRunner(
                strategy=object(),   # does not implement LiveStrategy
                adapter=mock_adapter,
                data_fetcher=lambda sym: self._make_df(),
                symbols=["AAPL"],
            )

    def test_tick_calls_on_bar_and_submits_orders(self, mock_adapter, monkeypatch):
        from quant_trading.execution.runner import StrategyRunner
        from quant_trading.execution.order_manager import OrderManager
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.delenv("MAX_ORDER_NOTIONAL", raising=False)
        monkeypatch.delenv("MAX_POSITIONS", raising=False)
        monkeypatch.delenv("MAX_POSITION_SIZE", raising=False)

        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
        strategy = self._make_strategy(orders=[order])
        manager = OrderManager(mock_adapter, min_order_gap=0)
        runner = StrategyRunner(
            strategy=strategy,
            adapter=mock_adapter,
            data_fetcher=lambda sym: self._make_df(),
            symbols=["AAPL"],
            order_manager=manager,
        )
        runner._tick()
        strategy.on_bar.assert_called_once()
        mock_adapter.submit_order.assert_called_once()

    def test_tick_no_orders_does_not_submit(self, mock_adapter):
        from quant_trading.execution.runner import StrategyRunner
        strategy = self._make_strategy(orders=[])
        runner = StrategyRunner(
            strategy=strategy,
            adapter=mock_adapter,
            data_fetcher=lambda sym: self._make_df(),
            symbols=["AAPL"],
        )
        runner._tick()
        mock_adapter.submit_order.assert_not_called()

    def test_start_and_stop(self, mock_adapter):
        import threading
        from quant_trading.execution.runner import StrategyRunner
        strategy = self._make_strategy()
        runner = StrategyRunner(
            strategy=strategy,
            adapter=mock_adapter,
            data_fetcher=lambda sym: self._make_df(),
            symbols=["AAPL"],
            interval_seconds=1,
        )
        t = threading.Thread(target=runner.start)
        t.start()
        # Let it run one tick then stop
        import time; time.sleep(0.1)
        runner.stop()
        t.join(timeout=3)
        assert not t.is_alive()
        assert not runner.is_running


# ---------------------------------------------------------------------------
# IBKRConnector — _derive_duration (pure function, no IB connection needed)
# ---------------------------------------------------------------------------

class TestDeriveDuration:
    """Tests for the _derive_duration helper in ibkr_connector."""

    def _derive(self, days_back: int, freq: str = "1d") -> str:
        from datetime import timezone
        from quant_trading.data.apis.ibkr_connector import _derive_duration
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)
        start = end - timedelta(days=days_back)
        return _derive_duration(start, end, freq)

    def test_short_range_returns_days(self) -> None:
        result = self._derive(days_back=7)
        assert result.endswith(" D")
        assert int(result.split()[0]) >= 7

    def test_medium_range_returns_months(self) -> None:
        result = self._derive(days_back=60)
        assert result.endswith(" M")

    def test_long_range_returns_years(self) -> None:
        result = self._derive(days_back=400)
        assert result.endswith(" Y")
        assert int(result.split()[0]) >= 1

    def test_single_day_range_minimum_one_day(self) -> None:
        result = self._derive(days_back=1)
        assert result.endswith(" D")
        assert int(result.split()[0]) >= 1

    def test_buffer_applied(self) -> None:
        # 10 days back should produce >= 11 days (10% buffer)
        result = self._derive(days_back=10)
        assert result.endswith(" D")
        assert int(result.split()[0]) >= 11

    def test_exactly_28_days_stays_in_days(self) -> None:
        result = self._derive(days_back=25)  # 25 * 1.1 = 27.5 → 28 D
        assert result.endswith(" D")

    def test_naive_datetimes_handled(self) -> None:
        from quant_trading.data.apis.ibkr_connector import _derive_duration
        # Should not raise even without tzinfo
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 1)
        result = _derive_duration(start, end, "1d")
        assert result.endswith(" M") or result.endswith(" D")


class TestIBKRConnectorMocked:
    """IBKRConnector.fetch_price_history with a fully mocked IB instance."""

    def _make_mock_ib(self, num_bars: int = 5):
        """Return a mock IB instance that returns synthetic bar data."""
        from unittest.mock import MagicMock
        import pandas as pd

        # Build synthetic bars as simple objects with the attributes ib_async returns
        bars = []
        for i in range(num_bars):
            bar = MagicMock()
            bar.date = pd.Timestamp(f"2024-01-{i + 1:02d}", tz="UTC")
            bar.open = 150.0 + i
            bar.high = 155.0 + i
            bar.low  = 148.0 + i
            bar.close = 152.0 + i
            bar.volume = 1_000_000 + i * 1000
            bars.append(bar)

        mock_ib = MagicMock()
        mock_ib.qualifyContracts.return_value = [MagicMock()]  # non-empty = success
        mock_ib.reqHistoricalData.return_value = bars

        # util.df converts the bars list to a proper DataFrame
        import pandas as pd

        def fake_util_df(bar_list):
            return pd.DataFrame(
                {
                    "date":   [b.date for b in bar_list],
                    "open":   [b.open for b in bar_list],
                    "high":   [b.high for b in bar_list],
                    "low":    [b.low  for b in bar_list],
                    "close":  [b.close for b in bar_list],
                    "volume": [b.volume for b in bar_list],
                }
            )

        return mock_ib, fake_util_df

    def test_returns_ohlcv_dataframe(self) -> None:
        from quant_trading.data.apis.ibkr_connector import IBKRConnector

        mock_ib, fake_util_df = self._make_mock_ib(5)
        connector = IBKRConnector(ib=mock_ib)

        with patch("ib_async.Stock"), \
             patch("ib_async.util") as mock_util:
            mock_util.df.side_effect = fake_util_df
            df = connector.fetch_price_history("AAPL", freq="1d")

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 5
        assert df.index.tzinfo is not None

    def test_unsupported_freq_raises_value_error(self) -> None:
        from quant_trading.data.apis.ibkr_connector import IBKRConnector

        mock_ib, _ = self._make_mock_ib()
        connector = IBKRConnector(ib=mock_ib)

        with pytest.raises(ValueError, match="Unsupported freq"):
            connector.fetch_price_history("AAPL", freq="2d")

    def test_qualify_failure_raises_runtime_error(self) -> None:
        from quant_trading.data.apis.ibkr_connector import IBKRConnector

        mock_ib, _ = self._make_mock_ib()
        mock_ib.qualifyContracts.return_value = []  # empty = qualification failed

        connector = IBKRConnector(ib=mock_ib)

        with patch("ib_async.Stock"), \
             patch("ib_async.util"):
            with pytest.raises(RuntimeError, match="could not qualify"):
                connector.fetch_price_history("INVALID", freq="1d")

    def test_empty_bars_raises_runtime_error(self) -> None:
        from quant_trading.data.apis.ibkr_connector import IBKRConnector

        mock_ib, _ = self._make_mock_ib()
        mock_ib.reqHistoricalData.return_value = []  # IB returned nothing

        connector = IBKRConnector(ib=mock_ib)

        with patch("ib_async.Stock"), \
             patch("ib_async.util"):
            with pytest.raises(RuntimeError, match="no data"):
                connector.fetch_price_history("AAPL", freq="1d")

    def test_start_date_trims_result(self) -> None:
        from quant_trading.data.apis.ibkr_connector import IBKRConnector

        mock_ib, fake_util_df = self._make_mock_ib(10)
        connector = IBKRConnector(ib=mock_ib)

        start = datetime(2024, 1, 5, tzinfo=timezone.utc)

        with patch("ib_async.Stock"), \
             patch("ib_async.util") as mock_util:
            mock_util.df.side_effect = fake_util_df
            df = connector.fetch_price_history("AAPL", freq="1d", start=start)

        assert df.index.min() >= start

    def test_default_duration_used_when_no_start(self) -> None:
        from quant_trading.data.apis.ibkr_connector import IBKRConnector, _IB_DEFAULT_DURATION

        mock_ib, fake_util_df = self._make_mock_ib(3)
        connector = IBKRConnector(ib=mock_ib)

        with patch("ib_async.Stock"), \
             patch("ib_async.util") as mock_util:
            mock_util.df.side_effect = fake_util_df
            connector.fetch_price_history("AAPL", freq="1d")

        _, kwargs = mock_ib.reqHistoricalData.call_args
        assert kwargs["durationStr"] == _IB_DEFAULT_DURATION["1d"]

    def test_fetch_fundamentals_raises_not_implemented(self) -> None:
        from quant_trading.data.apis.ibkr_connector import IBKRConnector

        connector = IBKRConnector(ib=MagicMock())
        with pytest.raises(NotImplementedError):
            connector.fetch_fundamentals("AAPL")
