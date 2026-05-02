"""Integration tests for IBKR connectivity — require a running IB Gateway.

These tests are SKIPPED unless the environment variable INTEGRATION_TESTS=1
is set AND IB_HOST / IB_PORT point to a reachable IB Gateway instance.

Run manually:
    INTEGRATION_TESTS=1 IB_HOST=127.0.0.1 IB_PORT=7497 pytest tests/integration/ -v

Never run these in CI without a dedicated paper-trading IB Gateway instance.
"""

import os

import pytest

INTEGRATION = os.getenv("INTEGRATION_TESTS", "0") == "1"
skip_reason = "Set INTEGRATION_TESTS=1 with a running IB Gateway to run these tests."


@pytest.mark.skipif(not INTEGRATION, reason=skip_reason)
class TestIBKRConnectivity:
    """Smoke tests against a live IB Gateway (paper trading account)."""

    @pytest.fixture(scope="class")
    def ib(self):
        """Connect a shared IB instance for the test class."""
        from ib_insync import IB  # type: ignore[import]
        ib = IB()
        host = os.getenv("IB_HOST", "127.0.0.1")
        port = int(os.getenv("IB_PORT", "7497"))
        client_id = int(os.getenv("IB_CLIENT_ID", "99"))  # use a distinct ID for tests
        ib.connect(host, port, clientId=client_id)
        yield ib
        ib.disconnect()

    @pytest.fixture(scope="class")
    def adapter(self, ib):
        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(ib=ib)
        adapter._connected = True   # already connected via shared ib fixture
        return adapter

    @pytest.fixture(scope="class")
    def connector(self, ib):
        from quant_trading.data.apis.ibkr_connector import IBKRConnector
        return IBKRConnector(ib=ib)

    def test_get_account_value(self, adapter):
        value = adapter.get_account_value()
        assert isinstance(value, float)
        assert value > 0, "Account value should be positive on a funded paper account."

    def test_get_positions(self, adapter):
        positions = adapter.get_positions()
        assert isinstance(positions, dict)
        # Positions may be empty — just check types
        for symbol, qty in positions.items():
            assert isinstance(symbol, str)
            assert isinstance(qty, (int, float))

    def test_get_open_orders(self, adapter):
        orders = adapter.get_open_orders()
        assert isinstance(orders, list)

    def test_fetch_historical_daily(self, connector):
        df = connector.fetch_price_history("AAPL", freq="1d")
        assert not df.empty
        assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
        assert df.index.tz is not None, "Index must be tz-aware."

    def test_fetch_historical_hourly(self, connector):
        df = connector.fetch_price_history("AAPL", freq="1h")
        assert not df.empty

    def test_submit_and_cancel_order(self, adapter):
        """Submit a tiny limit order far below market, then immediately cancel."""
        from quant_trading.execution.base_adapter import Order, OrderSide, OrderStatus, OrderType
        import os
        # Safety: this test must only run with DRY_RUN=false explicitly set
        assert os.getenv("DRY_RUN", "true").lower() == "false", (
            "Set DRY_RUN=false explicitly to run the order submission integration test."
        )
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            limit_price=1.0,   # far below market — will never fill
            strategy_id="integration_test",
        )
        submitted = adapter.submit_order(order)
        assert submitted.status == OrderStatus.SUBMITTED
        assert submitted.broker_order_id is not None

        # Cancel immediately
        adapter.cancel_order(submitted)
        assert submitted.status == OrderStatus.CANCELLED
