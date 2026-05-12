"""Integration tests for IBKR connectivity — require a running IB Gateway.

These tests are SKIPPED unless the environment variable INTEGRATION_TESTS=1
is set AND IB_HOST / IB_PORT point to a reachable IB Gateway instance.

Run manually (from repo root, with venv active and .env loaded):
    INTEGRATION_TESTS=1 pytest tests/integration/ -v

Set DRY_RUN=false in .env to also run the order submit/cancel test.
Never run these in CI without a dedicated paper-trading IB Gateway instance.
"""

import os

import pytest

INTEGRATION = os.getenv("INTEGRATION_TESTS", "0") == "1"
skip_reason = "Set INTEGRATION_TESTS=1 with a running IB Gateway to run these tests."


@pytest.mark.skipif(not INTEGRATION, reason=skip_reason)
class TestIBKRConnectivity:
    """Smoke tests against a live IB Gateway (paper trading account).

    All tests in this class share a single IB() connection (scope="class")
    to avoid hitting IB's per-client-ID connection limit and to minimise
    pacing violations from repeated connections.
    """

    @pytest.fixture(scope="class")
    def ib(self):
        """Connect a shared IB instance for the test class.

        Uses IB_CLIENT_ID_TEST (default: 99) to avoid conflicting with
        the main adapter connection (IB_CLIENT_ID, default: 1).
        """
        from ib_async import IB  # type: ignore[import]
        ib = IB()
        host = os.getenv("IB_HOST", "127.0.0.1")
        port = int(os.getenv("IB_PORT", "7497"))
        # Use a separate client ID for tests — must not collide with the main adapter
        client_id = int(os.getenv("IB_CLIENT_ID_TEST", "99"))
        ib.connect(host, port, clientId=client_id)
        # Paper accounts require delayed market data — set immediately after connect
        ib.reqMarketDataType(4)
        yield ib
        ib.disconnect()

    @pytest.fixture(scope="class")
    def adapter(self, ib):
        """IBKRAdapter wrapping the shared IB instance."""
        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        adapter = IBKRAdapter(ib=ib)
        # Mark as connected — the shared ib fixture already connected above
        adapter._connected = True
        return adapter

    @pytest.fixture(scope="class")
    def connector(self, ib):
        """IBKRConnector wrapping the shared IB instance."""
        from quant_trading.data.apis.ibkr_connector import IBKRConnector
        return IBKRConnector(ib=ib)

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------

    def test_get_account_value(self, adapter):
        """Net liquidation value must be a positive float on a funded paper account."""
        value = adapter.get_account_value()
        assert isinstance(value, float)
        assert value > 0, (
            f"Account value should be positive on a funded paper account, got {value}. "
            "Check that IB_ACCOUNT in .env matches the account shown in IB Gateway."
        )

    def test_get_positions(self, adapter):
        """Positions query must return a dict (may be empty on a fresh account)."""
        positions = adapter.get_positions()
        assert isinstance(positions, dict)
        for symbol, qty in positions.items():
            assert isinstance(symbol, str), f"Symbol must be str, got {type(symbol)}"
            assert isinstance(qty, (int, float)), f"Qty must be numeric, got {type(qty)}"

    def test_get_open_orders(self, adapter):
        """Open orders query must return a list (may be empty)."""
        orders = adapter.get_open_orders()
        assert isinstance(orders, list)

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def test_fetch_historical_daily(self, connector):
        """Daily bars for AAPL: non-empty, correct columns, tz-aware index."""
        df = connector.fetch_price_history("AAPL", freq="1d")
        assert not df.empty, "Expected daily bars for AAPL — got empty DataFrame."
        assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
        assert df.index.tzinfo is not None, "DatetimeIndex must be tz-aware (UTC)."
        assert (df["close"] > 0).all(), "All close prices must be positive."

    def test_fetch_historical_hourly(self, connector):
        """Hourly bars for AAPL: non-empty, tz-aware index."""
        df = connector.fetch_price_history("AAPL", freq="1h")
        assert not df.empty, "Expected hourly bars for AAPL — got empty DataFrame."
        assert df.index.tzinfo is not None

    def test_fetch_historical_with_start_date(self, connector):
        """Bars fetched with an explicit start date must not predate it."""
        from datetime import datetime, timezone
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        df = connector.fetch_price_history("AAPL", freq="1d", start=start)
        assert not df.empty
        assert df.index.min() >= start, (
            f"Earliest bar {df.index.min()} predates requested start {start}."
        )

    def test_invalid_symbol_raises(self, connector):
        """A clearly invalid symbol must raise RuntimeError (qualify or data error)."""
        with pytest.raises(RuntimeError):
            connector.fetch_price_history("ZZZZZINVALID999", freq="1d")

    # ------------------------------------------------------------------
    # Order submit / cancel  (requires DRY_RUN=false)
    # ------------------------------------------------------------------

    def test_submit_and_cancel_order(self, adapter):
        """Submit a limit order far below market, then cancel it immediately.

        The limit price of $1.00 ensures it will never fill on a paper account.
        Requires DRY_RUN=false to be set explicitly in .env.
        """
        from quant_trading.execution.base_adapter import Order, OrderSide, OrderStatus, OrderType

        dry_run = os.getenv("DRY_RUN", "true").strip().lower()
        if dry_run != "false":
            pytest.skip("Set DRY_RUN=false in .env to run the order submission test.")

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=1,
            order_type=OrderType.LIMIT,
            limit_price=1.00,    # far below market — will never fill
            strategy_id="integration_test",
        )

        submitted = adapter.submit_order(order)
        assert submitted.status == OrderStatus.SUBMITTED, (
            f"Expected SUBMITTED, got {submitted.status}. "
            "Check IB Gateway Activity Monitor for rejection reason."
        )
        assert submitted.broker_order_id is not None

        # Cancel immediately — do not leave open orders on the account
        adapter.cancel_order(submitted)
        assert submitted.status == OrderStatus.CANCELLED
