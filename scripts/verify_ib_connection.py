"""Verify IB Gateway connectivity and basic API functionality.

Run this script before any paper trading session to confirm:
  1. IB Gateway is reachable
  2. Account data is readable
  3. Delayed market data is available (paper accounts use delayed quotes)
  4. Historical bars can be fetched

Usage:
    # From repo root with venv active and .env loaded:
    python scripts/verify_ib_connection.py

    # Or with explicit .env loading:
    set -a && source .env && set +a && python scripts/verify_ib_connection.py

Required environment variables (from .env):
    IB_HOST       — IB Gateway host (default: 127.0.0.1)
    IB_PORT       — IB Gateway port (default: 7497 for paper)
    IB_CLIENT_ID  — Client ID (default: 1)
    IB_ACCOUNT    — Paper account ID (e.g. DU123456)

This script is read-only — it never places orders.
"""

import logging
import os
import sys

# ---------------------------------------------------------------------------
# Logging — plain output so results are easy to read in terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,   # suppress ib_async internal chatter
    format="%(levelname)s  %(name)s: %(message)s",
)
logger = logging.getLogger("verify_ib")
logger.setLevel(logging.DEBUG)   # our messages always shown


def _check_env() -> tuple[str, int, int, str]:
    """Read and validate required environment variables.

    Returns:
        Tuple of (host, port, client_id, account).

    Raises:
        SystemExit: If any required variable is missing or invalid.
    """
    host = os.getenv("IB_HOST", "127.0.0.1")
    account = os.getenv("IB_ACCOUNT", "")

    try:
        port = int(os.getenv("IB_PORT", "7497"))
    except ValueError:
        logger.error("IB_PORT must be an integer. Got: %r", os.getenv("IB_PORT"))
        sys.exit(1)

    try:
        client_id = int(os.getenv("IB_CLIENT_ID", "1"))
    except ValueError:
        logger.error("IB_CLIENT_ID must be an integer. Got: %r", os.getenv("IB_CLIENT_ID"))
        sys.exit(1)

    if not account:
        logger.error(
            "IB_ACCOUNT is not set. Add it to your .env file (e.g. IB_ACCOUNT=DU123456)."
        )
        sys.exit(1)

    return host, port, client_id, account


def _separator(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def main() -> None:
    # ------------------------------------------------------------------
    # Step 0 — environment check
    # ------------------------------------------------------------------
    _separator("Step 0 — Environment")
    host, port, client_id, account = _check_env()
    print(f"  IB_HOST      : {host}")
    print(f"  IB_PORT      : {port}")
    print(f"  IB_CLIENT_ID : {client_id}")
    print(f"  IB_ACCOUNT   : {account[:2]}{'*' * (len(account) - 2)}")  # mask middle digits
    print("  ✓ Environment variables loaded")

    # ------------------------------------------------------------------
    # Step 1 — import ib_async
    # ------------------------------------------------------------------
    _separator("Step 1 — Import ib_async")
    try:
        from ib_async import IB, Stock, util  # type: ignore[import]
        import ib_async
        print(f"  ib_async version : {ib_async.__version__}")
        print("  ✓ ib_async imported successfully")
    except ImportError as exc:
        logger.error("Could not import ib_async: %s", exc)
        logger.error("Run: pip install ib_async")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2 — connect to IB Gateway
    # ------------------------------------------------------------------
    _separator("Step 2 — Connect to IB Gateway")
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
        print(f"  Server version   : {ib.client.serverVersion()}")
        # print(f"  Connection time  : {ib.client.twsConnectionTime()}")
        print("  ✓ Connected to IB Gateway")
    except Exception as exc:
        logger.error("Connection failed: %s", exc)
        logger.error(
            "Checklist:\n"
            "  • Is IB Gateway running and logged in?\n"
            "  • Is API enabled? (Gateway → Settings → API → Enable ActiveX and Socket Clients)\n"
            "  • Is port %d configured in Gateway API settings?\n"
            "  • Is 127.0.0.1 in the trusted IPs list?",
            port,
        )
        sys.exit(1)

    try:
        # ------------------------------------------------------------------
        # Step 3 — request delayed market data (required for paper accounts)
        # ------------------------------------------------------------------
        _separator("Step 3 — Market Data Type")
        # 1=live, 2=frozen, 3=delayed, 4=delayed-frozen
        # Paper accounts do not have live data — delayed is correct.
        ib.reqMarketDataType(4)
        print("  Market data type : 4 (delayed-frozen — correct for paper accounts)")
        print("  ✓ Market data type set")

        # ------------------------------------------------------------------
        # Step 4 — account values
        # ------------------------------------------------------------------
        _separator("Step 4 — Account Values")
        account_values = ib.accountValues(account)
        if not account_values:
            logger.warning(
                "accountValues() returned empty. "
                "Check that IB_ACCOUNT matches the account shown in IB Gateway."
            )
        else:
            tags_to_show = {
                "NetLiquidation",
                "TotalCashValue",
                "BuyingPower",
                "UnrealizedPnL",
            }
            found: dict[str, str] = {}
            for av in account_values:
                if av.tag in tags_to_show and av.currency == "USD":
                    found[av.tag] = av.value

            for tag in sorted(tags_to_show):
                val = found.get(tag, "—")
                print(f"  {tag:<25}: {val}")

            if "NetLiquidation" in found:
                print("  ✓ Account values readable")
            else:
                logger.warning(
                    "NetLiquidation not found. "
                    "IB Gateway may still be loading account data — wait 10s and retry."
                )

        # ------------------------------------------------------------------
        # Step 5 — open positions
        # ------------------------------------------------------------------
        _separator("Step 5 — Positions")
        positions = ib.positions(account)
        if not positions:
            print("  No open positions (expected for a fresh paper account)")
        else:
            for pos in positions:
                print(f"  {pos.contract.symbol:<10} qty={pos.position:>10.2f}  "
                      f"avg_cost={pos.avgCost:.4f}")
        print("  ✓ Position query succeeded")

        # ------------------------------------------------------------------
        # Step 6 — historical data (AAPL, 5 days, daily bars)
        # ------------------------------------------------------------------
        _separator("Step 6 — Historical Data (AAPL, 5 days)")
        contract = Stock("AAPL", "SMART", "USD")
        ib.qualifyContracts(contract)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",          # "" = current time
            durationStr="5 D",       # 5 calendar days
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,             # regular trading hours
            formatDate=1,
        )

        if not bars:
            logger.warning(
                "reqHistoricalData returned no bars. "
                "This may be a pacing violation — wait 10 seconds and retry."
            )
        else:
            df = util.df(bars)
            print(f"  Rows returned    : {len(df)}")
            print(f"  Columns          : {list(df.columns)}")
            print(f"  Date range       : {df['date'].iloc[0]}  →  {df['date'].iloc[-1]}")
            print(f"  Latest close     : {df['close'].iloc[-1]:.2f}")
            print("  ✓ Historical data fetch succeeded")

        # ------------------------------------------------------------------
        # Step 7 — open orders (should be empty)
        # ------------------------------------------------------------------
        _separator("Step 7 — Open Orders")
        open_trades = ib.openTrades()
        if not open_trades:
            print("  No open orders (expected)")
        else:
            for trade in open_trades:
                print(
                    f"  order_id={trade.order.orderId}  "
                    f"symbol={trade.contract.symbol}  "
                    f"action={trade.order.action}  "
                    f"qty={trade.order.totalQuantity}  "
                    f"status={trade.orderStatus.status}"
                )
        print("  ✓ Open order query succeeded")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        _separator("Summary")
        print("  All checks passed. IB Gateway connection is healthy.")
        print("  You are ready to run paper trading scripts.\n")

    finally:
        ib.disconnect()
        logger.debug("Disconnected from IB Gateway.")


if __name__ == "__main__":
    main()
