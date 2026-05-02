"""IBKR data connector — historical and live bars via ib_insync.

Shares a single IB() instance with IBKRAdapter to avoid exceeding
IB's per-client-ID connection limit.

Supported frequencies (mapped from our internal freq strings):
    "1m"  → "1 min"
    "5m"  → "5 mins"
    "15m" → "15 mins"
    "30m" → "30 mins"
    "1h"  → "1 hour"
    "1d"  → "1 day"
    "1wk" → "1 week"

Note: IB enforces pacing limits on historical data requests
(~6 requests per second, max 60 requests per 10 minutes for identical
contracts). The connector does not currently implement automatic pacing —
use sparingly during live runs.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from quant_trading.data.apis.base_connector import BaseDataConnector, VALID_FREQS

logger = logging.getLogger(__name__)

# Maps internal freq strings → IB barSizeSetting strings
_IB_BAR_SIZE: dict[str, str] = {
    "1m":  "1 min",
    "5m":  "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h":  "1 hour",
    "1d":  "1 day",
    "1wk": "1 week",
}

# Maps internal freq → IB durationStr for ~1 year of data
_IB_DURATION: dict[str, str] = {
    "1m":  "1 W",    # 1 week (IB limits 1m to short windows)
    "5m":  "1 M",
    "15m": "1 M",
    "30m": "3 M",
    "1h":  "6 M",
    "1d":  "1 Y",
    "1wk": "3 Y",
}


class IBKRConnector(BaseDataConnector):
    """Fetches historical OHLCV data from IB Gateway via ib_insync.

    Designed to share an IB() instance with IBKRAdapter.

    Args:
        ib: An ib_insync.IB() instance. Must already be connected before
            calling fetch_price_history(). Passing an IBKRAdapter's IB
            instance is the recommended pattern.

    Example:
        from ib_insync import IB
        from quant_trading.execution.ibkr_adapter import IBKRAdapter
        from quant_trading.data.apis.ibkr_connector import IBKRConnector

        ib = IB()
        adapter = IBKRAdapter(ib=ib)
        connector = IBKRConnector(ib=ib)

        adapter.connect()
        df = connector.fetch_price_history("AAPL", freq="1d")
        adapter.disconnect()
    """

    def __init__(self, ib) -> None:
        self._ib = ib

    def fetch_price_history(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from IB.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start datetime (UTC). Currently ignored — IB uses
                   duration-based lookback. See ``_IB_DURATION``.
            end: End datetime (UTC). Defaults to now.
            freq: Bar frequency. One of: "1m","5m","15m","30m","1h","1d","1wk".

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, volume.

        Raises:
            ValueError: If freq is not supported.
            RuntimeError: If the IB data request fails or returns no data.
        """
        if freq not in _IB_BAR_SIZE:
            raise ValueError(
                f"Unsupported freq={freq!r}. Supported: {sorted(_IB_BAR_SIZE.keys())}"
            )

        from ib_insync import Stock, util  # type: ignore[import]

        contract = Stock(symbol, "SMART", "USD")
        end_str = ""  # empty = use current time
        if end is not None:
            end_str = end.strftime("%Y%m%d %H:%M:%S UTC")

        bar_size = _IB_BAR_SIZE[freq]
        duration = _IB_DURATION[freq]

        logger.info(
            "IBKRConnector: requesting %s bars for %s (duration=%s)",
            bar_size,
            symbol,
            duration,
        )

        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        except Exception as exc:
            raise RuntimeError(
                f"IB historical data request failed for {symbol}: {exc}"
            ) from exc

        if not bars:
            raise RuntimeError(
                f"IB returned no data for {symbol} at freq={freq!r}. "
                "Check that the symbol is valid and IB Gateway is connected."
            )

        df = util.df(bars)[["date", "open", "high", "low", "close", "volume"]]
        df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        df = self.normalize_ohlcv(df)

        logger.info("IBKRConnector: received %d bars for %s", len(df), symbol)
        return df
