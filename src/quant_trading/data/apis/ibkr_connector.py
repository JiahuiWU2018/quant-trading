"""IBKR data connector — historical and live bars via ib_async.

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
import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_trading.data.apis.base_connector import BaseDataConnector, VALID_FREQS

logger = logging.getLogger(__name__)

# Maps internal freq strings → IB barSizeSetting strings.
# These values are from the official IB API documentation and must match exactly.
_IB_BAR_SIZE: dict[str, str] = {
    "1m":  "1 min",
    "5m":  "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h":  "1 hour",
    "1d":  "1 day",
    "1wk": "1 week",
}

# Default IB durationStr when no start date is provided (~1 year lookback).
# IB imposes per-freq limits on how far back you can go at fine resolutions:
#   1m bars: max ~7 days; 5m/15m/30m: max ~1 month; 1h: max ~6 months
_IB_DEFAULT_DURATION: dict[str, str] = {
    "1m":  "1 W",
    "5m":  "1 M",
    "15m": "1 M",
    "30m": "3 M",
    "1h":  "6 M",
    "1d":  "1 Y",
    "1wk": "3 Y",
}


def _derive_duration(start: datetime, end: datetime, freq: str) -> str:
    """Derive an IB durationStr from a start/end datetime range.

    IB accepts durations in seconds (S), days (D), weeks (W),
    months (M), or years (Y). We choose the coarsest unit that
    covers the full requested range, adding 10% buffer to ensure
    the start date is always included.

    Args:
        start: Start datetime (UTC).
        end: End datetime (UTC, or current time if not specified).
        freq: Internal freq string (e.g. "1d"). Used for a sanity
              cap on fine-resolution requests.

    Returns:
        IB durationStr string (e.g. "30 D", "6 M", "2 Y").
    """
    # Ensure both are UTC-aware for safe subtraction
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    delta: timedelta = end - start
    total_days = max(delta.days, 1)  # minimum 1 day

    # Add 10% buffer so the start bar is always included
    buffered_days = math.ceil(total_days * 1.10)

    if buffered_days <= 28:
        return f"{buffered_days} D"
    elif buffered_days <= 364:
        months = math.ceil(buffered_days / 30)
        return f"{months} M"
    else:
        years = math.ceil(buffered_days / 365)
        return f"{years} Y"


class IBKRConnector(BaseDataConnector):
    """Fetches historical OHLCV data from IB Gateway via ib_async.

    Designed to share an IB() instance with IBKRAdapter.

    Args:
        ib: An ib_async.IB() instance. Must already be connected before
            calling fetch_price_history(). Passing an IBKRAdapter's IB
            instance is the recommended pattern.

    Example:
        from ib_async import IB
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

        When ``start`` is provided the durationStr is derived from the
        start/end range so the full window is always covered. When omitted,
        the default lookback for the given ``freq`` is used (see
        ``_IB_DEFAULT_DURATION``).

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start datetime (UTC). Optional — if omitted the default
                   lookback for the freq is used.
            end: End datetime (UTC). Defaults to now if not provided.
            freq: Bar frequency. One of: "1m","5m","15m","30m","1h","1d","1wk".

        Returns:
            DataFrame with tz-aware UTC DatetimeIndex and columns:
            open, high, low, close, volume.

        Raises:
            ValueError: If freq is not supported.
            RuntimeError: If the IB data request fails or returns no data.
        """
        if freq not in _IB_BAR_SIZE:
            raise ValueError(
                f"Unsupported freq={freq!r}. Supported: {sorted(_IB_BAR_SIZE.keys())}"
            )

        from ib_async import Stock, util  # type: ignore[import]

        # Resolve end datetime — IB uses "" to mean "now"
        end_dt = end if end is not None else datetime.now(tz=timezone.utc)
        end_str = "" if end is None else end_dt.strftime("%Y%m%d %H:%M:%S UTC")

        # Derive duration — from start/end range when available, else default
        if start is not None:
            duration = _derive_duration(start, end_dt, freq)
        else:
            duration = _IB_DEFAULT_DURATION[freq]

        bar_size = _IB_BAR_SIZE[freq]

        logger.info(
            "IBKRConnector: requesting %s bars for %s  duration=%s  end=%s",
            bar_size,
            symbol,
            duration,
            end_str or "now",
        )

        # qualifyContracts resolves any ambiguity in the contract definition
        # (e.g. multiple exchanges matching "AAPL"). Required before
        # reqHistoricalData to avoid IB error 200 "No security definition found".
        contract = Stock(symbol, "SMART", "USD")
        qualified = self._ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(
                f"IB could not qualify contract for symbol={symbol!r}. "
                "Check the symbol is valid and IB Gateway is connected."
            )

        try:
            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,       # regular trading hours only
                formatDate=1,
            )
        except Exception as exc:
            raise RuntimeError(
                f"IB historical data request failed for {symbol}: {exc}"
            ) from exc

        if not bars:
            raise RuntimeError(
                f"IB returned no data for {symbol} at freq={freq!r}. "
                "Possible causes: invalid symbol, pacing violation, or market closed. "
                "Wait 10 seconds and retry if this is a pacing issue."
            )

        df = util.df(bars)[["date", "open", "high", "low", "close", "volume"]]
        df = df.rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        df = self.normalize_ohlcv(df)

        # Trim to requested start if provided (duration may overshoot slightly)
        if start is not None:
            start_utc = start if start.tzinfo is not None else start.replace(tzinfo=timezone.utc)
            df = df[df.index >= start_utc]

        logger.info("IBKRConnector: received %d bars for %s", len(df), symbol)
        return df

    def fetch_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Not implemented — IB fundamental data requires a separate subscription.

        Use a dedicated fundamental data connector (e.g. Finnhub, FMP) instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "fetch_fundamentals is not implemented for IBKRConnector. "
            "Use a dedicated fundamental data connector instead."
        )
