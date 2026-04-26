"""YFinance data connector — free, no API key required."""

import logging
from datetime import datetime

import pandas as pd

from quant_trading.data.apis.base_connector import BaseDataConnector

logger = logging.getLogger(__name__)

# Map generic freq strings to yfinance interval strings
_FREQ_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}


class YFinanceConnector(BaseDataConnector):
    """Fetches market and fundamental data via yfinance (Yahoo Finance).

    No API key required. Rate limits apply for high-frequency or
    large-universe requests — use the cache layer to avoid repeated calls.

    Example:
        connector = YFinanceConnector()
        df = connector.fetch_price_history(
            "AAPL",
            start=datetime(2023, 1, 1),
            end=datetime(2023, 12, 31),
        )
    """

    def fetch_price_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV history via yfinance.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            freq: Bar frequency. See base_connector.VALID_FREQS.

        Returns:
            DataFrame with tz-aware UTC DatetimeIndex and columns:
            [open, high, low, close, volume].

        Raises:
            ValueError: If freq is not supported or result is empty.
            RuntimeError: If yfinance raises an unexpected error.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from exc

        self.validate_freq(freq)
        interval = _FREQ_MAP[freq]

        logger.debug(
            "YFinanceConnector: fetching %s from %s to %s (freq=%s)",
            symbol,
            start.date(),
            end.date(),
            freq,
        )

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=True,
                actions=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"yfinance failed to fetch history for '{symbol}': {exc}"
            ) from exc

        if df.empty:
            raise ValueError(
                f"yfinance returned no data for '{symbol}' between {start} and {end}."
            )

        return self.normalize_ohlcv(df)

    def fetch_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fetch key fundamental metrics via yfinance Ticker.info.

        Returns a single-row DataFrame with selected fields.
        Available fields depend on Yahoo Finance coverage and may change
        without notice.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").

        Returns:
            Single-row DataFrame with fundamental fields as columns.

        Raises:
            RuntimeError: If yfinance raises an unexpected error.
        """
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError(
                "yfinance is not installed. Run: pip install yfinance"
            ) from exc

        logger.debug("YFinanceConnector: fetching fundamentals for %s", symbol)

        try:
            info = yf.Ticker(symbol).info
        except Exception as exc:
            raise RuntimeError(
                f"yfinance failed to fetch fundamentals for '{symbol}': {exc}"
            ) from exc

        # Select a stable subset of fields; extend as needed
        fields = [
            "symbol",
            "shortName",
            "sector",
            "industry",
            "marketCap",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "trailingEps",
            "dividendYield",
            "beta",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
        ]
        row = {field: info.get(field) for field in fields}
        return pd.DataFrame([row])
