"""Abstract base class for all data connectors."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

VALID_FREQS = {"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"}


class BaseDataConnector(ABC):
    """Interface that every data source connector must implement.

    Concrete implementations (e.g. YFinanceConnector, IBKRConnector) live in
    this package. Strategy-specific connector wiring lives in the private repo.

    All connectors must:
    - Return DataFrames with a tz-aware DatetimeIndex (UTC).
    - Use column names: open, high, low, close, volume (lower-case).
    - Raise ValueError for unsupported freq values.
    - Raise RuntimeError for unrecoverable fetch failures.
    """

    @abstractmethod
    def fetch_price_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV price history for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start datetime (inclusive).
            end: End datetime (inclusive).
            freq: Bar frequency. Must be one of VALID_FREQS.

        Returns:
            DataFrame with tz-aware UTC DatetimeIndex and columns:
            [open, high, low, close, volume].

        Raises:
            ValueError: If freq is not supported.
            RuntimeError: If the data fetch fails unrecoverably.
        """
        ...

    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> pd.DataFrame:
        """Fetch fundamental data for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").

        Returns:
            DataFrame with fundamental fields as columns indexed by date or
            a single-row snapshot. Schema is connector-specific; callers
            should not rely on fixed column names without checking.

        Raises:
            RuntimeError: If the data fetch fails unrecoverably.
        """
        ...

    @staticmethod
    def validate_freq(freq: str) -> None:
        """Raise ValueError if freq is not in VALID_FREQS.

        Args:
            freq: Frequency string to validate.

        Raises:
            ValueError: If freq is not supported.
        """
        if freq not in VALID_FREQS:
            raise ValueError(
                f"Unsupported freq '{freq}'. Must be one of {sorted(VALID_FREQS)}."
            )

    @staticmethod
    def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize an OHLCV DataFrame for safe downstream use.

        Performs the following steps in order:
        1. Lower-case all column names.
        2. Validate required OHLCV columns are present.
        3. Coerce price and volume columns to float64 / int64 (replaces
           non-numeric values with NaN rather than raising).
        4. Drop rows where *all* OHLCV values are NaN.
        5. Ensure the DatetimeIndex is tz-aware UTC.
        6. Sort the index ascending and drop any duplicate timestamps,
           keeping the last occurrence.

        Args:
            df: Raw DataFrame from a data source.

        Returns:
            Clean DataFrame with lower-case columns, numeric dtypes,
            no all-NaN rows, a tz-aware UTC DatetimeIndex that is
            sorted and deduplicated.

        Raises:
            ValueError: If required OHLCV columns are missing after
                normalisation, or if the result is empty after cleaning.
        """
        df = df.copy()

        # 1. Lowercase columns
        df.columns = [c.lower() for c in df.columns]

        # 2. Validate presence
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        # 3. Coerce to numeric (non-numeric → NaN, not exception)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        # 4. Drop rows where every OHLCV value is NaN
        df = df.dropna(subset=required, how="all")

        # 5. Ensure tz-aware UTC DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # 6. Sort ascending, drop duplicate timestamps (keep last)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if df.empty:
            raise ValueError("OHLCV DataFrame is empty after normalization.")

        return df[["open", "high", "low", "close", "volume"]]
