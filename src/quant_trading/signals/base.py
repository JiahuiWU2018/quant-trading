"""Abstract base class for trading signals.

Signals transform price/feature data into actionable position scores or
directional indicators. Concrete signal implementations live in the private
strategy repository.
"""

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class BaseSignal(ABC):
    """Interface for all trading signal generators.

    A signal takes historical price data (and optionally other features) and
    produces a timestamp-aligned Series indicating position direction or score.

    Signals should be stateless or manage their own internal state explicitly.
    They must be safe to call multiple times with overlapping data.

    Example:
        class SimpleMASignal(BaseSignal):
            def __init__(self, fast=10, slow=50):
                self.fast = fast
                self.slow = slow

            def generate(self, prices: pd.DataFrame) -> pd.Series:
                close = prices["close"]
                ma_fast = close.rolling(self.fast).mean()
                ma_slow = close.rolling(self.slow).mean()
                signal = pd.Series(0, index=prices.index)
                signal[ma_fast > ma_slow] = 1
                signal[ma_fast < ma_slow] = -1
                return signal
    """

    @abstractmethod
    def generate(self, prices: pd.DataFrame, features: pd.DataFrame | None = None) -> pd.Series:
        """Generate trading signals from price and feature data.

        Args:
            prices: OHLCV DataFrame with DatetimeIndex and columns
                [open, high, low, close, volume].
            features: Optional DataFrame with additional features
                (fundamentals, alternative data, etc.) aligned by index.

        Returns:
            Series aligned to prices.index with signal values.
            Convention:
                +1 or positive: long / bullish
                 0: neutral / no position
                -1 or negative: short / bearish
            Continuous scores (e.g., [-1, 1]) are also valid.

        Raises:
            ValueError: If prices are invalid or insufficient history.
        """
        ...

    def validate_prices(self, prices: pd.DataFrame, min_rows: int = 1) -> None:
        """Validate that prices DataFrame meets minimum requirements.

        Args:
            prices: OHLCV DataFrame to validate.
            min_rows: Minimum number of rows required.

        Raises:
            ValueError: If prices is invalid.
        """
        if prices.empty:
            raise ValueError("prices DataFrame is empty.")
        if len(prices) < min_rows:
            raise ValueError(f"Insufficient data: {len(prices)} rows, need at least {min_rows}.")
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"prices DataFrame missing required columns: {missing}")
