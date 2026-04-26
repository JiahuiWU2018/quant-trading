"""Unified data loader interface.

This module provides a single entry point for all data fetching operations.
Callers do not need to interact with individual connectors directly.

Supported sources (Phase 1): "yfinance"
Additional sources will be added in Phase 3 (ibkr) and Phase 6 (alpha_vantage, finnhub).
"""

import logging
from datetime import datetime
from typing import Literal

import pandas as pd

from quant_trading.data.cache import cached_fetch

logger = logging.getLogger(__name__)

# Registry of available source identifiers → connector factory functions
# New connectors are registered here as they are implemented.
_CONNECTOR_REGISTRY: dict[str, type] = {}


def _get_connector(source: str):
    """Instantiate a connector for the given source identifier.

    Args:
        source: Source identifier string (e.g. "yfinance").

    Returns:
        An instantiated BaseDataConnector.

    Raises:
        ValueError: If the source is not registered.
    """
    # Lazy imports so optional dependencies don't break the package on import
    if source == "yfinance":
        from quant_trading.data.apis.yfinance_connector import YFinanceConnector
        return YFinanceConnector()

    raise ValueError(
        f"Unknown data source '{source}'. "
        f"Available sources: {list(_CONNECTOR_REGISTRY) + ['yfinance']}"
    )


def fetch_price_history(
    symbol: str,
    start: datetime,
    end: datetime,
    freq: str = "1d",
    source: str = "yfinance",
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV price history for a single symbol.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        start: Start datetime (inclusive).
        end: End datetime (inclusive).
        freq: Bar frequency. See data.apis.base_connector.VALID_FREQS.
        source: Data source to use. Defaults to "yfinance".
        cache: Whether to use the local parquet cache.

    Returns:
        DataFrame with tz-aware UTC DatetimeIndex and columns:
        [open, high, low, close, volume].

    Raises:
        ValueError: If the source or freq is unsupported, or result is empty.
        RuntimeError: If the connector fails unrecoverably.

    Example:
        from datetime import datetime
        from quant_trading.data.loaders import fetch_price_history

        df = fetch_price_history("AAPL", datetime(2023, 1, 1), datetime(2023, 12, 31))
    """
    connector = _get_connector(source)

    def _fetch() -> pd.DataFrame:
        return connector.fetch_price_history(symbol, start, end, freq)

    return cached_fetch(
        fetch_fn=_fetch,
        symbol=symbol,
        start=start,
        end=end,
        freq=freq,
        source=source,
        cache=cache,
    )


def fetch_fundamentals(
    symbol: str,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Fetch fundamental data for a single symbol.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        source: Data source to use. Defaults to "yfinance".

    Returns:
        DataFrame with fundamental fields as columns. Schema is
        connector-specific; see the connector's docstring for field details.

    Raises:
        ValueError: If the source is unsupported.
        RuntimeError: If the connector fails unrecoverably.
    """
    connector = _get_connector(source)
    return connector.fetch_fundamentals(symbol)


def fetch_multi(
    symbols: list[str],
    start: datetime,
    end: datetime,
    freq: str = "1d",
    source: str = "yfinance",
    cache: bool = True,
    field: str = "close",
) -> pd.DataFrame:
    """Fetch a single price field for multiple symbols and return a wide DataFrame.

    Args:
        symbols: List of ticker symbols.
        start: Start datetime (inclusive).
        end: End datetime (inclusive).
        freq: Bar frequency.
        source: Data source to use.
        cache: Whether to use the local parquet cache.
        field: OHLCV field to extract (e.g. "close", "volume").

    Returns:
        Wide DataFrame with DatetimeIndex and one column per symbol.
        Symbols that fail to fetch are logged and omitted.

    Example:
        prices = fetch_multi(["AAPL", "MSFT"], datetime(2023, 1, 1), datetime(2023, 12, 31))
    """
    frames: dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            df = fetch_price_history(symbol, start, end, freq, source, cache)
            frames[symbol] = df[field]
        except Exception as exc:
            logger.warning("Failed to fetch '%s' from %s: %s", symbol, source, exc)

    if not frames:
        raise RuntimeError(
            f"No data retrieved for any of the requested symbols: {symbols}"
        )

    return pd.DataFrame(frames)
