"""Local parquet cache for data connector results.

Cache keys encode (symbol, start, end, freq, source) so that calls with
different parameters never collide. A TTL-based invalidation strategy is used:
cached files older than ``ttl_hours`` are re-fetched on the next request.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# Default cache directory: respects XDG_CACHE_HOME or falls back to ~/.cache
_DEFAULT_CACHE_DIR = Path(
    os.getenv("QUANT_CACHE_DIR", Path.home() / ".cache" / "quant_trading")
)

# Default TTL in hours
_DEFAULT_TTL_HOURS = int(os.getenv("QUANT_CACHE_TTL_HOURS", "24"))


def _cache_key(symbol: str, start: datetime, end: datetime, freq: str, source: str) -> str:
    """Generate a deterministic filename-safe cache key.

    Args:
        symbol: Ticker symbol.
        start: Start datetime.
        end: End datetime.
        freq: Bar frequency string.
        source: Data source identifier.

    Returns:
        Hex digest string suitable for use as a filename.
    """
    raw = f"{symbol}|{start.date()}|{end.date()}|{freq}|{source}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(key: str, cache_dir: Path) -> Path:
    """Return the full path to a cached parquet file.

    Args:
        key: Cache key hex digest.
        cache_dir: Root cache directory.

    Returns:
        Path to the parquet file.
    """
    return cache_dir / f"{key}.parquet"


def _is_expired(path: Path, ttl_hours: int) -> bool:
    """Return True if the cached file is older than ttl_hours.

    Args:
        path: Path to the cached file.
        ttl_hours: TTL in hours.

    Returns:
        True if expired or file does not exist.
    """
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600
    return age_hours > ttl_hours


def cached_fetch(
    fetch_fn: Callable[[], pd.DataFrame],
    symbol: str,
    start: datetime,
    end: datetime,
    freq: str,
    source: str,
    cache: bool = True,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    ttl_hours: int = _DEFAULT_TTL_HOURS,
) -> pd.DataFrame:
    """Wrap a fetch function with local parquet caching.

    If a valid (non-expired) cached result exists, it is returned without
    calling ``fetch_fn``. Otherwise, ``fetch_fn`` is called, the result is
    saved to disk, and the result is returned.

    Args:
        fetch_fn: Zero-argument callable that returns a DataFrame.
        symbol: Ticker symbol (used only for cache key generation).
        start: Start datetime.
        end: End datetime.
        freq: Bar frequency string.
        source: Data source identifier (e.g. "yfinance").
        cache: If False, bypass the cache entirely.
        cache_dir: Directory to store parquet files.
        ttl_hours: Cache TTL in hours.

    Returns:
        DataFrame from cache or from fetch_fn.

    Raises:
        Any exception raised by fetch_fn propagates unchanged.
    """
    if not cache:
        logger.debug("Cache disabled; fetching %s directly.", symbol)
        return fetch_fn()

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(symbol, start, end, freq, source)
    path = _cache_path(key, cache_dir)

    if not _is_expired(path, ttl_hours):
        logger.debug("Cache hit for %s (%s); loading from %s", symbol, source, path)
        return pd.read_parquet(path)

    logger.debug("Cache miss for %s (%s); fetching from source.", symbol, source)
    df = fetch_fn()

    try:
        df.to_parquet(path)
        logger.debug("Cached %s (%s) to %s", symbol, source, path)
    except Exception as exc:
        # Non-fatal: log and continue with the fresh data
        logger.warning("Failed to write cache for %s: %s", symbol, exc)

    return df


def invalidate_cache(
    symbol: str,
    start: datetime,
    end: datetime,
    freq: str,
    source: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> bool:
    """Delete a specific cached entry.

    Args:
        symbol: Ticker symbol.
        start: Start datetime.
        end: End datetime.
        freq: Bar frequency string.
        source: Data source identifier.
        cache_dir: Root cache directory.

    Returns:
        True if the file was deleted, False if it did not exist.
    """
    key = _cache_key(symbol, start, end, freq, source)
    path = _cache_path(key, cache_dir)
    if path.exists():
        path.unlink()
        logger.info("Invalidated cache entry: %s", path)
        return True
    return False
