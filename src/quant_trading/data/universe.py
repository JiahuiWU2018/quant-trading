"""Asset universe management.

Provides a lightweight container for managing sets of tradeable instruments.
The private repository can instantiate Universe with its own symbol lists;
this module contains no hardcoded tickers or strategy-specific logic.
"""

import logging
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class Universe:
    """A named, immutable-by-default set of ticker symbols.

    Args:
        name: Human-readable label for this universe (e.g. "SP500").
        symbols: List of ticker symbols included in the universe.
        metadata: Optional dict of additional universe-level attributes
                  (e.g. {"exchange": "NYSE", "asset_class": "equity"}).

    Example:
        universe = Universe(name="test", symbols=["AAPL", "MSFT", "GOOG"])
        for symbol in universe:
            print(symbol)
    """

    name: str
    symbols: list[str]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for s in self.symbols:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        if len(deduped) != len(self.symbols):
            logger.warning(
                "Universe '%s': %d duplicate symbol(s) removed.",
                self.name,
                len(self.symbols) - len(deduped),
            )
        self.symbols = deduped

    def __len__(self) -> int:
        return len(self.symbols)

    def __iter__(self) -> Iterator[str]:
        return iter(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.symbols

    def __repr__(self) -> str:
        return f"Universe(name={self.name!r}, n={len(self.symbols)})"

    def filter(self, symbols: list[str]) -> "Universe":
        """Return a new Universe containing only symbols present in both sets.

        Args:
            symbols: List of symbols to keep.

        Returns:
            New Universe with the intersection of symbols.
        """
        kept = [s for s in self.symbols if s in set(symbols)]
        return Universe(name=f"{self.name}_filtered", symbols=kept, metadata=self.metadata.copy())

    def exclude(self, symbols: list[str]) -> "Universe":
        """Return a new Universe with the given symbols removed.

        Args:
            symbols: List of symbols to exclude.

        Returns:
            New Universe with the specified symbols removed.
        """
        excluded = set(symbols)
        kept = [s for s in self.symbols if s not in excluded]
        return Universe(name=f"{self.name}_excl", symbols=kept, metadata=self.metadata.copy())

    def union(self, other: "Universe") -> "Universe":
        """Return a new Universe combining both symbol sets (deduped).

        Args:
            other: Another Universe instance.

        Returns:
            New Universe with combined, deduplicated symbol list.
        """
        combined = self.symbols + [s for s in other.symbols if s not in set(self.symbols)]
        return Universe(
            name=f"{self.name}+{other.name}",
            symbols=combined,
            metadata={**self.metadata, **other.metadata},
        )
