from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    symbol: str
    quantity: int


class BaseStrategy(ABC):
    @abstractmethod
    def on_bar(self, bar: dict, portfolio: "Portfolio") -> Order | None:
        raise NotImplementedError

    def on_fill(self, order: Order, fill_price: float) -> None:
        return None

    @staticmethod
    def market_order(symbol: str, quantity: int) -> Order:
        return Order(symbol=symbol, quantity=quantity)
