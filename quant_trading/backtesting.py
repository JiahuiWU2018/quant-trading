from __future__ import annotations

from dataclasses import dataclass, field

from .risk import RiskManager
from .strategy import BaseStrategy


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        position_value = sum(qty * prices.get(symbol, 0.0) for symbol, qty in self.positions.items())
        return self.cash + position_value


@dataclass
class BacktestResult:
    portfolio: Portfolio
    equity_curve: list[float]

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.portfolio.cash


@dataclass
class BacktestEngine:
    risk_manager: RiskManager

    def run(self, market_data: list[dict], strategy: BaseStrategy, initial_cash: float) -> BacktestResult:
        portfolio = Portfolio(cash=initial_cash)
        prices: dict[str, float] = {}
        equity_curve: list[float] = []

        for bar in market_data:
            symbol = bar["symbol"]
            close = float(bar["close"])
            prices[symbol] = close

            order = strategy.on_bar(bar, portfolio)
            if order is not None:
                current_qty = portfolio.positions.get(order.symbol, 0)
                if self.risk_manager.allows(order.symbol, order.quantity, close, current_qty):
                    portfolio.cash -= order.quantity * close
                    portfolio.positions[order.symbol] = current_qty + order.quantity
                    strategy.on_fill(order, close)

            equity_curve.append(portfolio.equity(prices))

        return BacktestResult(portfolio=portfolio, equity_curve=equity_curve)
