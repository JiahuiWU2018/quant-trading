from __future__ import annotations

from dataclasses import dataclass, field

from .risk import RiskManager
from .strategy import BaseStrategy


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        missing_prices = [symbol for symbol in self.positions if symbol not in prices]
        if missing_prices:
            raise ValueError(f"Missing prices for symbols: {', '.join(sorted(missing_prices))}")
        position_value = sum(qty * prices[symbol] for symbol, qty in self.positions.items())
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
    allow_short_selling: bool = False

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
                fill_price = prices.get(order.symbol)
                if fill_price is None:
                    raise ValueError(f"Missing execution price for symbol: {order.symbol}")

                if self.risk_manager.allows(order.symbol, order.quantity, fill_price, current_qty):
                    projected_quantity = current_qty + order.quantity
                    if not self.allow_short_selling and projected_quantity < 0:
                        continue

                    # Cash decreases when buying (positive quantity) and increases when selling (negative quantity).
                    projected_cash = portfolio.cash - order.quantity * fill_price
                    if order.quantity > 0 and projected_cash < 0:
                        continue

                    portfolio.cash = projected_cash
                    new_quantity = projected_quantity
                    if new_quantity == 0:
                        portfolio.positions.pop(order.symbol, None)
                    else:
                        portfolio.positions[order.symbol] = new_quantity
                    strategy.on_fill(order, fill_price)

            equity_curve.append(portfolio.equity(prices))

        return BacktestResult(portfolio=portfolio, equity_curve=equity_curve)
