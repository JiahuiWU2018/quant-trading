"""Example smoke strategy that can be used with StrategyRunner.

This module provides a drop-in example implementing the LiveStrategy protocol.
It is intentionally conservative: it generates a single LIMIT BUY for 1 share
at $1.00 (well below market) so it will not execute on a live market.

Intended usage from project root:
python scripts/run_smoke_strategy.py --symbol AAPL
"""

from quant_trading.execution.base_adapter import Order, OrderSide, OrderType


class SmokeStrategy:
    strategy_id = "smoke-test"

    def __init__(self, symbol: str, limit_price: float = 1.0):
        self.symbol = symbol
        self.limit_price = float(limit_price)
        self._used = False

    def on_bar(self, data):
        if self._used:
            return []
        self._used = True
        return [
            Order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.LIMIT,
                limit_price=self.limit_price,
                strategy_id=self.strategy_id,
            )
        ]
