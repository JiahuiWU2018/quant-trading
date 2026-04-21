# quant-trading
Quantitative trading toolkit for backtesting and live execution. Features: backtesting engine, risk management, IB broker integration. Designed to be extended by private strategy repositories.

## Quick start

```python
from quant_trading import BacktestEngine, BaseStrategy, RiskManager


class BuyAndHold(BaseStrategy):
    def on_bar(self, bar, portfolio):
        if not portfolio.positions.get("AAPL"):
            return self.market_order("AAPL", 1)
        return None


engine = BacktestEngine(risk_manager=RiskManager(max_position_size=10))
result = engine.run(
    market_data=[
        {"symbol": "AAPL", "close": 100.0},
        {"symbol": "AAPL", "close": 101.0},
    ],
    strategy=BuyAndHold(),
    initial_cash=1000.0,
)
print(result.final_equity)
```
