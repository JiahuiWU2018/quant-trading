import unittest

from quant_trading import BacktestEngine, BaseStrategy, IBBrokerClient, RiskManager


class BuyFirstBarStrategy(BaseStrategy):
    def __init__(self):
        self.fills = []

    def on_bar(self, bar, portfolio):
        if not portfolio.positions.get(bar["symbol"]):
            return self.market_order(bar["symbol"], 1)
        return None

    def on_fill(self, order, fill_price):
        self.fills.append((order.symbol, fill_price))


class TestToolkit(unittest.TestCase):
    def test_risk_manager_blocks_oversized_orders(self):
        risk = RiskManager(max_position_size=1, max_notional_exposure=100.0)
        self.assertTrue(risk.allows("AAPL", quantity=1, price=100.0, current_quantity=0))
        self.assertFalse(risk.allows("AAPL", quantity=1, price=100.0, current_quantity=1))
        self.assertFalse(risk.allows("AAPL", quantity=2, price=60.0, current_quantity=0))

    def test_backtest_engine_executes_strategy_with_risk_checks(self):
        strategy = BuyFirstBarStrategy()
        engine = BacktestEngine(risk_manager=RiskManager(max_position_size=1))

        result = engine.run(
            market_data=[
                {"symbol": "AAPL", "close": 100.0},
                {"symbol": "AAPL", "close": 105.0},
            ],
            strategy=strategy,
            initial_cash=500.0,
        )

        self.assertEqual(result.portfolio.positions["AAPL"], 1)
        self.assertAlmostEqual(result.portfolio.cash, 400.0)
        self.assertEqual(len(result.equity_curve), 2)
        self.assertAlmostEqual(result.final_equity, 505.0)
        self.assertEqual(strategy.fills, [("AAPL", 100.0)])

    def test_backtest_engine_rejects_order_without_cash(self):
        strategy = BuyFirstBarStrategy()
        engine = BacktestEngine(risk_manager=RiskManager(max_position_size=10))

        result = engine.run(
            market_data=[{"symbol": "AAPL", "close": 100.0}],
            strategy=strategy,
            initial_cash=50.0,
        )

        self.assertEqual(result.portfolio.positions, {})
        self.assertAlmostEqual(result.portfolio.cash, 50.0)
        self.assertEqual(strategy.fills, [])

    def test_ib_broker_is_extension_point(self):
        class DemoIBClient(IBBrokerClient):
            def place_order(self, order):
                return f"accepted:{order.symbol}:{order.quantity}"

        strategy = BuyFirstBarStrategy()
        order = strategy.market_order("MSFT", 2)
        client = DemoIBClient()
        self.assertEqual(client.place_order(order), "accepted:MSFT:2")

    def test_backtest_engine_cleans_zero_positions(self):
        class BuyThenSellStrategy(BaseStrategy):
            def on_bar(self, bar, portfolio):
                current_quantity = portfolio.positions.get("AAPL", 0)
                if current_quantity == 0:
                    return self.market_order("AAPL", 1)
                return self.market_order("AAPL", -1)

        engine = BacktestEngine(risk_manager=RiskManager(max_position_size=5))
        result = engine.run(
            market_data=[
                {"symbol": "AAPL", "close": 100.0},
                {"symbol": "AAPL", "close": 101.0},
            ],
            strategy=BuyThenSellStrategy(),
            initial_cash=1000.0,
        )

        self.assertNotIn("AAPL", result.portfolio.positions)

    def test_backtest_engine_blocks_oversell_without_position(self):
        class SellFirstStrategy(BaseStrategy):
            def on_bar(self, bar, portfolio):
                return self.market_order("AAPL", -1)

        engine = BacktestEngine(risk_manager=RiskManager(max_position_size=5))
        result = engine.run(
            market_data=[{"symbol": "AAPL", "close": 100.0}],
            strategy=SellFirstStrategy(),
            initial_cash=1000.0,
        )

        self.assertEqual(result.portfolio.positions, {})
        self.assertAlmostEqual(result.portfolio.cash, 1000.0)


if __name__ == "__main__":
    unittest.main()
