"""Phase 3 — Live execution layer.

Public interface:
    BrokerAdapter   — ABC defining the broker contract
    IBKRAdapter     — ib_async-backed implementation (paper + live)
    OrderManager    — order lifecycle, retry, rate limiting
    SafetyChecker   — pre-trade kill-switch and limit enforcement
    StrategyRunner  — polling execution loop (one per strategy)
    start_all       — convenience helper to run multiple runners in parallel
"""

from quant_trading.execution.base_adapter import BrokerAdapter, Order, OrderSide, OrderStatus, OrderType
from quant_trading.execution.order_manager import OrderManager
from quant_trading.execution.runner import StrategyRunner, start_all
from quant_trading.execution.safety import SafetyChecker

__all__ = [
    "BrokerAdapter",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "OrderManager",
    "SafetyChecker",
    "StrategyRunner",
    "start_all",
]
