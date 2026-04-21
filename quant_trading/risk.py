from dataclasses import dataclass


@dataclass(frozen=True)
class RiskManager:
    max_position_size: int
    max_notional_exposure: float | None = None

    def allows(self, symbol: str, quantity: int, price: float, current_quantity: int) -> bool:
        projected_quantity = current_quantity + quantity
        if abs(projected_quantity) > self.max_position_size:
            return False
        if self.max_notional_exposure is None:
            return True
        return abs(projected_quantity * price) <= self.max_notional_exposure
