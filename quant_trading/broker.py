from abc import ABC, abstractmethod

from .strategy import Order


class IBBrokerClient(ABC):
    @abstractmethod
    def place_order(self, order: Order) -> str:
        raise NotImplementedError
