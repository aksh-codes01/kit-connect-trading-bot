"""Broker interface: the only thing strategy, backtest and live-engine code talk to."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


# Order states, spelled the way Kite reports them.
COMPLETE = "COMPLETE"
OPEN = "OPEN"
TRIGGER_PENDING = "TRIGGER PENDING"
CANCELLED = "CANCELLED"
REJECTED = "REJECTED"
FINAL_STATES = (COMPLETE, CANCELLED, REJECTED)


@dataclass(frozen=True)
class Order:
    order_id: str
    symbol: str
    side: Side
    quantity: int
    price: float  # fill price (paper broker) or the reference price the caller passed (live broker)


@dataclass(frozen=True)
class OrderStatus:
    order_id: str
    status: str  # COMPLETE, OPEN, TRIGGER PENDING, CANCELLED, REJECTED, or a transitional Kite state
    filled_quantity: int = 0
    average_price: float = 0.0
    message: str = ""

    @property
    def is_final(self) -> bool:
        return self.status in FINAL_STATES


class Broker(ABC):
    """Places orders and reports their state and the net positions.

    `price` is the caller's reference price. A paper broker fills at it (plus
    slippage); a live broker sends a market order and ignores it.
    """

    @abstractmethod
    def place_order(self, symbol: str, side: Side, quantity: int, price: float) -> Order:
        """Send an intraday market order."""

    @abstractmethod
    def place_stop_loss(self, symbol: str, side: Side, quantity: int, trigger_price: float) -> Order:
        """Rest a stop order at the exchange: it becomes a market order when the price reaches
        `trigger_price`. `side` is the side of the stop itself (BUY to protect a short)."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. False if it could not be cancelled (for example it already executed)."""

    @abstractmethod
    def order_status(self, order_id: str) -> OrderStatus:
        """Current state of an order, including how much filled and at what average price."""

    @abstractmethod
    def positions(self) -> dict[str, int]:
        """Net quantity per symbol (negative = short). Flat symbols are omitted."""
