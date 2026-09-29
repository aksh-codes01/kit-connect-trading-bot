"""Live broker: thin wrapper over an authenticated kiteconnect.KiteConnect object."""

from __future__ import annotations

import logging
from typing import Optional

from kite_bot.broker.base import Broker, Order, OrderStatus, Side

log = logging.getLogger(__name__)


class KiteBroker(Broker):
    """Places intraday (MIS) orders on NSE.

    The KiteConnect client is injected, so this class never reads credentials
    and can be tested with a fake client.

    Notes on the Kite API that shape this class:
      * The API rejects plain market orders unless they carry `market_protection`.
        -1 asks Kite to pick the protection band automatically; a positive number
        is a percentage. Needs a recent `kiteconnect` release that accepts the argument.
      * Stop orders are SL-M (stop-loss market), with the same protection.
      * Trigger prices must be a multiple of the tick size (0.05 for most stocks).

    It has NOT been exercised against the real Kite API. Try one share first.
    """

    def __init__(
        self,
        kite,
        exchange: Optional[str] = None,
        market_protection: float = -1,
        tag: str = "kitebot",  # Kite allows alphanumeric tags up to 20 characters
        tick_size: float = 0.05,
        tick_sizes: Optional[dict[str, float]] = None,
    ):
        self._kite = kite
        self._exchange = exchange or kite.EXCHANGE_NSE
        self._protection = market_protection
        self._tag = tag
        self._tick = tick_size
        self._ticks = tick_sizes or {}

    def place_order(self, symbol: str, side: Side, quantity: int, price: float) -> Order:
        order_id = self._kite.place_order(
            variety=self._kite.VARIETY_REGULAR,
            exchange=self._exchange,
            tradingsymbol=symbol,
            transaction_type=self._transaction(side),
            quantity=quantity,
            product=self._kite.PRODUCT_MIS,
            order_type=self._kite.ORDER_TYPE_MARKET,
            market_protection=self._protection,
            tag=self._tag,
        )
        return Order(str(order_id), symbol, side, quantity, price)

    def place_stop_loss(self, symbol: str, side: Side, quantity: int, trigger_price: float) -> Order:
        trigger = self._round_to_tick(symbol, trigger_price)
        order_id = self._kite.place_order(
            variety=self._kite.VARIETY_REGULAR,
            exchange=self._exchange,
            tradingsymbol=symbol,
            transaction_type=self._transaction(side),
            quantity=quantity,
            product=self._kite.PRODUCT_MIS,
            order_type=self._kite.ORDER_TYPE_SLM,
            trigger_price=trigger,
            market_protection=self._protection,
            tag=self._tag,
        )
        return Order(str(order_id), symbol, side, quantity, trigger)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._kite.cancel_order(variety=self._kite.VARIETY_REGULAR, order_id=order_id)
            return True
        except Exception as error:  # Kite raises when the order already executed or is not open
            log.warning("cancel_order(%s) failed: %s", order_id, error)
            return False

    def order_status(self, order_id: str) -> OrderStatus:
        latest = self._kite.order_history(order_id)[-1]
        return OrderStatus(
            order_id=str(order_id),
            status=str(latest["status"]),
            filled_quantity=int(latest.get("filled_quantity") or 0),
            average_price=float(latest.get("average_price") or 0.0),
            message=str(latest.get("status_message") or ""),
        )

    def positions(self) -> dict[str, int]:
        """Open intraday (MIS) positions on our exchange; unrelated positions (F&O, delivery) are ignored."""
        net = self._kite.positions()["net"]
        return {
            p["tradingsymbol"]: int(p["quantity"])
            for p in net
            if int(p["quantity"]) != 0 and p.get("product", self._kite.PRODUCT_MIS) == self._kite.PRODUCT_MIS
            and p.get("exchange", self._exchange) == self._exchange
        }

    # ------------------------------------------------------------------ helpers

    def _transaction(self, side: Side) -> str:
        return self._kite.TRANSACTION_TYPE_BUY if side == Side.BUY else self._kite.TRANSACTION_TYPE_SELL

    def _round_to_tick(self, symbol: str, price: float) -> float:
        tick = self._ticks.get(symbol, self._tick)
        return round(round(price / tick) * tick, 2)
