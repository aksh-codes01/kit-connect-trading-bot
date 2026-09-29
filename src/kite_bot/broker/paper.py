"""Simulated broker: fills instantly at the given price and tracks P&L."""

from __future__ import annotations

from kite_bot.broker.base import CANCELLED, COMPLETE, TRIGGER_PENDING, Broker, Order, OrderStatus, Side


class PaperBroker(Broker):
    """In-memory broker for tests, backtests and dry runs.

    Market orders fill immediately. Stop orders rest until `on_price` is called
    with a price that reaches their trigger, which mimics an exchange-side stop.

    Args:
        commission_pct: fee per fill as a fraction of traded value (0.0003 = 0.03%).
        slippage_pct: adverse price move per fill; buys fill higher, sells lower.
    """

    def __init__(self, commission_pct: float = 0.0, slippage_pct: float = 0.0):
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.orders: list[Order] = []
        self._qty: dict[str, int] = {}
        self._avg: dict[str, float] = {}
        self._realized = 0.0
        self._status: dict[str, OrderStatus] = {}
        self._stops: dict[str, tuple[str, Side, int, float]] = {}  # order_id -> symbol, side, qty, trigger

    @property
    def realized_pnl(self) -> float:
        """Profit from closed quantity, net of commissions on every fill so far."""
        return self._realized

    # ---------------------------------------------------------------- orders

    def place_order(self, symbol: str, side: Side, quantity: int, price: float) -> Order:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        return self._execute(self._new_id(), symbol, side, quantity, price)

    def place_stop_loss(self, symbol: str, side: Side, quantity: int, trigger_price: float) -> Order:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        order = Order(self._new_id(), symbol, side, quantity, trigger_price)
        self.orders.append(order)
        self._stops[order.order_id] = (symbol, side, quantity, trigger_price)
        self._status[order.order_id] = OrderStatus(order.order_id, TRIGGER_PENDING)
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self._stops:
            return False
        del self._stops[order_id]
        self._status[order_id] = OrderStatus(order_id, CANCELLED)
        return True

    def order_status(self, order_id: str) -> OrderStatus:
        try:
            return self._status[order_id]
        except KeyError:
            raise KeyError(f"unknown order id: {order_id}") from None

    def positions(self) -> dict[str, int]:
        return {symbol: qty for symbol, qty in self._qty.items() if qty != 0}

    def on_price(self, symbol: str, price: float) -> list[Order]:
        """Feed a market price; any resting stop it reaches executes at that price."""
        triggered = []
        for order_id, (sym, side, qty, trigger) in list(self._stops.items()):
            if sym != symbol:
                continue
            hit = price >= trigger if side == Side.BUY else price <= trigger
            if hit:
                del self._stops[order_id]
                triggered.append(self._execute(order_id, sym, side, qty, price))
        return triggered

    # -------------------------------------------------------------- internals

    def _new_id(self) -> str:
        return f"paper-{len(self.orders) + 1}"

    def _execute(self, order_id: str, symbol: str, side: Side, quantity: int, price: float) -> Order:
        fill = price * (1 + self.slippage_pct) if side == Side.BUY else price * (1 - self.slippage_pct)
        self._apply_fill(symbol, quantity if side == Side.BUY else -quantity, fill)
        self._realized -= self.commission_pct * fill * quantity
        order = Order(order_id, symbol, side, quantity, fill)
        if not any(o.order_id == order_id for o in self.orders):
            self.orders.append(order)
        self._status[order_id] = OrderStatus(order_id, COMPLETE, quantity, fill)
        return order

    def _apply_fill(self, symbol: str, signed_qty: int, price: float) -> None:
        held = self._qty.get(symbol, 0)
        avg = self._avg.get(symbol, 0.0)

        if held == 0 or (held > 0) == (signed_qty > 0):  # opening or adding
            total = abs(held) + abs(signed_qty)
            self._avg[symbol] = (abs(held) * avg + abs(signed_qty) * price) / total
        else:  # reducing or flipping
            closing = min(abs(signed_qty), abs(held))
            direction = 1 if held > 0 else -1
            self._realized += closing * (price - avg) * direction
            if abs(signed_qty) > abs(held):  # flipped: remainder opens at this price
                self._avg[symbol] = price
            elif abs(signed_qty) == abs(held):
                self._avg[symbol] = 0.0
        self._qty[symbol] = held + signed_qty
