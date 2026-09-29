"""The live trading engine: turns a tick stream into orders, safely.

Daily flow (the same code runs on live ticks, paper trading and replays):

    screen()         evening/pre-market: run the daily screener, keep the shortlist
    load_warmup()    pre-market: give each shortlisted stock its recent 2-minute bars
    recover()        after a restart: reconcile the database with the broker
    on_tick()        every tick: manage open positions, build bars, look for entries
    check_time()     once a second: time-based exits even when a stock stops trading
    end_of_day()     flush bars, close anything still open, summarise

Entry: when a 2-minute bar completes and EMA(fast) has just crossed below EMA(slow)
on it. A bar is completed by the first tick of the next bar, so that tick's price is
the next bar's open, the same fill price the backtester assumes.

Safety rules, in the order they matter:
  * The trade is written to the database BEFORE its protective stop is placed, so a
    crash in between is repaired on restart instead of leaving an untracked position.
  * Every entry gets an exchange-side stop-loss. If one cannot be placed, the
    position is closed immediately.
  * A resting stop is cancelled before covering. If the cancel fails, the stop is
    checked first: if it already fired the position is closed and we do NOT buy again.
  * If a position cannot be closed, the engine halts and logs a CRITICAL event.
  * A halt (kill switch, loss limit, mismatch after restart) blocks new entries;
    exits carry on.
  * One trade per stock per day, and at most `max_trades_per_day` overall.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from dataclasses import dataclass
from typing import Mapping, Optional

import pandas as pd

from kite_bot.broker import CANCELLED, COMPLETE, REJECTED, Broker, OrderStatus, Side
from kite_bot.config import LiveConfig
from kite_bot.feed import Bar, BarBuilder, Tick, bar_start
from kite_bot.ohlc import ist_now, naive_ist, resample_ohlc
from kite_bot.screener import find_setup
from kite_bot.store import Store
from kite_bot.strategy import bearish_cross

log = logging.getLogger(__name__)


@dataclass
class OpenTrade:
    trade_id: int
    symbol: str
    quantity: int
    entry_time: dt.datetime
    entry_price: float
    target: float
    stop: float
    stop_order_id: Optional[str]


class LiveEngine:
    """One instance per trading day. Thread-safe: ticks arrive on the market-data
    thread while `check_time` is called from the main thread."""

    def __init__(
        self,
        broker: Broker,
        store: Store,
        day: dt.date,
        config: LiveConfig = LiveConfig(),
        sleep=time.sleep,
    ):
        self.broker = broker
        self.store = store
        self.day = day
        self.config = config
        self._sleep = sleep
        self._lock = threading.RLock()

        self.symbols: set[str] = set()
        self.now: Optional[dt.datetime] = None  # time of the latest tick
        self._builders: dict[str, BarBuilder] = {}
        self._closes: dict[str, list[float]] = {}
        self._last_price: dict[str, float] = {}
        self._open: dict[str, OpenTrade] = {}
        self._traded: set[str] = set()
        self._trade_count = 0
        self._realized = 0.0
        self._last_skip: dict[str, str] = {}
        self._halt_key = f"halted:{day.isoformat()}"
        self.halted: Optional[str] = store.get(self._halt_key)

    # ------------------------------------------------------------ preparation

    def screen(self, daily: Mapping[str, pd.DataFrame]) -> list[str]:
        """Run the daily screener on data up to the day before `day`; returns the shortlist."""
        matches = []
        for symbol in sorted(daily):
            history = daily[symbol]
            history = history[history.index.date < self.day]
            setup = find_setup(history, "short", self.config.ma_period, self.config.slope_window)
            if setup is not None:
                matches.append((symbol, setup))
        with self._lock:
            self.store.save_shortlist(self.day, matches)
            self.symbols = {symbol for symbol, _ in matches}
        return sorted(self.symbols)

    def load_warmup(self, symbol: str, minute_bars: pd.DataFrame, now: Optional[dt.datetime] = None) -> int:
        """Seed `symbol` with 2-minute bars built from recent 1-minute history.

        Bars whose window has not finished by `now` are left out, because live ticks
        will build them. Returns how many bars were loaded.
        """
        bars = resample_ohlc(naive_ist(minute_bars))
        cutoff = bar_start(now, 2) if now is not None and now.date() == self.day else dt.datetime.combine(self.day, dt.time.min)
        bars = bars[bars.index < cutoff]
        with self._lock:
            self._closes[symbol] = bars["close"].astype(float).tolist()
            self._builders[symbol] = BarBuilder(2)
        needed = self.config.slow
        if len(bars) < needed:
            self.store.log_event("WARNING", f"{symbol}: only {len(bars)} warm-up bars, the {needed}-bar EMA is not valid yet")
        return len(bars)

    # --------------------------------------------------------------- recovery

    def recover(self) -> list[str]:
        """Reconcile today's open trades in the database with the broker's positions.

        Returns human-readable notes. Call once after start-up, before ticks arrive.
        """
        notes: list[str] = []
        with self._lock:
            for trade in self.store.trades(self.day):
                self._traded.add(trade.symbol)
                self._trade_count += 1
            self._realized = self.store.realized_pnl(self.day)

            positions = self.broker.positions()
            tracked = set()
            for row in self.store.open_trades(self.day):
                tracked.add(row.symbol)
                held = positions.get(row.symbol, 0)
                trade = OpenTrade(row.id, row.symbol, row.quantity, dt.datetime.fromisoformat(row.entry_time),
                                  row.entry_price, row.target_price, row.stop_price, row.stop_order_id)
                if held == -row.quantity:
                    self._resume(trade, notes)
                elif held == 0:
                    self._closed_while_down(trade, notes)
                else:
                    note = f"{row.symbol}: database says short {row.quantity}, broker holds {held}"
                    self.halt(f"position mismatch, {note}")
                    self._event("CRITICAL", note + "; not managed, check manually", notes)
            for symbol, qty in positions.items():
                if symbol not in tracked:
                    self.halt(f"unexpected position {symbol} {qty}")
                    self._event("CRITICAL", f"{symbol}: broker holds {qty} but the database has no open trade; not managed", notes)
        return notes

    def _resume(self, trade: OpenTrade, notes: list[str]) -> None:
        status = self._status(trade.stop_order_id) if trade.stop_order_id else None
        if status is None or status.status in (CANCELLED, REJECTED):
            order = self._place_stop(trade.symbol, trade.quantity, trade.stop, trade.trade_id, self.now or ist_now())
            trade.stop_order_id = order.order_id if order else None
            self._event("WARNING", f"{trade.symbol}: stop order was missing, placed a new one" if order else f"{trade.symbol}: stop order missing and could not be replaced", notes)
        self._open[trade.symbol] = trade
        self.symbols.add(trade.symbol)
        self._event("INFO", f"{trade.symbol}: resumed short {trade.quantity} from {trade.entry_price:.2f}", notes)

    def _closed_while_down(self, trade: OpenTrade, notes: list[str]) -> None:
        status = self._status(trade.stop_order_id) if trade.stop_order_id else None
        if status is not None and status.status == COMPLETE:
            pnl = (trade.entry_price - status.average_price) * trade.quantity
            self.store.close_trade(trade.trade_id, self.now or ist_now(), status.average_price, "stop_loss", pnl)
            self._realized += pnl
            self._event("INFO", f"{trade.symbol}: stop-loss executed while the bot was down ({pnl:.2f})", notes)
        else:
            self.store.close_trade(trade.trade_id, self.now or ist_now(), None, "closed_externally", None)
            self._event("WARNING", f"{trade.symbol}: position is gone and the exit price is unknown", notes)

    # ------------------------------------------------------------------ ticks

    def on_tick(self, tick: Tick) -> None:
        with self._lock:
            if tick.symbol not in self.symbols:
                return
            try:
                self._handle_tick(tick)
            except Exception as error:  # a bug must never stop the stream: record it and carry on
                log.exception("error handling tick for %s", tick.symbol)
                self.store.log_event("ERROR", f"{tick.symbol}: {type(error).__name__}: {error}", tick.timestamp)

    def _handle_tick(self, tick: Tick) -> None:
        symbol, price = tick.symbol, tick.price
        self.now = tick.timestamp
        self._last_price[symbol] = price
        if self.config.record_ticks:
            self.store.save_ticks([tick])
        if hasattr(self.broker, "on_price"):  # paper broker: fire resting stops like an exchange would
            self.broker.on_price(symbol, price)

        trade = self._open.get(symbol)
        if trade is not None:
            self._manage(trade, price, tick.timestamp)

        builder = self._builders.get(symbol)
        if builder is not None:
            bar = builder.update(tick)
            if bar is not None:
                self._on_bar(symbol, bar, tick)

        if tick.timestamp.time() >= self.config.square_off:
            self._square_off_all(tick.timestamp)
        self._risk_check()

    def check_time(self, now: dt.datetime) -> None:
        """Time-based exits for stocks that have stopped ticking. Call about once a second."""
        with self._lock:
            if self.now is None or now > self.now:
                self.now = now
            if now.time() >= self.config.square_off:
                self._square_off_all(now)

    def _square_off_all(self, when: dt.datetime) -> None:
        for trade in list(self._open.values()):
            self._close(trade, "square_off", self._last_price.get(trade.symbol, trade.entry_price))

    def _manage(self, trade: OpenTrade, price: float, when: dt.datetime) -> None:
        if when.time() >= self.config.square_off:
            self._close(trade, "square_off", price)
        elif price >= trade.stop:
            self._close(trade, "stop_loss", price)
        elif price <= trade.target:
            self._close(trade, "take_profit", price)

    # ---------------------------------------------------------------- signals

    def _on_bar(self, symbol: str, bar: Bar, tick: Tick) -> None:
        self.store.save_bar(symbol, bar)
        closes = self._closes.setdefault(symbol, [])
        closes.append(bar.close)
        if not bool(bearish_cross(pd.Series(closes), self.config.fast, self.config.slow).iloc[-1]):
            return
        reason = self._entry_block(symbol, bar)
        if reason:
            if self._last_skip.get(symbol) != reason:  # log each distinct reason once, not on every bar
                self._last_skip[symbol] = reason
                self.store.log_signal(tick.timestamp, symbol, "skipped", tick.price, reason)
            return
        self._enter(symbol, tick)

    def _entry_block(self, symbol: str, bar: Bar) -> Optional[str]:
        if self.halted:
            return f"halted: {self.halted}"
        if bar.start.date() != self.day:
            return "signal is not from today"
        if bar.start.time() > self.config.last_entry:
            return "after the last entry time"
        if symbol in self._open:
            return "already in a position"
        if symbol in self._traded:
            return "already traded this stock today"
        if self._trade_count >= self.config.max_trades_per_day:
            return "max trades per day reached"
        return None

    # ----------------------------------------------------------------- orders

    def _enter(self, symbol: str, tick: Tick) -> None:
        cfg = self.config
        quantity = int(cfg.capital_per_trade // tick.price)
        if quantity < 1:
            self.store.log_signal(tick.timestamp, symbol, "skipped", tick.price, "capital per trade is below one share")
            return
        self._traded.add(symbol)  # one attempt per stock per day, even if the order fails

        try:
            order = self.broker.place_order(symbol, Side.SELL, quantity, tick.price)
        except Exception as error:
            self._event("ERROR", f"{symbol}: entry order failed: {error}")
            self.store.log_signal(tick.timestamp, symbol, "skipped", tick.price, f"entry order failed: {error}")
            return
        self.store.log_order(order, "entry", tick.timestamp)
        status = self._await_final(order.order_id)
        if status.filled_quantity <= 0:
            self.store.log_signal(tick.timestamp, symbol, "skipped", tick.price, f"entry not filled ({status.status} {status.message})".strip())
            return

        filled = status.filled_quantity
        entry_price = status.average_price or order.price
        target = entry_price * (1 - cfg.take_profit_pct)
        stop = entry_price * (1 + cfg.stop_loss_pct)
        trade_id = self.store.open_trade(self.day, symbol, filled, tick.timestamp, entry_price, target, stop, None)
        trade = OpenTrade(trade_id, symbol, filled, tick.timestamp, entry_price, target, stop, None)
        self._open[symbol] = trade
        self._trade_count += 1

        protection = self._place_stop(symbol, filled, stop, trade_id, tick.timestamp)
        if protection is None:
            self._event("CRITICAL", f"{symbol}: could not place the stop-loss, closing the position")
            self._close(trade, "no_stop_protection", tick.price)
            return
        trade.stop_order_id = protection.order_id
        self.store.log_signal(tick.timestamp, symbol, "entered", entry_price, f"qty {filled}, target {target:.2f}, stop {stop:.2f}")

    def _place_stop(self, symbol: str, quantity: int, stop: float, trade_id: int, when: dt.datetime):
        for attempt in (1, 2):
            try:
                order = self.broker.place_stop_loss(symbol, Side.BUY, quantity, stop)
            except Exception as error:
                log.warning("stop-loss attempt %d for %s failed: %s", attempt, symbol, error)
                continue
            self.store.log_order(order, "stop", when, trade_id)
            self.store.set_stop_order(trade_id, order.order_id)
            return order
        return None

    def _close(self, trade: OpenTrade, reason: str, price: float) -> bool:
        """Exit a position. Returns True if it is now closed."""
        outcome = self._neutralise_stop(trade)
        if outcome == "filled":  # the exchange stop already executed: do not buy a second time
            status = self._status(trade.stop_order_id)
            exit_price = (status.average_price if status and status.average_price else price)
            self._finish(trade, "stop_loss", exit_price)
            return True
        if outcome == "failed":
            self.halt(f"{trade.symbol}: stop order could not be cancelled")
            self._event("CRITICAL", f"{trade.symbol}: cannot cancel the stop-loss, not covering (would risk a double buy). Check manually")
            return False

        remaining, covered, value = trade.quantity, 0, 0.0
        for _ in range(self.config.cover_retries):
            try:
                order = self.broker.place_order(trade.symbol, Side.BUY, remaining, price)
            except Exception as error:
                log.warning("cover order for %s failed: %s", trade.symbol, error)
                continue
            self.store.log_order(order, "exit", self.now or ist_now(), trade.trade_id)
            status = self._await_final(order.order_id)
            if status.filled_quantity > 0:
                covered += status.filled_quantity
                value += status.filled_quantity * (status.average_price or order.price)
                remaining -= status.filled_quantity
            if remaining <= 0:
                break

        if remaining > 0:
            self.halt(f"{trade.symbol}: could not close the position")
            self._event("CRITICAL", f"{trade.symbol}: {remaining} of {trade.quantity} still short after {self.config.cover_retries} attempts. Close it manually")
            restored = self._place_stop(trade.symbol, remaining, trade.stop, trade.trade_id, self.now or ist_now())
            trade.stop_order_id = restored.order_id if restored else None
            return False
        self._finish(trade, reason, value / covered)
        return True

    def _neutralise_stop(self, trade: OpenTrade) -> str:
        """Make sure the resting stop cannot fire after we cover. Returns 'none', 'cancelled', 'filled' or 'failed'."""
        stop_id = trade.stop_order_id
        if not stop_id:
            return "none"
        for _ in range(3):
            if self.broker.cancel_order(stop_id):
                return "cancelled"
            status = self._status(stop_id)
            if status is not None:
                if status.status == COMPLETE:
                    return "filled"
                if status.status in (CANCELLED, REJECTED):
                    return "cancelled"
            self._sleep(self.config.fill_poll_s)
        # Still unresolved. If the position has vanished the stop must have executed.
        try:
            return "filled" if self.broker.positions().get(trade.symbol, 0) == 0 else "failed"
        except Exception:
            return "failed"

    def _finish(self, trade: OpenTrade, reason: str, exit_price: float) -> None:
        when = self.now or ist_now()
        pnl = (trade.entry_price - exit_price) * trade.quantity
        self.store.close_trade(trade.trade_id, when, exit_price, reason, pnl)
        self._open.pop(trade.symbol, None)
        self._realized += pnl
        self.store.log_signal(when, trade.symbol, f"exited: {reason}", exit_price, f"pnl {pnl:.2f}")

    def _await_final(self, order_id: str) -> OrderStatus:
        """Wait for an order to reach a final state; cancel it if it does not in time."""
        cfg = self.config
        polls = max(1, int(cfg.fill_timeout_s / cfg.fill_poll_s))
        status: Optional[OrderStatus] = None
        for _ in range(polls):
            status = self._status(order_id) or status
            if status is not None and status.is_final:
                return status
            self._sleep(cfg.fill_poll_s)
        self.broker.cancel_order(order_id)
        status = self._status(order_id)
        if status is None:
            self.halt(f"order {order_id}: state unknown")
            self._event("CRITICAL", f"order {order_id}: could not read its state; check the broker for a stray position")
            return OrderStatus(order_id, "UNKNOWN")
        return status

    def _status(self, order_id: Optional[str]) -> Optional[OrderStatus]:
        if not order_id:
            return None
        try:
            return self.broker.order_status(order_id)
        except Exception as error:
            log.warning("order_status(%s) failed: %s", order_id, error)
            return None

    # ------------------------------------------------------------------ risk

    def _risk_check(self) -> None:
        limit = self.config.max_daily_loss
        if limit is None or self.halted:
            return
        open_loss = sum((t.entry_price - self._last_price.get(t.symbol, t.entry_price)) * t.quantity for t in self._open.values())
        if self._realized + open_loss <= -limit:
            self.halt(f"daily loss limit of {limit:.0f} reached")
            if self.config.flatten_on_loss_limit:
                self.flatten("loss_limit")

    def halt(self, reason: str) -> None:
        """Kill switch: no new entries until `resume()`. Exits keep working. Survives a restart."""
        with self._lock:
            if self.halted:
                return
            self.halted = reason
            self.store.set(self._halt_key, reason)
            self._event("WARNING", f"trading halted: {reason}")

    def resume(self) -> None:
        with self._lock:
            self.halted = None
            self.store.set(self._halt_key, None)
            self._event("INFO", "trading resumed")

    def flatten(self, reason: str = "flatten") -> None:
        """Close every open position at the last seen price."""
        with self._lock:
            for trade in list(self._open.values()):
                self._close(trade, reason, self._last_price.get(trade.symbol, trade.entry_price))

    # ----------------------------------------------------------------- finish

    def end_of_day(self) -> dict:
        """Save the bars in progress, close anything still open and return a summary."""
        with self._lock:
            for symbol, builder in self._builders.items():
                bar = builder.flush()
                if bar is not None:
                    self.store.save_bar(symbol, bar)
            self.flatten("end_of_day")
            return self.summary()

    def summary(self) -> dict:
        trades = self.store.trades(self.day)
        closed = [t for t in trades if t.status == "closed" and t.pnl is not None]
        return {
            "day": self.day.isoformat(),
            "shortlist": sorted(self.symbols),
            "trades": len(trades),
            "open": len(self._open),
            "wins": sum(1 for t in closed if t.pnl > 0),
            "realized_pnl": round(sum(t.pnl for t in closed), 2),
            "halted": self.halted,
        }

    # ---------------------------------------------------------------- helpers

    def _event(self, level: str, message: str, notes: Optional[list[str]] = None) -> None:
        self.store.log_event(level, message, self.now)
        {"INFO": log.info, "WARNING": log.warning}.get(level, log.error)(message)
        if notes is not None:
            notes.append(f"{level}: {message}")
