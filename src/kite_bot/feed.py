"""Live market data: ticks in, 2-minute bars out.

    Tick           one price update for a symbol
    BarBuilder     turns a tick stream into fixed-width bars (aligned like `resample_ohlc`)
    TickFeed       interface every data source implements
    KiteTickerFeed streams from Kite's WebSocket (KiteTicker)
    ReplayFeed     replays historical 1-minute bars as ticks, for offline runs and tests

How live data reaches the bot
    Kite pushes binary tick packets over one WebSocket (up to 3000 instruments per
    connection, 3 connections per API key). The KiteTicker library decodes them and
    calls `on_ticks` on its own thread. `KiteTickerFeed` converts each tick to a
    `Tick` and hands it on. No polling of the REST API is needed for prices.
"""

from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from kite_bot.ohlc import ist_now, naive_ist, to_ist_naive

log = logging.getLogger(__name__)

SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 30)


@dataclass(frozen=True)
class Tick:
    symbol: str
    price: float
    timestamp: dt.datetime  # naive India time
    volume: int = 0  # quantity of this trade, when the feed provides it


@dataclass(frozen=True)
class Bar:
    start: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0


def bar_start(ts: dt.datetime, minutes: int = 2, session_open: dt.time = SESSION_OPEN) -> dt.datetime:
    """Start time of the `minutes`-wide bar containing `ts` (bars are anchored at the session open)."""
    opened = dt.datetime.combine(ts.date(), session_open)
    elapsed = int((ts - opened).total_seconds() // 60)
    return opened + dt.timedelta(minutes=(elapsed // minutes) * minutes)


class BarBuilder:
    """Builds `minutes`-wide bars from ticks, with bars starting at 09:15, 09:17, ... for 2 minutes.

    `update` returns the previous bar when a tick arrives that belongs to a later
    bar, so a bar is handed over at the moment the next one opens. Ticks outside
    the trading session, and ticks older than the current bar, are ignored.
    """

    def __init__(self, minutes: int = 2, session_open: dt.time = SESSION_OPEN, session_close: dt.time = SESSION_CLOSE):
        self.minutes = minutes
        self._open = session_open
        self._close = session_close
        self._bar: Optional[Bar] = None

    def update(self, tick: Tick) -> Optional[Bar]:
        if not (self._open <= tick.timestamp.time() < self._close):
            return None
        start = bar_start(tick.timestamp, self.minutes, self._open)
        current = self._bar
        if current is None or start > current.start:
            self._bar = Bar(start, tick.price, tick.price, tick.price, tick.price, tick.volume)
            return current
        if start == current.start:
            self._bar = Bar(
                current.start,
                current.open,
                max(current.high, tick.price),
                min(current.low, tick.price),
                tick.price,
                current.volume + tick.volume,
            )
        return None

    def flush(self) -> Optional[Bar]:
        """Hand over the bar in progress (end of day)."""
        bar, self._bar = self._bar, None
        return bar


class TickFeed(ABC):
    """A source of ticks. `start` begins delivering to the callback; `stop` ends it."""

    finished: bool = False  # True once a finite feed (a replay) has delivered everything
    realtime: bool = True  # False for replays, which run on historical time rather than the wall clock

    @abstractmethod
    def start(self, symbols: Iterable[str], on_tick: Callable[[Tick], None]) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


class ReplayFeed(TickFeed):
    """Replays 1-minute bars of one day as a chronological tick stream.

    Each bar becomes ticks along open -> first extreme -> second extreme -> close
    (low first for a green bar, high first for a red bar), sampled `substeps`
    times between those points. The bar's own open, high, low and close are all
    hit exactly, so bars rebuilt from these ticks equal the historical bars.
    """

    realtime = False

    def __init__(self, minute: Mapping[str, pd.DataFrame], day: dt.date, substeps: int = 4):
        self._minute = minute
        self._day = day
        self._substeps = substeps

    def ticks(self, symbols: Iterable[str]) -> list[Tick]:
        steps = 3 * self._substeps
        spacing = 60.0 / (steps + 1)
        waypoints_at = [0.0, 1 / 3, 2 / 3, 1.0]
        out: list[Tick] = []
        for symbol in symbols:
            frame = naive_ist(self._minute[symbol])
            frame = frame[frame.index.date == self._day]
            for ts, row in frame.iterrows():
                o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
                first, second = (l, h) if c >= o else (h, l)
                prices = np.interp(np.arange(steps + 1) / steps, waypoints_at, [o, first, second, c])
                per_tick = int(row["volume"] // (steps + 1)) if "volume" in frame.columns else 0
                base = ts.to_pydatetime()
                for k, price in enumerate(prices):
                    out.append(Tick(symbol, float(price), base + dt.timedelta(seconds=k * spacing), per_tick))
        out.sort(key=lambda t: (t.timestamp, t.symbol))
        return out

    def start(self, symbols: Iterable[str], on_tick: Callable[[Tick], None]) -> None:
        self.finished = False
        for tick in self.ticks(list(symbols)):
            on_tick(tick)
        self.finished = True

    def stop(self) -> None:
        self.finished = True


class KiteTickerFeed(TickFeed):
    """Streams live ticks from Kite's WebSocket using the `KiteTicker` client.

    Args:
        api_key, access_token: the session credentials.
        token_to_symbol: {instrument_token: trading symbol}; only symbols passed to
            `start` are subscribed.
        on_status: optional callback(str) told about connect, close, reconnect and
            give-up events, so the engine can react to a lost connection.
        ticker_factory: builds the KiteTicker (injectable so tests need no network).
        clock: used for ticks that arrive without an exchange timestamp.

    Subscribes in FULL mode, the only mode whose packets carry the exchange timestamp.
    KiteTicker reconnects by itself after a drop. Its reactor cannot be restarted
    inside one process, so after `stop()` start a new process instead of a new feed.
    Untested against the live service.
    """

    def __init__(
        self,
        api_key: str,
        access_token: str,
        token_to_symbol: Mapping[int, str],
        on_status: Optional[Callable[[str], None]] = None,
        ticker_factory: Optional[Callable] = None,
        clock: Callable[[], dt.datetime] = ist_now,
    ):
        self._api_key = api_key
        self._access_token = access_token
        self._token_to_symbol = dict(token_to_symbol)
        self._on_status = on_status or (lambda message: None)
        self._factory = ticker_factory or self._default_factory
        self._clock = clock
        self._ticker = None

    @staticmethod
    def _default_factory(api_key: str, access_token: str):
        from kiteconnect import KiteTicker

        return KiteTicker(api_key, access_token)

    def start(self, symbols: Iterable[str], on_tick: Callable[[Tick], None]) -> None:
        wanted = set(symbols)
        subscribed = {token: sym for token, sym in self._token_to_symbol.items() if sym in wanted}
        tokens = list(subscribed)
        ticker = self._factory(self._api_key, self._access_token)
        self._ticker = ticker

        def on_ticks(ws, ticks):
            for raw in ticks:
                symbol = subscribed.get(raw.get("instrument_token"))
                if symbol is None:
                    continue
                try:
                    on_tick(self._convert(symbol, raw))
                except Exception:  # never let one bad tick kill the WebSocket thread
                    log.exception("error handling tick for %s", symbol)

        def on_connect(ws, response):
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            self._on_status(f"connected, subscribed to {len(tokens)} instruments")

        ticker.on_ticks = on_ticks
        ticker.on_connect = on_connect
        ticker.on_close = lambda ws, code, reason: self._on_status(f"closed: {code} {reason}")
        ticker.on_error = lambda ws, code, reason: self._on_status(f"error: {code} {reason}")
        ticker.on_reconnect = lambda ws, attempts: self._on_status(f"reconnecting (attempt {attempts})")
        ticker.on_noreconnect = lambda ws: self._on_status("gave up reconnecting")
        ticker.connect(threaded=True)

    def _convert(self, symbol: str, raw: Mapping) -> Tick:
        stamp = raw.get("exchange_timestamp") or raw.get("last_trade_time")
        when = to_ist_naive(stamp) if stamp else self._clock()
        return Tick(symbol, float(raw["last_price"]), when, int(raw.get("last_traded_quantity") or 0))

    def stop(self) -> None:
        if self._ticker is not None:
            self._ticker.close()
