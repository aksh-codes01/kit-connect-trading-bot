"""Runs one trading session end to end: screen, warm up, stream, trade, wrap up.

    MarketData        where history comes from (Kite in production, dataframes in tests)
    run_session       the daily routine, identical for live, paper and replay runs
    load_universe     read the list of stocks to screen from a text file

Which Kite APIs a live session touches
    REST, before the open:   instruments (once), historical_data for each stock in the
                             universe (daily bars) and each shortlisted stock (1-minute bars)
    WebSocket, all day:      ticks for the shortlisted stocks only
    REST, when trading:      place_order, order_history, cancel_order, positions
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

from kite_bot.engine import LiveEngine
from kite_bot.feed import TickFeed
from kite_bot.ohlc import fetch_ohlc, ist_now

log = logging.getLogger(__name__)

MARKET_CLOSE = dt.time(15, 31)


class MarketData(ABC):
    @abstractmethod
    def daily(self, symbol: str) -> pd.DataFrame:
        """Daily bars, oldest first, including recent history for the 100-day average."""

    @abstractmethod
    def minute_history(self, symbol: str) -> pd.DataFrame:
        """Recent 1-minute bars covering the warm-up sessions."""


class KiteMarketData(MarketData):
    """History from the Kite REST API. Pauses between stocks to respect the rate limit
    (Kite allows about 3 historical-data requests per second). Untested against the live API."""

    def __init__(self, kite, instruments: pd.DataFrame, daily_days: int = 300, minute_days: int = 12,
                 pause_s: float = 0.4, sleep: Callable[[float], None] = time.sleep):
        self._kite, self._instruments = kite, instruments
        self._daily_days, self._minute_days = daily_days, minute_days
        self._pause, self._sleep = pause_s, sleep

    def daily(self, symbol: str) -> pd.DataFrame:
        frame = fetch_ohlc(self._kite, self._instruments, symbol, "day", days=self._daily_days, sleep=self._sleep)
        self._sleep(self._pause)
        return frame

    def minute_history(self, symbol: str) -> pd.DataFrame:
        frame = fetch_ohlc(self._kite, self._instruments, symbol, "minute", days=self._minute_days, sleep=self._sleep)
        self._sleep(self._pause)
        return frame


class FrameMarketData(MarketData):
    """History served from dataframes already in memory (replays and tests)."""

    def __init__(self, daily: Mapping[str, pd.DataFrame], minute: Mapping[str, pd.DataFrame]):
        self._daily, self._minute = daily, minute

    def daily(self, symbol: str) -> pd.DataFrame:
        return self._daily[symbol]

    def minute_history(self, symbol: str) -> pd.DataFrame:
        return self._minute[symbol]


def load_universe(path) -> list[str]:
    """Stocks to screen: one trading symbol per line; blank lines and # comments are ignored."""
    symbols = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            symbols.append(line.upper())
    return list(dict.fromkeys(symbols))  # keep order, drop repeats


def run_session(
    engine: LiveEngine,
    feed: TickFeed,
    data: MarketData,
    universe: Iterable[str],
    clock: Callable[[], dt.datetime] = ist_now,
    sleep: Callable[[float], None] = time.sleep,
    stop_at: dt.time = MARKET_CLOSE,
    notify: Callable[[str], None] = print,
) -> dict:
    """Run a whole session and return the day's summary.

    Live feeds run in the background, so this loops once a second doing the time
    checks until the market closes. A replay delivers everything inside `start`.
    """
    for note in engine.recover():
        notify(note)

    daily = {}
    for symbol in universe:
        try:
            daily[symbol] = data.daily(symbol)
        except Exception as error:  # one bad symbol must not stop the session
            engine.store.log_event("WARNING", f"{symbol}: could not load daily data: {error}")
            notify(f"skipping {symbol}: {error}")
    shortlist = engine.screen(daily)
    notify(f"shortlist ({len(shortlist)}): {', '.join(shortlist) or 'none'}")

    now = clock() if feed.realtime else None
    for symbol in list(shortlist):
        try:
            engine.load_warmup(symbol, data.minute_history(symbol), now=now)
        except Exception as error:
            engine.store.log_event("WARNING", f"{symbol}: could not load warm-up data: {error}")
            engine.symbols.discard(symbol)
            shortlist.remove(symbol)
            notify(f"dropping {symbol}: {error}")

    if not shortlist:
        return engine.end_of_day()

    feed.start(shortlist, engine.on_tick)
    while not feed.finished and clock().time() < stop_at:
        sleep(1)
        engine.check_time(clock())
    feed.stop()
    return engine.end_of_day()


def status_handler(engine: LiveEngine, notify: Callable[[str], None] = print) -> Callable[[str], None]:
    """Feed-status callback: report events, and stop taking new trades if the data connection is lost for good.

    Open positions stay protected by their exchange-side stop-loss, and the square-off
    time still closes them (`check_time` does not need ticks).
    """

    def handle(message: str) -> None:
        notify(f"feed: {message}")
        engine.store.log_event("INFO", f"feed: {message}")
        if "gave up" in message:
            engine.halt("market data connection lost")

    return handle
