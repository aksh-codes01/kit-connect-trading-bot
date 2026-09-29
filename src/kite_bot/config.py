"""Settings shared by the backtester and the live engine."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StrategyConfig:
    """Everything that defines the strategy itself, so a backtest and a live run behave alike."""

    ma_period: int = 100  # daily screener: moving average length
    slope_window: int = 5  # daily screener: bars over which the MA slope is measured
    fast: int = 100  # 2-minute trigger: fast EMA
    slow: int = 500  # 2-minute trigger: slow EMA
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.02
    square_off: dt.time = dt.time(15, 15)  # close everything from this time on
    last_entry: dt.time = dt.time(14, 30)  # no new entries after a signal bar starting later
    capital_per_trade: float = 100_000.0
    max_trades_per_day: int = 5
    warmup_sessions: int = 6  # earlier sessions of minute data loaded for the slow EMA


@dataclass(frozen=True)
class LiveConfig(StrategyConfig):
    """Strategy settings plus the safety limits that only matter when trading live."""

    max_daily_loss: Optional[float] = None  # rupees; new entries stop once realized + open loss reaches it
    flatten_on_loss_limit: bool = True  # also close open positions when the limit is hit
    fill_timeout_s: float = 10.0  # how long to wait for an order to fill before cancelling it
    fill_poll_s: float = 0.5
    cover_retries: int = 3  # attempts to close a position before giving up and halting
    record_ticks: bool = False  # store every tick in the database (large; off by default)
