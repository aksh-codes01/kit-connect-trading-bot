"""Intraday short strategy for stocks shortlisted by the daily screener.

Flow, for a stock shortlisted the previous evening:
    1. Load 1-minute bars (several past sessions for warm-up plus today) and
       resample them to 2-minute bars with `kite_bot.ohlc.resample_ohlc`.
    2. `find_short_entry` waits for a bearish crossover: the 100-period line
       crossing below the 500-period line. It shorts at the open of the bar
       after the signal bar, because the signal is only known once its bar closes.
    3. `simulate_short` manages the trade: 2% take-profit, 2% stop-loss, both
       measured from the short price, and a forced exit before the broker's
       intraday square-off time.

Everything here is pure (DataFrames in, results out) so it can be unit tested
and reused by a backtester or a live engine.

The "MACD 100 / MACD 500" line is computed as EMA(100) - EMA(500). It crosses
zero exactly when EMA(100) crosses EMA(500), so that is what `bearish_cross` tests.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from kite_bot.indicators import ema


@dataclass(frozen=True)
class ShortEntry:
    signal_time: pd.Timestamp  # bar on which the crossover was detected
    time: pd.Timestamp  # bar we sell on (the next bar)
    price: float  # open of that bar


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    reason: str  # "take_profit", "stop_loss", "square_off" or "end_of_data"
    pnl_pct: float  # profit as a fraction of the short price (0.02 = +2%), before costs


def bearish_cross(close: pd.Series, fast: int = 100, slow: int = 500) -> pd.Series:
    """True on the bar where EMA(fast) crosses from above to below EMA(slow).

    Bars before both EMAs have `slow` observations are never a signal.
    """
    diff = ema(close, fast) - ema(close, slow)
    prev = diff.shift(1)
    return (diff < 0) & (prev >= 0)


def find_short_entry(
    bars: pd.DataFrame,
    trade_date: dt.date,
    fast: int = 100,
    slow: int = 500,
    last_entry: dt.time = dt.time(14, 30),
) -> Optional[ShortEntry]:
    """First bearish crossover on `trade_date`, or None.

    `bars` are ascending 2-minute OHLC bars that include earlier sessions for
    warm-up. Crossovers on other days are ignored, and so are signals after
    `last_entry`. Entry is at the next bar's open, on the same day.
    """
    signals = np.flatnonzero(bearish_cross(bars["close"], fast, slow).to_numpy())
    index = bars.index
    opens = bars["open"].to_numpy(dtype=float)

    for i in signals:
        ts = index[i]
        if ts.date() != trade_date:
            continue
        if ts.time() > last_entry:
            break
        if i + 1 >= len(bars) or index[i + 1].date() != trade_date:
            break
        return ShortEntry(signal_time=ts, time=index[i + 1], price=float(opens[i + 1]))
    return None


def simulate_short(
    bars: pd.DataFrame,
    entry: ShortEntry,
    take_profit_pct: float = 0.02,
    stop_loss_pct: float = 0.02,
    square_off: dt.time = dt.time(15, 15),
) -> Trade:
    """Walk forward from the entry bar and return how the short trade ended.

    Conservative fill rules:
      * Take-profit and stop-loss are placed at +/- pct from the short price.
      * If one bar touches both levels, the stop-loss is assumed to hit first.
      * A gap through a level fills at the bar's open, not at the level.
      * At `square_off` the position is closed at that bar's open.
    """
    stop = entry.price * (1 + stop_loss_pct)
    target = entry.price * (1 - take_profit_pct)

    day = bars[(bars.index >= entry.time) & (bars.index.date == entry.time.date())]
    exit_time, exit_price, reason = day.index[-1], float(day["close"].iloc[-1]), "end_of_data"

    for ts, row in day.iterrows():
        if ts.time() >= square_off:
            exit_time, exit_price, reason = ts, float(row["open"]), "square_off"
            break
        if row["high"] >= stop:
            exit_time, exit_price, reason = ts, float(max(row["open"], stop)), "stop_loss"
            break
        if row["low"] <= target:
            exit_time, exit_price, reason = ts, float(min(row["open"], target)), "take_profit"
            break

    return Trade(
        entry_time=entry.time,
        entry_price=entry.price,
        exit_time=exit_time,
        exit_price=exit_price,
        reason=reason,
        pnl_pct=(entry.price - exit_price) / entry.price,
    )
