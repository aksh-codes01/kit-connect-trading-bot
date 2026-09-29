"""Daily-timeframe screener: first close across MA100 after the MA's slope turns.

Setup definition (default ``side="short"``)
-------------------------------------------
1. The MA's slope turns from up to down. The slope is the change in the MA over
   ``slope_window`` bars, so a single wiggle does not count as a turn.
2. Since that turn, the stock's close crosses ABOVE the MA for the FIRST time.
   A cross is a close above the MA when the previous close was at or below it.
3. That first cross happened on the last traded day. The stock is then
   shortlisted for a short sale on the next day's intraday session.

``side="long"`` is the mirror image: the slope turns up, the first close
BELOW the MA is a pullback, and the stock is shortlisted for a buy.

This module is pure: it takes DataFrames and returns results. No broker calls,
no file access, so it can be unit tested without a Kite account.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from kite_bot.indicators import sma

VALID_SIDES = ("short", "long")


@dataclass(frozen=True)
class Setup:
    """A stock that matched the screening criteria."""

    side: str
    turn_date: pd.Timestamp  # first bar of the new MA slope direction
    cross_date: pd.Timestamp  # first close across the MA since the turn
    close: float  # close on the last bar
    ma: float  # MA value on the last bar


def _slope_direction(ma: pd.Series, window: int) -> np.ndarray:
    """+1 where the MA is higher than `window` bars ago, -1 lower, 0 if flat/unknown."""
    diff = (ma - ma.shift(window)).to_numpy(dtype=float)
    direction = np.sign(diff)
    direction[np.isnan(diff)] = 0.0
    return direction


def find_setup(
    df: pd.DataFrame,
    side: str = "short",
    ma_period: int = 100,
    slope_window: int = 5,
    max_age_bars: int = 0,
) -> Optional[Setup]:
    """Return a Setup if the last bar of `df` completes the pattern, else None.

    Args:
        df: daily bars sorted oldest to newest, with a ``close`` column and a
            date-like index (the last row is the last traded day).
        side: "short" (slope turns down, first close above MA) or "long" (mirror).
        ma_period: simple moving average length in bars.
        slope_window: bars over which the MA slope is measured.
        max_age_bars: how many bars ago the first cross may have happened.
            0 means it must be the last bar.
    """
    if side not in VALID_SIDES:
        raise ValueError(f"side must be one of {VALID_SIDES}, got {side!r}")
    if "close" not in df.columns:
        raise ValueError("df needs a 'close' column")
    if len(df) < ma_period + slope_window + 2:
        return None

    close = df["close"].astype(float)
    ma = sma(close, ma_period)
    direction = _slope_direction(ma, slope_window)

    want = -1.0 if side == "short" else 1.0  # slope direction we need now

    # 1) current slope must match, and the run of it must start with a real turn
    last = len(df) - 1
    if direction[last] != want:
        return None
    turn = last
    while turn > 0 and direction[turn - 1] == want:
        turn -= 1
    if turn == 0 or direction[turn - 1] != -want:
        return None

    # 2) cross events (short: close crosses above MA, long: crosses below)
    if side == "short":
        state = (close > ma).to_numpy()
    else:
        state = (close < ma).to_numpy()
    valid = ma.notna().to_numpy()
    prev_state = np.concatenate(([False], state[:-1]))
    prev_valid = np.concatenate(([False], valid[:-1]))
    cross = state & valid & prev_valid & ~prev_state

    # 3) first cross strictly after the turn bar
    candidates = np.flatnonzero(cross[turn + 1 :])
    if candidates.size == 0:
        return None
    first_cross = turn + 1 + int(candidates[0])

    # 4) it must be fresh
    if last - first_cross > max_age_bars:
        return None

    return Setup(
        side=side,
        turn_date=pd.Timestamp(df.index[turn]),
        cross_date=pd.Timestamp(df.index[first_cross]),
        close=float(close.iloc[last]),
        ma=float(ma.iloc[last]),
    )


def scan(
    universe: Mapping[str, pd.DataFrame],
    **kwargs,
) -> list[tuple[str, Setup]]:
    """Run find_setup over {symbol: daily_df}; return the (symbol, Setup) matches."""
    matches: list[tuple[str, Setup]] = []
    for symbol, df in universe.items():
        setup = find_setup(df, **kwargs)
        if setup is not None:
            matches.append((symbol, setup))
    return matches
