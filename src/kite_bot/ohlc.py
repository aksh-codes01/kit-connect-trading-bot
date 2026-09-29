"""Everything about OHLC bars, in one place.

    Instruments   load_instruments, instrument_token, instrument_tokens, symbol_for_token
    Fetching      fetch_ohlc (any interval and range), fetch_history (by token)
    Reshaping     resample_ohlc (1-minute -> 2-minute etc.), ticks_to_ohlc
    Files         load_bars, save_bars, load_directory, write_market

Frames are indexed by timestamp (oldest first) with columns open, high, low,
close and usually volume, the same shape Kite's historical_data returns.

CSV layout used by the runner (one folder, two files per symbol):
    <SYMBOL>_day.csv    daily bars
    <SYMBOL>_1min.csv   1-minute bars
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Callable, Optional, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

REQUIRED = ("open", "high", "low", "close")

# Most days Kite lets one historical_data call span, per interval. These are kept
# a little below the documented limits; check the current Kite Connect docs.
CHUNK_DAYS = {
    "minute": 55,
    "3minute": 95,
    "5minute": 95,
    "10minute": 95,
    "15minute": 190,
    "30minute": 190,
    "60minute": 390,
    "day": 1900,
}


# ------------------------------------------------------------------ instruments


def load_instruments(kite, exchange: str = "NSE") -> pd.DataFrame:
    """Download the instrument list for an exchange (one call; Kite refreshes it daily)."""
    return pd.DataFrame(kite.instruments(exchange))


def instrument_token(instruments: pd.DataFrame, symbol: str) -> int:
    """Instrument token for a trading symbol, from the frame `load_instruments` returns."""
    rows = instruments[instruments["tradingsymbol"] == symbol]
    if rows.empty:
        raise KeyError(f"unknown trading symbol: {symbol!r}")
    return int(rows["instrument_token"].iloc[0])


def instrument_tokens(instruments: pd.DataFrame, symbols: Sequence[str]) -> list[int]:
    """Tokens for several symbols, in the same order."""
    return [instrument_token(instruments, symbol) for symbol in symbols]


def symbol_for_token(instruments: pd.DataFrame, token: int) -> str:
    """Trading symbol for an instrument token (used to label streaming ticks)."""
    rows = instruments[instruments["instrument_token"] == token]
    if rows.empty:
        raise KeyError(f"unknown instrument token: {token!r}")
    return str(rows["tradingsymbol"].iloc[0])


# --------------------------------------------------------------------- fetching


def fetch_history(
    kite,
    token: int,
    start: dt.date,
    end: dt.date,
    interval: str,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download bars for one instrument token, in windows the API allows, stitched together.

    Sleeps between calls to stay under Kite's request rate limit. Untested against
    the live API.
    """
    if interval not in CHUNK_DAYS:
        raise ValueError(f"interval must be one of {sorted(CHUNK_DAYS)}")
    step = dt.timedelta(days=CHUNK_DAYS[interval])
    frames = []
    window_start = start
    while window_start <= end:
        window_end = min(window_start + step - dt.timedelta(days=1), end)
        rows = kite.historical_data(token, window_start, window_end, interval)
        if rows:
            frames.append(pd.DataFrame(rows))
        window_start = window_end + dt.timedelta(days=1)
        if window_start <= end:
            sleep(0.4)
    if not frames:
        return pd.DataFrame(columns=list(REQUIRED) + ["volume"])
    return pd.concat(frames).drop_duplicates(subset="date").set_index("date").sort_index()


def fetch_ohlc(
    kite,
    instruments: pd.DataFrame,
    symbol: str,
    interval: str,
    days: Optional[int] = None,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Bars for a trading symbol over the last `days` days, or from `start` to `end`.

    Long ranges are split into as many API calls as needed, so this replaces both
    the old fetchOHLC (recent data) and fetchOHLCExtended (data since inception).
    """
    if (days is None) == (start is None):
        raise ValueError("give either `days` or `start`")
    end = end or dt.date.today()
    if days is not None:
        start = end - dt.timedelta(days=days)
    return fetch_history(kite, instrument_token(instruments, symbol), start, end, interval, sleep)


# ------------------------------------------------------------------- reshaping


def resample_ohlc(bars: pd.DataFrame, rule: str = "2min") -> pd.DataFrame:
    """Aggregate bars to a coarser interval (for example 1-minute to 2-minute).

    Bins are anchored at the first bar (09:15), so 2-minute bars start at
    09:15, 09:17, and so on. Empty bins (overnight) are dropped.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in bars.columns:
        agg["volume"] = "sum"
    out = bars.resample(rule, origin="start").agg(agg)
    return out.dropna(subset=["close"])


def naive_ist(df: pd.DataFrame) -> pd.DataFrame:
    """Give a frame a timezone-free index showing India time.

    Kite's historical data is timezone-aware (+05:30) while its live ticks are
    naive local times; the live engine works with naive India time throughout.
    """
    if getattr(df.index, "tz", None) is None:
        return df
    out = df.copy()
    out.index = out.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return out


def to_ist_naive(ts: dt.datetime) -> dt.datetime:
    """Convert one timestamp to naive India time.

    A naive input is taken to be in the machine's local timezone, which is how the
    Kite ticker library builds tick times. That keeps bars correct on a UTC server.
    """
    return ts.astimezone(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)


def ist_now() -> dt.datetime:
    """Current India time as a naive datetime, whatever timezone the machine is set to."""
    return dt.datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)


def ticks_to_ohlc(prices: pd.Series, rule: str = "5min") -> pd.DataFrame:
    """Build OHLC bars from a series of tick prices indexed by timestamp."""
    return prices.resample(rule).ohlc().dropna()


# ------------------------------------------------------------------------ files


def load_bars(path) -> pd.DataFrame:
    """Read a bar CSV into a DataFrame indexed by date, sorted oldest first."""
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df.set_index("date").sort_index()


def save_bars(df: pd.DataFrame, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.rename_axis("date").to_csv(path)


def load_directory(data_dir) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load every <SYMBOL>_day.csv and <SYMBOL>_1min.csv in a folder."""
    folder = Path(data_dir)
    daily = {p.name[: -len("_day.csv")]: load_bars(p) for p in sorted(folder.glob("*_day.csv"))}
    minute = {p.name[: -len("_1min.csv")]: load_bars(p) for p in sorted(folder.glob("*_1min.csv"))}
    if not daily:
        raise FileNotFoundError(f"no *_day.csv files found in {folder}")
    return daily, minute


def write_market(daily, minute, data_dir) -> None:
    folder = Path(data_dir)
    for symbol, df in daily.items():
        save_bars(df, folder / f"{symbol}_day.csv")
    for symbol, df in minute.items():
        save_bars(df, folder / f"{symbol}_1min.csv")
