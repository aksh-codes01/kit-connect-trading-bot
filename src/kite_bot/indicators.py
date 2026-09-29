"""Every technical indicator in the project, in one place.

All functions take an OHLC DataFrame (columns ``open, high, low, close``, oldest
row first) or a price Series and return new objects; inputs are never modified.

Formulas are ports of the original exploratory scripts, so numbers match them.
Where a choice looks odd (for example ATR smoothing with ``com=n`` instead of
Wilder's ``com=n-1``) it is kept on purpose and noted in the docstring.

Contents
    Moving averages   sma, ema
    Volatility        true_range, atr, bollinger_bands
    Momentum / trend  macd, rsi, adx, supertrend, slope
    Levels            pivot_levels
    Renko             renko, renko_brick_size
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- moving averages


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average over `n` observations."""
    return series.rolling(n).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average with span `n`; NaN until `n` observations exist."""
    return series.ewm(span=n, min_periods=n).mean()


# ------------------------------------------------------------------- volatility


def true_range(df: pd.DataFrame) -> pd.Series:
    """Largest of high-low, |high - previous close|, |low - previous close|."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [(df["high"] - df["low"]).abs(), (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1, skipna=False)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Average true range: exponential average of the true range, ``com=n``.

    The original course formula uses ``com=n`` (alpha = 1/(n+1)), which is not
    Wilder's smoothing (``com=n-1``). Kept as is so results match the old scripts.
    """
    return true_range(df).ewm(com=n, min_periods=n).mean()


def bollinger_bands(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Bollinger bands (2 population standard deviations) around the n-bar mean.

    Returns a copy of `df` with MA, BB_up, BB_dn and BB_width; warm-up rows are dropped.
    """
    out = df.copy()
    out["MA"] = out["close"].rolling(n).mean()
    spread = 2 * out["close"].rolling(n).std(ddof=0)  # ddof=0: population, not sample
    out["BB_up"] = out["MA"] + spread
    out["BB_dn"] = out["MA"] - spread
    out["BB_width"] = out["BB_up"] - out["BB_dn"]
    return out.dropna()


# ------------------------------------------------------------- momentum / trend


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line (fast EMA - slow EMA) and its signal line.

    Returns a copy of `df` with MA_Fast, MA_Slow, MACD and Signal; warm-up rows are dropped.
    """
    out = df.copy()
    out["MA_Fast"] = ema(out["close"], fast)
    out["MA_Slow"] = ema(out["close"], slow)
    out["MACD"] = out["MA_Fast"] - out["MA_Slow"]
    out["Signal"] = ema(out["MACD"], signal)
    return out.dropna()


def rsi(df: pd.DataFrame, n: int) -> pd.Series:
    """Relative strength index. The first average gain/loss is a plain mean of `n` values."""
    delta = df["close"].diff().dropna()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    gain.iloc[n - 1] = gain.iloc[:n].mean()
    loss.iloc[n - 1] = loss.iloc[:n].mean()
    gain = gain.iloc[n - 1 :]
    loss = loss.iloc[n - 1 :]
    rs = gain.ewm(com=n, min_periods=n).mean() / loss.ewm(com=n, min_periods=n).mean()
    return 100 - 100 / (1 + rs)


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    """Average directional index (Wilder's smoothing, seeded with a plain sum/mean)."""
    high, low = df["high"], df["low"]
    tr = true_range(df).to_numpy(dtype=float)
    up_move = (high - high.shift(1)).to_numpy(dtype=float)
    down_move = (low.shift(1) - low).to_numpy(dtype=float)
    dm_plus = np.where(up_move > down_move, up_move, 0.0)
    dm_minus = np.where(down_move > up_move, down_move, 0.0)
    dm_plus = np.where(dm_plus < 0, 0.0, dm_plus)
    dm_minus = np.where(dm_minus < 0, 0.0, dm_minus)

    size = len(df)
    tr_n = np.full(size, np.nan)
    dmp_n = np.full(size, np.nan)
    dmm_n = np.full(size, np.nan)
    if size > n:
        tr_n[n] = pd.Series(tr).rolling(n).sum().iloc[n]
        dmp_n[n] = pd.Series(dm_plus).rolling(n).sum().iloc[n]
        dmm_n[n] = pd.Series(dm_minus).rolling(n).sum().iloc[n]
        for i in range(n + 1, size):
            tr_n[i] = tr_n[i - 1] - tr_n[i - 1] / n + tr[i]
            dmp_n[i] = dmp_n[i - 1] - dmp_n[i - 1] / n + dm_plus[i]
            dmm_n[i] = dmm_n[i - 1] - dmm_n[i - 1] / n + dm_minus[i]

    di_plus = 100 * dmp_n / tr_n
    di_minus = 100 * dmm_n / tr_n
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

    result = np.full(size, np.nan)
    first = 2 * n - 1
    if size > first:
        result[first] = dx[first - n + 1 : first + 1].mean()
        for j in range(first + 1, size):
            result[j] = ((n - 1) * result[j - 1] + dx[j]) / n
    return pd.Series(result, index=df.index, name="ADX")


def supertrend(df: pd.DataFrame, n: int, m: float) -> pd.Series:
    """Supertrend line. `n` is the ATR period (7 is common), `m` the multiplier (2 or 3).

    Returns the active band on every bar from the first trend flip onward, and
    NaN before it. If price closes above the line the trend is up (line below
    price), below it the trend is down.
    """
    close = df["close"].to_numpy(dtype=float)
    mid = ((df["high"] + df["low"]) / 2).to_numpy(dtype=float)
    band = m * atr(df, n).to_numpy(dtype=float)
    basic_upper = mid + band
    basic_lower = mid - band
    upper = basic_upper.copy()
    lower = basic_lower.copy()
    size = len(df)

    for i in range(n, size):
        upper[i] = min(basic_upper[i], upper[i - 1]) if close[i - 1] <= upper[i - 1] else basic_upper[i]
    for i in range(n, size):
        lower[i] = max(basic_lower[i], lower[i - 1]) if close[i - 1] >= lower[i - 1] else basic_lower[i]

    line = np.full(size, np.nan)
    start = size  # index after the first flip; stays `size` if there is none
    for t in range(n, size):
        if close[t - 1] <= upper[t - 1] and close[t] > upper[t]:
            line[t], start = lower[t], t + 1
            break
        if close[t - 1] >= lower[t - 1] and close[t] < lower[t]:
            line[t], start = upper[t], t + 1
            break

    for i in range(start, size):
        if line[i - 1] == upper[i - 1] and close[i] <= upper[i]:
            line[i] = upper[i]
        elif line[i - 1] == upper[i - 1] and close[i] >= upper[i]:
            line[i] = lower[i]
        elif line[i - 1] == lower[i - 1] and close[i] >= lower[i]:
            line[i] = lower[i]
        elif line[i - 1] == lower[i - 1] and close[i] <= lower[i]:
            line[i] = upper[i]
    return pd.Series(line, index=df.index, name="Strend")


def slope(df: pd.DataFrame, n: int) -> float:
    """Angle in degrees of the regression line through the last `n` candle midpoints.

    Prices and bar numbers are both scaled to 0..1 first, so the angle is
    comparable across stocks. A perfectly flat window returns 0.0.
    """
    tail = df.iloc[-n:]
    y = ((tail["open"] + tail["close"]) / 2).to_numpy(dtype=float)
    if y.max() == y.min():
        return 0.0
    y_scaled = (y - y.min()) / (y.max() - y.min())
    x = np.arange(n, dtype=float)
    x_scaled = (x - x.min()) / (x.max() - x.min())
    gradient = np.polyfit(x_scaled, y_scaled, 1)[0]  # least-squares slope, same as OLS with a constant
    return float(np.rad2deg(np.arctan(gradient)))


# ----------------------------------------------------------------------- levels


class PivotLevels(NamedTuple):
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


def pivot_levels(day_df: pd.DataFrame) -> PivotLevels:
    """Classic pivot point with three resistance and support levels.

    Uses the last row of `day_df`, which should be the previous session's daily bar.
    """
    high = round(float(day_df["high"].iloc[-1]), 2)
    low = round(float(day_df["low"].iloc[-1]), 2)
    close = round(float(day_df["close"].iloc[-1]), 2)
    pivot = round((high + low + close) / 3, 2)
    return PivotLevels(
        pivot=pivot,
        r1=round(2 * pivot - low, 2),
        r2=round(pivot + (high - low), 2),
        r3=round(high + 2 * (pivot - low), 2),
        s1=round(2 * pivot - high, 2),
        s2=round(pivot - (high - low), 2),
        s3=round(low - 2 * (high - pivot), 2),
    )


# ------------------------------------------------------------------------ renko


def renko(df: pd.DataFrame, brick_size: float = 10) -> pd.DataFrame:
    """Convert OHLC bars to Renko bricks using the `stocktrends` package.

    `stocktrends` is imported here so the rest of the module works without it
    (``pip install stocktrends``). The index of `df` becomes the ``date`` column.
    """
    from stocktrends import Renko

    frame = df.copy()
    frame.index.name = "date"
    frame = frame.reset_index()
    converter = Renko(frame)
    converter.brick_size = brick_size
    return converter.get_ohlc_data()


def renko_brick_size(df: pd.DataFrame, atr_period: int = 200, multiplier: float = 1.5, lowest: float = 1, highest: float = 10) -> float:
    """Brick size from recent volatility: ``multiplier * latest ATR``, rounded and clamped."""
    latest = float(atr(df, atr_period).iloc[-1])
    return float(min(highest, max(lowest, round(multiplier * latest, 0))))
