"""Every candlestick pattern in the project, in one place.

Single-candle detectors (`doji`, `maru_bozu`, `hammer`, `shooting_star`) take an
OHLC DataFrame and return a copy with one extra column, so they can be applied
to a whole history at once. `candle_type` and `candle_pattern` classify the
LAST candle of a DataFrame, using earlier candles for context.

Thresholds are ports of the original scanner script so results match it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from kite_bot.indicators import pivot_levels

# ------------------------------------------------------ single-candle detectors


def _typical_body(df: pd.DataFrame) -> float:
    """Median absolute candle body, used as the yardstick for "small" and "big" candles."""
    return float((df["close"] - df["open"]).abs().median())


def doji(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a boolean `doji` column: body no bigger than 5% of the typical body."""
    df = ohlc_df.copy()
    df["doji"] = (df["close"] - df["open"]).abs() <= 0.05 * _typical_body(df)
    return df


def maru_bozu(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `maru_bozu` column: "maru_bozu_green", "maru_bozu_red" or False.

    A marubozu is a long body (over twice the typical body) with almost no shadow.

    NOTE: mirrors the original scanner. For green candles only the upper shadow is
    checked (the lower-shadow term is never positive); for red candles both are.
    """
    df = ohlc_df.copy()
    typical = _typical_body(df)
    body_up = df["close"] - df["open"]
    body_down = df["open"] - df["close"]
    upper_shadow_green = df["high"] - df["close"]
    lower_shadow_green = df["low"] - df["open"]
    upper_shadow_red = df["high"] - df["open"]
    lower_shadow_red = df["low"] - df["close"]

    green = (body_up > 2 * typical) & (pd.concat([upper_shadow_green, lower_shadow_green], axis=1).max(axis=1) < 0.005 * typical)
    red = (body_down > 2 * typical) & (
        pd.concat([upper_shadow_red, lower_shadow_red], axis=1).abs().max(axis=1) < 0.005 * typical
    )

    labels = np.full(len(df), False, dtype=object)
    labels[green.to_numpy()] = "maru_bozu_green"
    labels[(red & ~green).to_numpy()] = "maru_bozu_red"
    df["maru_bozu"] = labels
    return df


def hammer(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a boolean `hammer` column: small body near the top, long lower shadow."""
    df = ohlc_df.copy()
    span = df["high"] - df["low"]
    df["hammer"] = (
        (span > 3 * (df["open"] - df["close"]))
        & ((df["close"] - df["low"]) / (0.001 + span) > 0.6)
        & ((df["open"] - df["low"]) / (0.001 + span) > 0.6)
        & ((df["close"] - df["open"]).abs() > 0.1 * span)
    )
    return df


def shooting_star(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a boolean `sstar` column: small body near the bottom, long upper shadow."""
    df = ohlc_df.copy()
    span = df["high"] - df["low"]
    df["sstar"] = (
        (span > 3 * (df["open"] - df["close"]))
        & ((df["high"] - df["close"]) / (0.001 + span) > 0.6)
        & ((df["high"] - df["open"]) / (0.001 + span) > 0.6)
        & ((df["close"] - df["open"]).abs() > 0.1 * span)
    )
    return df


# ------------------------------------------------------------ market structure


def trend(ohlc_df: pd.DataFrame, n: int) -> Optional[str]:
    """Return "uptrend" or "downtrend" from the last `n` candles, else None.

    Uptrend: last candle is green and at least 70% of the last `n` candles made a
    higher-or-equal low. Downtrend: last candle is red and at least 70% made a
    lower-or-equal high.
    """
    df = ohlc_df.copy()
    df["up"] = np.where(df["low"] >= df["low"].shift(1), 1, 0)
    df["dn"] = np.where(df["high"] <= df["high"].shift(1), 1, 0)
    last_close, last_open = df["close"].iloc[-1], df["open"].iloc[-1]
    if last_close > last_open:
        if df["up"].iloc[-n:].sum() >= 0.7 * n:
            return "uptrend"
    elif last_open > last_close:
        if df["dn"].iloc[-n:].sum() >= 0.7 * n:
            return "downtrend"
    return None


def support_resistance(ohlc_df: pd.DataFrame, ohlc_day: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """Nearest pivot-level support (below) and resistance (above) for the last candle.

    Returns (support, resistance); either is None if the candle sits outside all levels.
    """
    last = ohlc_df.iloc[-1]
    level = ((last["close"] + last["open"]) / 2 + (last["high"] + last["low"]) / 2) / 2
    levels = pivot_levels(ohlc_day)._asdict()
    below = [value for value in levels.values() if value < level]
    above = [value for value in levels.values() if value > level]
    return (max(below) if below else None, min(above) if above else None)


# -------------------------------------------------------------- classification


def candle_type(ohlc_df: pd.DataFrame) -> Optional[str]:
    """Type of the last candle. If several match, the later of these wins:
    doji, maru_bozu_green, maru_bozu_red, shooting_star, hammer."""
    candle = None
    if bool(doji(ohlc_df)["doji"].iloc[-1]):
        candle = "doji"
    marubozu = maru_bozu(ohlc_df)["maru_bozu"].iloc[-1]
    if marubozu == "maru_bozu_green":
        candle = "maru_bozu_green"
    if marubozu == "maru_bozu_red":
        candle = "maru_bozu_red"
    if bool(shooting_star(ohlc_df)["sstar"].iloc[-1]):
        candle = "shooting_star"
    if bool(hammer(ohlc_df)["hammer"].iloc[-1]):
        candle = "hammer"
    return candle


@dataclass(frozen=True)
class CandlePattern:
    pattern: Optional[str]  # e.g. "hammer_bullish", or None
    significance: str  # "HIGH" near a pivot support/resistance level, otherwise "low"

    def __str__(self) -> str:
        return f"Significance - {self.significance}, Pattern - {self.pattern}"


def candle_pattern(ohlc_df: pd.DataFrame, ohlc_day: pd.DataFrame) -> CandlePattern:
    """Identify the pattern formed by the last candle.

    Args:
        ohlc_df: intraday (or any) bars, oldest first.
        ohlc_day: daily bars whose last row is the previous session, used for pivot levels.
    """
    typical = _typical_body(ohlc_df)
    support, resistance = support_resistance(ohlc_df, ohlc_day)
    close = ohlc_df["close"].iloc[-1]

    significance = "low"
    for level in (support, resistance):
        if level is not None and (level - 1.5 * typical) < close < (level + 1.5 * typical):
            significance = "HIGH"

    kind = candle_type(ohlc_df)
    prior_trend = trend(ohlc_df.iloc[:-1, :], 7)
    last, prev = ohlc_df.iloc[-1], ohlc_df.iloc[-2]

    pattern = None
    if kind == "doji" and last["close"] > prev["close"] and last["close"] > last["open"]:
        pattern = "doji_bullish"
    if kind == "doji" and last["close"] < prev["close"] and last["close"] < last["open"]:
        pattern = "doji_bearish"
    if kind == "maru_bozu_green":
        pattern = "maru_bozu_bullish"
    if kind == "maru_bozu_red":
        pattern = "maru_bozu_bearish"
    if prior_trend == "uptrend" and kind == "hammer":
        pattern = "hanging_man_bearish"
    if prior_trend == "downtrend" and kind == "hammer":
        pattern = "hammer_bullish"
    if prior_trend == "uptrend" and kind == "shooting_star":
        pattern = "shooting_star_bearish"
    if prior_trend == "uptrend" and kind == "doji" and last["high"] < prev["close"] and last["low"] > prev["open"]:
        pattern = "harami_cross_bearish"
    if prior_trend == "downtrend" and kind == "doji" and last["high"] < prev["open"] and last["low"] > prev["close"]:
        pattern = "harami_cross_bullish"
    if prior_trend == "uptrend" and kind != "doji" and last["open"] > prev["high"] and last["close"] < prev["low"]:
        pattern = "engulfing_bearish"
    if prior_trend == "downtrend" and kind != "doji" and last["close"] > prev["high"] and last["open"] < prev["low"]:
        pattern = "engulfing_bullish"

    return CandlePattern(pattern=pattern, significance=significance)
