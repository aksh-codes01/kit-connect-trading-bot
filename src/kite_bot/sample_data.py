"""Synthetic market data for demos and tests (no broker account needed).

Three fake stocks, all sharing one trade day:
    ALPHA    passes the daily screen; the intraday crossover is followed by a fall -> take-profit.
    BRAVO    passes the daily screen; the crossover is followed by a rebound     -> stop-loss.
    CHARLIE  same intraday pattern as ALPHA, but its daily trend fails the screen -> never traded.

The prices are hand-shaped line segments, not a model of a real market.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

TRADE_DAY = dt.date(2025, 12, 26)
_LAST_SCREEN_DAY = dt.date(2025, 12, 25)  # last traded day before TRADE_DAY

# (minutes, end_price) pieces; each list adds up to one 375-minute session, starting from 140.
_MINUTE_PATHS = {
    "ALPHA": [(30, 142), (150, 122), (195, 108)],
    "BRAVO": [(30, 142), (150, 122), (60, 117), (135, 138)],
    "CHARLIE": [(30, 142), (150, 122), (195, 108)],
}


def _linear(segments, start):
    values = [float(start)]
    for n, end in segments:
        values += list(np.linspace(values[-1], end, n + 1)[1:])
    return np.array(values)


def _ohlc(closes, index, opens=None, pad=0.02):
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = np.concatenate(([closes[0]], closes[:-1]))
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + pad,
            "low": np.minimum(opens, closes) - pad,
            "close": closes,
            "volume": 1000,
        },
        index=index,
    )


def _daily(symbol: str) -> pd.DataFrame:
    if symbol == "CHARLIE":
        closes = _linear([(256, 250)], start=100)  # steady uptrend: MA never turns down
    else:
        # up to 200, slide to 120, flat, then a jump to 140 on the last day
        closes = _linear([(150, 200), (45, 120), (60, 120), (1, 140)], start=100)
    index = pd.bdate_range(end=_LAST_SCREEN_DAY.isoformat(), periods=len(closes))
    return _ohlc(closes, index, pad=0.5)


def _minute_session(day: dt.date, segments, start_price: float) -> pd.DataFrame:
    closes = _linear(segments, start_price)[1:]
    assert len(closes) == 375
    index = pd.date_range(pd.Timestamp(f"{day.isoformat()} 09:15"), periods=375, freq="1min")
    opens = np.concatenate(([start_price], closes[:-1]))
    return _ohlc(closes, index, opens=opens)


def _minute(symbol: str) -> pd.DataFrame:
    sessions = pd.bdate_range(end=_LAST_SCREEN_DAY.isoformat(), periods=6)
    frames = [_minute_session(d.date(), [(375, 120.0)], 120.0) for d in sessions[:-1]]
    frames.append(_minute_session(sessions[-1].date(), [(375, 140.0)], 120.0))  # ramp to 140
    frames.append(_minute_session(TRADE_DAY, _MINUTE_PATHS[symbol], 140.0))
    return pd.concat(frames)


def make_sample_market() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return (daily, minute) dicts keyed by symbol."""
    symbols = list(_MINUTE_PATHS)
    return {s: _daily(s) for s in symbols}, {s: _minute(s) for s in symbols}
