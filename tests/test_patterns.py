import pandas as pd
import pytest

from kite_bot import patterns as pat


def uptrend(n=10):
    """Green candles, each with a higher low than the one before."""
    rows = []
    for i in range(n):
        o = 100 + 2 * i
        rows.append((o, o + 1.3, o - 0.3, o + 1.0))
    return rows


def downtrend(n=10):
    """Red candles, each with a lower high than the one before."""
    rows = []
    for i in range(n):
        o = 120 - 2 * i
        rows.append((o, o + 0.3, o - 1.3, o - 1.0))
    return rows


def frame(context, last):
    return pd.DataFrame(context + [last], columns=["open", "high", "low", "close"])


# Daily bar whose pivot levels are 20, 50, 80, 110 (pivot), 140, 170, 200.
DAY = pd.DataFrame({"open": [110.0], "high": [140.0], "low": [80.0], "close": [110.0]})

# (name, context, (open, high, low, close)). "no_pattern" is an ordinary candle.
CASES = [
    ("hanging_man_bearish", uptrend(), (119.0, 119.7, 116.0, 119.6)),
    ("shooting_star_bearish", uptrend(), (119.6, 122.5, 118.9, 119.0)),
    ("doji_bullish", uptrend(), (119.2, 120.0, 118.6, 119.24)),
    ("doji_bearish", uptrend(), (118.8, 119.3, 118.4, 118.76)),
    ("harami_cross_bearish", uptrend(), (118.5, 118.9, 118.2, 118.53)),
    ("maru_bozu_bullish", uptrend(), (119.5, 122.0, 119.5, 122.0)),
    ("maru_bozu_bearish", uptrend(), (119.2, 119.2, 116.7, 116.7)),
    ("engulfing_bearish", uptrend(), (120.0, 120.6, 116.5, 117.0)),
    ("hammer_bullish", downtrend(), (100.8, 101.5, 98.0, 101.4)),
    ("harami_cross_bullish", downtrend(), (101.5, 101.8, 101.2, 101.52)),
    ("engulfing_bullish", downtrend(), (100.0, 104.2, 99.4, 103.5)),
    ("no_pattern", uptrend(), (119.0, 120.2, 118.7, 119.6)),
]


@pytest.mark.parametrize("name,context,last", CASES, ids=[c[0] for c in CASES])
def test_candle_pattern_labels(name, context, last):
    result = pat.candle_pattern(frame(context, last), DAY)
    assert result.pattern == (None if name == "no_pattern" else name)


# ---- single-candle detectors ----

def test_single_candle_detectors_flag_only_the_right_rows():
    df = frame(uptrend(), (119.0, 119.7, 116.0, 119.6))  # last row is a hammer shape
    assert pat.hammer(df)["hammer"].tolist() == [False] * 10 + [True]
    assert not pat.shooting_star(df)["sstar"].iloc[-1]
    assert not pat.doji(df)["doji"].iloc[-1]

    star = frame(uptrend(), (119.6, 122.5, 118.9, 119.0))
    assert pat.shooting_star(star)["sstar"].iloc[-1]

    tiny = frame(uptrend(), (119.2, 120.0, 118.6, 119.24))
    assert pat.doji(tiny)["doji"].iloc[-1] and not pat.doji(tiny)["doji"].iloc[:-1].any()


def test_maru_bozu_column_values():
    green = pat.maru_bozu(frame(uptrend(), (119.5, 122.0, 119.5, 122.0)))["maru_bozu"]
    red = pat.maru_bozu(frame(uptrend(), (119.2, 119.2, 116.7, 116.7)))["maru_bozu"]
    assert green.iloc[-1] == "maru_bozu_green" and not green.iloc[0]
    assert red.iloc[-1] == "maru_bozu_red"


def test_detectors_do_not_modify_their_input():
    df = frame(uptrend(), (119.0, 119.7, 116.0, 119.6))
    before = df.copy()
    for func in (pat.doji, pat.hammer, pat.shooting_star, pat.maru_bozu):
        func(df)
    pd.testing.assert_frame_equal(df, before)


# ---- market structure ----

def test_trend_detection():
    assert pat.trend(frame(uptrend(), (119.0, 120.0, 118.9, 119.9)), 7) == "uptrend"
    assert pat.trend(frame(downtrend(), (101.0, 101.3, 99.4, 99.6)), 7) == "downtrend"
    assert pat.trend(frame(uptrend(), (119.0, 120.0, 118.9, 119.0)), 7) is None  # open == close


def test_support_resistance_picks_nearest_levels_either_side():
    df = frame(uptrend(), (119.0, 120.2, 118.7, 119.6))
    support, resistance = pat.support_resistance(df, DAY)
    assert (support, resistance) == (110.0, 140.0)


def test_support_resistance_when_price_is_beyond_every_level():
    far_below = pd.DataFrame({"open": [10.0], "high": [12.0], "low": [8.0], "close": [10.0]})
    df = frame(uptrend(), (119.0, 120.2, 118.7, 119.6))
    support, resistance = pat.support_resistance(df, far_below)
    assert resistance is None and support is not None
    # the original scanner raised ValueError here; the port carries on
    assert pat.candle_pattern(df, far_below).significance == "low"


def test_significance_is_high_only_near_a_pivot_level():
    near_pivot = frame(uptrend(), (108.8, 109.4, 108.0, 108.9))  # ~1 point from the 110 pivot
    assert pat.candle_pattern(near_pivot, DAY).significance == "HIGH"
    far = frame(uptrend(), (119.0, 120.2, 118.7, 119.6))
    assert pat.candle_pattern(far, DAY).significance == "low"


def test_candle_pattern_string_keeps_the_old_output_format():
    df = frame(uptrend(), (119.0, 119.7, 116.0, 119.6))
    assert str(pat.candle_pattern(df, DAY)) == "Significance - low, Pattern - hanging_man_bearish"
    assert pat.candle_type(df) == "hammer"
