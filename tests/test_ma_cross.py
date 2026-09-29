import numpy as np
import pandas as pd
import pytest

from kite_bot.screener import find_setup, scan


def make_prices(segments, start=100.0):
    """Build a daily close series from (n_bars, end_price) linear segments."""
    values = [start]
    for n_bars, end_price in segments:
        values += list(np.linspace(values[-1], end_price, n_bars + 1)[1:])
    index = pd.bdate_range("2025-01-01", periods=len(values))
    return pd.DataFrame({"close": values}, index=index)


# Uptrend to 200, slide to 120, then sit at 120 while MA100 keeps falling.
DOWNTREND = [(150, 200), (45, 120), (60, 120)]


def test_short_setup_fresh_cross_on_last_bar():
    df = make_prices(DOWNTREND + [(1, 140)])
    setup = find_setup(df)
    assert setup is not None
    assert setup.side == "short"
    assert setup.cross_date == df.index[-1]
    assert setup.turn_date < setup.cross_date
    assert setup.close > setup.ma


def test_no_setup_before_price_crosses_ma():
    assert find_setup(make_prices(DOWNTREND)) is None


def test_stale_cross_is_rejected_by_default_but_allowed_with_max_age():
    df = make_prices(DOWNTREND + [(1, 140), (3, 140)])  # cross was 3 bars ago
    assert find_setup(df) is None
    setup = find_setup(df, max_age_bars=3)
    assert setup is not None
    assert setup.cross_date == df.index[-4]


def test_second_cross_does_not_count_as_first():
    df = make_prices(DOWNTREND + [(1, 140), (3, 120), (1, 140)])
    assert find_setup(df) is None


def test_no_setup_if_ma_slope_is_still_up():
    # sharp dip and recovery inside an uptrend: price crosses MA but MA never turned down
    df = make_prices([(150, 200), (3, 110), (1, 190)])
    assert find_setup(df) is None


def test_long_setup_is_the_mirror_image():
    df = make_prices([(150, 100), (60, 170), (1, 120)], start=200)
    setup = find_setup(df, side="long")
    assert setup is not None
    assert setup.side == "long"
    assert setup.cross_date == df.index[-1]
    assert setup.close < setup.ma
    assert find_setup(df, side="short") is None


def test_long_setup_needs_close_below_ma():
    df = make_prices([(150, 100), (60, 170), (1, 190)], start=200)
    assert find_setup(df, side="long") is None


def test_too_little_history_returns_none():
    assert find_setup(make_prices([(50, 120)])) is None


def test_invalid_side_and_missing_column_raise():
    df = make_prices(DOWNTREND + [(1, 140)])
    with pytest.raises(ValueError):
        find_setup(df, side="sideways")
    with pytest.raises(ValueError):
        find_setup(df.rename(columns={"close": "price"}))


def test_scan_returns_only_matching_symbols():
    universe = {
        "MATCH": make_prices(DOWNTREND + [(1, 140)]),
        "NOCROSS": make_prices(DOWNTREND),
        "SHORTDATA": make_prices([(50, 120)]),
    }
    results = scan(universe)
    assert [symbol for symbol, _ in results] == ["MATCH"]
