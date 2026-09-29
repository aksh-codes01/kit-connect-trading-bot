import numpy as np
import pandas as pd
import pytest

from kite_bot import indicators as ind


def det_ohlc(n=120):
    """Deterministic OHLC series (no randomness) used for the frozen reference numbers."""
    i = np.arange(n, dtype=float)
    close = 100 + 8 * np.sin(i / 7) + 0.08 * i + 3 * np.cos(i / 2.3)
    open_ = close + 1.2 * np.sin(i * 1.7)
    high = np.maximum(open_, close) + 0.5 + 0.4 * np.abs(np.sin(i * 0.9))
    low = np.minimum(open_, close) - 0.5 - 0.4 * np.abs(np.cos(i * 1.3))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def bars(closes, spread=1.0):
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": closes, "high": closes + spread, "low": closes - spread, "close": closes})


# ---- frozen reference values: produced by the ORIGINAL scripts' functions on det_ohlc() ----

def test_matches_original_scripts_on_reference_data():
    df = det_ohlc()
    assert ind.atr(df, 14).iloc[-1] == pytest.approx(2.5289939596739375)
    assert ind.rsi(df, 14).iloc[-1] == pytest.approx(27.610356353808058)
    adx = ind.adx(df, 14)
    assert adx.iloc[-1] == pytest.approx(30.877403036182976)
    assert adx.first_valid_index() == 27
    st = ind.supertrend(df, 10, 3)
    assert st.iloc[-1] == pytest.approx(110.15514826314386)
    assert int(st.isna().sum()) == 19
    macd = ind.macd(df, 12, 26, 9)
    assert macd["MACD"].iloc[-1] == pytest.approx(-1.615462013019112)
    assert macd["Signal"].iloc[-1] == pytest.approx(-0.649973446324481)
    bb = ind.bollinger_bands(df, 20)
    assert bb["BB_up"].iloc[-1] == pytest.approx(119.94875614380673)
    assert bb["BB_dn"].iloc[-1] == pytest.approx(99.44294193972739)
    assert ind.slope(df, 10) == pytest.approx(-37.39461766930129)
    assert tuple(ind.pivot_levels(df)) == pytest.approx((102.44, 103.55, 104.98, 106.09, 101.01, 99.9, 98.47))


def test_works_the_same_on_a_date_index():
    plain = det_ohlc()
    dated = plain.copy()
    dated.index = pd.bdate_range("2025-01-01", periods=len(dated))
    for func, args in ((ind.atr, (14,)), (ind.rsi, (14,)), (ind.adx, (14,)), (ind.supertrend, (10, 3))):
        np.testing.assert_allclose(func(plain, *args).to_numpy(), func(dated, *args).to_numpy(), equal_nan=True)


# ---- hand-checkable behaviour ----

def test_sma_and_ema_warm_up():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ind.sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert ind.sma(s, 3).iloc[:2].isna().all()
    assert ind.ema(s, 3).iloc[:2].isna().all() and not ind.ema(s, 3).iloc[2:].isna().any()
    assert ind.ema(pd.Series([7.0] * 10), 4).iloc[-1] == pytest.approx(7.0)


def test_true_range_takes_the_largest_of_three_measures():
    df = pd.DataFrame({"open": [10, 10], "high": [11, 16], "low": [9, 12], "close": [10, 15]})
    tr = ind.true_range(df)
    assert np.isnan(tr.iloc[0])  # no previous close on the first bar
    assert tr.iloc[1] == 6  # |high - prev close| = 6 beats high-low = 4


def test_atr_of_constant_range_is_that_range():
    assert ind.atr(bars([50] * 30, spread=1.0), 7).iloc[-1] == pytest.approx(2.0)


def test_bollinger_bands_known_values():
    out = ind.bollinger_bands(bars([1, 2, 3, 4, 5]), 5)
    row = out.iloc[0]
    assert row["MA"] == 3
    assert row["BB_up"] == pytest.approx(3 + 2 * np.sqrt(2))  # population std of 1..5 is sqrt(2)
    assert row["BB_width"] == pytest.approx(4 * np.sqrt(2))
    flat = ind.bollinger_bands(bars([9] * 10), 5)
    assert (flat["BB_width"] == 0).all()


def test_macd_columns_and_definition():
    out = ind.macd(det_ohlc(), 12, 26, 9)
    assert {"MA_Fast", "MA_Slow", "MACD", "Signal"} <= set(out.columns)
    np.testing.assert_allclose(out["MACD"], out["MA_Fast"] - out["MA_Slow"])
    assert not out[["MACD", "Signal"]].isna().any().any()


def test_rsi_extremes_and_bounds():
    rising = ind.rsi(bars(np.linspace(100, 150, 60)), 14)
    falling = ind.rsi(bars(np.linspace(150, 100, 60)), 14)
    assert rising.iloc[-1] == pytest.approx(100.0)
    assert falling.iloc[-1] == pytest.approx(0.0)
    mixed = ind.rsi(det_ohlc(), 14).dropna()
    assert ((mixed >= 0) & (mixed <= 100)).all()


def test_adx_is_high_in_a_trend_and_low_in_a_range():
    trend_bars = bars(np.linspace(100, 200, 120), spread=0.5)
    range_bars = bars(100 + 3 * np.sin(np.arange(120) / 2.0), spread=0.5)
    assert ind.adx(trend_bars, 14).iloc[-1] > 40
    assert ind.adx(range_bars, 14).iloc[-1] < 25
    valid = ind.adx(det_ohlc(), 14).dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_supertrend_sits_below_price_in_uptrend_above_in_downtrend():
    closes = np.concatenate([np.linspace(120, 90, 40), np.linspace(90, 140, 60)])  # down, then up
    df = bars(closes, spread=0.6)
    st = ind.supertrend(df, 7, 3)
    assert st.iloc[:7].isna().all()
    assert st.iloc[-1] < df["close"].iloc[-1]  # uptrend: line under price
    down = bars(np.concatenate([np.linspace(90, 120, 40), np.linspace(120, 80, 60)]), spread=0.6)
    st_down = ind.supertrend(down, 7, 3)
    assert st_down.iloc[-1] > down["close"].iloc[-1]  # downtrend: line above price


def test_supertrend_with_too_little_data_is_all_nan():
    assert ind.supertrend(bars([100, 101, 102]), 7, 3).isna().all()


def test_slope_angles():
    assert ind.slope(bars(np.linspace(100, 110, 20)), 10) == pytest.approx(45.0)
    assert ind.slope(bars(np.linspace(110, 100, 20)), 10) == pytest.approx(-45.0)
    assert ind.slope(bars([100.0] * 20), 10) == 0.0


def test_pivot_levels_known_values_and_unpacking():
    day = pd.DataFrame({"open": [110.0], "high": [140.0], "low": [80.0], "close": [110.0]})
    levels = ind.pivot_levels(day)
    assert tuple(levels) == (110.0, 140.0, 170.0, 200.0, 80.0, 50.0, 20.0)
    pivot, r1, r2, r3, s1, s2, s3 = levels  # tuple-compatible with the old scripts
    assert (levels.pivot, levels.r1, levels.s3) == (pivot, r1, s3)


def test_renko_bricks_and_brick_size():
    pytest.importorskip("stocktrends")
    df = bars(np.linspace(100, 130, 40), spread=0.0)
    df.index = pd.bdate_range("2025-01-01", periods=40)
    bricks = ind.renko(df, brick_size=10)
    assert bricks["close"].tolist() == [100.0, 110.0, 120.0, 130.0]  # first brick anchors at the start price
    assert (bricks["high"] - bricks["low"]).eq(10.0).all() and bricks["uptrend"].all()
    calm = bars([100.0] * 260, spread=0.2)  # ATR = 0.4 -> 1.5 * 0.4 rounds to 1 (the floor)
    wild = bars([100.0] * 260, spread=20.0)  # ATR = 40 -> capped at 10
    assert ind.renko_brick_size(calm) == 1.0
    assert ind.renko_brick_size(wild) == 10.0
