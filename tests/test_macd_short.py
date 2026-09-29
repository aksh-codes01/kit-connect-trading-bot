import datetime as dt

import numpy as np
import pandas as pd
import pytest

from kite_bot.strategy import (
    ShortEntry,
    bearish_cross,
    find_short_entry,
    simulate_short,
)

DAY1 = dt.date(2025, 3, 3)
DAY2 = dt.date(2025, 3, 4)


def session_index(day, n_bars=188):
    start = pd.Timestamp(day.isoformat() + " 09:15")
    return pd.date_range(start, periods=n_bars, freq="2min")


def bars_from_closes(day, closes):
    """2-minute bars whose open is the previous close."""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 0.05,
            "low": np.minimum(opens, closes) - 0.05,
            "close": closes,
        },
        index=session_index(day, len(closes)),
    )


def rise_then_fall(n_up, n_down, n_flat, start=100.0, step=0.2):
    up = start + step * np.arange(n_up)
    down = up[-1] - step * 2 * np.arange(1, n_down + 1)
    flat = np.full(n_flat, down[-1])
    return np.concatenate([up, down, flat])


def manual_bars(rows, start="2025-03-04 10:00"):
    """rows: list of (open, high, low, close), 2-minute spacing."""
    idx = pd.date_range(pd.Timestamp(start), periods=len(rows), freq="2min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


# ---------- crossover ----------

def test_bearish_cross_fires_once_after_reversal():
    closes = rise_then_fall(60, 60, 0)
    signal = bearish_cross(pd.Series(closes), fast=3, slow=8)
    fired = np.flatnonzero(signal.to_numpy())
    assert len(fired) == 1
    assert fired[0] > 59  # only after the peak


def test_no_cross_in_steady_trends_or_before_warmup():
    up = pd.Series(np.linspace(100, 120, 80))
    down = pd.Series(np.linspace(120, 100, 80))
    assert not bearish_cross(up, 3, 8).any()
    assert not bearish_cross(down, 3, 8).any()
    assert not bearish_cross(pd.Series([1.0, 2.0, 1.0, 0.5]), 3, 8).any()  # shorter than slow


# ---------- entry ----------

def make_two_days(day1_closes, day2_closes):
    return pd.concat([bars_from_closes(DAY1, day1_closes), bars_from_closes(DAY2, day2_closes)])


def reversal_day():
    """Day 1 drifts up to 110; day 2 continues from 110, rises, then reverses mid-session."""
    return make_two_days(np.linspace(100, 110, 188), rise_then_fall(40, 40, 108, start=110))


def test_entry_is_next_bar_open_on_trade_date():
    bars = reversal_day()
    entry = find_short_entry(bars, DAY2, fast=3, slow=8)
    assert entry is not None
    assert entry.signal_time.date() == DAY2
    assert entry.signal_time.time() > dt.time(10, 0)  # a real mid-session reversal, not a gap at the open
    assert entry.time == entry.signal_time + pd.Timedelta("2min")
    assert entry.price == bars.loc[entry.time, "open"]


def test_crossover_on_earlier_day_is_ignored():
    day1 = rise_then_fall(40, 40, 108)  # crosses on day 1
    day2 = np.linspace(day1[-1], day1[-1] + 3, 188)  # gentle rise, no cross
    bars = make_two_days(day1, day2)
    assert find_short_entry(bars, DAY1, fast=3, slow=8) is not None
    assert find_short_entry(bars, DAY2, fast=3, slow=8) is None


def test_signal_after_last_entry_time_is_rejected():
    bars = reversal_day()
    assert find_short_entry(bars, DAY2, fast=3, slow=8) is not None
    assert find_short_entry(bars, DAY2, fast=3, slow=8, last_entry=dt.time(9, 30)) is None


# ---------- trade management ----------

def entry_at(price, ts="2025-03-04 10:00"):
    t = pd.Timestamp(ts)
    return ShortEntry(signal_time=t - pd.Timedelta("2min"), time=t, price=price)


def test_take_profit_at_two_percent():
    bars = manual_bars([(100, 100.5, 99.5, 100), (100, 100.4, 97.0, 97.5)])
    trade = simulate_short(bars, entry_at(100))
    assert trade.reason == "take_profit"
    assert trade.exit_price == pytest.approx(98.0)
    assert trade.pnl_pct == pytest.approx(0.02)


def test_stop_loss_at_two_percent():
    bars = manual_bars([(100, 100.5, 99.5, 100), (100, 103.0, 99.8, 102.5)])
    trade = simulate_short(bars, entry_at(100))
    assert trade.reason == "stop_loss"
    assert trade.exit_price == pytest.approx(102.0)
    assert trade.pnl_pct == pytest.approx(-0.02)


def test_bar_touching_both_levels_counts_as_stop_loss():
    bars = manual_bars([(100, 103.0, 97.0, 100)])
    assert simulate_short(bars, entry_at(100)).reason == "stop_loss"


def test_gap_through_stop_fills_at_open():
    bars = manual_bars([(100, 100.2, 99.9, 100), (104.0, 104.5, 103.5, 104.2)])
    trade = simulate_short(bars, entry_at(100))
    assert trade.reason == "stop_loss"
    assert trade.exit_price == pytest.approx(104.0)


def test_square_off_closes_at_that_bars_open():
    bars = manual_bars(
        [(100, 100.3, 99.7, 100), (100.1, 100.4, 99.8, 100.2), (100.2, 100.5, 99.9, 100.4)],
        start="2025-03-04 15:11",
    )
    trade = simulate_short(bars, entry_at(100, "2025-03-04 15:11"), square_off=dt.time(15, 15))
    assert trade.reason == "square_off"
    assert trade.exit_time == pd.Timestamp("2025-03-04 15:15")
    assert trade.exit_price == pytest.approx(100.2)


def test_data_ending_before_any_exit_returns_end_of_data():
    bars = manual_bars([(100, 100.3, 99.7, 100), (100, 100.2, 99.8, 99.9)])
    trade = simulate_short(bars, entry_at(100))
    assert trade.reason == "end_of_data"
    assert trade.exit_price == pytest.approx(99.9)
