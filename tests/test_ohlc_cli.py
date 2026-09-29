import datetime as dt

import pandas as pd
import pytest

from kite_bot.cli import main
from kite_bot.ohlc import (
    fetch_history,
    fetch_ohlc,
    instrument_token,
    instrument_tokens,
    load_bars,
    load_directory,
    load_instruments,
    resample_ohlc,
    symbol_for_token,
    ticks_to_ohlc,
    write_market,
)
from kite_bot.sample_data import make_sample_market


class FakeHistoryKite:
    def __init__(self):
        self.windows = []
        self.calls = []

    def historical_data(self, token, start, end, interval):
        self.windows.append((start, end))
        self.calls.append((token, interval))
        stamp = dt.datetime.combine(start, dt.time(9, 15))
        return [{"date": stamp, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}]


def test_fetch_history_splits_long_ranges_into_contiguous_windows():
    kite, pauses = FakeHistoryKite(), []
    start, end = dt.date(2025, 1, 1), dt.date(2025, 5, 10)  # 130 days
    df = fetch_history(kite, 99, start, end, "minute", sleep=pauses.append)

    assert len(kite.windows) == 3
    assert kite.windows[0][0] == start and kite.windows[-1][1] == end
    for (_, prev_end), (next_start, _) in zip(kite.windows, kite.windows[1:]):
        assert next_start == prev_end + dt.timedelta(days=1)
    assert all((e - s).days + 1 <= 55 for s, e in kite.windows)
    assert len(pauses) == 2  # a pause between calls, none after the last
    assert len(df) == 3 and df.index.is_monotonic_increasing


def test_fetch_history_single_window_and_bad_interval():
    kite = FakeHistoryKite()
    fetch_history(kite, 1, dt.date(2025, 1, 1), dt.date(2025, 1, 10), "minute", sleep=lambda s: None)
    assert len(kite.windows) == 1
    with pytest.raises(ValueError):
        fetch_history(kite, 1, dt.date(2025, 1, 1), dt.date(2025, 1, 10), "week")


# ---------- reshaping ----------

def test_resample_1min_to_2min_aggregates_ohlcv():
    idx = pd.date_range("2025-03-04 09:15", periods=4, freq="1min")
    one_min = pd.DataFrame(
        {
            "open": [10, 11, 12, 13],
            "high": [11, 13, 12.5, 15],
            "low": [9, 10.5, 11, 12],
            "close": [11, 12, 13, 14],
            "volume": [100, 200, 300, 400],
        },
        index=idx,
    )
    out = resample_ohlc(one_min)
    assert list(out.index) == [pd.Timestamp("2025-03-04 09:15"), pd.Timestamp("2025-03-04 09:17")]
    first = out.iloc[0]
    assert (first.open, first.high, first.low, first.close, first.volume) == (10, 13, 9, 12, 300)
    second = out.iloc[1]
    assert (second.open, second.high, second.low, second.close, second.volume) == (12, 15, 11, 14, 700)


def test_ticks_to_ohlc_builds_bars_from_prices():
    idx = pd.to_datetime(["2025-03-04 09:15:10", "2025-03-04 09:16:00", "2025-03-04 09:19:50", "2025-03-04 09:20:05"])
    bars = ticks_to_ohlc(pd.Series([100.0, 103.0, 99.0, 101.0], index=idx), "5min")
    assert bars.loc["2025-03-04 09:15"].tolist() == [100.0, 103.0, 99.0, 99.0]
    assert bars.loc["2025-03-04 09:20"].tolist() == [101.0, 101.0, 101.0, 101.0]


# ---------- instruments and fetch_ohlc ----------

INSTRUMENTS = pd.DataFrame(
    {"tradingsymbol": ["INFY", "TCS", "SBIN"], "instrument_token": [408065, 2953217, 779521]}
)


def test_load_instruments_wraps_the_kite_dump():
    class K:
        def instruments(self, exchange):
            assert exchange == "NSE"
            return [{"tradingsymbol": "INFY", "instrument_token": 408065}]

    assert instrument_token(load_instruments(K()), "INFY") == 408065


def test_instrument_lookups():
    assert instrument_token(INSTRUMENTS, "TCS") == 2953217
    assert instrument_tokens(INSTRUMENTS, ["SBIN", "INFY"]) == [779521, 408065]
    assert symbol_for_token(INSTRUMENTS, 408065) == "INFY"
    with pytest.raises(KeyError):
        instrument_token(INSTRUMENTS, "NOPE")
    with pytest.raises(KeyError):
        symbol_for_token(INSTRUMENTS, 1)


def test_fetch_ohlc_recent_days_uses_the_symbols_token():
    kite = FakeHistoryKite()
    end = dt.date(2025, 6, 30)
    df = fetch_ohlc(kite, INSTRUMENTS, "INFY", "day", days=300, end=end)
    assert kite.calls == [(408065, "day")]
    assert kite.windows == [(end - dt.timedelta(days=300), end)]
    assert len(df) == 1


def test_fetch_ohlc_long_history_from_a_start_date_is_chunked():
    kite = FakeHistoryKite()
    fetch_ohlc(kite, INSTRUMENTS, "SBIN", "5minute", start=dt.date(2025, 1, 1), end=dt.date(2025, 12, 31), sleep=lambda s: None)
    assert len(kite.windows) > 3  # 364 days of 5-minute data needs several requests
    assert all((e - s).days + 1 <= 95 for s, e in kite.windows)
    assert kite.windows[0][0] == dt.date(2025, 1, 1) and kite.windows[-1][1] == dt.date(2025, 12, 31)


def test_fetch_ohlc_needs_exactly_one_of_days_or_start():
    kite = FakeHistoryKite()
    with pytest.raises(ValueError):
        fetch_ohlc(kite, INSTRUMENTS, "INFY", "day")
    with pytest.raises(ValueError):
        fetch_ohlc(kite, INSTRUMENTS, "INFY", "day", days=5, start=dt.date(2025, 1, 1))
    assert kite.calls == []  # nothing was requested


def test_csv_round_trip(tmp_path):
    daily, minute = make_sample_market()
    write_market(daily, minute, tmp_path)
    daily_back, minute_back = load_directory(tmp_path)
    assert set(daily_back) == set(daily) and set(minute_back) == set(minute)
    pd.testing.assert_frame_equal(daily_back["ALPHA"], daily["ALPHA"], check_freq=False, check_dtype=False, check_names=False)
    pd.testing.assert_frame_equal(minute_back["BRAVO"], minute["BRAVO"], check_freq=False, check_dtype=False, check_names=False)


def test_load_bars_requires_ohlc_columns(tmp_path):
    path = tmp_path / "X_day.csv"
    path.write_text("date,open,close\n2025-01-01,1,2\n")
    with pytest.raises(ValueError):
        load_bars(path)


def test_empty_folder_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path)


def test_cli_demo_shows_both_outcomes(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "ALPHA" in out and "BRAVO" in out and "CHARLIE" not in out
    assert "take_profit" in out and "stop_loss" in out


def test_cli_screen_and_backtest_on_csv_files(tmp_path, capsys):
    daily, minute = make_sample_market()
    write_market(daily, minute, tmp_path)

    main(["screen", "--data-dir", str(tmp_path)])
    screen_out = capsys.readouterr().out
    assert "ALPHA" in screen_out and "BRAVO" in screen_out and "CHARLIE" not in screen_out

    trades_csv = tmp_path / "trades.csv"
    main(["backtest", "--data-dir", str(tmp_path), "--trades-csv", str(trades_csv)])
    backtest_out = capsys.readouterr().out
    assert "trades 2" in backtest_out
    assert len(pd.read_csv(trades_csv)) == 2
