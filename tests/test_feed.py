import datetime as dt

import pandas as pd
import pytest

from kite_bot.feed import BarBuilder, KiteTickerFeed, ReplayFeed, Tick
from kite_bot.ohlc import naive_ist, resample_ohlc, to_ist_naive
from kite_bot.sample_data import TRADE_DAY, make_sample_market


def tick(hms, price, symbol="ABC", day="2025-03-04", volume=1):
    return Tick(symbol, price, dt.datetime.fromisoformat(f"{day} {hms}"), volume)


# ---------------------------------------------------------------- BarBuilder

def test_bar_is_handed_over_when_the_next_bar_opens():
    builder = BarBuilder(2)
    assert builder.update(tick("09:15:10", 100.0)) is None
    assert builder.update(tick("09:15:40", 103.0)) is None
    assert builder.update(tick("09:16:30", 99.0)) is None
    done = builder.update(tick("09:17:05", 101.0))
    assert (done.start, done.open, done.high, done.low, done.close, done.volume) == (
        dt.datetime(2025, 3, 4, 9, 15), 100.0, 103.0, 99.0, 99.0, 3)
    last = builder.flush()
    assert (last.start, last.open, last.close) == (dt.datetime(2025, 3, 4, 9, 17), 101.0, 101.0)
    assert builder.flush() is None


def test_ticks_outside_the_session_or_out_of_order_are_ignored():
    builder = BarBuilder(2)
    assert builder.update(tick("09:14:59", 50.0)) is None      # before the open
    assert builder.update(tick("15:30:00", 50.0)) is None      # at or after the close
    builder.update(tick("09:17:10", 100.0))
    assert builder.update(tick("09:15:30", 1.0)) is None       # older than the current bar
    assert builder.flush().low == 100.0


def test_bars_skip_over_gaps_without_inventing_bars():
    builder = BarBuilder(2)
    builder.update(tick("09:15:00", 100.0))
    done = builder.update(tick("09:31:00", 105.0))  # nothing traded for a while
    assert done.start == dt.datetime(2025, 3, 4, 9, 15)
    assert builder.flush().start == dt.datetime(2025, 3, 4, 9, 31)


# ---------------------------------------------------------------- ReplayFeed

def test_replay_ticks_are_chronological_and_hit_every_ohlc_point():
    _, minute = make_sample_market()
    feed = ReplayFeed(minute, TRADE_DAY)
    ticks = feed.ticks(["ALPHA", "BRAVO"])
    assert [t.timestamp for t in ticks] == sorted(t.timestamp for t in ticks)
    assert {t.symbol for t in ticks} == {"ALPHA", "BRAVO"}

    frame = minute["ALPHA"]
    row = frame[frame.index.date == TRADE_DAY].iloc[10]
    in_minute = [t.price for t in ticks if t.symbol == "ALPHA" and t.timestamp.replace(second=0, microsecond=0) == row.name.to_pydatetime()]
    assert in_minute[0] == pytest.approx(row["open"]) and in_minute[-1] == pytest.approx(row["close"])
    assert max(in_minute) == pytest.approx(row["high"]) and min(in_minute) == pytest.approx(row["low"])


def test_bars_rebuilt_from_replayed_ticks_equal_the_historical_resample():
    _, minute = make_sample_market()
    builder, rebuilt = BarBuilder(2), []
    for t in ReplayFeed(minute, TRADE_DAY).ticks(["ALPHA"]):
        bar = builder.update(t)
        if bar:
            rebuilt.append(bar)
    rebuilt.append(builder.flush())

    frame = naive_ist(minute["ALPHA"])
    expected = resample_ohlc(frame[frame.index.date == TRADE_DAY])
    got = pd.DataFrame([vars(b) for b in rebuilt]).set_index("start")
    assert len(got) == len(expected)
    pd.testing.assert_frame_equal(got[["open", "high", "low", "close"]], expected[["open", "high", "low", "close"]], check_freq=False, check_names=False, atol=1e-9)


def test_replay_start_delivers_everything_then_finishes():
    _, minute = make_sample_market()
    received = []
    feed = ReplayFeed(minute, TRADE_DAY)
    assert feed.finished is False
    feed.start(["ALPHA"], received.append)
    assert feed.finished and len(received) == len(feed.ticks(["ALPHA"])) > 0


def test_replay_accepts_timezone_aware_history():
    _, minute = make_sample_market()
    aware = minute["ALPHA"].copy()
    aware.index = aware.index.tz_localize("Asia/Kolkata")
    assert [t.timestamp for t in ReplayFeed({"ALPHA": aware}, TRADE_DAY).ticks(["ALPHA"])][:2] == \
           [t.timestamp for t in ReplayFeed(minute, TRADE_DAY).ticks(["ALPHA"])][:2]


# ------------------------------------------------------------- KiteTickerFeed

class FakeTicker:
    MODE_FULL = "full"

    def __init__(self, api_key, access_token):
        self.credentials = (api_key, access_token)
        self.subscribed, self.modes, self.closed, self.connect_args = None, None, False, None

    def connect(self, threaded=False):
        self.connect_args = threaded

    def subscribe(self, tokens):
        self.subscribed = tokens

    def set_mode(self, mode, tokens):
        self.modes = (mode, tokens)

    def close(self):
        self.closed = True


def make_feed(**kwargs):
    holder, status = {}, []

    def factory(api_key, access_token):
        holder["ticker"] = FakeTicker(api_key, access_token)
        return holder["ticker"]

    feed = KiteTickerFeed("key", "token", {111: "ALPHA", 222: "BRAVO", 333: "OTHER"}, on_status=status.append, ticker_factory=factory, **kwargs)
    return feed, holder, status


def test_kite_feed_connects_threaded_and_subscribes_only_wanted_symbols():
    feed, holder, status = make_feed()
    feed.start(["ALPHA", "BRAVO"], lambda t: None)
    ticker = holder["ticker"]
    assert ticker.credentials == ("key", "token") and ticker.connect_args is True
    ticker.on_connect(ticker, {})
    assert sorted(ticker.subscribed) == [111, 222]
    assert ticker.modes == ("full", ticker.subscribed)
    assert "subscribed to 2" in status[-1]


def test_kite_feed_converts_ticks_and_drops_unknown_instruments():
    feed, holder, _ = make_feed()
    received = []
    feed.start(["ALPHA"], received.append)
    ticker = holder["ticker"]
    stamp = dt.datetime(2025, 3, 4, 9, 20, 5)  # naive: treated as machine-local time, like KiteTicker builds it
    ticker.on_ticks(ticker, [
        {"instrument_token": 111, "last_price": 101.5, "exchange_timestamp": stamp, "last_traded_quantity": 7},
        {"instrument_token": 222, "last_price": 5.0, "exchange_timestamp": stamp},  # not subscribed
        {"instrument_token": 999, "last_price": 5.0},                                # unknown
    ])
    assert len(received) == 1
    (t,) = received
    assert (t.symbol, t.price, t.volume) == ("ALPHA", 101.5, 7)
    assert t.timestamp == to_ist_naive(stamp)


def test_kite_feed_handles_aware_timestamps_and_missing_timestamps():
    fixed = dt.datetime(2025, 3, 4, 10, 0, 0)
    feed, holder, _ = make_feed(clock=lambda: fixed)
    received = []
    feed.start(["ALPHA"], received.append)
    ticker = holder["ticker"]
    aware = dt.datetime(2025, 3, 4, 3, 50, 0, tzinfo=dt.timezone.utc)  # 09:20 in India
    ticker.on_ticks(ticker, [
        {"instrument_token": 111, "last_price": 100.0, "exchange_timestamp": aware},
        {"instrument_token": 111, "last_price": 101.0},
    ])
    assert received[0].timestamp == dt.datetime(2025, 3, 4, 9, 20, 0)
    assert received[1].timestamp == fixed


def test_a_failing_consumer_does_not_break_the_stream():
    feed, holder, _ = make_feed()
    seen = []

    def flaky(tick):
        if tick.price == 1.0:
            raise RuntimeError("boom")
        seen.append(tick.price)

    feed.start(["ALPHA"], flaky)
    ticker = holder["ticker"]
    stamp = dt.datetime(2025, 3, 4, 9, 20)
    ticker.on_ticks(ticker, [{"instrument_token": 111, "last_price": p, "exchange_timestamp": stamp} for p in (1.0, 2.0)])
    assert seen == [2.0]


def test_kite_feed_reports_connection_events_and_stops():
    feed, holder, status = make_feed()
    feed.start(["ALPHA"], lambda t: None)
    ticker = holder["ticker"]
    ticker.on_close(ticker, 1006, "abnormal")
    ticker.on_reconnect(ticker, 3)
    ticker.on_noreconnect(ticker)
    assert status == ["closed: 1006 abnormal", "reconnecting (attempt 3)", "gave up reconnecting"]
    feed.stop()
    assert ticker.closed
