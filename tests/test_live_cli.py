import datetime as dt

import pandas as pd
import pytest

from kite_bot.broker import PaperBroker
from kite_bot.cli import main
from kite_bot.config import LiveConfig
from kite_bot.engine import LiveEngine
from kite_bot.feed import ReplayFeed, TickFeed
from kite_bot.live import FrameMarketData, KiteMarketData, MarketData, load_universe, run_session, status_handler
from kite_bot.ohlc import write_market
from kite_bot.sample_data import TRADE_DAY, make_sample_market
from kite_bot.store import Store


def new_engine(store=None, config=LiveConfig()):
    return LiveEngine(PaperBroker(), store or Store(), TRADE_DAY, config, sleep=lambda s: None)


# ------------------------------------------------------------- universe file

def test_load_universe_ignores_comments_blanks_and_repeats(tmp_path):
    path = tmp_path / "universe.txt"
    path.write_text("# my list\ninfy\nTCS   # large cap\n\n  sbin \nINFY\n")
    assert load_universe(path) == ["INFY", "TCS", "SBIN"]


# ---------------------------------------------------------------- run_session

def test_replayed_session_screens_trades_and_summarises():
    daily, minute = make_sample_market()
    store, notes = Store(), []
    engine = new_engine(store)
    summary = run_session(engine, ReplayFeed(minute, TRADE_DAY), FrameMarketData(daily, minute), sorted(daily),
                          sleep=lambda s: None, notify=notes.append)
    assert summary["shortlist"] == ["ALPHA", "BRAVO"] and summary["trades"] == 2
    assert summary["wins"] == 1 and summary["open"] == 0 and summary["halted"] is None
    assert any("shortlist (2): ALPHA, BRAVO" in n for n in notes)
    assert {t.exit_reason for t in store.trades(TRADE_DAY)} == {"take_profit", "stop_loss"}


class FailingData(MarketData):
    """Daily data fails for one stock, warm-up data fails for another."""

    def __init__(self, inner):
        self.inner = inner

    def daily(self, symbol):
        if symbol == "CHARLIE":
            raise RuntimeError("no data")
        return self.inner.daily(symbol)

    def minute_history(self, symbol):
        if symbol == "BRAVO":
            raise RuntimeError("history unavailable")
        return self.inner.minute_history(symbol)


def test_one_bad_symbol_does_not_stop_the_session():
    daily, minute = make_sample_market()
    store, notes = Store(), []
    engine = new_engine(store)
    summary = run_session(engine, ReplayFeed(minute, TRADE_DAY), FailingData(FrameMarketData(daily, minute)), sorted(daily),
                          sleep=lambda s: None, notify=notes.append)
    assert summary["shortlist"] == ["ALPHA"] and [t.symbol for t in store.trades(TRADE_DAY)] == ["ALPHA"]
    warnings = store.events("WARNING")["message"].tolist()
    assert any("CHARLIE" in w and "no data" in w for w in warnings)
    assert any("BRAVO" in w and "warm-up" in w for w in warnings)
    assert any("dropping BRAVO" in n for n in notes)


def test_an_empty_shortlist_ends_quietly_without_starting_the_feed():
    daily, minute = make_sample_market()
    feed = ReplayFeed(minute, TRADE_DAY)
    summary = run_session(new_engine(), feed, FrameMarketData(daily, minute), ["CHARLIE"], sleep=lambda s: None, notify=lambda m: None)
    assert summary["trades"] == 0 and summary["shortlist"] == [] and feed.finished is False


class ScriptedFeed(TickFeed):
    """A live-style feed: delivers ticks a little at a time while the session loop sleeps."""

    realtime = True

    def __init__(self, ticks):
        self._ticks, self.stopped, self._on_tick = sorted(ticks, key=lambda t: t.timestamp), False, None

    def start(self, symbols, on_tick):
        self._on_tick = on_tick

    def deliver_until(self, moment):
        while self._ticks and self._ticks[0].timestamp <= moment:
            self._on_tick(self._ticks.pop(0))

    def stop(self):
        self.stopped = True


def test_when_ticks_stop_the_clock_still_closes_open_positions():
    daily, minute = make_sample_market()
    cutoff = dt.datetime.combine(TRADE_DAY, dt.time(13, 0))
    ticks = [t for t in ReplayFeed(minute, TRADE_DAY).ticks(["ALPHA", "BRAVO"]) if t.timestamp < cutoff]  # the feed dies at 13:00
    feed, store = ScriptedFeed(ticks), Store()
    clock = {"now": dt.datetime.combine(TRADE_DAY, dt.time(9, 0))}

    def sleep(_):
        clock["now"] += dt.timedelta(minutes=5)
        feed.deliver_until(clock["now"])

    engine = new_engine(store)
    summary = run_session(engine, feed, FrameMarketData(daily, minute), sorted(daily), clock=lambda: clock["now"], sleep=sleep, notify=lambda m: None)

    assert feed.stopped and summary["open"] == 0 and summary["trades"] == 2
    exits = {t.symbol: (t.exit_reason, dt.datetime.fromisoformat(t.exit_time).time()) for t in store.trades(TRADE_DAY)}
    assert all(reason == "square_off" and moment >= dt.time(15, 15) for reason, moment in exits.values())
    assert engine.broker.positions() == {}


def test_losing_the_data_connection_for_good_halts_new_trades():
    engine = new_engine()
    messages = []
    handle = status_handler(engine, notify=messages.append)
    handle("reconnecting (attempt 2)")
    assert engine.halted is None
    handle("gave up reconnecting")
    assert engine.halted == "market data connection lost"
    assert messages == ["feed: reconnecting (attempt 2)", "feed: gave up reconnecting"]


# -------------------------------------------------------------- KiteMarketData

class FakeKite:
    def __init__(self):
        self.calls = []

    def historical_data(self, token, start, end, interval):
        self.calls.append((token, interval))
        return [{"date": dt.datetime.combine(start, dt.time(9, 15)), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1}]


def test_kite_market_data_pauses_between_requests_to_respect_the_rate_limit():
    instruments = pd.DataFrame({"tradingsymbol": ["INFY", "TCS"], "instrument_token": [408065, 2953217]})
    kite, pauses = FakeKite(), []
    data = KiteMarketData(kite, instruments, daily_days=300, minute_days=12, pause_s=0.4, sleep=pauses.append)
    data.daily("INFY")
    data.minute_history("TCS")
    assert kite.calls == [(408065, "day"), (2953217, "minute")]
    assert pauses.count(0.4) == 2  # one pause after each stock


# ------------------------------------------------------------------------ CLI

@pytest.fixture()
def sample_dir(tmp_path):
    daily, minute = make_sample_market()
    write_market(daily, minute, tmp_path / "data")
    return str(tmp_path / "data")


def test_replay_command_logs_to_sqlite_and_report_reads_it_back(sample_dir, tmp_path, capsys):
    db = str(tmp_path / "bot.db")
    assert main(["replay", "--data-dir", sample_dir, "--db", db]) == 0
    out = capsys.readouterr().out
    assert "take_profit" in out and "stop_loss" in out and "trades 2" in out

    store = Store(db)
    assert store.shortlist(TRADE_DAY) == ["ALPHA", "BRAVO"]
    assert len(store.trades(TRADE_DAY)) == 2 and not store.bars("ALPHA").empty

    main(["report", "--db", db])
    report = capsys.readouterr().out
    assert f"Trades on {TRADE_DAY}" in report and "ALPHA" in report and "realized P&L" in report


def test_replay_respects_the_daily_loss_limit(sample_dir, capsys):
    # wide stop so the loss limit fires first; it applies to net day P&L (ALPHA's profit offsets BRAVO's loss for a while)
    main(["replay", "--data-dir", sample_dir, "--max-daily-loss", "500", "--stop-loss", "0.5"])
    out = capsys.readouterr().out
    assert "loss_limit" in out and "HALTED: daily loss limit of 500 reached" in out


def test_report_on_an_empty_database(tmp_path, capsys):
    main(["report", "--db", str(tmp_path / "empty.db")])
    assert "No trades" in capsys.readouterr().out


def test_live_mode_refuses_to_start_without_explicit_confirmation(tmp_path):
    universe = tmp_path / "u.txt"
    universe.write_text("INFY\n")
    with pytest.raises(SystemExit, match="--yes-real-money"):
        main(["live", "--universe", str(universe), "--mode", "live"])


def test_real_money_needs_an_explicit_capital_and_loss_limit(tmp_path):
    universe = tmp_path / "u.txt"
    universe.write_text("INFY\n")
    base = ["live", "--universe", str(universe), "--mode", "live", "--yes-real-money"]
    with pytest.raises(SystemExit, match="--capital"):
        main(base)
    with pytest.raises(SystemExit, match="--max-daily-loss"):
        main(base + ["--capital", "20000"])


def test_demo_shows_the_live_engine_matching_the_backtest(capsys):
    main(["demo"])
    out = capsys.readouterr().out
    assert "Step 3" in out and "IDENTICAL" in out
