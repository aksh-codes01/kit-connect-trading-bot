import datetime as dt
import threading

import numpy as np
import pandas as pd
import pytest

from kite_bot.backtest import run_backtest
from kite_bot.broker import CANCELLED, COMPLETE, REJECTED, TRIGGER_PENDING, OrderStatus, PaperBroker, Side
from kite_bot.config import LiveConfig
from kite_bot.engine import LiveEngine
from kite_bot.feed import ReplayFeed, Tick
from kite_bot.sample_data import TRADE_DAY, make_sample_market
from kite_bot.store import Store

PREV = dt.date(2025, 3, 3)
DAY = dt.date(2025, 3, 4)

# Small EMAs keep the price paths short and readable; capital 10,000 buys ~90 shares.
CFG = LiveConfig(fast=3, slow=8, capital_per_trade=10_000, fill_poll_s=0.01, fill_timeout_s=0.05)


# ---------------------------------------------------------------- data helpers

def minute_frame(day, closes, start="09:15", first_open=None):
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate(([closes[0] if first_open is None else first_open], closes[:-1]))
    index = pd.date_range(pd.Timestamp(f"{day.isoformat()} {start}"), periods=len(closes), freq="1min")
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) + 0.01, "low": np.minimum(opens, closes) - 0.01,
                         "close": closes, "volume": 1000}, index=index)


def warmup_frame():
    return minute_frame(PREV, np.linspace(100, 110, 375))  # a quiet uptrend the day before


def today_path(rise=30, fall=100, tail=245, low=100.0):
    """110 -> 112 -> `low`, then flat: EMA(3) drops through EMA(8) on the way down."""
    up = np.linspace(110, 112, rise)
    down = np.linspace(112, low, fall)
    return np.concatenate([up, down, np.full(tail, low)])


def rebound_path():
    """A shallow dip (just enough for the crossover), then a hard rally: stop-loss territory."""
    return np.concatenate([np.linspace(110, 112, 30), np.linspace(112, 110.9, 12), np.linspace(110.9, 125, 60), np.full(273, 125.0)])


def flat_after_cross_path():
    """A shallow dip for the crossover, then flat: neither target nor stop is touched, so only the clock ends the trade."""
    return np.concatenate([np.linspace(110, 112, 30), np.linspace(112, 110.9, 12), np.full(333, 110.9)])


def make_engine(broker=None, store=None, config=CFG, symbols=("ABC",), warm=True):
    broker = broker or PaperBroker()
    store = store or Store()
    engine = LiveEngine(broker, store, DAY, config, sleep=lambda s: None)
    engine.symbols = set(symbols)
    if warm:
        for symbol in symbols:
            engine.load_warmup(symbol, warmup_frame())
    return engine, broker, store


def run_day(engine, closes_by_symbol):
    frames = {s: minute_frame(DAY, c, first_open=110.0) for s, c in closes_by_symbol.items()}
    ReplayFeed(frames, DAY).start(list(frames), engine.on_tick)
    return frames


def only_trade(store):
    (trade,) = store.trades(DAY)
    return trade


# ---------------------------------------------------------------------- exits

def test_ticks_and_clock_checks_from_different_threads_do_not_conflict():
    """In production the WebSocket thread delivers ticks while the main thread calls check_time."""
    engine, broker, store = make_engine()
    ticks = ReplayFeed({"ABC": minute_frame(DAY, today_path(), first_open=110.0)}, DAY).ticks(["ABC"])
    finished = threading.Event()

    def pump():
        for tick in ticks:
            engine.on_tick(tick)
        finished.set()

    worker = threading.Thread(target=pump)
    worker.start()
    while not finished.is_set():
        engine.check_time(dt.datetime(2025, 3, 4, 9, 30))  # before square-off, so it must change nothing
    worker.join()
    assert only_trade(store).exit_reason == "take_profit" and broker.positions() == {}
    assert store.events("ERROR").empty


def test_entry_then_take_profit_leaves_a_clean_audit_trail():
    engine, broker, store = make_engine()
    run_day(engine, {"ABC": today_path()})

    trade = only_trade(store)
    assert (trade.symbol, trade.status, trade.exit_reason) == ("ABC", "closed", "take_profit")
    assert trade.pnl > 0 and trade.pnl == pytest.approx((trade.entry_price - trade.exit_price) * trade.quantity)
    assert trade.target_price == pytest.approx(trade.entry_price * 0.98) and trade.stop_price == pytest.approx(trade.entry_price * 1.02)
    assert dt.datetime.fromisoformat(trade.entry_time).second == 0  # the first tick of a bar: its open

    assert broker.positions() == {}
    orders = store.orders()
    assert orders["purpose"].tolist() == ["entry", "stop", "exit"]
    assert orders["side"].tolist() == ["SELL", "BUY", "BUY"]
    assert broker.order_status(trade.stop_order_id).status == CANCELLED  # the resting stop was cancelled, not left behind
    actions = store.signals()["action"].tolist()
    assert actions == ["entered", "exited: take_profit"]
    assert store.events("ERROR").empty


def test_stop_loss_fires_once_and_never_buys_twice():
    engine, broker, store = make_engine()
    run_day(engine, {"ABC": rebound_path()})

    trade = only_trade(store)
    assert trade.exit_reason == "stop_loss" and trade.pnl < 0
    assert broker.positions() == {}  # flat, not long: the stop and the cover did not both buy
    assert store.orders()["purpose"].tolist() == ["entry", "stop"]  # no separate cover order was needed
    assert broker.order_status(trade.stop_order_id).status == COMPLETE


def test_time_exit_at_square_off_even_without_price_moves():
    engine, broker, store = make_engine()
    run_day(engine, {"ABC": flat_after_cross_path()})
    trade = only_trade(store)
    assert trade.exit_reason == "square_off"
    assert dt.datetime.fromisoformat(trade.exit_time).time() >= dt.time(15, 15)
    assert broker.positions() == {}


def test_check_time_closes_positions_when_the_stock_stops_ticking():
    engine, broker, store = make_engine()
    closes = flat_after_cross_path()[:110]  # stops well before 15:15
    frames = {"ABC": minute_frame(DAY, closes, first_open=110.0)}
    ReplayFeed(frames, DAY).start(["ABC"], engine.on_tick)
    assert store.open_trades(DAY) and broker.positions() != {}
    engine.check_time(dt.datetime(2025, 3, 4, 15, 14, 59))
    assert store.open_trades(DAY)
    engine.check_time(dt.datetime(2025, 3, 4, 15, 15, 1))
    assert only_trade(store).exit_reason == "square_off" and broker.positions() == {}


def test_end_of_day_flattens_and_summarises():
    engine, broker, store = make_engine()
    frames = {"ABC": minute_frame(DAY, flat_after_cross_path()[:110], first_open=110.0)}
    ReplayFeed(frames, DAY).start(["ABC"], engine.on_tick)
    summary = engine.end_of_day()
    assert broker.positions() == {} and summary["open"] == 0 and summary["trades"] == 1
    assert only_trade(store).exit_reason == "end_of_day"
    assert not store.bars("ABC").empty  # bars were saved


# ---------------------------------------------------------------- entry rules

def test_unlisted_symbols_are_ignored():
    engine, broker, store = make_engine()
    engine.on_tick(Tick("OTHER", 100.0, dt.datetime(2025, 3, 4, 10, 0)))
    assert engine.now is None and store.trades() == []


def test_one_trade_per_stock_per_day():
    twice = np.concatenate([today_path(tail=0, low=104.0), np.linspace(104, 113, 40), np.linspace(113, 100, 100), np.full(110, 100.0)])
    engine, broker, store = make_engine()
    run_day(engine, {"ABC": twice})
    assert len(store.trades(DAY)) == 1
    reasons = store.signals()["detail"].tolist()
    assert any("already traded" in r for r in reasons if r)


def test_max_trades_per_day_skips_later_signals():
    engine, broker, store = make_engine(config=LiveConfig(**{**CFG.__dict__, "max_trades_per_day": 1}), symbols=("ABC", "XYZ"))
    run_day(engine, {"ABC": today_path(), "XYZ": today_path()})
    assert len(store.trades(DAY)) == 1
    skipped = store.signals()
    assert "max trades per day reached" in skipped[skipped["action"] == "skipped"]["detail"].tolist()


def test_no_entry_after_the_last_entry_time():
    late = LiveConfig(**{**CFG.__dict__, "last_entry": dt.time(9, 30)})
    engine, broker, store = make_engine(config=late)
    run_day(engine, {"ABC": today_path()})
    assert store.trades(DAY) == []
    assert "after the last entry time" in store.signals()["detail"].tolist()


def test_capital_too_small_for_one_share_is_skipped():
    tiny = LiveConfig(**{**CFG.__dict__, "capital_per_trade": 50})
    engine, broker, store = make_engine(config=tiny)
    run_day(engine, {"ABC": today_path()})
    assert store.trades(DAY) == [] and broker.orders == []


def test_too_little_history_is_flagged_and_produces_no_signal():
    engine, broker, store = make_engine(warm=False)
    engine.load_warmup("ABC", minute_frame(PREV, np.linspace(100, 110, 6)))  # 3 bars, needs 8
    assert "warm-up bars" in store.events("WARNING").loc[0, "message"]
    run_day(engine, {"ABC": today_path(rise=2, fall=6, tail=0)})
    assert store.trades(DAY) == []


def test_warmup_leaves_out_bars_still_being_built_from_live_ticks():
    engine, *_ = make_engine(warm=False)
    frame = pd.concat([minute_frame(PREV, np.linspace(100, 110, 375)), minute_frame(DAY, np.linspace(110, 111, 5))])
    now = dt.datetime(2025, 3, 4, 9, 19, 30)  # the 09:19 bar is still open
    loaded = engine.load_warmup("ABC", frame, now=now)
    yesterday = 188
    assert loaded == yesterday + 2  # today's 09:15 and 09:17 bars only


# ------------------------------------------------------------------ kill switch

def test_halt_blocks_entries_and_survives_a_restart():
    engine, broker, store = make_engine()
    engine.halt("manual stop")
    run_day(engine, {"ABC": today_path()})
    assert store.trades(DAY) == []
    assert any("halted: manual stop" in (d or "") for d in store.signals()["detail"])

    restarted, _, _ = make_engine(store=store, broker=broker)
    assert restarted.halted == "manual stop"
    restarted.resume()
    assert make_engine(store=store, broker=broker)[0].halted is None


def test_daily_loss_limit_halts_and_flattens_an_open_position():
    limit = LiveConfig(**{**CFG.__dict__, "max_daily_loss": 100.0, "stop_loss_pct": 0.5})  # wide stop: the loss limit fires first
    engine, broker, store = make_engine(config=limit)
    run_day(engine, {"ABC": rebound_path()})
    trade = only_trade(store)
    assert trade.exit_reason == "loss_limit" and trade.pnl < 0
    assert engine.halted and "loss limit" in engine.halted
    assert broker.positions() == {}


# --------------------------------------------------------- failures at the broker

class RejectingBroker(PaperBroker):
    def place_order(self, symbol, side, quantity, price):
        order = super().place_order(symbol, side, quantity, price) if side == Side.BUY else self._rejected(symbol, side, quantity, price)
        return order

    def _rejected(self, symbol, side, quantity, price):
        from kite_bot.broker import Order
        order = Order(self._new_id(), symbol, side, quantity, price)
        self.orders.append(order)
        self._status[order.order_id] = OrderStatus(order.order_id, REJECTED, 0, 0.0, "MIS shorting not allowed")
        return order


def test_a_rejected_entry_creates_no_trade_and_is_not_retried():
    engine, broker, store = make_engine(broker=RejectingBroker())
    run_day(engine, {"ABC": today_path()})
    assert store.trades(DAY) == [] and broker.positions() == {}
    detail = store.signals()["detail"].tolist()
    assert any("REJECTED" in d and "shorting" in d for d in detail)
    assert len(broker.orders) == 1  # tried once, then the stock is done for the day


class NoStopBroker(PaperBroker):
    def place_stop_loss(self, *args, **kwargs):
        raise RuntimeError("exchange refused the stop order")


def test_a_position_that_cannot_be_protected_is_closed_immediately():
    engine, broker, store = make_engine(broker=NoStopBroker())
    run_day(engine, {"ABC": today_path()})
    trade = only_trade(store)
    assert trade.exit_reason == "no_stop_protection"
    assert broker.positions() == {}
    assert "could not place the stop-loss" in store.events("CRITICAL").loc[0, "message"]


class StuckStopBroker(PaperBroker):
    """The stop cannot be cancelled and still reports TRIGGER PENDING."""

    def cancel_order(self, order_id):
        return False if order_id in self._stops else super().cancel_order(order_id)


def test_if_the_stop_cannot_be_cancelled_it_does_not_cover_and_halts():
    engine, broker, store = make_engine(broker=StuckStopBroker())
    run_day(engine, {"ABC": today_path()})
    assert broker.positions() != {}  # still short, deliberately
    purposes = store.orders()["purpose"].tolist()
    assert "exit" not in purposes  # covering now could leave a second buy waiting in the stop
    assert engine.halted and "could not be cancelled" in engine.halted
    assert "double buy" in store.events("CRITICAL").loc[0, "message"]


class UnfillableCoverBroker(PaperBroker):
    def place_order(self, symbol, side, quantity, price):
        if side == Side.SELL:
            return super().place_order(symbol, side, quantity, price)
        raise RuntimeError("exchange is down")


def test_if_a_position_cannot_be_closed_the_engine_halts_and_restores_the_stop():
    engine, broker, store = make_engine(broker=UnfillableCoverBroker())
    run_day(engine, {"ABC": today_path()})
    assert broker.positions() != {}
    assert engine.halted and "could not close" in engine.halted
    assert "Close it manually" in store.events("CRITICAL").loc[0, "message"]
    trade = store.open_trades(DAY)[0]
    assert broker.order_status(trade.stop_order_id).status == TRIGGER_PENDING  # protection put back


def test_a_bug_while_handling_a_tick_is_logged_not_raised():
    engine, broker, store = make_engine()

    def boom(tick):
        raise ValueError("bug")

    engine._handle_tick = boom
    engine.on_tick(Tick("ABC", 100.0, dt.datetime(2025, 3, 4, 10, 0)))
    assert "ValueError: bug" in store.events("ERROR").loc[0, "message"]


# ----------------------------------------------------------------- crash recovery

def open_a_trade():
    engine, broker, store = make_engine()
    frames = {"ABC": minute_frame(DAY, flat_after_cross_path()[:110], first_open=110.0)}
    ReplayFeed(frames, DAY).start(["ABC"], engine.on_tick)
    assert len(store.open_trades(DAY)) == 1
    return engine, broker, store, frames


def test_restart_resumes_a_live_position_and_finishes_the_trade():
    engine, broker, store, _ = open_a_trade()
    fresh = LiveEngine(broker, store, DAY, CFG, sleep=lambda s: None)
    fresh.symbols = {"ABC"}
    notes = fresh.recover()
    assert any("resumed short" in n for n in notes)
    fresh.on_tick(Tick("ABC", 100.0, dt.datetime(2025, 3, 4, 11, 30)))  # far below the target
    assert only_trade(store).exit_reason == "take_profit" and broker.positions() == {}


def test_restart_finds_that_the_stop_fired_while_the_bot_was_down():
    engine, broker, store, _ = open_a_trade()
    broker.on_price("ABC", 130.0)  # the exchange stop executes with nobody watching
    fresh = LiveEngine(broker, store, DAY, CFG, sleep=lambda s: None)
    notes = fresh.recover()
    trade = only_trade(store)
    assert trade.status == "closed" and trade.exit_reason == "stop_loss" and trade.pnl < 0
    assert any("while the bot was down" in n for n in notes)


def test_restart_when_the_position_disappeared_without_a_known_exit():
    engine, broker, store, _ = open_a_trade()
    trade = store.open_trades(DAY)[0]
    broker.cancel_order(trade.stop_order_id)
    broker.place_order("ABC", Side.BUY, trade.quantity, 108.0)  # closed by hand
    fresh = LiveEngine(broker, store, DAY, CFG, sleep=lambda s: None)
    fresh.recover()
    closed = only_trade(store)
    assert closed.exit_reason == "closed_externally" and closed.pnl is None


def test_restart_replaces_a_missing_stop():
    engine, broker, store, _ = open_a_trade()
    trade = store.open_trades(DAY)[0]
    broker.cancel_order(trade.stop_order_id)
    fresh = LiveEngine(broker, store, DAY, CFG, sleep=lambda s: None)
    notes = fresh.recover()
    new_id = store.open_trades(DAY)[0].stop_order_id
    assert new_id != trade.stop_order_id and broker.order_status(new_id).status == TRIGGER_PENDING
    assert any("placed a new one" in n for n in notes)


def test_restart_with_a_position_mismatch_halts_and_touches_nothing():
    engine, broker, store, _ = open_a_trade()
    trade = store.open_trades(DAY)[0]
    broker.place_order("ABC", Side.BUY, 5, 108.0)  # someone covered part of it
    before = list(broker.orders)
    fresh = LiveEngine(broker, store, DAY, CFG, sleep=lambda s: None)
    fresh.recover()
    assert fresh.halted and "mismatch" in fresh.halted
    assert broker.orders == before and store.open_trades(DAY)[0].id == trade.id


def test_restart_with_an_unknown_position_halts():
    broker = PaperBroker()
    broker.place_order("MANUAL", Side.SELL, 10, 100.0)  # not ours
    engine = LiveEngine(broker, Store(), DAY, CFG, sleep=lambda s: None)
    notes = engine.recover()
    assert engine.halted and "unexpected position" in engine.halted
    assert any("not managed" in n for n in notes)


def test_restart_does_not_trade_the_same_stock_twice():
    engine, broker, store, _ = open_a_trade()
    broker.on_price("ABC", 130.0)
    fresh = LiveEngine(broker, store, DAY, CFG, sleep=lambda s: None)
    fresh.symbols = {"ABC"}
    fresh.recover()
    assert "ABC" in fresh._traded and fresh._trade_count == 1


# ---------------------------------------------------------- screening + equivalence

def test_screen_saves_the_shortlist_using_only_earlier_days():
    daily, _ = make_sample_market()
    engine = LiveEngine(PaperBroker(), Store(), TRADE_DAY, sleep=lambda s: None)
    assert engine.screen(daily) == ["ALPHA", "BRAVO"]  # CHARLIE fails the screen
    assert engine.store.shortlist(TRADE_DAY) == ["ALPHA", "BRAVO"]
    # data from the trade day itself must not leak into the screen
    extra = daily["CHARLIE"].copy()
    assert LiveEngine(PaperBroker(), Store(), TRADE_DAY, sleep=lambda s: None).screen({"CHARLIE": extra}) == []


def test_live_engine_reproduces_the_backtest_on_the_sample_market():
    daily, minute = make_sample_market()
    expected = {r.symbol: r for r in run_backtest(daily, minute).records}

    store, broker = Store(), PaperBroker()
    engine = LiveEngine(broker, store, TRADE_DAY, sleep=lambda s: None)
    for symbol in engine.screen(daily):
        engine.load_warmup(symbol, minute[symbol][minute[symbol].index.date < TRADE_DAY])
    ReplayFeed(minute, TRADE_DAY).start(sorted(engine.symbols), engine.on_tick)
    engine.end_of_day()

    trades = {t.symbol: t for t in store.trades(TRADE_DAY)}
    assert set(trades) == set(expected) == {"ALPHA", "BRAVO"}
    for symbol, live in trades.items():
        ref = expected[symbol]
        assert live.exit_reason == ref.trade.reason
        assert pd.Timestamp(live.entry_time) == ref.trade.entry_time
        assert live.entry_price == pytest.approx(ref.trade.entry_price)
        assert live.quantity == ref.quantity
        assert live.exit_price == pytest.approx(ref.trade.exit_price, abs=0.1)  # ticks cross the level a hair past it
    assert store.events("ERROR").empty and broker.positions() == {}
