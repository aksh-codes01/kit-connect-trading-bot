import datetime as dt
import threading

import pandas as pd
import pytest

from kite_bot.broker import Order, Side
from kite_bot.feed import Bar, Tick
from kite_bot.screener import Setup
from kite_bot.store import Store

DAY = dt.date(2025, 12, 26)
T0 = dt.datetime(2025, 12, 26, 12, 55)


def setup_for(close=140.0):
    return Setup("short", pd.Timestamp("2025-09-10"), pd.Timestamp("2025-12-25"), close, 133.4)


def test_shortlist_round_trip_and_per_day_isolation():
    store = Store()
    store.save_shortlist(DAY, [("BRAVO", setup_for()), ("ALPHA", setup_for())])
    assert store.shortlist(DAY) == ["ALPHA", "BRAVO"]
    assert store.shortlist(dt.date(2025, 12, 27)) == []
    store.save_shortlist(DAY, [("ALPHA", setup_for(141.0))])  # re-running the screen replaces, not duplicates
    assert store.shortlist(DAY) == ["ALPHA", "BRAVO"]


def test_trade_lifecycle_and_pnl():
    store = Store()
    trade_id = store.open_trade(DAY, "ALPHA", 840, T0, 118.98, 116.6, 121.36, "stop-1")
    (row,) = store.open_trades(DAY)
    assert (row.id, row.symbol, row.quantity, row.status, row.stop_order_id) == (trade_id, "ALPHA", 840, "open", "stop-1")
    assert store.realized_pnl(DAY) == 0.0

    store.close_trade(trade_id, T0 + dt.timedelta(minutes=34), 116.6, "take_profit", 1999.2)
    assert store.open_trades(DAY) == []
    (closed,) = store.trades(DAY)
    assert (closed.status, closed.exit_reason, closed.exit_price, closed.pnl) == ("closed", "take_profit", 116.6, 1999.2)
    assert store.realized_pnl(DAY) == pytest.approx(1999.2)

    second = store.open_trade(DAY, "BRAVO", 842, T0, 118.67, 116.3, 121.04, None)
    store.close_trade(second, T0, 121.04, "stop_loss", -1998.3)
    assert store.realized_pnl(DAY) == pytest.approx(0.9)
    assert store.realized_pnl(dt.date(2025, 12, 27)) == 0.0


def test_stop_order_can_be_replaced_and_cleared():
    store = Store()
    trade_id = store.open_trade(DAY, "ALPHA", 1, T0, 100.0, 98.0, 102.0, "a")
    store.set_stop_order(trade_id, "b")
    assert store.open_trades(DAY)[0].stop_order_id == "b"
    store.set_stop_order(trade_id, None)
    assert store.open_trades(DAY)[0].stop_order_id is None


def test_orders_signals_bars_events_and_kv():
    store = Store()
    store.log_order(Order("o-1", "ALPHA", Side.SELL, 10, 118.9), "entry", T0, trade_id=1)
    orders = store.orders()
    assert orders.loc[0, ["order_id", "side", "purpose", "trade_id"]].tolist() == ["o-1", "SELL", "entry", 1]

    store.log_signal(T0, "ALPHA", "entered", 118.9)
    store.log_signal(T0, "BRAVO", "skipped", detail="max trades reached")
    assert store.signals("BRAVO").loc[0, "detail"] == "max trades reached"
    assert len(store.signals()) == 2

    store.save_bar("ALPHA", Bar(T0, 1.0, 2.0, 0.5, 1.5, 100))
    store.save_bar("ALPHA", Bar(T0, 1.0, 2.5, 0.5, 1.6, 120))  # same bar again overwrites
    bars = store.bars("ALPHA")
    assert len(bars) == 1 and bars.iloc[0]["high"] == 2.5
    assert store.bars("NONE").empty

    store.log_event("WARNING", "stop order missing", T0)
    store.log_event("INFO", "started")
    assert store.events("WARNING")["message"].tolist() == ["stop order missing"]

    assert store.get("halted") is None and store.get("halted", "no") == "no"
    store.set("halted", "loss limit")
    assert store.get("halted") == "loss limit"
    store.set("halted", None)
    assert store.get("halted") is None


def test_ticks_can_be_stored_and_read_back_since_a_time():
    store = Store()
    ticks = [Tick("ALPHA", 100.0 + i, T0 + dt.timedelta(seconds=i), 5) for i in range(4)]
    store.save_ticks(ticks)
    assert len(store.ticks("ALPHA")) == 4
    assert store.ticks("ALPHA", since=T0 + dt.timedelta(seconds=2))["price"].tolist() == [102.0, 103.0]
    assert store.ticks("OTHER").empty


def test_data_survives_reopening_a_file_database(tmp_path):
    path = str(tmp_path / "bot.db")
    first = Store(path)
    first.open_trade(DAY, "ALPHA", 10, T0, 100.0, 98.0, 102.0, "s")
    first.set("halted", "manual")
    first.close()

    second = Store(path)
    assert [t.symbol for t in second.open_trades(DAY)] == ["ALPHA"]
    assert second.get("halted") == "manual"


def test_concurrent_writers_do_not_lose_rows():
    store = Store()

    def writer(name):
        for i in range(50):
            store.log_event("INFO", f"{name}-{i}")

    threads = [threading.Thread(target=writer, args=(n,)) for n in "abcd"]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(store.events()) == 200
