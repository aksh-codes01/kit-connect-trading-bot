import pytest

from kite_bot.broker import CANCELLED, COMPLETE, TRIGGER_PENDING, KiteBroker, PaperBroker, Side


def test_short_then_cover_realizes_profit():
    broker = PaperBroker()
    broker.place_order("ABC", Side.SELL, 10, 100.0)
    assert broker.positions() == {"ABC": -10}
    broker.place_order("ABC", Side.BUY, 10, 98.0)
    assert broker.positions() == {}
    assert broker.realized_pnl == pytest.approx(20.0)


def test_short_covered_higher_is_a_loss():
    broker = PaperBroker()
    broker.place_order("ABC", Side.SELL, 5, 100.0)
    broker.place_order("ABC", Side.BUY, 5, 103.0)
    assert broker.realized_pnl == pytest.approx(-15.0)


def test_long_round_trip_and_partial_close():
    broker = PaperBroker()
    broker.place_order("ABC", Side.BUY, 10, 50.0)
    broker.place_order("ABC", Side.SELL, 4, 55.0)
    assert broker.positions() == {"ABC": 6}
    assert broker.realized_pnl == pytest.approx(20.0)


def test_average_price_when_adding_to_a_position():
    broker = PaperBroker()
    broker.place_order("ABC", Side.SELL, 10, 100.0)
    broker.place_order("ABC", Side.SELL, 10, 110.0)  # average short price 105
    broker.place_order("ABC", Side.BUY, 20, 100.0)
    assert broker.realized_pnl == pytest.approx(100.0)


def test_flipping_from_long_to_short():
    broker = PaperBroker()
    broker.place_order("ABC", Side.BUY, 10, 100.0)
    broker.place_order("ABC", Side.SELL, 15, 110.0)  # closes 10 (+100), opens 5 short at 110
    assert broker.positions() == {"ABC": -5}
    assert broker.realized_pnl == pytest.approx(100.0)
    broker.place_order("ABC", Side.BUY, 5, 108.0)
    assert broker.realized_pnl == pytest.approx(110.0)


def test_commission_and_slippage_reduce_profit():
    broker = PaperBroker(commission_pct=0.001, slippage_pct=0.001)
    sell = broker.place_order("ABC", Side.SELL, 100, 100.0)
    buy = broker.place_order("ABC", Side.BUY, 100, 98.0)
    assert sell.price == pytest.approx(99.9)   # sells fill lower
    assert buy.price == pytest.approx(98.098)  # buys fill higher
    gross = (99.9 - 98.098) * 100
    fees = 0.001 * (99.9 + 98.098) * 100
    assert broker.realized_pnl == pytest.approx(gross - fees)


def test_invalid_quantity_rejected():
    with pytest.raises(ValueError):
        PaperBroker().place_order("ABC", Side.BUY, 0, 100.0)


# ---- paper broker: order states and resting stops ----

def test_market_orders_report_complete_with_fill_details():
    broker = PaperBroker(slippage_pct=0.001)
    order = broker.place_order("ABC", Side.SELL, 10, 100.0)
    status = broker.order_status(order.order_id)
    assert (status.status, status.filled_quantity, status.is_final) == (COMPLETE, 10, True)
    assert status.average_price == pytest.approx(99.9)


def test_stop_order_rests_until_price_reaches_the_trigger():
    broker = PaperBroker()
    broker.place_order("ABC", Side.SELL, 10, 100.0)
    stop = broker.place_stop_loss("ABC", Side.BUY, 10, 102.0)
    assert broker.order_status(stop.order_id).status == TRIGGER_PENDING
    assert broker.on_price("ABC", 101.5) == []  # not reached
    assert broker.positions() == {"ABC": -10}

    (filled,) = broker.on_price("ABC", 102.3)  # gapped through the trigger: fills at the market price
    assert filled.price == pytest.approx(102.3)
    assert broker.positions() == {}
    assert broker.realized_pnl == pytest.approx(-23.0)
    assert broker.order_status(stop.order_id).status == COMPLETE
    assert broker.on_price("ABC", 110.0) == []  # a triggered stop does not fire twice


def test_sell_stop_triggers_on_a_fall_and_other_symbols_are_untouched():
    broker = PaperBroker()
    broker.place_order("ABC", Side.BUY, 5, 100.0)
    broker.place_order("XYZ", Side.SELL, 5, 50.0)
    broker.place_stop_loss("ABC", Side.SELL, 5, 98.0)
    assert broker.on_price("XYZ", 10.0) == []
    assert len(broker.on_price("ABC", 97.9)) == 1
    assert broker.positions() == {"XYZ": -5}


def test_cancel_stop_order():
    broker = PaperBroker()
    stop = broker.place_stop_loss("ABC", Side.BUY, 10, 102.0)
    assert broker.cancel_order(stop.order_id) is True
    assert broker.order_status(stop.order_id).status == CANCELLED
    assert broker.on_price("ABC", 105.0) == []
    assert broker.cancel_order(stop.order_id) is False  # already gone
    assert broker.cancel_order("nope") is False


def test_cancel_fails_once_the_stop_has_triggered():
    broker = PaperBroker()
    stop = broker.place_stop_loss("ABC", Side.BUY, 10, 102.0)
    broker.on_price("ABC", 103.0)
    assert broker.cancel_order(stop.order_id) is False
    assert broker.order_status(stop.order_id).status == COMPLETE


def test_unknown_order_id_is_an_error():
    with pytest.raises(KeyError):
        PaperBroker().order_status("missing")


# ---- Kite broker, against a fake client ----

class FakeKite:
    EXCHANGE_NSE = "NSE"
    TRANSACTION_TYPE_BUY = "BUY"
    TRANSACTION_TYPE_SELL = "SELL"
    VARIETY_REGULAR = "regular"
    PRODUCT_MIS = "MIS"
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_SLM = "SL-M"

    def __init__(self):
        self.calls = []
        self.cancelled = []
        self.fail_cancel = False
        self.history = {}

    def place_order(self, **kwargs):
        self.calls.append(kwargs)
        return 12345

    def cancel_order(self, variety, order_id):
        if self.fail_cancel:
            raise RuntimeError("order already executed")
        self.cancelled.append((variety, order_id))

    def order_history(self, order_id):
        return self.history[order_id]

    def positions(self):
        return {"net": [
            {"tradingsymbol": "ABC", "quantity": -10, "product": "MIS", "exchange": "NSE"},
            {"tradingsymbol": "XYZ", "quantity": 0, "product": "MIS", "exchange": "NSE"},
            {"tradingsymbol": "NIFTYFUT", "quantity": 50, "product": "NRML", "exchange": "NFO"},
            {"tradingsymbol": "HOLD", "quantity": 5, "product": "CNC", "exchange": "NSE"},
        ]}


def test_kite_broker_market_order_carries_market_protection_and_tag():
    kite = FakeKite()
    order = KiteBroker(kite).place_order("ABC", Side.SELL, 7, 101.5)
    assert kite.calls == [
        {
            "variety": "regular",
            "exchange": "NSE",
            "tradingsymbol": "ABC",
            "transaction_type": "SELL",
            "quantity": 7,
            "product": "MIS",
            "order_type": "MARKET",
            "market_protection": -1,  # the API rejects market orders without it
            "tag": "kitebot",
        }
    ]
    assert (order.order_id, order.symbol, order.side, order.quantity) == ("12345", "ABC", Side.SELL, 7)


def test_kite_broker_market_protection_is_configurable():
    kite = FakeKite()
    KiteBroker(kite, market_protection=1.5).place_order("ABC", Side.BUY, 1, 100.0)
    assert kite.calls[0]["market_protection"] == 1.5


def test_kite_broker_stop_loss_is_slm_with_a_tick_aligned_trigger():
    kite = FakeKite()
    order = KiteBroker(kite).place_stop_loss("ABC", Side.BUY, 7, 102.0311)
    call = kite.calls[0]
    assert (call["order_type"], call["transaction_type"], call["product"]) == ("SL-M", "BUY", "MIS")
    assert call["trigger_price"] == 102.05  # rounded to the 0.05 tick
    assert "price" not in call
    assert order.price == 102.05


def test_kite_broker_per_symbol_tick_size():
    kite = FakeKite()
    KiteBroker(kite, tick_sizes={"ABC": 0.1}).place_stop_loss("ABC", Side.BUY, 1, 102.04)
    assert kite.calls[0]["trigger_price"] == 102.0


def test_kite_broker_cancel_reports_failure_instead_of_raising():
    kite = FakeKite()
    broker = KiteBroker(kite)
    assert broker.cancel_order("1") is True and kite.cancelled == [("regular", "1")]
    kite.fail_cancel = True
    assert broker.cancel_order("1") is False


def test_kite_broker_order_status_reads_the_latest_history_entry():
    kite = FakeKite()
    kite.history["9"] = [
        {"status": "OPEN", "filled_quantity": 0, "average_price": 0},
        {"status": "COMPLETE", "filled_quantity": 7, "average_price": 101.35, "status_message": None},
    ]
    status = KiteBroker(kite).order_status("9")
    assert (status.status, status.filled_quantity, status.average_price, status.is_final) == ("COMPLETE", 7, 101.35, True)
    kite.history["10"] = [{"status": "REJECTED", "status_message": "insufficient margin"}]
    rejected = KiteBroker(kite).order_status("10")
    assert rejected.is_final and rejected.filled_quantity == 0 and "margin" in rejected.message


def test_kite_broker_positions_only_count_intraday_positions_on_its_exchange():
    assert KiteBroker(FakeKite()).positions() == {"ABC": -10}
