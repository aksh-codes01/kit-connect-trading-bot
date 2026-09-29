import datetime as dt

import pytest

from kite_bot.backtest import BacktestConfig, BacktestResult, run_backtest
from kite_bot.broker import PaperBroker
from kite_bot.sample_data import TRADE_DAY, make_sample_market


@pytest.fixture(scope="module")
def market():
    return make_sample_market()


def test_sample_market_gives_one_take_profit_and_one_stop_loss(market):
    daily, minute = market
    result = run_backtest(daily, minute)
    by_symbol = {r.symbol: r for r in result.records}

    assert set(by_symbol) == {"ALPHA", "BRAVO"}  # CHARLIE fails the daily screen
    assert by_symbol["ALPHA"].trade.reason == "take_profit"
    assert by_symbol["BRAVO"].trade.reason == "stop_loss"
    assert all(r.day == TRADE_DAY for r in result.records)
    assert by_symbol["ALPHA"].trade.pnl_pct == pytest.approx(0.02)
    assert by_symbol["BRAVO"].trade.pnl_pct == pytest.approx(-0.02)


def test_net_pnl_matches_the_brokers_books(market):
    daily, minute = market
    broker = PaperBroker()
    result = run_backtest(daily, minute, broker=broker)
    assert sum(r.net_pnl for r in result.records) == pytest.approx(broker.realized_pnl)
    assert broker.positions() == {}  # every trade was closed
    assert len(broker.orders) == 4  # a sell and a buy per trade


def test_quantity_comes_from_capital_per_trade(market):
    daily, minute = market
    result = run_backtest(daily, minute, BacktestConfig(capital_per_trade=50_000))
    for r in result.records:
        assert r.quantity == int(50_000 // r.trade.entry_price)


def test_capital_too_small_for_one_share_skips_the_trade(market):
    daily, minute = market
    assert run_backtest(daily, minute, BacktestConfig(capital_per_trade=50)).records == []


def test_max_trades_per_day_keeps_the_earliest_signal(market):
    daily, minute = market
    result = run_backtest(daily, minute, BacktestConfig(max_trades_per_day=1))
    assert [r.symbol for r in result.records] == ["BRAVO"]  # BRAVO enters two minutes before ALPHA


def test_costs_reduce_pnl(market):
    daily, minute = market
    free = run_backtest(daily, minute).summary()["total_pnl"]
    costly = run_backtest(daily, minute, broker=PaperBroker(commission_pct=0.001)).summary()["total_pnl"]
    assert costly < free


def test_date_range_filters_days(market):
    daily, minute = market
    after = TRADE_DAY + dt.timedelta(days=1)
    assert run_backtest(daily, minute, start=after).records == []
    assert len(run_backtest(daily, minute, start=TRADE_DAY, end=TRADE_DAY).records) == 2


def test_summary_statistics(market):
    daily, minute = market
    summary = run_backtest(daily, minute).summary()
    assert summary["trades"] == 2
    assert summary["win_rate"] == 0.5
    assert summary["exits"] == {"take_profit": 1, "stop_loss": 1}
    assert summary["worst_trade"] < 0 < summary["best_trade"]
    assert summary["max_drawdown"] == pytest.approx(-summary["worst_trade"], rel=1e-6)


def test_empty_result_summary_and_frame():
    result = BacktestResult()
    assert result.summary() == {"trades": 0, "total_pnl": 0.0}
    assert result.to_frame().empty
