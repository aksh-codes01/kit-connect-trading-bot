"""Replay historical bars through the screener, the strategy and a paper broker.

For each trading day D found in the minute data:
    1. Screen every symbol on daily bars up to the day before D.
    2. For shortlisted symbols, look for the bearish crossover on D's 2-minute bars.
    3. Sell short through the broker, walk the trade forward to its exit, cover.

Simplifications: trades are independent (no shared margin or concurrency
limit beyond `max_trades_per_day`), and the signal-to-fill delay is one 2-minute bar.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Mapping, Optional

import pandas as pd

from kite_bot.broker import PaperBroker, Side
from kite_bot.config import StrategyConfig
from kite_bot.ohlc import resample_ohlc
from kite_bot.screener import find_setup
from kite_bot.strategy import Trade, find_short_entry, simulate_short


BacktestConfig = StrategyConfig  # same settings object the live engine uses


@dataclass(frozen=True)
class TradeRecord:
    symbol: str
    day: dt.date
    quantity: int
    trade: Trade
    net_pnl: float  # rupees, after the broker's commission and slippage


@dataclass
class BacktestResult:
    records: list[TradeRecord] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "day": r.day,
                "symbol": r.symbol,
                "quantity": r.quantity,
                "entry_time": r.trade.entry_time,
                "entry_price": r.trade.entry_price,
                "exit_time": r.trade.exit_time,
                "exit_price": r.trade.exit_price,
                "reason": r.trade.reason,
                "pnl_pct": r.trade.pnl_pct,
                "net_pnl": r.net_pnl,
            }
            for r in self.records
        ]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        n = len(self.records)
        if n == 0:
            return {"trades": 0, "total_pnl": 0.0}
        pnls = pd.Series([r.net_pnl for r in self.records])
        equity = pnls.cumsum()
        drawdown = (equity.cummax().clip(lower=0) - equity).max()
        reasons = pd.Series([r.trade.reason for r in self.records]).value_counts().to_dict()
        return {
            "trades": n,
            "win_rate": float((pnls > 0).mean()),
            "total_pnl": float(pnls.sum()),
            "avg_pnl_pct": float(pd.Series([r.trade.pnl_pct for r in self.records]).mean()),
            "best_trade": float(pnls.max()),
            "worst_trade": float(pnls.min()),
            "max_drawdown": float(drawdown),
            "exits": reasons,
        }


def _dates(df: pd.DataFrame):
    return df.index.date


def run_backtest(
    daily: Mapping[str, pd.DataFrame],
    minute: Mapping[str, pd.DataFrame],
    config: BacktestConfig = BacktestConfig(),
    broker: Optional[PaperBroker] = None,
    start: Optional[dt.date] = None,
    end: Optional[dt.date] = None,
) -> BacktestResult:
    """daily / minute map symbol -> bars (ascending, DatetimeIndex, open/high/low/close)."""
    broker = broker or PaperBroker()
    result = BacktestResult()

    days = sorted({d for df in minute.values() for d in _dates(df)})
    days = [d for d in days if (start is None or d >= start) and (end is None or d <= end)]

    for day in days:
        candidates = []
        for symbol in sorted(daily):
            if symbol not in minute:
                continue
            history = daily[symbol]
            history = history[_dates(history) < day]
            if find_setup(history, "short", config.ma_period, config.slope_window) is None:
                continue

            mins = minute[symbol]
            sessions = sorted({d for d in _dates(mins) if d <= day})[-(config.warmup_sessions + 1):]
            window = mins[pd.Index(_dates(mins)).isin(sessions)]
            bars = resample_ohlc(window)
            entry = find_short_entry(bars, day, config.fast, config.slow, config.last_entry)
            if entry is not None:
                candidates.append((entry.time, symbol, bars, entry))

        for _, symbol, bars, entry in sorted(candidates, key=lambda c: (c[0], c[1]))[: config.max_trades_per_day]:
            quantity = int(config.capital_per_trade // entry.price)
            if quantity < 1:
                continue
            before = broker.realized_pnl
            broker.place_order(symbol, Side.SELL, quantity, entry.price)
            trade = simulate_short(bars, entry, config.take_profit_pct, config.stop_loss_pct, config.square_off)
            broker.place_order(symbol, Side.BUY, quantity, trade.exit_price)
            result.records.append(TradeRecord(symbol, day, quantity, trade, broker.realized_pnl - before))

    return result
