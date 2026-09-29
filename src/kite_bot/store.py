"""SQLite persistence for the live engine: what was shortlisted, signalled, ordered and traded.

One small file (or ``:memory:`` for tests) holds everything needed to audit a day
and to recover after a crash. Python's built-in ``sqlite3`` is used, guarded by a
lock because the market-data thread and the main thread both write.

Tables
    shortlist   stocks that passed the daily screen, per day
    signals     crossovers seen and whether they were acted on (with the reason if not)
    orders      every order sent to the broker
    trades      one row per position: entry, stop/target, exit, P&L, status open/closed
    bars        completed 2-minute bars built from the tick stream
    ticks       optional raw tick log
    events      timestamped log lines (warnings, halts, recoveries)
    kv          small key/value state such as the kill switch
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS shortlist (
    day TEXT NOT NULL, symbol TEXT NOT NULL, turn_date TEXT, cross_date TEXT, close REAL, ma REAL,
    PRIMARY KEY (day, symbol)
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, symbol TEXT NOT NULL,
    action TEXT NOT NULL, price REAL, detail TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, ts TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
    quantity INTEGER NOT NULL, price REAL, purpose TEXT NOT NULL, trade_id INTEGER
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL,
    quantity INTEGER NOT NULL, entry_time TEXT NOT NULL, entry_price REAL NOT NULL,
    target_price REAL NOT NULL, stop_price REAL NOT NULL, stop_order_id TEXT,
    status TEXT NOT NULL DEFAULT 'open', exit_time TEXT, exit_price REAL, exit_reason TEXT, pnl REAL
);
CREATE INDEX IF NOT EXISTS trades_day ON trades (day, status);
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL, start TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    PRIMARY KEY (symbol, start)
);
CREATE TABLE IF NOT EXISTS ticks (ts TEXT NOT NULL, symbol TEXT NOT NULL, price REAL NOT NULL, volume INTEGER);
CREATE INDEX IF NOT EXISTS ticks_symbol_ts ON ticks (symbol, ts);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass(frozen=True)
class TradeRow:
    id: int
    day: str
    symbol: str
    quantity: int
    entry_time: str
    entry_price: float
    target_price: float
    stop_price: float
    stop_order_id: Optional[str]
    status: str
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None


def _iso(value) -> str:
    if isinstance(value, dt.datetime):
        return value.isoformat(sep=" ")
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


class Store:
    """Thread-safe SQLite store. ``Store()`` keeps everything in memory; pass a path to keep it."""

    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _run(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._db.execute(sql, tuple(params))
            self._db.commit()
            return cursor

    def _query(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(params)).fetchall()

    # ------------------------------------------------------------ shortlist

    def save_shortlist(self, day: dt.date, entries: Iterable[tuple[str, object]]) -> None:
        """entries: (symbol, screener.Setup)."""
        rows = [
            (day.isoformat(), symbol, _iso(s.turn_date.date()), _iso(s.cross_date.date()), s.close, s.ma)
            for symbol, s in entries
        ]
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO shortlist VALUES (?,?,?,?,?,?)", rows)
            self._db.commit()

    def shortlist(self, day: dt.date) -> list[str]:
        return [r["symbol"] for r in self._query("SELECT symbol FROM shortlist WHERE day=? ORDER BY symbol", [day.isoformat()])]

    # -------------------------------------------------------------- signals

    def log_signal(self, ts, symbol: str, action: str, price: Optional[float] = None, detail: str = "") -> None:
        self._run("INSERT INTO signals (ts, symbol, action, price, detail) VALUES (?,?,?,?,?)", [_iso(ts), symbol, action, price, detail])

    def signals(self, symbol: Optional[str] = None) -> pd.DataFrame:
        sql, params = "SELECT ts, symbol, action, price, detail FROM signals", []
        if symbol:
            sql, params = sql + " WHERE symbol=?", [symbol]
        return pd.DataFrame([dict(r) for r in self._query(sql + " ORDER BY id", params)])

    # --------------------------------------------------------------- orders

    def log_order(self, order, purpose: str, ts, trade_id: Optional[int] = None) -> None:
        self._run(
            "INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?,?,?)",
            [order.order_id, _iso(ts), order.symbol, order.side.value, order.quantity, order.price, purpose, trade_id],
        )

    def orders(self) -> pd.DataFrame:
        return pd.DataFrame([dict(r) for r in self._query("SELECT * FROM orders ORDER BY ts, order_id")])

    # --------------------------------------------------------------- trades

    def open_trade(self, day: dt.date, symbol: str, quantity: int, entry_time, entry_price: float,
                   target_price: float, stop_price: float, stop_order_id: Optional[str]) -> int:
        cursor = self._run(
            "INSERT INTO trades (day, symbol, side, quantity, entry_time, entry_price, target_price, stop_price, stop_order_id)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [day.isoformat(), symbol, "SHORT", quantity, _iso(entry_time), entry_price, target_price, stop_price, stop_order_id],
        )
        return int(cursor.lastrowid)

    def close_trade(self, trade_id: int, exit_time, exit_price: Optional[float], reason: str, pnl: Optional[float]) -> None:
        self._run(
            "UPDATE trades SET status='closed', exit_time=?, exit_price=?, exit_reason=?, pnl=? WHERE id=?",
            [_iso(exit_time), exit_price, reason, pnl, trade_id],
        )

    def set_stop_order(self, trade_id: int, stop_order_id: Optional[str]) -> None:
        self._run("UPDATE trades SET stop_order_id=? WHERE id=?", [stop_order_id, trade_id])

    @staticmethod
    def _trade(row: sqlite3.Row) -> TradeRow:
        return TradeRow(**{k: row[k] for k in TradeRow.__dataclass_fields__})

    def open_trades(self, day: dt.date) -> list[TradeRow]:
        rows = self._query("SELECT * FROM trades WHERE day=? AND status='open' ORDER BY id", [day.isoformat()])
        return [self._trade(r) for r in rows]

    def trades(self, day: Optional[dt.date] = None) -> list[TradeRow]:
        sql, params = "SELECT * FROM trades", []
        if day:
            sql, params = sql + " WHERE day=?", [day.isoformat()]
        return [self._trade(r) for r in self._query(sql + " ORDER BY id", params)]

    def realized_pnl(self, day: dt.date) -> float:
        row = self._query("SELECT COALESCE(SUM(pnl), 0) AS total FROM trades WHERE day=? AND status='closed'", [day.isoformat()])[0]
        return float(row["total"])

    # ----------------------------------------------------------------- bars

    def save_bar(self, symbol: str, bar) -> None:
        self._run(
            "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)",
            [symbol, _iso(bar.start), bar.open, bar.high, bar.low, bar.close, bar.volume],
        )

    def bars(self, symbol: str) -> pd.DataFrame:
        rows = self._query("SELECT start, open, high, low, close, volume FROM bars WHERE symbol=? ORDER BY start", [symbol])
        frame = pd.DataFrame([dict(r) for r in rows])
        if frame.empty:
            return frame
        frame["start"] = pd.to_datetime(frame["start"])
        return frame.set_index("start")

    # ---------------------------------------------------------------- ticks

    def save_ticks(self, ticks: Iterable) -> None:
        rows = [(_iso(t.timestamp), t.symbol, t.price, t.volume) for t in ticks]
        with self._lock:
            self._db.executemany("INSERT INTO ticks VALUES (?,?,?,?)", rows)
            self._db.commit()

    def ticks(self, symbol: str, since: Optional[dt.datetime] = None) -> pd.DataFrame:
        sql, params = "SELECT ts, price, volume FROM ticks WHERE symbol=?", [symbol]
        if since is not None:
            sql, params = sql + " AND ts>=?", params + [_iso(since)]
        frame = pd.DataFrame([dict(r) for r in self._query(sql + " ORDER BY ts", params)])
        if frame.empty:
            return frame
        frame["ts"] = pd.to_datetime(frame["ts"])
        return frame.set_index("ts")

    # --------------------------------------------------------------- events

    def log_event(self, level: str, message: str, ts=None) -> None:
        self._run("INSERT INTO events (ts, level, message) VALUES (?,?,?)", [_iso(ts or dt.datetime.now()), level, message])

    def events(self, level: Optional[str] = None) -> pd.DataFrame:
        sql, params = "SELECT ts, level, message FROM events", []
        if level:
            sql, params = sql + " WHERE level=?", [level]
        return pd.DataFrame([dict(r) for r in self._query(sql + " ORDER BY id", params)])

    # ------------------------------------------------------------------- kv

    def set(self, key: str, value: Optional[str]) -> None:
        if value is None:
            self._run("DELETE FROM kv WHERE key=?", [key])
        else:
            self._run("INSERT OR REPLACE INTO kv VALUES (?,?)", [key, value])

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        rows = self._query("SELECT value FROM kv WHERE key=?", [key])
        return rows[0]["value"] if rows else default
