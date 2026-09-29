"""Command-line runner:  python -m kite_bot <command>

    demo       run the whole pipeline on built-in synthetic data (no account needed)
    screen     list stocks from daily CSVs that match the MA100 screener
    backtest   replay CSV data through screener + strategy + paper broker
    replay     run the LIVE engine over one recorded day of CSV data (paper, logs to SQLite)
    live       trade today: --mode paper (real data, simulated orders) or --mode live (real orders)
    report     print trades and warnings from a bot database
    download   fetch daily and 1-minute bars from Kite into CSV files (needs login files)
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from kite_bot.backtest import BacktestConfig, run_backtest
from kite_bot.broker import KiteBroker, PaperBroker
from kite_bot.config import LiveConfig
from kite_bot.engine import LiveEngine
from kite_bot.feed import KiteTickerFeed, ReplayFeed
from kite_bot.live import FrameMarketData, KiteMarketData, load_universe, run_session, status_handler
from kite_bot.ohlc import fetch_ohlc, ist_now, load_directory, load_instruments, write_market
from kite_bot.sample_data import TRADE_DAY, make_sample_market
from kite_bot.screener import scan
from kite_bot.store import Store


def _date(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


# ------------------------------------------------------------------- printing

def _print_screen(daily, side: str) -> int:
    matches = scan(daily, side=side)
    if not matches:
        print("No stocks match the screener.")
        return 0
    print(f"{'symbol':<12}{'turn date':<12}{'cross date':<12}{'close':>10}{'MA100':>10}")
    for symbol, s in matches:
        print(f"{symbol:<12}{s.turn_date.date()!s:<12}{s.cross_date.date()!s:<12}{s.close:>10.2f}{s.ma:>10.2f}")
    return len(matches)


def _print_backtest(result) -> None:
    frame = result.to_frame()
    if frame.empty:
        print("No trades.")
        return
    show = frame[["day", "symbol", "quantity", "entry_price", "exit_price", "reason", "pnl_pct", "net_pnl"]].copy()
    show["pnl_pct"] = (show["pnl_pct"] * 100).round(2)
    show[["entry_price", "exit_price", "net_pnl"]] = show[["entry_price", "exit_price", "net_pnl"]].round(2)
    print(show.to_string(index=False))
    s = result.summary()
    print(
        f"\ntrades {s['trades']} | win rate {s['win_rate']:.0%} | total P&L {s['total_pnl']:.2f} "
        f"| worst trade {s['worst_trade']:.2f} | max drawdown {s['max_drawdown']:.2f} | exits {s['exits']}"
    )


def _print_store_trades(store: Store, day: dt.date) -> None:
    trades = store.trades(day)
    if not trades:
        print("No trades.")
        return
    rows = [
        {"symbol": t.symbol, "qty": t.quantity, "entry": t.entry_time[11:19], "entry_price": round(t.entry_price, 2),
         "exit": (t.exit_time or "")[11:19], "exit_price": None if t.exit_price is None else round(t.exit_price, 2),
         "reason": t.exit_reason or t.status, "pnl": None if t.pnl is None else round(t.pnl, 2)}
        for t in trades
    ]
    print(pd.DataFrame(rows).to_string(index=False))


def _print_summary(summary: dict) -> None:
    print(
        f"\n{summary['day']}: shortlist {len(summary['shortlist'])} | trades {summary['trades']} | wins {summary['wins']} "
        f"| realized P&L {summary['realized_pnl']:.2f} | still open {summary['open']}"
        + (f" | HALTED: {summary['halted']}" if summary["halted"] else "")
    )


def _capital(args) -> float:
    return 100_000.0 if args.capital is None else args.capital


def _config(args) -> BacktestConfig:
    return BacktestConfig(
        fast=args.fast,
        slow=args.slow,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        capital_per_trade=_capital(args),
    )


def _live_config(args) -> LiveConfig:
    return LiveConfig(
        fast=args.fast, slow=args.slow, take_profit_pct=args.take_profit, stop_loss_pct=args.stop_loss,
        capital_per_trade=_capital(args), max_trades_per_day=args.max_trades,
        max_daily_loss=args.max_daily_loss, record_ticks=getattr(args, "record_ticks", False),
    )


# ------------------------------------------------------------------- commands

def _cmd_demo(args) -> None:
    daily, minute = make_sample_market()
    if args.write_data:
        write_market(daily, minute, args.write_data)
        print(f"Sample CSVs written to {args.write_data}\n")
    print("Step 1: daily screen (data up to the day before the trade day)")
    last_day = max(df.index[-1] for df in daily.values()).date()
    print(f"        last traded day in the data: {last_day}")
    _print_screen(daily, "short")
    print("\nStep 2: backtest of the trade day (bars in, paper broker)")
    result = run_backtest(daily, minute, BacktestConfig(), PaperBroker())
    _print_backtest(result)

    print("\nStep 3: the same day through the LIVE engine (replayed as ticks, paper broker, logged to SQLite)")
    store = Store()
    engine = LiveEngine(PaperBroker(), store, TRADE_DAY, sleep=lambda s: None)
    run_session(engine, ReplayFeed(minute, TRADE_DAY), FrameMarketData(daily, minute), sorted(daily), sleep=lambda s: None, notify=lambda m: None)
    _print_store_trades(store, TRADE_DAY)
    live = {t.symbol: (t.exit_reason, t.quantity) for t in store.trades(TRADE_DAY)}
    reference = {r.symbol: (r.trade.reason, r.quantity) for r in result.records}
    print("\nLive engine vs backtest (symbol, exit reason, quantity):", "IDENTICAL" if live == reference else f"DIFFERENT {live} vs {reference}")
    print("Synthetic data: this shows how the pieces connect, not how the strategy performs.")


def _cmd_screen(args) -> None:
    daily, _ = load_directory(args.data_dir)
    _print_screen(daily, args.side)


def _cmd_backtest(args) -> None:
    daily, minute = load_directory(args.data_dir)
    broker = PaperBroker(commission_pct=args.commission, slippage_pct=args.slippage)
    result = run_backtest(daily, minute, _config(args), broker, args.start, args.end)
    _print_backtest(result)
    if args.trades_csv:
        result.to_frame().to_csv(args.trades_csv, index=False)
        print(f"Trades written to {args.trades_csv}")


def _cmd_replay(args) -> None:
    daily, minute = load_directory(args.data_dir)
    day = args.date or max(df.index[-1].date() for df in minute.values())
    store = Store(args.db)
    engine = LiveEngine(PaperBroker(commission_pct=args.commission), store, day, _live_config(args), sleep=lambda s: None)
    summary = run_session(engine, ReplayFeed(minute, day), FrameMarketData(daily, minute), sorted(daily), sleep=lambda s: None)
    _print_store_trades(store, day)
    _print_summary(summary)


def _cmd_report(args) -> None:
    store = Store(args.db)
    days = sorted({t.day for t in store.trades()})
    if not days:
        print("No trades in this database.")
        return
    day = args.day or dt.date.fromisoformat(days[-1])
    print(f"Trades on {day}")
    _print_store_trades(store, day)
    trades = store.trades(day)
    closed = [t for t in trades if t.pnl is not None]
    print(f"\ntrades {len(trades)} | realized P&L {sum(t.pnl for t in closed):.2f} | days recorded: {', '.join(days)}")
    problems = pd.concat([store.events("WARNING"), store.events("ERROR"), store.events("CRITICAL")])
    if not problems.empty:
        print("\nWarnings and errors:")
        print(problems.sort_index().to_string(index=False))


def _kite_session(auth_dir: str):
    """Log in with the token files the login scripts write. Returns (kite, api_key, access_token)."""
    from kiteconnect import KiteConnect  # imported here so the other commands work without it

    auth = Path(auth_dir)
    api_key = (auth / "api_key.txt").read_text().split()[0]
    access_token = (auth / "access_token.txt").read_text().strip()
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite, api_key, access_token


def _cmd_download(args) -> None:
    kite, _, _ = _kite_session(args.auth_dir)
    instruments = load_instruments(kite)
    daily, minute = {}, {}
    for symbol in args.symbols:
        print(f"downloading {symbol} ...")
        try:
            daily[symbol] = fetch_ohlc(kite, instruments, symbol, "day", days=args.daily_days)
            minute[symbol] = fetch_ohlc(kite, instruments, symbol, "minute", days=args.minute_days)
        except KeyError:
            print(f"skipping {symbol}: not found in the NSE instrument list")
            daily.pop(symbol, None)
    write_market(daily, minute, args.data_dir)
    print(f"Saved {len(daily)} symbols to {args.data_dir}")


def _cmd_live(args) -> None:
    if args.mode == "live" and not args.yes_real_money:
        raise SystemExit("Refusing to send real orders without --yes-real-money. Use --mode paper to dry-run on live data.")
    if args.mode == "live" and (args.capital is None or args.max_daily_loss is None):
        raise SystemExit("Real-money runs need an explicit --capital (rupees per trade) and --max-daily-loss (rupees).")
    kite, api_key, access_token = _kite_session(args.auth_dir)
    instruments = load_instruments(kite)
    universe = load_universe(args.universe)
    subset = instruments[instruments["tradingsymbol"].isin(universe)]
    token_map = {int(r.instrument_token): r.tradingsymbol for r in subset.itertuples()}
    tick_sizes = dict(zip(subset["tradingsymbol"], subset["tick_size"])) if "tick_size" in subset.columns else None

    broker = KiteBroker(kite, tick_sizes=tick_sizes) if args.mode == "live" else PaperBroker()
    engine = LiveEngine(broker, Store(args.db), ist_now().date(), _live_config(args))
    feed = KiteTickerFeed(api_key, access_token, token_map, on_status=status_handler(engine))
    print(f"{args.mode.upper()} session for {len(universe)} stocks, logging to {args.db}")
    summary = run_session(engine, feed, KiteMarketData(kite, instruments), universe)
    _print_summary(summary)


# --------------------------------------------------------------------- parser

def _strategy_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--capital", type=float, default=None, help="rupees per trade (default 100000; required for --mode live)")
    p.add_argument("--take-profit", type=float, default=0.02)
    p.add_argument("--stop-loss", type=float, default=0.02)
    p.add_argument("--fast", type=int, default=100)
    p.add_argument("--slow", type=int, default=500)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kite_bot", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run on built-in synthetic data")
    demo.add_argument("--write-data", metavar="DIR", help="also save the sample data as CSV files")
    demo.set_defaults(func=_cmd_demo)

    screen = sub.add_parser("screen", help="run the daily screener on CSV data")
    screen.add_argument("--data-dir", required=True)
    screen.add_argument("--side", choices=["short", "long"], default="short")
    screen.set_defaults(func=_cmd_screen)

    bt = sub.add_parser("backtest", help="backtest on CSV data")
    bt.add_argument("--data-dir", required=True)
    bt.add_argument("--start", type=_date)
    bt.add_argument("--end", type=_date)
    _strategy_options(bt)
    bt.add_argument("--commission", type=float, default=0.0, help="fraction per fill, e.g. 0.0003")
    bt.add_argument("--slippage", type=float, default=0.0, help="fraction per fill, e.g. 0.0005")
    bt.add_argument("--trades-csv", metavar="PATH", help="save the trade list")
    bt.set_defaults(func=_cmd_backtest)

    rp = sub.add_parser("replay", help="run the live engine over one recorded day (paper)")
    rp.add_argument("--data-dir", required=True)
    rp.add_argument("--date", type=_date, help="the trade day (default: last day in the minute data)")
    rp.add_argument("--db", default=":memory:", help="SQLite file to log to (default: in memory)")
    _strategy_options(rp)
    rp.add_argument("--max-trades", type=int, default=5)
    rp.add_argument("--max-daily-loss", type=float, default=None, help="rupees; halts new entries and flattens")
    rp.add_argument("--commission", type=float, default=0.0)
    rp.set_defaults(func=_cmd_replay)

    rep = sub.add_parser("report", help="print trades and warnings from a bot database")
    rep.add_argument("--db", required=True)
    rep.add_argument("--day", type=_date)
    rep.set_defaults(func=_cmd_report)

    live = sub.add_parser("live", help="trade today (paper on live data, or real orders)")
    live.add_argument("--universe", required=True, help="text file: one NSE symbol per line")
    live.add_argument("--mode", choices=["paper", "live"], default="paper")
    live.add_argument("--yes-real-money", action="store_true", help="required for --mode live")
    live.add_argument("--db", default="bot.db", help="SQLite file for the trade log (default: bot.db)")
    live.add_argument("--auth-dir", default=".", help="folder holding api_key.txt and access_token.txt")
    _strategy_options(live)
    live.add_argument("--max-trades", type=int, default=5)
    live.add_argument("--max-daily-loss", type=float, default=None, help="rupees; halts new entries and flattens")
    live.add_argument("--record-ticks", action="store_true", help="store every tick in the database (large)")
    live.set_defaults(func=_cmd_live)

    dl = sub.add_parser("download", help="download bars from Kite to CSV")
    dl.add_argument("--symbols", nargs="+", required=True)
    dl.add_argument("--data-dir", required=True)
    dl.add_argument("--auth-dir", default=".", help="folder holding api_key.txt and access_token.txt")
    dl.add_argument("--daily-days", type=int, default=300)
    dl.add_argument("--minute-days", type=int, default=20)
    dl.set_defaults(func=_cmd_download)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
