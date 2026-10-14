"""
Milestone 3: backtesting.

  python backtest.py download                  # Fetch history into PostgreSQL
  python backtest.py run                       # Backtest all strategies on it
  python backtest.py run --strategy trend      # One strategy
  python backtest.py walk-forward --strategy breakout
  python backtest.py run --synthetic           # Try it on made-up data, no DB

Defaults: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT and DOGEUSDT, 1-hour candles,
from 2021-01-01 (which takes in the 2021 bull run and May crash, the 2022
bear market with the LUNA and FTX crashes, the 2023 recovery and the 2024
rally). No orders are ever placed.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import structlog

from config.settings import AppSettings, get_settings

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT"
DEFAULT_START = "2021-01-01"


def _date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _quiet(verbose: bool) -> None:
    level = 20 if verbose else 30  # The Risk Manager logs every verdict at INFO
    structlog.configure(
        processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )


async def _open_db(settings: AppSettings):
    from database import MarketDataRepository, ResearchRepository
    repo = MarketDataRepository.from_url(settings.database.url)
    try:
        await repo.create_schema()
    except Exception as e:
        await repo.close()
        sys.exit(f"Cannot reach the database ({e}). Start PostgreSQL with "
                 "'docker compose -f docker/docker-compose.yml up -d', or use --synthetic.")
    return repo, ResearchRepository(repo.engine)


# ── download ───────────────────────────────────────────────────────────

async def cmd_download(args, settings: AppSettings) -> None:
    from events import EventBus
    from exchanges.bybit import BybitAdapter
    from research.history import HistoryDownloader

    repo, research = await _open_db(settings)
    exchange = BybitAdapter(settings.bybit, EventBus())
    await exchange.connect()
    try:
        dl = HistoryDownloader(exchange, repo, research)
        for sym in args.symbols.split(","):
            s = await dl.download(sym.strip(), args.interval, _date(args.start),
                                  _date(args.end))
            print(f"{s['symbol']}: {s['new_candles']} new candles, {s['stored_candles']} stored"
                  f" ({s['first']} to {s['last']}), {s['funding_rows']} funding rates")
    finally:
        await exchange.disconnect()
        await repo.close()


# ── run / walk-forward ─────────────────────────────────────────────────

async def _load(args, settings, bt):
    """History (or synthetic data) and an optional repository to save to."""
    if args.synthetic:
        from research.synthetic import synthetic_history
        from strategies.features import interval_minutes
        data = synthetic_history(bar_minutes=interval_minutes(args.interval))
        return data, None, None, None
    from research.history import load_history
    repo, research = await _open_db(settings)
    data = await load_history(research, "bybit", [s.strip() for s in args.symbols.split(",")],
                              args.interval, _date(args.start), _date(args.end))
    if not data:
        await repo.close()
        sys.exit("No stored history for these symbols. Run: python backtest.py download")
    return data, repo, research, _date(args.start)


def _config(args):
    from research.backtester import BacktestConfig
    from research.simulator import CostModel
    costs = CostModel(spread_bps=args.spread_bps, base_slippage_bps=args.slippage_bps)
    return BacktestConfig(interval=args.interval, initial_equity=args.equity, costs=costs,
                          breaker_reset_daily=not args.keep_breaker,
                          entry_delay_bars=args.delay_bars)


async def cmd_run(args, settings: AppSettings) -> None:
    from research import metrics as perf
    from research import report
    from research.backtester import Backtester
    from research.monte_carlo import monte_carlo
    from strategies import build_strategies

    bt = Backtester(_config(args), settings.risk)
    data, repo, research, start = await _load(args, settings, bt)
    try:
        strategies = build_strategies(args.strategy)
        res = await bt.run(strategies, data, start, _date(args.end))
        m = res.metrics()
        mc = monte_carlo(res.trades)

        span = f"{res.equity.index[0]:%Y-%m-%d} to {res.equity.index[-1]:%Y-%m-%d}" \
            if len(res.equity) else "no bars"
        print(f"\n=== BACKTEST {'(SYNTHETIC DATA) ' if args.synthetic else ''}==="
              f"  {', '.join(s.name for s in strategies)} on {', '.join(res.symbols)},"
              f" {args.interval}m bars, {span}\n")
        print("\n".join(report.summary_lines(m)))
        print("\n".join(report.monte_carlo_lines(mc)))
        print()
        print("\n".join(report.table(res.by_phase(), "By market phase at entry"
                                     " (a trade can be in several)", "phase")))
        print()
        print("\n".join(report.table(perf.by_strategy(res.trades), "By strategy", "strategy")))
        print()
        print("\n".join(report.table(perf.by_regime(res.trades),
                                     "By scanner regime at entry", "regime")))
        print()
        print("\n".join(report.coverage_lines(res.coverage)))
        if res.rejection_reasons:
            print("\nRisk Manager rejections: " + ", ".join(
                f"{k} x{v}" for k, v in res.rejection_reasons.most_common()))
        if res.cancelled_entries:
            print("Cancelled entries: " + ", ".join(
                f"{k} x{v}" for k, v in res.cancelled_entries.most_common()))
        if args.csv:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        if args.csv and report.write_trades_csv(args.csv, res.trades):
            print(f"\nTrades written to {args.csv}")
        if research is not None:
            run_id = await research.save_run(
                "backtest", res.strategies, res.symbols, args.interval, start, _date(args.end),
                res.config.as_dict(), {**m, "monte_carlo": mc}, res.by_phase())
            await research.save_trades(run_id, "backtest", "bybit", res.trades)
            print(f"Saved as research run {run_id} (tables research_runs, trade_journal)")
        print("\nThese are simulated results. Real fills, outages and order book"
              " depth will differ; paper trade before trusting them.")
    finally:
        if repo is not None:
            await repo.close()


async def cmd_walk_forward(args, settings: AppSettings) -> None:
    from research import metrics as perf
    from research import report
    from research.backtester import Backtester
    from research.walk_forward import WalkForwardConfig, walk_forward
    from strategies import STRATEGIES

    if args.strategy == "all":
        names = list(STRATEGIES)
    else:
        names = [n.strip() for n in args.strategy.split(",")]
    for n in names:
        if n not in STRATEGIES:
            sys.exit(f"Unknown strategy {n}; choose from {sorted(STRATEGIES)}")

    bt = Backtester(_config(args), settings.risk)
    data, repo, research, start = await _load(args, settings, bt)
    try:
        prepared = bt.prepare(data)
        first = max(p.times[0] for p in prepared.values()).to_pydatetime()
        last = max(p.times[-1] for p in prepared.values()).to_pydatetime()
        lo = max(start, first) if start else first
        hi = _date(args.end) or last
        cfg = WalkForwardConfig(train_days=args.train_days, test_days=args.test_days)
        for name in names:
            res = await walk_forward(bt, STRATEGIES[name], prepared, lo, hi, cfg)
            print(f"\n=== WALK-FORWARD {'(SYNTHETIC DATA) ' if args.synthetic else ''}==="
                  f"  {name}: train {cfg.train_days}d, test {cfg.test_days}d,"
                  f" {len(res.windows)} windows\n")
            if not res.windows:
                print("Not enough history for one window. Download more, or shorten"
                      " --train-days / --test-days.")
                continue
            for w in res.windows:
                o, i = w.out_sample, w.in_sample
                changed = {k: v for k, v in w.params.items()
                           if STRATEGIES[name].DEFAULT_PARAMS.get(k) != v}
                print(f"  test {w.test_start:%Y-%m-%d}..{w.test_end:%Y-%m-%d}"
                      f"  in-sample {i.get('trades', 0):>4} trades {i.get('expectancy_r', 0):+.3f} R"
                      f"  | out-of-sample {o.get('trades', 0):>4} trades"
                      f" {o.get('expectancy_r', 0):+.3f} R  net {o.get('net_pnl', 0):+.2f}"
                      f"  params {changed or 'defaults'}")
            print("\nOut-of-sample only, all windows chained:")
            print("\n".join(report.summary_lines(res.metrics)))
            print(f"\nValidation gate: {'PASSED' if res.passed else 'NOT PASSED'}"
                  " (passing means worth paper trading, not worth real money)")
            print("\n".join(report.gate_lines(res.gate)))
            if research is not None:
                run_id = await research.save_run(
                    "walk_forward", [{"name": name, "windows": [
                        {"test_start": str(w.test_start), "params": w.params}
                        for w in res.windows]}],
                    sorted(prepared), args.interval, lo, hi, bt.config.as_dict(),
                    {**res.metrics, "gate": res.gate, "passed": res.passed,
                     "in_sample_expectancy_r": res.in_sample_expectancy_r},
                    perf.by_phase(res.trades))
                await research.save_trades(run_id, "walk_forward", "bybit", res.trades)
                print(f"Saved as research run {run_id}")
    finally:
        if repo is not None:
            await repo.close()


def main(argv=None) -> None:
    settings = get_settings()
    assert not settings.execution_enabled, "Execution must be DISABLED until Milestone 5"

    ap = argparse.ArgumentParser(description="Backtesting (Milestone 3). No orders are placed.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--symbols", default=DEFAULT_SYMBOLS)
        p.add_argument("--interval", default="60", help="Candle minutes: 5, 15, 60, 240, D")
        p.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD (UTC)")
        p.add_argument("--end", default=None, help="YYYY-MM-DD (UTC), default: now")
        p.add_argument("--verbose", action="store_true")

    d = sub.add_parser("download", help="Fetch candle and funding history from Bybit")
    common(d)

    for name in ("run", "walk-forward"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--strategy", default="all",
                       help="trend, breakout, mean_reversion, or all")
        p.add_argument("--equity", type=float, default=settings.paper_equity)
        p.add_argument("--spread-bps", type=float, default=2.0)
        p.add_argument("--slippage-bps", type=float, default=1.0)
        p.add_argument("--delay-bars", type=int, default=0,
                       help="Extra latency: fill this many bars after the signal")
        p.add_argument("--keep-breaker", action="store_true",
                       help="Don't reset a tripped circuit breaker the next day")
        p.add_argument("--synthetic", action="store_true",
                       help="Made-up data with known phases; no database needed")
        if name == "run":
            p.add_argument("--csv", default=None,
                           help="Write every trade to this CSV file, e.g. results/trades.csv")
        else:
            p.add_argument("--train-days", type=int, default=365)
            p.add_argument("--test-days", type=int, default=90)

    args = ap.parse_args(argv)
    _quiet(args.verbose or args.cmd == "download")
    if args.cmd == "download":
        asyncio.run(cmd_download(args, settings))
    elif args.cmd == "run":
        asyncio.run(cmd_run(args, settings))
    else:
        asyncio.run(cmd_walk_forward(args, settings))


if __name__ == "__main__":
    main()
