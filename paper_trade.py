"""
Milestone 3: paper trading on live Bybit market data. No real money, no
orders: the account is simulated, the prices and order books are real.

  python paper_trade.py                       # Run until Ctrl+C
  python paper_trade.py --strategy trend --interval 240 --params fast=10,stop_atr=2.0
  python paper_trade.py --once                # One cycle, then exit
  python paper_trade.py --status              # Equity, positions, journal stats
  python paper_trade.py --close-all           # Close every paper position now
  python paper_trade.py --reset-breaker "what you checked"
  python paper_trade.py --reset-account       # Start a fresh paper account

The account, open positions, circuit breaker and loss streak are kept in
PostgreSQL, so stopping and starting again carries on where it left off.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import sys

import structlog

from config.settings import AccountEnv, get_settings

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT"
PREFIX = "paper_trading:"

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)
logger = structlog.get_logger()


async def build(args, settings):
    from database import MarketDataRepository, ResearchRepository, RiskRepository
    from events import EventBus, EventType
    from exchanges.bybit import BybitAdapter
    from paper.engine import PaperTrader
    from research.simulator import CostModel
    from risk.circuit_breaker import CircuitBreaker
    from risk.drawdown import DrawdownTracker
    from risk.risk_manager import RiskManager
    from risk.state import InMemoryRiskStore, PrefixedRiskStore
    from strategies import build_strategies, parse_params

    try:
        strategies = build_strategies(args.strategy, parse_params(args.params))
    except ValueError as e:
        sys.exit(str(e))
    bus = EventBus()
    repo = journal = None
    if args.no_db:
        base = InMemoryRiskStore()
        logger.warning("paper_state_not_persisted",
                       msg="--no-db: the paper account is lost when this process exits")
    else:
        repo = MarketDataRepository.from_url(settings.database.url)
        try:
            await repo.create_schema()
        except Exception as e:
            await repo.close()
            sys.exit(f"Cannot reach the database ({e}). Start PostgreSQL with "
                     "'docker compose -f docker/docker-compose.yml up -d', or use --no-db.")
        risk_repo = RiskRepository(repo.engine)
        base = risk_repo
        journal = ResearchRepository(repo.engine)

        async def record(event):  # Every paper verdict goes to risk_decisions too
            await risk_repo.save_decision(event.data)
        bus.subscribe(EventType.RISK_APPROVED, record)
        bus.subscribe(EventType.RISK_REJECTED, record)

    store = PrefixedRiskStore(base, PREFIX)
    risk = RiskManager(settings.risk, bus, breaker=CircuitBreaker(store, bus),
                       drawdown=DrawdownTracker(store, source="paper_trading"), store=store)
    # Market data only. The adapter's account side is never called here.
    exchange = BybitAdapter(settings.bybit, bus)
    trader = PaperTrader(
        exchange, risk, strategies,
        [s.strip() for s in args.symbols.split(",")], settings, store,
        interval=args.interval, costs=CostModel(), journal=journal,
        initial_equity=args.equity)
    return trader, exchange, risk, store, repo


async def run(args) -> None:
    settings = get_settings()
    trader, exchange, risk, store, repo = await build(args, settings)
    await exchange.connect()
    try:
        if args.reset_account:
            answer = input("Delete the paper account and its open positions? Type yes: ")
            if answer.strip().lower() != "yes":
                print("Nothing changed.")
                return
            await store.set_state("account", {})
            await store.set_state("drawdown:paper_trading", {})  # New high-water marks
            await store.set_state("loss_streak", {"count": 0})
            print("Paper account reset. The journal keeps the old trades;"
                  " a tripped paper breaker still needs --reset-breaker.")
            return

        acct = await trader.load()
        if args.reset_breaker is not None:
            if not args.reset_breaker.strip():
                sys.exit('Say what you checked: python paper_trade.py --reset-breaker "note"')
            if not risk.is_circuit_breaker_active:
                print("The paper circuit breaker is not active. Nothing to reset.")
                return
            await risk.reset_circuit_breaker(getpass.getuser(), args.reset_breaker)
            print("Paper circuit breaker reset.")
            return
        if args.close_all:
            closed = await trader.close_all()
            print(f"Closed {len(closed)} paper positions.")
            return
        if args.status:
            show_status(trader, risk)
            return

        logger.info("=== PAPER TRADING (no orders are placed) ===",
                    strategies=args.strategy, params=args.params or "defaults",
                    symbols=args.symbols, interval=args.interval,
                    equity=round(acct.equity(), 2), open_positions=len(acct.positions),
                    breaker=risk.breaker.reason if risk.is_circuit_breaker_active else "off")
        while True:
            report = await trader.cycle()
            for v in report.verdicts:
                p = v.original_proposal
                if v.decision.value == "rejected":
                    logger.info("REJECTED", symbol=p.symbol, strategy=p.strategy,
                                reasons=v.rejection_reasons)
            logger.info("paper_cycle", **trader.stats(),
                        opened=report.opened, closed=[t.symbol for t in report.closed],
                        errors=report.errors or None)
            if args.once:
                break
            await asyncio.sleep(args.poll)
    finally:
        await exchange.disconnect()
        if repo is not None:
            await repo.close()


def show_status(trader, risk) -> None:
    acct = trader.account
    print(f"Paper account since {acct.started_at}")
    print(f"  equity {acct.equity():,.2f}  balance {acct.balance:,.2f}"
          f"  (started at {acct.initial_equity:,.2f})")
    print(f"  closed trades {acct.closed_trades}, realized net {acct.realized_net:+,.2f}")
    print(f"  circuit breaker: {risk.breaker.reason if risk.is_circuit_breaker_active else 'off'}")
    for sym, p in acct.positions.items():
        print(f"  OPEN {sym} {p.side.value} {p.strategy} qty {p.quantity} entry {p.entry_price:.6g}"
              f" stop {p.stop:.6g} target {p.target if p.target is None else f'{p.target:.6g}'}"
              f" unrealized {p.unrealized():+.2f} bars held {p.bars_held}")
    if acct.run_id:
        print(f"  trades are in trade_journal with run_id = {acct.run_id}")


def main() -> None:
    settings = get_settings()
    assert not settings.execution_enabled, "Execution must be DISABLED until Milestone 5"
    assert settings.bybit.account_env != AccountEnv.MAINNET, \
        "Real-money account is not allowed yet (use demo or testnet)"

    ap = argparse.ArgumentParser(description="Paper trading on live Bybit data. No orders.")
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--params", default=None,
                    help="Override strategy settings, e.g. fast=10,stop_atr=2.0"
                         " (needs a single --strategy)")
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--interval", default="60", help="Strategy candle minutes")
    ap.add_argument("--poll", type=int, default=30, help="Seconds between cycles")
    ap.add_argument("--equity", type=float, default=settings.paper_equity,
                    help="Starting equity of a NEW paper account")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--close-all", action="store_true")
    ap.add_argument("--reset-breaker", default=None, metavar="NOTE")
    ap.add_argument("--reset-account", action="store_true")
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped. The paper account is saved; run again to carry on.")


if __name__ == "__main__":
    main()
