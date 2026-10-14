"""
Crypto Agent — Main Entry Point

MILESTONE 1: Market Intelligence Engine
  - Scan the Bybit universe, compute features, score opportunities, store it all

MILESTONE 2: Risk + Portfolio Engine
  - Read equity and positions (Bybit Demo account, or paper equity without a key)
  - Compute drawdown, exposure, leverage, correlation and liquidity limits
  - Run the top opportunities through the allocator and Risk Manager
  - Show what would be approved, at what size, and why the rest is rejected
  - NO orders are placed

MILESTONE 3: Backtesting + Paper Trading (separate entry points)
  - python backtest.py ...     history download, backtests, walk-forward
  - python paper_trade.py ...  strategies on live data, simulated account

MILESTONE 4: Claude integration (--ai, here and in paper_trade.py)
  - Claude reads a structured brief of each top candidate and answers
    TRADE or NO TRADE, with reasoning, confidence, expected R, invalidation,
    regime and portfolio impact
  - Only TRADE answers go on to the Risk Manager, which still decides
  - Claude never gets an exchange key; execution stays disabled

Usage:
  python main.py                        # Scan once, then risk preview
  python main.py --loop                 # Continuous loop
  python main.py --ai                   # Claude reviews the candidates first
  python main.py --ai-status            # Claude's recent decisions and cost
  python main.py --no-db                # Dry run without storing anything
  python main.py --reset-breaker "note" # Reset a tripped circuit breaker
"""
from __future__ import annotations

import asyncio
import getpass
import sys
from dataclasses import dataclass
from typing import Optional

import structlog
from dotenv import load_dotenv

from agents.market_scanner import CANDLE_INTERVAL, MarketScanner
from agents.order_flow import OrderFlowAgent
from agents.quant import QuantEngine
from agents.risk_preview import run_risk_preview
from ai.factory import build_advisor
from database import AIDecisionRepository, DataRecorder, MarketDataRepository, RiskRepository
from config.settings import AccountEnv, AppSettings, get_settings
from events import EventBus
from exchanges.bybit import BybitAdapter
from models import RiskDecision, RiskVerdict
from portfolio.portfolio import PortfolioService
from risk.circuit_breaker import CircuitBreaker
from risk.drawdown import DrawdownTracker
from risk.risk_manager import RiskManager
from risk.state import InMemoryRiskStore

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
)
logger = structlog.get_logger()


async def open_repository(settings: AppSettings) -> Optional[MarketDataRepository]:
    """Connect to the database and create tables. Fails loudly if storage is
    enabled but unreachable, rather than silently scanning without storing."""
    if not settings.database.enabled:
        logger.warning("storage_disabled", msg="Nothing will be stored")
        return None
    repo = MarketDataRepository.from_url(settings.database.url)
    try:
        await repo.create_schema()
    except Exception as e:
        await repo.close()
        raise RuntimeError(
            f"Cannot reach the database ({e}). Start PostgreSQL "
            "(docker compose -f docker/docker-compose.yml up -d), "
            "set DATABASE_URL, or run with --no-db."
        ) from e
    logger.info("storage_ready")
    return repo


@dataclass
class Runtime:
    """Long-lived pieces shared by every scan cycle."""
    settings: AppSettings
    event_bus: EventBus
    repo: Optional[MarketDataRepository]
    risk: RiskManager
    has_credentials: bool
    advisor: Optional[object] = None  # ai.ClaudeAdvisor with --ai

    async def close(self) -> None:
        if self.repo is not None:
            await self.repo.close()


async def build_runtime(settings: AppSettings) -> Runtime:
    repo = await open_repository(settings)
    event_bus = EventBus()
    has_credentials = bool(settings.bybit.api_key and settings.bybit.api_secret)
    if repo is not None:
        store = RiskRepository(repo.engine)
        DataRecorder(repo, "bybit", store).attach(event_bus)
    else:
        store = InMemoryRiskStore()
        logger.warning("risk_state_not_persisted",
                       msg="Circuit breaker and drawdown reset when this process exits")
    risk = RiskManager(
        settings.risk, event_bus,
        breaker=CircuitBreaker(store, event_bus),
        drawdown=DrawdownTracker(store, source="exchange" if has_credentials else "paper"),
        store=store,
    )
    await risk.load()
    if risk.is_circuit_breaker_active:
        logger.critical("CIRCUIT BREAKER IS ACTIVE: every trade will be rejected",
                        reason=risk.breaker.reason,
                        reset='python main.py --reset-breaker "what you checked"')
    advisor = build_advisor(settings, "preview", repo.engine if repo else None)
    if advisor is not None:
        await advisor.load()
        logger.info("claude_enabled", model=settings.ai.model, effort=settings.ai.effort,
                    calls_today=advisor.calls_today,
                    max_calls_per_day=settings.ai.max_calls_per_day)
    return Runtime(settings, event_bus, repo, risk, has_credentials, advisor)


async def run_scan(rt: Runtime) -> None:
    """Run a single scan cycle, then the risk preview."""
    settings = rt.settings
    exchange = BybitAdapter(settings.bybit, rt.event_bus)
    quant = QuantEngine(min_candles=50, bar_minutes=int(CANDLE_INTERVAL))
    order_flow = OrderFlowAgent()
    scanner = MarketScanner(exchange, rt.event_bus, quant, order_flow, settings)
    portfolio_service = PortfolioService(exchange, rt.has_credentials, settings.paper_equity)

    try:
        await exchange.connect()

        logger.info("=== SCAN STARTING ===",
                     exchange=exchange.name,
                     account_env=settings.bybit.account_env.value,
                     portfolio=portfolio_service.source,
                     trading_enabled=settings.trading_enabled,
                     execution_enabled=settings.execution_enabled)

        # Run the scan
        opportunities = await scanner.scan()

        # Display results
        logger.info("=== TOP OPPORTUNITIES ===")
        for i, opp in enumerate(opportunities[:15], 1):
            logger.info(
                f"#{i}",
                symbol=opp.symbol,
                score=opp.overall_score,
                direction=opp.direction.value,
                regime=opp.market_regime.value,
                trend=opp.trend_score,
                momentum=opp.momentum_score,
                order_flow=opp.order_flow_score,
                liquidity=opp.liquidity_score,
                volume=opp.volume_score,
                volatility=opp.volatility_score,
                derivatives=opp.derivatives_score,
            )

        logger.info("=== SCAN COMPLETE ===",
                     total_opportunities=len(opportunities),
                     above_threshold=sum(
                         1 for o in opportunities
                         if o.overall_score >= settings.risk.min_opportunity_score
                     ))

        # Risk preview: what the risk engine would do with the top ideas
        try:
            verdicts, reviews = await run_risk_preview(
                exchange, scanner, rt.risk, portfolio_service, settings, opportunities,
                advisor=rt.advisor)
        except Exception as e:
            logger.error("risk_preview_error", error=str(e))
            verdicts, reviews = [], []
        show_reviews(reviews)
        show_verdicts(verdicts, rt.risk)

    finally:
        await exchange.disconnect()


def show_reviews(reviews) -> None:
    if not reviews:
        return
    logger.info("=== CLAUDE REVIEW (TRADE goes on to the Risk Manager) ===")
    for r in reviews:
        d = r.decision
        logger.info(d.decision.value, symbol=r.candidate.symbol,
                    side=r.candidate.direction.value, confidence=d.confidence,
                    expected_r=d.expected_r, cached=r.cached or None, error=r.error,
                    reasoning=d.reasoning, invalidation=d.invalidation or None)


async def show_ai_status(settings: AppSettings) -> None:
    from datetime import datetime, timedelta, timezone
    repo = await open_repository(settings)
    if repo is None:
        sys.exit("--ai-status needs the database (Claude's decisions are stored there).")
    try:
        journal = AIDecisionRepository(repo.engine)
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        for label, since in (("today (UTC)", today), ("last 7 days", now - timedelta(days=7))):
            s = await journal.summary(since)
            print(f"Claude, {label}: {s['TRADE']} TRADE, {s['NO_TRADE']} NO TRADE "
                  f"({s['errors']} of them errors or skips), ~${s['cost_usd']:.2f}, "
                  f"{s['input_tokens']:,} tokens in, {s['output_tokens']:,} out")
        print(f"Daily limit: {settings.ai.max_calls_per_day} calls (AI_MAX_CALLS_PER_DAY)")
        print("\nLatest decisions:")
        for r in await journal.recent(15):
            why = r["error"] or ((r["response"] or {}).get("reasoning") or "")
            risk = f" -> risk {r['risk_decision']}" if r["risk_decision"] else ""
            conf = f" {r['confidence']:.2f}" if r["confidence"] is not None else ""
            print(f"  {r['created_at']:%m-%d %H:%M} {r['mode']:7} {r['symbol']:12} "
                  f"{r['direction']:4} {r['strategy']:14} {r['decision']:8}{conf}{risk}")
            print(f"      {why[:160]}")
    finally:
        await repo.close()


def show_verdicts(verdicts: list[RiskVerdict], risk: RiskManager) -> None:
    if not verdicts:
        return
    dd = risk.drawdown.state
    logger.info("=== RISK PREVIEW (no orders placed) ===",
                equity=dd.equity if dd else None,
                daily_drawdown=f"{dd.daily_drawdown_pct:.2%}" if dd else None,
                total_drawdown=f"{dd.total_drawdown_pct:.2%}" if dd else None,
                circuit_breaker=risk.breaker.reason if risk.is_circuit_breaker_active else "off")
    for v in verdicts:
        p, m = v.original_proposal, v.risk_metrics
        if v.decision == RiskDecision.REJECTED:
            logger.info("REJECTED", symbol=p.symbol, direction=p.direction.value,
                        score=p.opportunity_score, reasons=v.rejection_reasons)
        else:
            logger.info(v.decision.value.upper(), symbol=p.symbol,
                        direction=p.direction.value, score=p.opportunity_score,
                        size_usd=round(v.approved_size_usd, 2), qty=v.approved_quantity,
                        entry=p.entry_price, stop=round(p.stop_loss, 6),
                        risk_usd=round(m["trade_risk_usd"], 2),
                        portfolio_risk=f"{m['portfolio_risk_pct']:.3%}",
                        correlated_with=m["correlated_with"])


async def run_once(settings: AppSettings) -> None:
    rt = await build_runtime(settings)
    try:
        await run_scan(rt)
    finally:
        await rt.close()


async def run_loop(settings: AppSettings) -> None:
    """Continuous scanning loop."""
    logger.info("Starting continuous scan loop",
                 interval=settings.scan_interval_seconds)
    rt = await build_runtime(settings)

    try:
        while True:
            try:
                await run_scan(rt)
            except Exception as e:
                logger.error("scan_cycle_error", error=str(e))

            logger.info(f"Next scan in {settings.scan_interval_seconds}s...")
            await asyncio.sleep(settings.scan_interval_seconds)
    finally:
        await rt.close()


async def reset_breaker(settings: AppSettings, note: str) -> None:
    rt = await build_runtime(settings)
    try:
        if not rt.risk.is_circuit_breaker_active:
            print("The circuit breaker is not active. Nothing to reset.")
            return
        reason = rt.risk.breaker.reason
        await rt.risk.reset_circuit_breaker(getpass.getuser(), note)
        print(f"Circuit breaker reset (it was tripped by {reason}).")
    finally:
        await rt.close()


def main():
    load_dotenv()  # API keys from the project's .env file (never committed)
    settings = get_settings()
    if "--no-db" in sys.argv:
        settings.database.enabled = False
    if "--ai" in sys.argv:
        settings.ai.enabled = True

    # Safety checks
    assert not settings.execution_enabled, \
        "Execution must be DISABLED until Milestone 5"
    assert settings.bybit.account_env != AccountEnv.MAINNET, \
        "Real-money account is not allowed yet (use demo or testnet)"

    if "--reset-breaker" in sys.argv:
        i = sys.argv.index("--reset-breaker")
        note = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
        if not note.strip() or note.startswith("--"):
            sys.exit('Say what you checked: python main.py --reset-breaker "note"')
        asyncio.run(reset_breaker(settings, note))
    elif "--ai-status" in sys.argv:
        asyncio.run(show_ai_status(settings))
    elif "--loop" in sys.argv:
        asyncio.run(run_loop(settings))
    else:
        asyncio.run(run_once(settings))


if __name__ == "__main__":
    main()
