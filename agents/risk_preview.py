"""
Risk Preview.

There are no strategies yet (Milestone 3 onwards), so this turns the top
scanner opportunities into simple placeholder proposals and runs them
through the portfolio allocator and Risk Manager. Nothing is executed: the
output shows what the risk engine would approve, at what size, and why it
rejects the rest.

Placeholder proposal: entry at the current mid, stop PREVIEW_STOP_ATR x ATR
away, size equal to the full 0.05% risk budget. Real strategies replace this.

With an advisor (main.py --ai, Milestone 4), Claude reviews the best
candidates first, and only those it answers TRADE go to the Risk Manager.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from agents.market_scanner import CANDLE_INTERVAL, MarketScanner
from ai.briefing import Candidate, MarketSnapshot, build_brief, risk_state
from config.settings import AppSettings
from exchanges.base import ExchangeAdapter
from models import (
    MarketContext, OpportunityScore, RiskVerdict, Side, TechnicalFeatures, TradeProposal,
)
from portfolio.allocation import PortfolioAllocator
from portfolio.correlation import CorrelationMatrix
from portfolio.portfolio import PortfolioService, next_risk_budget
from risk.risk_manager import RiskManager

logger = structlog.get_logger()

PREVIEW_STOP_ATR = 2.0
PREVIEW_MAX_CANDIDATES = 10


def build_preview_proposal(
    opp: OpportunityScore,
    tech: Optional[TechnicalFeatures],
    mid: Optional[float],
    equity: float,
    risk_pct: float,
) -> Optional[TradeProposal]:
    if tech is None or not tech.atr or not mid:
        return None
    distance = PREVIEW_STOP_ATR * tech.atr
    stop = mid - distance if opp.direction == Side.BUY else mid + distance
    size = next_risk_budget(equity, risk_pct) / (distance / mid)
    return TradeProposal(
        symbol=opp.symbol,
        direction=opp.direction,
        strategy="risk_preview",
        entry_price=mid,
        stop_loss=stop,
        position_size_usd=size,
        confidence=opp.confidence,
        market_regime=opp.market_regime,
        opportunity_score=opp.overall_score,
        reasoning=f"Preview: stop {PREVIEW_STOP_ATR}x ATR from mid",
    )


async def run_risk_preview(
    exchange: ExchangeAdapter,
    scanner: MarketScanner,
    risk: RiskManager,
    portfolio_service: PortfolioService,
    settings: AppSettings,
    opportunities: list[OpportunityScore],
    advisor=None,  # ai.ClaudeAdvisor, or None for the deterministic preview only
) -> tuple[list[RiskVerdict], list]:
    """(Risk Manager verdicts, Claude reviews). Reviews are empty without an advisor."""
    candidates = [o for o in opportunities
                  if o.overall_score >= settings.risk.min_opportunity_score]
    candidates = candidates[:PREVIEW_MAX_CANDIDATES]
    if not candidates:
        logger.info("risk_preview_skipped", msg="No opportunity above the minimum score")
        return [], []

    reviews = []
    if advisor is not None:
        candidates, reviews = await claude_review(
            exchange, scanner, risk, portfolio_service, settings, candidates, advisor)
        if not candidates:
            return [], reviews

    # Fresh order books and order rules for the candidates only
    markets: dict[str, MarketContext] = {}
    for opp in candidates:
        try:
            book = await exchange.get_orderbook(opp.symbol, depth=settings.orderbook_depth)
            rules = await exchange.get_instrument_rules(opp.symbol)
        except Exception as e:
            logger.error("risk_preview_market_error", symbol=opp.symbol, error=str(e))
            book, rules = None, None
        ticker = scanner.ticker(opp.symbol)
        markets[opp.symbol] = MarketContext(
            symbol=opp.symbol, orderbook=book, instrument=rules,
            turnover_24h=ticker.turnover_24h if ticker else None)

    # Read the account last, so it's as fresh as the order books
    portfolio = await portfolio_service.snapshot()
    proposals: list[TradeProposal] = []
    for opp in candidates:
        book = markets[opp.symbol].orderbook
        proposal = build_preview_proposal(
            opp, scanner.technical(opp.symbol), book.mid_price if book else None,
            portfolio.equity, settings.risk.risk_per_trade_pct)
        if proposal is not None:
            proposals.append(proposal)

    # Correlations across the candidates and anything already held
    candles = {o.symbol: scanner.candles(o.symbol) for o in candidates}
    for pos in portfolio.positions:
        if pos.symbol not in candles:
            candles[pos.symbol] = scanner.candles(pos.symbol) or await exchange.get_candles(
                pos.symbol, interval=CANDLE_INTERVAL, limit=200)
    correlations = CorrelationMatrix.from_candles(
        candles, min_samples=settings.risk.min_correlation_samples)

    verdicts = await PortfolioAllocator(risk).allocate(
        proposals, portfolio, markets, correlations)
    if advisor is not None:
        await advisor.record_verdicts(reviews, verdicts)
    return verdicts, reviews


def scanner_candidate(opp: OpportunityScore, scanner: MarketScanner) -> Optional[Candidate]:
    """The preview's placeholder trade, as a candidate for Claude."""
    tech, candles = scanner.technical(opp.symbol), scanner.candles(opp.symbol)
    if tech is None or not tech.atr or not candles:
        return None
    entry = candles[-1].close
    distance = PREVIEW_STOP_ATR * tech.atr
    return Candidate(
        symbol=opp.symbol, direction=opp.direction, strategy="scanner", entry=entry,
        stop=entry - distance if opp.direction == Side.BUY else entry + distance,
        target=None, score=opp.overall_score, confidence=opp.confidence,
        expected_r=opp.expected_r,
        reason=(f"Scanner score {opp.overall_score:.0f}: trend {opp.trend_score:.0f}, "
                f"momentum {opp.momentum_score:.0f}, order flow {opp.order_flow_score:.0f}, "
                f"liquidity {opp.liquidity_score:.0f}, volume {opp.volume_score:.0f}, "
                f"volatility {opp.volatility_score:.0f}, derivatives "
                f"{opp.derivatives_score:.0f}. Placeholder stop {PREVIEW_STOP_ATR} ATR, "
                "no target (exits are not defined in the preview)."),
        regime=opp.market_regime.value, bar_time=candles[-1].timestamp,
        bar_minutes=int(CANDLE_INTERVAL))


async def claude_review(exchange, scanner, risk, portfolio_service, settings,
                        candidates: list[OpportunityScore], advisor):
    """Ask Claude about the best candidates. Returns (the opportunities it
    answered TRADE for, all reviews)."""
    cands = [c for c in (scanner_candidate(o, scanner) for o in candidates) if c]
    if not cands:
        return [], []
    portfolio = await portfolio_service.snapshot()
    correlations = CorrelationMatrix.from_candles(
        {c.symbol: scanner.candles(c.symbol) for c in cands}
        | {p.symbol: scanner.candles(p.symbol) for p in portfolio.positions},
        min_samples=settings.risk.min_correlation_samples)
    rstate = risk_state(risk, settings.risk)
    briefs = {}
    for c in advisor.top(cands):
        market = MarketSnapshot(
            technical=scanner.technical(c.symbol), candles=scanner.candles(c.symbol),
            order_flow=scanner.order_flow(c.symbol), derivatives=scanner.derivatives(c.symbol),
            ticker=scanner.ticker(c.symbol))
        try:
            market.orderbook = await exchange.get_orderbook(c.symbol, depth=settings.orderbook_depth)
        except Exception as e:
            logger.warning("ai_brief_data_missing", symbol=c.symbol, error=str(e))
        briefs[c.key] = build_brief(c, market, portfolio, rstate, correlations, cands)
    reviews = await advisor.review(cands, briefs, risk.is_circuit_breaker_active,
                                   now=datetime.now(timezone.utc))
    keep = {r.candidate.symbol for r in reviews if r.approved}
    return [o for o in candidates if o.symbol in keep], reviews
