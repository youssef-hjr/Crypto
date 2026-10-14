"""
Risk Preview.

There are no strategies yet (Milestone 3 onwards), so this turns the top
scanner opportunities into simple placeholder proposals and runs them
through the portfolio allocator and Risk Manager. Nothing is executed: the
output shows what the risk engine would approve, at what size, and why it
rejects the rest.

Placeholder proposal: entry at the current mid, stop PREVIEW_STOP_ATR x ATR
away, size equal to the full 0.05% risk budget. Real strategies replace this.
"""
from __future__ import annotations

from typing import Optional

import structlog

from agents.market_scanner import CANDLE_INTERVAL, MarketScanner
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
) -> list[RiskVerdict]:
    candidates = [o for o in opportunities
                  if o.overall_score >= settings.risk.min_opportunity_score]
    candidates = candidates[:PREVIEW_MAX_CANDIDATES]
    if not candidates:
        logger.info("risk_preview_skipped", msg="No opportunity above the minimum score")
        return []

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

    return await PortfolioAllocator(risk).allocate(
        proposals, portfolio, markets, correlations)
