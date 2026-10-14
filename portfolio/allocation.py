"""
Portfolio Allocation.

Ranks trade proposals and runs them through the Risk Manager best first.
Each approved trade is added to a working copy of the portfolio before the
next one is checked, so five correlated longs can't each pass on their own
as if the others didn't exist.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from models import (
    MarketContext, PortfolioState, Position, PositionSide, RiskDecision,
    RiskVerdict, Side, TradeProposal,
)
from portfolio.correlation import CorrelationMatrix
from risk.risk_manager import RiskManager


class PortfolioAllocator:

    def __init__(self, risk_manager: RiskManager):
        self._risk = risk_manager

    async def allocate(
        self,
        proposals: list[TradeProposal],
        portfolio: PortfolioState,
        markets: dict[str, MarketContext],
        correlations: Optional[CorrelationMatrix] = None,
        now: Optional[datetime] = None,
    ) -> list[RiskVerdict]:
        """Verdicts in rank order (highest opportunity score first). `now` is
        the time the data is judged at; backtests pass the simulated time."""
        working = portfolio.model_copy(deep=True)
        verdicts = []
        for p in sorted(proposals, key=lambda p: p.opportunity_score, reverse=True):
            v = await self._risk.evaluate(p, working, markets.get(p.symbol), correlations, now)
            verdicts.append(v)
            if v.decision in (RiskDecision.APPROVED, RiskDecision.REDUCED):
                working.positions.append(Position(
                    symbol=p.symbol,
                    side=PositionSide.LONG if p.direction == Side.BUY else PositionSide.SHORT,
                    entry_price=p.entry_price,
                    quantity=v.approved_quantity,
                    leverage=v.approved_leverage or 1.0,
                    stop_loss=p.stop_loss,
                    strategy=p.strategy,
                ))
        return verdicts
