"""
Portfolio state and accounting.

`PortfolioService` reads the current equity and open positions from the
exchange account (or uses a paper equity when no API key is set).
`DailyAccount` and `net_realized_pnl` implement the PDF's profit-locking and
daily accounting rules: P&L is counted after fees and funding, and the next
risk budget always comes from current equity, never the starting capital.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from exchanges.base import ExchangeAdapter
from models import PortfolioState


def net_realized_pnl(gross_pnl: float, fees: float, funding: float) -> float:
    """Realized P&L - fees - funding paid. Funding received is a negative cost."""
    return gross_pnl - fees - funding


def next_risk_budget(equity: float, risk_pct: float) -> float:
    """Risk budget for the next trade: current equity x risk percentage."""
    return max(0.0, equity) * risk_pct


@dataclass
class DailyAccount:
    """End-of-cycle accounting, as in the PDF:
    starting equity + realized + unrealized - fees - funding = net equity."""
    starting_equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees: float = 0.0
    funding: float = 0.0

    def record_close(self, gross_pnl: float, fees: float, funding: float) -> float:
        self.realized_pnl += gross_pnl
        self.fees += fees
        self.funding += funding
        return net_realized_pnl(gross_pnl, fees, funding)

    @property
    def net_equity(self) -> float:
        return (self.starting_equity + self.realized_pnl + self.unrealized_pnl
                - self.fees - self.funding)

    @property
    def daily_return_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.net_equity - self.starting_equity) / self.starting_equity


class PortfolioService:
    """Builds a PortfolioState from the exchange account."""

    def __init__(self, exchange: ExchangeAdapter, has_credentials: bool, paper_equity: float):
        self._exchange = exchange
        self._has_credentials = has_credentials
        self._paper_equity = paper_equity

    @property
    def source(self) -> str:
        return "exchange" if self._has_credentials else "paper"

    async def snapshot(self) -> PortfolioState:
        """Current equity and positions. Raises if the account can't be read:
        an API error must never look like an empty account."""
        now = datetime.now(timezone.utc)
        if not self._has_credentials:
            return PortfolioState(
                equity=self._paper_equity, available_balance=self._paper_equity,
                unrealized_pnl=0.0, realized_pnl_today=0.0, source="paper", timestamp=now)
        equity = await self._exchange.get_equity()
        positions = await self._exchange.get_positions()
        return PortfolioState(
            equity=equity,
            available_balance=equity,
            unrealized_pnl=sum(p.unrealized_pnl for p in positions),
            realized_pnl_today=0.0,
            positions=positions,
            source="exchange",
            timestamp=now,
        )
