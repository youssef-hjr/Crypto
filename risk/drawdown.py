"""
Drawdown Tracker.

Computes drawdown from the equity the system actually observes, instead of
trusting numbers handed in with the portfolio. High-water marks are
persisted, so a restart doesn't wipe out an ongoing drawdown.

  daily drawdown = drop from the highest equity seen since 00:00 UTC
                   (the day starts at the first equity reading of the day)
  total drawdown = drop from the highest equity ever seen

Deposits and withdrawals look like gains and losses here. Reset the
tracker (`reset_peaks`) after moving money in or out of the account.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import structlog

from risk.state import InMemoryRiskStore, RiskStateStore

logger = structlog.get_logger()


@dataclass
class DrawdownState:
    equity: float
    peak_equity: float
    day: date
    day_start_equity: float
    day_peak_equity: float

    @property
    def daily_drawdown_pct(self) -> float:
        if self.day_peak_equity <= 0:
            return 0.0
        return max(0.0, (self.day_peak_equity - self.equity) / self.day_peak_equity)

    @property
    def total_drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.equity) / self.peak_equity)

    @property
    def daily_pnl(self) -> float:
        return self.equity - self.day_start_equity


class DrawdownTracker:

    def __init__(self, store: Optional[RiskStateStore] = None, source: str = "exchange"):
        # Paper and exchange equity keep separate high-water marks, so a
        # paper preview can never fake a drawdown on the real account.
        self._store = store or InMemoryRiskStore()
        self._source = source
        self._key = f"drawdown:{source}"
        self._state: Optional[DrawdownState] = None

    async def load(self) -> None:
        saved = await self._store.get_state(self._key)
        if saved:
            self._state = DrawdownState(
                equity=saved["equity"],
                peak_equity=saved["peak_equity"],
                day=date.fromisoformat(saved["day"]),
                day_start_equity=saved["day_start_equity"],
                day_peak_equity=saved["day_peak_equity"],
            )

    async def update(self, equity: float, at: Optional[datetime] = None) -> DrawdownState:
        at = at or datetime.now(timezone.utc)
        today = at.astimezone(timezone.utc).date()
        s = self._state
        if s is None:
            s = DrawdownState(equity, equity, today, equity, equity)
        else:
            if today > s.day:  # New UTC day: start from the first reading
                s.day = today
                s.day_start_equity = equity
                s.day_peak_equity = equity
            s.equity = equity
            s.peak_equity = max(s.peak_equity, equity)
            s.day_peak_equity = max(s.day_peak_equity, equity)
        self._state = s
        await self._store.set_state(self._key, {
            "equity": s.equity,
            "peak_equity": s.peak_equity,
            "day": s.day.isoformat(),
            "day_start_equity": s.day_start_equity,
            "day_peak_equity": s.day_peak_equity,
        })
        await self._store.save_equity(at, equity, self._source)
        return s

    async def reset_peaks(self, equity: float, operator: str, note: str) -> None:
        """Start the high-water marks again from `equity`, for example after
        a deposit or withdrawal. Logged as a risk event."""
        await self._store.log_event("drawdown_reset", "manual",
                                    {"operator": operator, "note": note, "equity": equity})
        self._state = None
        await self.update(equity)

    @property
    def state(self) -> Optional[DrawdownState]:
        return self._state
