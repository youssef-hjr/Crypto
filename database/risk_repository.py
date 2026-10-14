"""
PostgreSQL storage for the risk engine: persistent state, risk events,
every risk verdict, and equity snapshots.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from database.models import equity_snapshots, risk_decisions, risk_events, risk_state
from database.repository import MarketDataRepository
from models import RiskVerdict


class RiskRepository:
    """Implements `risk.state.RiskStateStore` on the shared database."""
    persistent = True

    def __init__(self, engine: AsyncEngine):
        self._engine = engine
        # Reuse the dialect-aware upsert from the market-data repository.
        self._insert = MarketDataRepository(engine)._insert

    async def get_state(self, key: str) -> Optional[dict]:
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                sa.select(risk_state.c.value).where(risk_state.c.key == key)
            )).first()
        return dict(row[0]) if row else None

    async def set_state(self, key: str, value: dict) -> None:
        now = datetime.now(timezone.utc)
        stmt = self._insert(risk_state).values(key=key, value=value, updated_at=now)
        stmt = stmt.on_conflict_do_update(
            index_elements=["key"], set_={"value": value, "updated_at": now}
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def log_event(self, kind: str, reason: str, details: Optional[dict] = None) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(risk_events.insert().values(
                occurred_at=datetime.now(timezone.utc),
                kind=kind, reason=reason, details=details or {},
            ))

    async def save_equity(self, at: datetime, equity: float, source: str) -> None:
        stmt = self._insert(equity_snapshots).values(
            captured_at=at, equity=equity, source=source
        ).on_conflict_do_nothing(index_elements=["captured_at"])
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def save_decision(self, verdict: RiskVerdict) -> None:
        p = verdict.original_proposal
        async with self._engine.begin() as conn:
            await conn.execute(risk_decisions.insert().values(
                decided_at=verdict.timestamp,
                symbol=p.symbol,
                direction=p.direction.value,
                decision=verdict.decision.value,
                approved_size_usd=verdict.approved_size_usd,
                reasons=verdict.rejection_reasons,
                proposal=p.model_dump(mode="json"),
                metrics=_jsonable(verdict.risk_metrics),
            ))


def _jsonable(d: dict) -> dict:
    """Risk metrics hold plain numbers, strings and lists; anything else is
    stored as text rather than failing the write."""
    return json.loads(json.dumps(d, default=str))
