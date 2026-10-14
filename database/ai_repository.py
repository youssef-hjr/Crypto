"""
Storage for Claude's decisions: one row per question asked, updated with
the Risk Manager's verdict once there is one.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from database.models import ai_decisions
from database.research_repository import _json


class AIDecisionRepository:

    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    async def save(self, row: dict) -> int:
        values = dict(row)
        for k in ("candidate", "brief", "response", "risk_reasons"):
            if k in values:
                values[k] = _json(values[k])
        async with self._engine.begin() as conn:
            result = await conn.execute(
                ai_decisions.insert().values(**values).returning(ai_decisions.c.id))
            return int(result.scalar_one())

    async def set_risk_verdict(self, row_id: int, decision: str, reasons: list[str]) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(ai_decisions.update().where(ai_decisions.c.id == row_id)
                               .values(risk_decision=decision, risk_reasons=_json(reasons)))

    async def calls_since(self, since: datetime) -> int:
        """Questions actually sent to the API since `since` (errors count too:
        they may have been billed)."""
        a = ai_decisions.c
        async with self._engine.connect() as conn:
            return int((await conn.execute(
                sa.select(sa.func.count()).where(a.created_at >= since, a.model.is_not(None))
            )).scalar_one())

    async def summary(self, since: datetime) -> dict:
        a = ai_decisions.c
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                sa.select(a.decision, a.error.is_not(None), sa.func.count(),
                          sa.func.coalesce(sa.func.sum(a.cost_usd), 0.0),
                          sa.func.coalesce(sa.func.sum(a.input_tokens), 0),
                          sa.func.coalesce(sa.func.sum(a.output_tokens), 0))
                .where(a.created_at >= since).group_by(a.decision, a.error.is_not(None))
            )).all()
        out = {"TRADE": 0, "NO_TRADE": 0, "errors": 0, "cost_usd": 0.0,
               "input_tokens": 0, "output_tokens": 0}
        for decision, failed, n, cost, tin, tout in rows:
            out[decision] = out.get(decision, 0) + n
            out["errors"] += n if failed else 0
            out["cost_usd"] += float(cost)
            out["input_tokens"] += int(tin)
            out["output_tokens"] += int(tout)
        return out

    async def recent(self, limit: int = 10, mode: Optional[str] = None) -> list[dict]:
        a = ai_decisions.c
        q = (sa.select(a.created_at, a.mode, a.symbol, a.strategy, a.direction, a.decision,
                       a.confidence, a.expected_r, a.error, a.risk_decision, a.response)
             .order_by(a.created_at.desc()).limit(limit))
        if mode:
            q = q.where(a.mode == mode)
        async with self._engine.connect() as conn:
            return [dict(r._mapping) for r in (await conn.execute(q)).all()]
