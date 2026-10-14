"""
Async persistence for market data and scanner output.
Works with PostgreSQL in production (asyncpg) and SQLite in tests.
"""
from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from database.models import candles, feature_snapshots, metadata, opportunities
from models import (
    Candle, DerivativesFeatures, MarketRegime, OpportunityScore,
    OrderFlowFeatures, TechnicalFeatures,
)

logger = structlog.get_logger()


class MarketDataRepository:
    """Writes candles, feature snapshots and scored opportunities."""

    def __init__(self, engine: AsyncEngine):
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> "MarketDataRepository":
        return cls(create_async_engine(url, pool_pre_ping=True))

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def create_schema(self) -> None:
        """Create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    def _insert(self, table):
        dialect = self._engine.dialect.name
        if dialect == "postgresql":
            return postgresql.insert(table)
        if dialect == "sqlite":
            return sqlite.insert(table)
        raise NotImplementedError(f"Unsupported database dialect: {dialect}")

    async def save_candles(
        self, exchange: str, symbol: str, interval: str, rows: list[Candle]
    ) -> None:
        """Insert closed candles. Candles already stored are left untouched,
        so overlapping fetches on every scan are harmless."""
        if not rows:
            return
        values = [{
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "open_time": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "turnover": c.turnover,
        } for c in rows]
        stmt = self._insert(candles).values(values).on_conflict_do_nothing(
            index_elements=["exchange", "symbol", "interval", "open_time"]
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def save_features(
        self,
        exchange: str,
        technical: TechnicalFeatures,
        order_flow: OrderFlowFeatures,
        derivatives: DerivativesFeatures,
        regime: MarketRegime,
    ) -> None:
        stmt = feature_snapshots.insert().values(
            exchange=exchange,
            symbol=technical.symbol,
            captured_at=technical.timestamp,
            market_regime=regime.value,
            technical=_dump(technical),
            order_flow=_dump(order_flow),
            derivatives=_dump(derivatives),
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def save_opportunity(self, exchange: str, opp: OpportunityScore) -> None:
        stmt = opportunities.insert().values(
            exchange=exchange,
            symbol=opp.symbol,
            scored_at=opp.timestamp,
            direction=opp.direction.value,
            overall_score=opp.overall_score,
            market_regime=opp.market_regime.value,
            scores=_dump(opp),
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)


def _dump(model: Any) -> dict:
    return model.model_dump(mode="json")
