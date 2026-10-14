"""
Tests for market-data storage.
Runs on in-memory SQLite by default. Set TEST_DATABASE_URL to a PostgreSQL
URL (postgresql+asyncpg://...) to run the same tests against Postgres.
"""
import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
import sqlalchemy as sa

from database import DataRecorder, MarketDataRepository
from database.models import candles, feature_snapshots, metadata, opportunities
from events import Event, EventBus, EventType
from models import (
    Candle, DerivativesFeatures, MarketRegime, OpportunityScore,
    OrderFlowFeatures, Side, TechnicalFeatures,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def repo():
    url = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    r = MarketDataRepository.from_url(url)
    async with r.engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await r.create_schema()
    yield r
    async with r.engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await r.close()


async def count(repo, table):
    async with repo.engine.connect() as conn:
        return (await conn.execute(sa.select(sa.func.count()).select_from(table))).scalar()


def make_candles(n, start=T0):
    return [Candle(timestamp=start + timedelta(minutes=5 * i), open=1, high=2,
                   low=0.5, close=1.5, volume=10, turnover=15, symbol="BTCUSDT")
            for i in range(n)]


@pytest.mark.asyncio
async def test_overlapping_candle_fetches_do_not_duplicate(repo):
    await repo.save_candles("bybit", "BTCUSDT", "5", make_candles(10))
    # Next scan fetches a window overlapping the first by 8 candles
    await repo.save_candles("bybit", "BTCUSDT", "5",
                            make_candles(10, start=T0 + timedelta(minutes=10)))
    assert await count(repo, candles) == 12


@pytest.mark.asyncio
async def test_recorder_stores_scanner_events(repo):
    bus = EventBus()
    DataRecorder(repo, "bybit").attach(bus)

    await bus.publish(Event(EventType.MARKET_DATA_UPDATED, {
        "symbol": "BTCUSDT", "interval": "5", "candles": make_candles(3)}))
    await bus.publish(Event(EventType.FEATURES_CALCULATED, {
        "symbol": "BTCUSDT",
        "technical": TechnicalFeatures(symbol="BTCUSDT", timestamp=T0, rsi=55.0),
        "order_flow": OrderFlowFeatures(symbol="BTCUSDT", timestamp=T0),
        "derivatives": DerivativesFeatures(symbol="BTCUSDT", timestamp=T0, funding_rate=0.0001),
        "regime": MarketRegime.WEAK_BULLISH,
    }))
    await bus.publish(Event(EventType.OPPORTUNITY_SCORED, OpportunityScore(
        symbol="BTCUSDT", overall_score=72.5, direction=Side.SELL, timestamp=T0)))

    assert await count(repo, candles) == 3
    assert await count(repo, feature_snapshots) == 1
    assert await count(repo, opportunities) == 1

    async with repo.engine.connect() as conn:
        row = (await conn.execute(sa.select(opportunities))).mappings().one()
        feat = (await conn.execute(sa.select(feature_snapshots))).mappings().one()
    assert row["direction"] == "sell"
    assert row["scores"]["overall_score"] == 72.5
    assert feat["technical"]["rsi"] == 55.0
    assert feat["market_regime"] == "weak_bullish"
