"""
Persistence of risk state: the circuit breaker and drawdown high-water
marks survive a restart. Runs on in-memory SQLite by default; set
TEST_DATABASE_URL to run against PostgreSQL.
"""
import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
import sqlalchemy as sa

from config.settings import RiskSettings
from database import DataRecorder, MarketDataRepository, RiskRepository
from database.models import equity_snapshots, metadata, risk_decisions, risk_events
from events import EventBus
from risk.circuit_breaker import CircuitBreaker
from risk.drawdown import DrawdownTracker
from risk.risk_manager import RiskManager
from tests.helpers import market, portfolio, proposal


@pytest_asyncio.fixture
async def engine():
    url = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    repo = MarketDataRepository.from_url(url)
    async with repo.engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await repo.create_schema()
    yield repo.engine
    async with repo.engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await repo.close()


async def rows(engine, table):
    async with engine.connect() as conn:
        return (await conn.execute(sa.select(table))).mappings().all()


@pytest.mark.asyncio
async def test_breaker_survives_restart(engine):
    store = RiskRepository(engine)
    first = CircuitBreaker(store)
    await first.load()
    await first.trip("DAILY_DRAWDOWN_LIMIT", {"daily_drawdown_pct": 0.06})

    restarted = CircuitBreaker(RiskRepository(engine))
    await restarted.load()
    assert restarted.is_active
    assert restarted.reason == "DAILY_DRAWDOWN_LIMIT"

    await restarted.reset("esteban", "checked positions")
    again = CircuitBreaker(RiskRepository(engine))
    await again.load()
    assert not again.is_active

    kinds = [r["kind"] for r in await rows(engine, risk_events)]
    assert kinds == ["breaker_tripped", "breaker_reset"]


@pytest.mark.asyncio
async def test_breaker_fails_closed_when_state_unreadable():
    class BrokenStore:
        async def get_state(self, key):
            raise ConnectionError("db down")

    breaker = CircuitBreaker(BrokenStore())
    await breaker.load()
    assert breaker.is_active
    assert breaker.reason == "STATE_UNAVAILABLE"


@pytest.mark.asyncio
async def test_drawdown_survives_restart(engine):
    t0 = datetime(2026, 10, 14, 9, tzinfo=timezone.utc)
    tracker = DrawdownTracker(RiskRepository(engine))
    await tracker.update(10_000, t0)
    await tracker.update(10_500, t0 + timedelta(hours=1))

    restarted = DrawdownTracker(RiskRepository(engine))
    await restarted.load()
    s = await restarted.update(9_975, t0 + timedelta(hours=2))
    assert s.peak_equity == 10_500
    assert s.daily_drawdown_pct == pytest.approx(0.05)
    assert len(await rows(engine, equity_snapshots)) == 3


@pytest.mark.asyncio
async def test_new_utc_day_restarts_daily_drawdown_only():
    t0 = datetime(2026, 10, 14, 22, tzinfo=timezone.utc)
    tracker = DrawdownTracker()
    await tracker.update(10_000, t0)
    await tracker.update(9_000, t0 + timedelta(hours=1))
    s = await tracker.update(9_000, t0 + timedelta(hours=3))  # next UTC day
    assert s.daily_drawdown_pct == 0.0
    assert s.day_start_equity == 9_000
    assert s.total_drawdown_pct == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_paper_and_exchange_peaks_are_separate():
    from risk.state import InMemoryRiskStore
    store = InMemoryRiskStore()
    await DrawdownTracker(store, source="paper").update(10_000)
    real = DrawdownTracker(store, source="exchange")
    await real.load()
    s = await real.update(5_000)
    assert s.total_drawdown_pct == 0.0


@pytest.mark.asyncio
async def test_loss_streak_survives_restart(engine):
    rm = RiskManager(RiskSettings(), EventBus(), store=RiskRepository(engine))
    await rm.load()
    for pnl in (-1, -1, -1):
        await rm.record_trade_result(pnl)

    restarted = RiskManager(RiskSettings(), EventBus(), store=RiskRepository(engine))
    await restarted.load()
    v = await restarted.evaluate(proposal(), portfolio(), market())
    assert v.risk_metrics["loss_streak_reduction"] == 0.5


@pytest.mark.asyncio
async def test_every_verdict_is_stored(engine):
    bus = EventBus()
    store = RiskRepository(engine)
    DataRecorder(MarketDataRepository(engine), "bybit", store).attach(bus)
    rm = RiskManager(RiskSettings(), bus, store=store)
    await rm.evaluate(proposal(), portfolio(), market())
    await rm.evaluate(proposal(stop=0), portfolio(), market())

    stored = await rows(engine, risk_decisions)
    assert [r["decision"] for r in stored] == ["approved", "rejected"]
    assert stored[1]["reasons"] == ["NO_STOP_LOSS"]
    assert stored[0]["metrics"]["trade_risk_usd"] == pytest.approx(5.0)
    assert stored[0]["proposal"]["symbol"] == "BTCUSDT"
