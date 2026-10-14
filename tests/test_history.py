"""History download (paging, incremental), research storage and loading
history back for a backtest. SQLite by default; set TEST_DATABASE_URL to
run against PostgreSQL."""
import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
import sqlalchemy as sa

from config.settings import BybitSettings
from database import MarketDataRepository, ResearchRepository
from database.models import metadata, research_runs, trade_journal
from events import EventBus
from exchanges.base import ExchangeAdapter
from exchanges.bybit import BybitAdapter
from models import Candle, InstrumentRules
from research.history import HistoryDownloader, load_history

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
HOUR_MS = 3_600_000


@pytest_asyncio.fixture
async def repos():
    url = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    r = MarketDataRepository.from_url(url)
    async with r.engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await r.create_schema()
    yield r, ResearchRepository(r.engine)
    async with r.engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await r.close()


@pytest.mark.asyncio
async def test_bybit_candle_history_pages_backwards(monkeypatch):
    monkeypatch.setattr("exchanges.bybit.HISTORY_PAUSE_S", 0)
    adapter = BybitAdapter(BybitSettings(), EventBus())
    start_ms = int(T0.timestamp() * 1000)
    all_rows = [[str(start_ms + i * HOUR_MS), "1", "2", "0.5", "1.5", "10", "15"]
                for i in range(2500)]
    calls = []

    async def fake_request(method, path, params=None, signed=False):
        calls.append(dict(params))
        rows = [r for r in all_rows if params["start"] <= int(r[0]) <= params["end"]]
        return {"retCode": 0, "result": {"list": list(reversed(rows))[:params["limit"]]}}

    monkeypatch.setattr(adapter, "_request", fake_request)
    out = await adapter.get_candle_history("BTCUSDT", "60", T0, T0 + timedelta(hours=2500))
    assert len(out) == 2500 and len(calls) == 3
    assert out[0].timestamp == T0 and out[-1].timestamp == T0 + timedelta(hours=2499)
    assert all(a.timestamp < b.timestamp for a, b in zip(out, out[1:]))


@pytest.mark.asyncio
async def test_bybit_history_errors_raise(monkeypatch):
    adapter = BybitAdapter(BybitSettings(), EventBus())

    async def failing(method, path, params=None, signed=False):
        return {"retCode": 10001, "retMsg": "bad", "result": {}}

    monkeypatch.setattr(adapter, "_request", failing)
    with pytest.raises(Exception):
        await adapter.get_funding_history("BTCUSDT", T0, T0 + timedelta(days=1))


class FakeHistoryExchange(ExchangeAdapter):
    """Only the history methods are used by the downloader."""

    def __init__(self, first: datetime, last: datetime):
        self.first, self.last = first, last
        self.requested: list[tuple[datetime, datetime]] = []

    @property
    def name(self):
        return "bybit"

    async def get_candle_history(self, symbol, interval, start, end):
        self.requested.append((start, end))
        out, t = [], max(start, self.first)
        while t < min(end, self.last):
            out.append(Candle(timestamp=t, open=1, high=2, low=0.5, close=1.5,
                              volume=10, turnover=15, symbol=symbol))
            t += timedelta(hours=1)
        return out

    async def get_funding_history(self, symbol, start, end):
        return [(T0 + timedelta(hours=8 * i), 0.0001) for i in range(6)]

    async def get_instrument_rules(self, symbol):
        return InstrumentRules(symbol=symbol, qty_step=0.001, min_qty=0.001, min_notional=5)

    # Unused by the downloader
    connect = disconnect = get_balance = get_equity = get_positions = None
    get_ticker = get_tickers = get_orderbook = get_recent_trades = get_candles = None
    get_funding_rate = get_open_interest = subscribe_orderbook = subscribe_trades = None
    subscribe_ticker = place_order = cancel_order = get_order_status = None
    get_exchange_info = is_connected = None


@pytest.mark.asyncio
async def test_download_is_incremental_and_loads_back(repos):
    market, research = repos
    ex = FakeHistoryExchange(first=T0 + timedelta(days=10), last=T0 + timedelta(days=60))
    dl = HistoryDownloader(ex, market, research)
    s = await dl.download("BTCUSDT", "60", T0, T0 + timedelta(days=50))
    assert s["new_candles"] == 40 * 24  # Nothing before the contract existed
    assert s["first"] == T0 + timedelta(days=10)

    ex.requested.clear()
    s = await dl.download("BTCUSDT", "60", T0, T0 + timedelta(days=55))
    assert s["new_candles"] == 5 * 24  # Only the missing end
    assert all(lo >= T0 + timedelta(days=50) or hi <= T0 + timedelta(days=10)
               for lo, hi in ex.requested)

    data = await load_history(research, "bybit", ["BTCUSDT", "ETHUSDT"], "60",
                              T0 + timedelta(days=45), T0 + timedelta(days=55))
    assert list(data) == ["BTCUSDT"]  # No history for ETH: skipped with a warning
    d = data["BTCUSDT"]
    assert d.candles.index[0] == T0 + timedelta(days=15)  # 30 days of warm-up included
    assert d.candles.index[-1] == T0 + timedelta(days=55) - timedelta(hours=1)
    assert d.rules.qty_step == 0.001
    assert d.funding[T0 + timedelta(hours=8)] == 0.0001


@pytest.mark.asyncio
async def test_runs_and_journal_are_saved(repos):
    from research.backtester import Backtester, BacktestConfig
    from research.synthetic import synthetic_history
    from strategies import build_strategies
    from config.settings import RiskSettings

    market, research = repos
    res = await Backtester(BacktestConfig(), RiskSettings()).run(
        build_strategies("breakout"), synthetic_history(symbols=["BTCUSDT"]))
    m = res.metrics()
    run_id = await research.save_run("backtest", res.strategies, res.symbols, "60", None,
                                     None, res.config.as_dict(), m, res.by_phase())
    await research.save_trades(run_id, "backtest", "bybit", res.trades)
    async with market.engine.connect() as conn:
        n = (await conn.execute(sa.select(sa.func.count()).select_from(trade_journal))).scalar()
        row = (await conn.execute(sa.select(research_runs))).first()
    assert n == len(res.trades) > 0
    assert row.metrics["trades"] == len(res.trades)
    assert row.config["costs"]["taker_fee"] == 0.00055
