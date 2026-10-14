"""
Paper trading: live-style cycles against a fake exchange. Every trade goes
through the Risk Manager, fills against the order book, survives a restart,
and no order is ever sent.
"""
from datetime import datetime, timedelta, timezone

import pytest

from config.settings import AppSettings, RiskSettings
from events import EventBus
from exchanges.base import ExchangeAdapter
from models import Candle, InstrumentRules, OrderBookLevel, OrderBookSnapshot, TickerInfo
from paper.engine import PaperTrader
from risk.circuit_breaker import CircuitBreaker
from risk.drawdown import DrawdownTracker
from risk.risk_manager import RiskManager
from risk.state import InMemoryRiskStore, PrefixedRiskStore
from tests.research_helpers import Scripted

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)
M = timedelta(minutes=1)


class FakeLive(ExchangeAdapter):
    """Flat market at `price`; minute bars can be overridden to move it."""

    def __init__(self, price=100.0):
        self.price = price
        self.now = T0 + 300 * H + M
        self.minute_bars: dict[datetime, tuple] = {}
        self.orders = 0

    @property
    def name(self):
        return "bybit"

    def _bar(self, t, minutes):
        o, h, l, c = self.minute_bars.get(t, (self.price,) * 4) if minutes == 1 else (
            self.price, self.price * 1.001, self.price * 0.999, self.price)
        return Candle(timestamp=t, open=o, high=h, low=l, close=c, volume=1_000,
                      turnover=1_000 * c, symbol="BTCUSDT")

    async def get_candles(self, symbol, interval="5", limit=200):
        step = timedelta(minutes=int(interval))
        last = T0 + ((self.now - T0) // step) * step - step  # Last CLOSED bar
        return [self._bar(last - k * step, int(interval)) for k in reversed(range(limit))]

    async def get_candle_history(self, symbol, interval, start, end):
        out, t = [], start
        while t + M <= end:
            out.append(self._bar(t, 1))
            t += M
        return out

    async def get_orderbook(self, symbol, depth=50):
        p = self.price
        return OrderBookSnapshot(
            symbol=symbol, timestamp=self.now,
            bids=[OrderBookLevel(price=p - 0.01 - i * 0.05, quantity=50) for i in range(5)],
            asks=[OrderBookLevel(price=p + 0.01 + i * 0.05, quantity=50) for i in range(5)])

    async def get_ticker(self, symbol):
        return TickerInfo(symbol=symbol, last_price=self.price, volume_24h=1e7,
                          turnover_24h=1e9, high_24h=self.price, low_24h=self.price)

    async def get_instrument_rules(self, symbol):
        return InstrumentRules(symbol=symbol, qty_step=0.001, min_qty=0.001, min_notional=5)

    async def get_funding_history(self, symbol, start, end):
        return []

    async def place_order(self, order):
        self.orders += 1
        raise AssertionError("Paper trading must never place an order")

    connect = disconnect = get_balance = get_equity = get_positions = None
    get_tickers = get_recent_trades = get_funding_rate = get_open_interest = None
    subscribe_orderbook = subscribe_trades = subscribe_ticker = None
    cancel_order = get_order_status = get_exchange_info = is_connected = None


def make_trader(ex, base=None, risk_settings=None, **strategy):
    base = base or InMemoryRiskStore()
    store = PrefixedRiskStore(base, "paper_trading:")
    settings = AppSettings()
    if risk_settings is not None:
        settings.risk = risk_settings
    bus = EventBus()
    risk = RiskManager(settings.risk, bus, breaker=CircuitBreaker(store, bus),
                       drawdown=DrawdownTracker(store, source="paper_trading"), store=store)
    strat = Scripted(bars=(299,), **strategy)
    return PaperTrader(ex, risk, [strat], ["BTCUSDT"], settings, store), base


@pytest.mark.asyncio
async def test_signal_opens_a_paper_position_through_the_risk_manager():
    ex = FakeLive()
    trader, _ = make_trader(ex)
    await trader.load()
    report = await trader.cycle(ex.now)
    assert report.opened == ["BTCUSDT"]
    assert [v.decision.value for v in report.verdicts] in (["approved"], ["reduced"])
    pos = trader.account.positions["BTCUSDT"]
    assert pos.entry_price == pytest.approx(100.01)  # Bought at the ask
    assert pos.risk_usd == pytest.approx(5.0, rel=0.02)  # 0.05% of 10,000
    assert trader.account.balance < 10_000  # Taker fee paid
    assert ex.orders == 0


@pytest.mark.asyncio
async def test_stop_on_a_minute_bar_closes_the_trade_and_counts_the_loss():
    ex = FakeLive()
    trader, _ = make_trader(ex)
    await trader.load()
    await trader.cycle(ex.now)
    stop = trader.account.positions["BTCUSDT"].stop
    bar_t = ex.now.replace(second=0, microsecond=0) + M
    ex.minute_bars[bar_t] = (100.0, 100.0, stop - 0.5, 99.0)
    ex.now = bar_t + 2 * M
    report = await trader.cycle(ex.now)
    assert [t.exit_reason for t in report.closed] == ["stop"]
    t = report.closed[0]
    assert t.r_multiple < -1.0  # Fees and slippage on top of the 1R loss
    assert "BTCUSDT" not in trader.account.positions
    assert trader.account.closed_trades == 1
    assert trader._risk._recent_losses == 1


@pytest.mark.asyncio
async def test_account_and_positions_survive_a_restart():
    ex = FakeLive()
    trader, base = make_trader(ex)
    await trader.load()
    await trader.cycle(ex.now)
    again, _ = make_trader(ex, base=base)
    acct = await again.load()
    assert set(acct.positions) == {"BTCUSDT"}
    assert acct.positions["BTCUSDT"] == trader.account.positions["BTCUSDT"]
    assert acct.balance == pytest.approx(trader.account.balance)


@pytest.mark.asyncio
async def test_old_signals_are_not_traded_after_a_restart():
    ex = FakeLive()
    ex.now = T0 + 300 * H + 30 * M  # The bar closed 30 minutes ago
    trader, _ = make_trader(ex)
    await trader.load()
    report = await trader.cycle(ex.now)
    assert report.opened == [] and not report.verdicts


@pytest.mark.asyncio
async def test_rejected_by_the_risk_manager_means_no_trade():
    ex = FakeLive()
    trader, _ = make_trader(ex, risk_settings=RiskSettings(min_opportunity_score=90))
    await trader.load()
    report = await trader.cycle(ex.now)
    assert report.opened == []
    assert "LOW_OPPORTUNITY_SCORE" in report.verdicts[0].rejection_reasons[0]


@pytest.mark.asyncio
async def test_paper_breaker_is_separate_from_the_real_one():
    base = InMemoryRiskStore()
    trader, _ = make_trader(FakeLive(), base=base)
    await trader.load()
    await trader._risk.breaker.trip("TEST")
    assert "paper_trading:circuit_breaker" in base.state
    live = CircuitBreaker(base)
    await live.load()
    assert not live.is_active


@pytest.mark.asyncio
async def test_downtime_is_replayed_minute_by_minute():
    ex = FakeLive()
    trader, _ = make_trader(ex)
    await trader.load()
    await trader.cycle(ex.now)
    stop = trader.account.positions["BTCUSDT"].stop
    hit = ex.now.replace(second=0, microsecond=0) + 30 * M
    ex.minute_bars[hit] = (100.0, 100.0, stop - 1, 99.0)
    ex.now = hit + 5 * H  # Down for five hours: more than 200 minutes
    ex.price = 100.0
    report = await trader.cycle(ex.now)
    assert [t.exit_reason for t in report.closed] == ["stop"]
    assert report.closed[0].closed_at == hit + M
