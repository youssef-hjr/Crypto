"""
Tests for correlation, exposure, accounting and allocation.
"""
import numpy as np
import pytest

from config.settings import RiskSettings
from events import EventBus
from models import Position, PositionSide, RiskDecision, Side
from portfolio import exposure as ex
from portfolio.allocation import PortfolioAllocator
from portfolio.correlation import CorrelationMatrix
from portfolio.portfolio import DailyAccount, net_realized_pnl, next_risk_budget
from risk.risk_manager import RiskManager
from tests.helpers import candles_from_returns, market, portfolio, proposal


# ── Correlation ────────────────────────────────────────────────────────

def test_correlation_of_related_and_unrelated_series():
    rng = np.random.default_rng(0)
    beta = rng.normal(0, 0.01, 300)
    corr = CorrelationMatrix.from_candles({
        "BTC": candles_from_returns(beta),
        "ETH": candles_from_returns(beta * 1.2 + rng.normal(0, 0.002, 300)),
        "INV": candles_from_returns(-beta),
        "RND": candles_from_returns(rng.normal(0, 0.01, 300)),
    })
    assert corr.get("BTC", "ETH") > 0.9
    assert corr.get("ETH", "BTC") == corr.get("BTC", "ETH")
    assert corr.get("BTC", "INV") < -0.99
    assert abs(corr.get("BTC", "RND")) < 0.2
    assert corr.get("BTC", "BTC") == 1.0


def test_correlation_unknown_with_too_little_history():
    rng = np.random.default_rng(0)
    corr = CorrelationMatrix.from_candles({
        "A": candles_from_returns(rng.normal(0, 0.01, 50)),
        "B": candles_from_returns(rng.normal(0, 0.01, 50)),
    }, min_samples=100)
    assert corr.get("A", "B") is None
    assert corr.samples("A", "B") == 50


def test_correlation_aligns_on_candle_time():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 200)
    a = candles_from_returns(r)
    b = candles_from_returns(r)[50:]  # B listed later: only the overlap counts
    corr = CorrelationMatrix.from_candles({"A": a, "B": b})
    assert corr.get("A", "B") == pytest.approx(1.0)
    assert corr.samples("A", "B") == 150


# ── Exposure ───────────────────────────────────────────────────────────

def test_exposure_long_and_short():
    items = ex.exposures([
        Position(symbol="BTC", side=PositionSide.LONG, entry_price=100, quantity=2,
                 stop_loss=95, mark_price=110),
        Position(symbol="ETH", side=PositionSide.SHORT, entry_price=50, quantity=4,
                 stop_loss=52),
    ])
    assert ex.gross_exposure(items) == pytest.approx(220 + 200)
    assert ex.net_exposure(items) == pytest.approx(220 - 200)
    assert ex.open_risk(items) == pytest.approx(2 * 15 + 4 * 2)


def test_short_stop_below_price_risks_nothing():
    p = Position(symbol="ETH", side=PositionSide.SHORT, entry_price=50, quantity=4,
                 stop_loss=48, mark_price=45)
    assert p.risk_amount == pytest.approx(12.0)
    p.mark_price = 49  # Price went past the stop: nothing left to risk
    assert p.risk_amount == 0.0


# ── Accounting (PDF sections 23, 26, 27) ───────────────────────────────

@pytest.mark.parametrize("equity,budget", [(1_000, 0.50), (1_500, 0.75), (800, 0.40)])
def test_risk_budget_follows_current_equity(equity, budget):
    assert next_risk_budget(equity, 0.0005) == pytest.approx(budget)


def test_net_realized_pnl_after_fees_and_funding():
    assert net_realized_pnl(100, fees=3, funding=2) == pytest.approx(95)
    assert net_realized_pnl(100, fees=3, funding=-1) == pytest.approx(98)  # funding received


def test_daily_account():
    day = DailyAccount(starting_equity=10_000)
    day.record_close(gross_pnl=50, fees=2, funding=1)
    day.record_close(gross_pnl=-20, fees=2, funding=0)
    day.unrealized_pnl = 15
    assert day.net_equity == pytest.approx(10_000 + 30 + 15 - 4 - 1)
    assert day.daily_return_pct == pytest.approx(40 / 10_000)


# ── Allocation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allocation_ranks_and_stops_at_correlated_cap():
    """Six correlated longs: the five best are approved, the sixth hits the
    correlated-risk cap even though each looked fine on its own."""
    rm = RiskManager(RiskSettings(), EventBus())
    symbols = [f"C{i}USDT" for i in range(6)]
    proposals = [proposal(symbol=s, entry=100, stop=99, score=90 - i)
                 for i, s in enumerate(symbols)]
    markets = {s: market(s, mid=100) for s in symbols}

    verdicts = await PortfolioAllocator(rm).allocate(
        list(reversed(proposals)), portfolio(), markets, correlations=None)

    assert [v.original_proposal.symbol for v in verdicts] == symbols  # best first
    decisions = [v.decision for v in verdicts]
    assert decisions[:5] == [RiskDecision.APPROVED] * 5
    assert decisions[5] == RiskDecision.REJECTED
    assert any(r.startswith("CORRELATED_RISK_EXCEEDED") for r in verdicts[5].rejection_reasons)


@pytest.mark.asyncio
async def test_allocation_does_not_change_the_real_portfolio():
    rm = RiskManager(RiskSettings(), EventBus())
    pf = portfolio()
    await PortfolioAllocator(rm).allocate([proposal()], pf, {"BTCUSDT": market()})
    assert pf.positions == []


@pytest.mark.asyncio
async def test_allocation_lets_hedged_directions_through():
    rm = RiskManager(RiskSettings(), EventBus())
    longs = [proposal(symbol=f"L{i}USDT", entry=100, stop=99, score=90) for i in range(5)]
    short = proposal(symbol="S0USDT", entry=100, stop=101, score=80, direction=Side.SELL)
    markets = {p.symbol: market(p.symbol, mid=100) for p in longs + [short]}
    verdicts = await PortfolioAllocator(rm).allocate(longs + [short], portfolio(), markets)
    assert all(v.decision == RiskDecision.APPROVED for v in verdicts)
