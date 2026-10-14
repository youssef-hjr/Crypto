"""
Tests for the Risk Manager.
Validates position sizing, veto authority, anti-martingale, circuit breakers,
and the Milestone 2 portfolio, correlation, liquidity and drawdown limits.
"""
from datetime import datetime, timedelta, timezone

import pytest

from config.settings import RiskSettings
from events import EventBus
from models import Position, PositionSide, RiskDecision, Side
from portfolio.correlation import CorrelationMatrix
from risk.risk_manager import RiskManager
from tests.helpers import candles_from_returns, market, portfolio as make_portfolio, proposal

make_proposal = proposal


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def risk_settings():
    return RiskSettings(
        risk_per_trade_pct=0.0005,  # 0.05%
        max_portfolio_risk_pct=0.02,
        max_daily_drawdown_pct=0.05,
        max_total_drawdown_pct=0.15,
        max_single_asset_pct=0.20,
        max_leverage=5.0,
        max_positions=10,
        min_opportunity_score=70.0,
    )


@pytest.fixture
def risk_manager(risk_settings, event_bus):
    return RiskManager(risk_settings, event_bus)


@pytest.fixture
def portfolio():
    return make_portfolio()


async def evaluate(rm, p, pf, mkt="default", corr=None):
    if mkt == "default":
        mkt = market(p.symbol, mid=p.entry_price if p.entry_price > 0 else 50_000.0)
    return await rm.evaluate(p, pf, mkt, corr)


def pos(symbol, side=PositionSide.LONG, entry=100.0, qty=1.0, stop=None, mark=None):
    return Position(symbol=symbol, side=side, entry_price=entry, quantity=qty,
                    stop_loss=stop, mark_price=mark)


# ── Position Sizing ────────────────────────────────────────────────────

class TestPositionSizing:

    def test_position_size_from_stop_distance(self, risk_manager):
        """Position size = risk_budget / stop_distance, NOT arbitrary % of equity."""
        # Equity $10,000, risk 0.05% = $5 budget
        # Stop distance 1% -> size = $5 / 0.01 = $500
        size = risk_manager.calculate_position_size(10_000, 0.01)
        assert size == pytest.approx(500.0)

    def test_position_size_tight_stop(self, risk_manager):
        # Stop 0.5% -> size = $5 / 0.005 = $1000
        size = risk_manager.calculate_position_size(10_000, 0.005)
        assert size == pytest.approx(1000.0)

    def test_position_size_wide_stop(self, risk_manager):
        # Stop 5% -> size = $5 / 0.05 = $100
        size = risk_manager.calculate_position_size(10_000, 0.05)
        assert size == pytest.approx(100.0)

    def test_position_size_zero_stop(self, risk_manager):
        size = risk_manager.calculate_position_size(10_000, 0.0)
        assert size == 0.0

    def test_compounding_up(self, risk_manager):
        """After profits, risk budget should grow (controlled compounding)."""
        size_before = risk_manager.calculate_position_size(10_000, 0.01)
        size_after = risk_manager.calculate_position_size(15_000, 0.01)
        assert size_after > size_before
        assert size_after == pytest.approx(750.0)

    def test_compounding_down(self, risk_manager):
        """After losses, risk budget should shrink."""
        size_before = risk_manager.calculate_position_size(10_000, 0.01)
        size_after = risk_manager.calculate_position_size(8_000, 0.01)
        assert size_after < size_before
        assert size_after == pytest.approx(400.0)


# ── Risk Manager Verdicts ──────────────────────────────────────────────

class TestRiskVerdicts:

    @pytest.mark.asyncio
    async def test_approve_valid_trade(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(), portfolio)
        assert verdict.decision == RiskDecision.APPROVED
        assert verdict.approved_quantity == pytest.approx(0.01)
        assert verdict.risk_metrics["trade_risk_usd"] == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_reject_no_stop_loss(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(stop=0), portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert "NO_STOP_LOSS" in verdict.rejection_reasons

    @pytest.mark.asyncio
    async def test_reject_excessive_leverage(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(leverage=20.0), portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert any(r.startswith("LEVERAGE_EXCEEDED") for r in verdict.rejection_reasons)

    @pytest.mark.asyncio
    async def test_reject_low_opportunity_score(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(score=40.0), portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert any("LOW_OPPORTUNITY_SCORE" in r for r in verdict.rejection_reasons)

    @pytest.mark.asyncio
    async def test_reject_max_positions(self, risk_manager, portfolio):
        portfolio.positions = [pos(f"ASSET{i}USDT", stop=99.99) for i in range(10)]
        verdict = await evaluate(risk_manager, make_proposal(), portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert "MAX_POSITIONS_REACHED" in verdict.rejection_reasons

    @pytest.mark.asyncio
    async def test_reduce_oversized_position(self, risk_manager, portfolio):
        # Stop distance = 1%, risk budget = $5, max size = $500
        # Proposal asks for $2000 -> should be reduced
        verdict = await evaluate(risk_manager, make_proposal(size=2000), portfolio)
        assert verdict.decision == RiskDecision.REDUCED
        assert verdict.approved_size_usd == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_reject_stop_too_wide(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(stop=40_000), portfolio)
        assert "STOP_DISTANCE_TOO_WIDE" in verdict.rejection_reasons


# ── Drawdown is computed, not trusted ──────────────────────────────────

class TestDrawdown:

    @pytest.mark.asyncio
    async def test_daily_drawdown_trips_breaker(self, risk_manager):
        t0 = datetime.now(timezone.utc)
        await risk_manager.observe(make_portfolio(10_000, timestamp=t0 - timedelta(minutes=1)))
        verdict = await evaluate(risk_manager, make_proposal(),
                                 make_portfolio(9_400, timestamp=t0))  # -6% > 5%
        assert verdict.decision == RiskDecision.REJECTED
        assert risk_manager.is_circuit_breaker_active
        assert risk_manager.breaker.reason == "DAILY_DRAWDOWN_LIMIT"

    @pytest.mark.asyncio
    async def test_small_drawdown_still_trades(self, risk_manager):
        t0 = datetime.now(timezone.utc)
        await risk_manager.observe(make_portfolio(10_000, timestamp=t0 - timedelta(minutes=1)))
        verdict = await evaluate(risk_manager, make_proposal(),
                                 make_portfolio(9_900, timestamp=t0))
        assert verdict.decision != RiskDecision.REJECTED
        assert verdict.risk_metrics["daily_drawdown_pct"] == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_no_room_left_today_for_all_stops(self, risk_settings, event_bus):
        # 4.95% down today: one more 0.05% trade on top of 0.02% open risk
        # would end the day past 5% if every stop were hit.
        rm = RiskManager(risk_settings, event_bus)
        t0 = datetime.now(timezone.utc)
        await rm.observe(make_portfolio(10_000, timestamp=t0 - timedelta(minutes=1)))
        pf = make_portfolio(9_505, timestamp=t0,
                            positions=[pos("ETHUSDT", entry=2000, qty=0.1, stop=1980)])
        verdict = await evaluate(rm, make_proposal(), pf)
        assert "DAILY_LOSS_HEADROOM" in verdict.rejection_reasons
        assert not rm.is_circuit_breaker_active


# ── Anti-Martingale ────────────────────────────────────────────────────

class TestAntiMartingale:

    @pytest.mark.asyncio
    async def test_reduce_after_consecutive_losses(self, risk_manager, portfolio):
        """After 3+ consecutive losses, size MUST decrease."""
        for pnl in (-10, -8, -12):
            await risk_manager.record_trade_result(pnl)
        verdict = await evaluate(risk_manager, make_proposal(size=500), portfolio)
        assert verdict.decision == RiskDecision.REDUCED
        assert verdict.approved_size_usd == pytest.approx(250.0)

    @pytest.mark.asyncio
    async def test_win_resets_loss_counter(self, risk_manager, portfolio):
        """A winning trade resets the consecutive loss counter."""
        for pnl in (-10, -8, 20):
            await risk_manager.record_trade_result(pnl)
        verdict = await evaluate(risk_manager, make_proposal(size=500), portfolio)
        assert verdict.decision == RiskDecision.APPROVED


# ── Circuit Breaker ────────────────────────────────────────────────────

class TestCircuitBreaker:

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_all_trades(self, risk_manager, portfolio):
        await risk_manager.breaker.trip("test")
        verdict = await evaluate(risk_manager, make_proposal(), portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert "CIRCUIT_BREAKER_ACTIVE" in verdict.rejection_reasons

    @pytest.mark.asyncio
    async def test_circuit_breaker_manual_reset(self, risk_manager, portfolio):
        await risk_manager.breaker.trip("test")
        await risk_manager.reset_circuit_breaker("esteban", "checked the account")
        verdict = await evaluate(risk_manager, make_proposal(), portfolio)
        assert verdict.decision == RiskDecision.APPROVED

    @pytest.mark.asyncio
    async def test_reset_needs_operator_and_note(self, risk_manager):
        await risk_manager.breaker.trip("test")
        with pytest.raises(ValueError):
            await risk_manager.reset_circuit_breaker("esteban", "  ")
        assert risk_manager.is_circuit_breaker_active


# ── Price sanity ───────────────────────────────────────────────────────

class TestPriceSanity:

    @pytest.mark.asyncio
    async def test_reject_long_with_stop_above_entry(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(entry=100.0, stop=101.0, size=400.0),
                                 portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert "STOP_ON_WRONG_SIDE" in verdict.rejection_reasons

    @pytest.mark.asyncio
    async def test_reject_short_with_stop_below_entry(self, risk_manager, portfolio):
        p = make_proposal(entry=100.0, stop=99.0, size=400.0, direction=Side.SELL)
        verdict = await evaluate(risk_manager, p, portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert "STOP_ON_WRONG_SIDE" in verdict.rejection_reasons

    @pytest.mark.asyncio
    async def test_approve_valid_short(self, risk_manager, portfolio):
        p = make_proposal(entry=50000.0, stop=50500.0, size=500.0, direction=Side.SELL)
        verdict = await evaluate(risk_manager, p, portfolio)
        assert verdict.decision == RiskDecision.APPROVED

    @pytest.mark.asyncio
    async def test_reject_zero_entry_price(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(entry=0.0, stop=0.0), portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert "INVALID_ENTRY_PRICE" in verdict.rejection_reasons

    @pytest.mark.asyncio
    async def test_reject_target_on_wrong_side(self, risk_manager, portfolio):
        p = make_proposal()
        p.take_profit = 49000.0
        verdict = await evaluate(risk_manager, p, portfolio)
        assert verdict.decision == RiskDecision.REJECTED
        assert any(r.startswith("TARGET_ON_WRONG_SIDE") for r in verdict.rejection_reasons)


# ── Market data and liquidity ──────────────────────────────────────────

class TestMarketData:

    @pytest.mark.asyncio
    async def test_reject_without_market_data(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(), portfolio, mkt=None)
        assert verdict.rejection_reasons == ["MARKET_DATA_MISSING"]

    @pytest.mark.asyncio
    async def test_reject_stale_orderbook(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(), portfolio,
                                 mkt=market(age_s=60))
        assert verdict.rejection_reasons[0].startswith("STALE_MARKET_DATA")

    @pytest.mark.asyncio
    async def test_reject_stale_portfolio(self, risk_manager):
        old = make_portfolio(timestamp=datetime.now(timezone.utc) - timedelta(minutes=5))
        verdict = await evaluate(risk_manager, make_proposal(), old)
        assert verdict.rejection_reasons == ["STALE_PORTFOLIO_STATE"]

    @pytest.mark.asyncio
    async def test_reject_crossed_book(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(), portfolio,
                                 mkt=market(half_spread=-5, step=1))
        assert verdict.rejection_reasons == ["ORDERBOOK_CROSSED"]

    @pytest.mark.asyncio
    async def test_reject_wide_spread(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(), portfolio,
                                 mkt=market(half_spread=50))  # 20 bps
        assert any(r.startswith("SPREAD_TOO_WIDE") for r in verdict.rejection_reasons)

    @pytest.mark.asyncio
    async def test_reject_entry_far_from_market(self, risk_manager, portfolio):
        p = make_proposal(entry=50_000, stop=49_500)
        verdict = await evaluate(risk_manager, p, portfolio, mkt=market(mid=51_000))
        assert any(r.startswith("ENTRY_FAR_FROM_MARKET") for r in verdict.rejection_reasons)

    @pytest.mark.asyncio
    async def test_thin_book_caps_size(self, risk_manager, portfolio):
        # $50k x 0.01 x 5 levels = $2,500 of asks; 10% of that is $250
        verdict = await evaluate(risk_manager, make_proposal(size=500), portfolio,
                                 mkt=market(qty=0.01))
        assert verdict.decision == RiskDecision.REDUCED
        assert verdict.approved_size_usd <= 250.0
        assert verdict.risk_metrics["liquidity_capped"]

    @pytest.mark.asyncio
    async def test_low_turnover_caps_size(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(size=500), portfolio,
                                 mkt=market(turnover=100_000))  # 0.1% = $100
        assert verdict.approved_size_usd <= 100.0

    @pytest.mark.asyncio
    async def test_reject_below_exchange_minimum(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(size=500), portfolio,
                                 mkt=market(qty_step=0.1, min_qty=0.1))  # 0.1 BTC = $5,000
        assert "BELOW_MIN_ORDER_SIZE" in verdict.rejection_reasons

    @pytest.mark.asyncio
    async def test_quantity_rounded_down_to_step(self, risk_manager, portfolio):
        verdict = await evaluate(risk_manager, make_proposal(size=520), portfolio,
                                 mkt=market(qty_step=0.003))
        assert verdict.approved_quantity == pytest.approx(0.009)


# ── Portfolio limits ───────────────────────────────────────────────────

class TestPortfolioLimits:

    @pytest.mark.asyncio
    async def test_reject_when_open_position_has_no_stop(self, risk_manager):
        pf = make_portfolio(positions=[pos("ETHUSDT", entry=2000, qty=0.1)])
        verdict = await evaluate(risk_manager, make_proposal(), pf)
        assert "UNPROTECTED_POSITION: ETHUSDT" in verdict.rejection_reasons

    @pytest.mark.asyncio
    async def test_portfolio_risk_computed_from_stops(self, risk_manager):
        # $199 already at risk across stops; +$5 crosses the 2% ($200) limit
        pf = make_portfolio(positions=[pos("ETHUSDT", entry=2000, qty=1.99, stop=1900)])
        verdict = await evaluate(risk_manager, make_proposal(), pf)
        assert any(r.startswith("PORTFOLIO_RISK_EXCEEDED") for r in verdict.rejection_reasons)

    @pytest.mark.asyncio
    async def test_open_risk_measured_from_current_price(self, risk_manager):
        # Long from 2000, now 2050, stop at 2040: $10 can still be lost
        pf = make_portfolio(positions=[
            pos("ETHUSDT", entry=2000, qty=1.0, stop=2040, mark=2050)])
        verdict = await evaluate(risk_manager, make_proposal(symbol="SOLUSDT", entry=150,
                                                             stop=148.5), pf)
        assert verdict.risk_metrics["open_risk_usd"] == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_reject_gross_leverage(self, risk_manager):
        # $29,800 notional on $10k equity; +$500 goes past 3x
        pf = make_portfolio(positions=[pos("ETHUSDT", entry=100, qty=298, stop=99.99)])
        verdict = await evaluate(risk_manager, make_proposal(), pf)
        assert any(r.startswith("GROSS_LEVERAGE_EXCEEDED") for r in verdict.rejection_reasons)

    @pytest.mark.asyncio
    async def test_reject_single_asset_concentration(self, risk_manager):
        pf = make_portfolio(positions=[pos("BTCUSDT", entry=50_000, qty=0.04, stop=49_999)])
        verdict = await evaluate(risk_manager, make_proposal(), pf)  # $2,000 + $500 > 20%
        assert any(r.startswith("CONCENTRATION_EXCEEDED") for r in verdict.rejection_reasons)


# ── Correlation ────────────────────────────────────────────────────────

def correlated_longs(n, risk_each=5.0):
    """n long positions each risking `risk_each`."""
    return [pos(f"ALT{i}USDT", entry=100, qty=risk_each, stop=99) for i in range(n)]


class TestCorrelation:

    @pytest.mark.asyncio
    async def test_unknown_correlation_counts_as_correlated(self, risk_manager):
        # Five longs at $5 risk = 0.25%; a sixth long goes past the cap
        pf = make_portfolio(positions=correlated_longs(5))
        verdict = await evaluate(risk_manager, make_proposal(), pf, corr=None)
        assert any(r.startswith("CORRELATED_RISK_EXCEEDED") for r in verdict.rejection_reasons)

    @pytest.mark.asyncio
    async def test_short_is_not_in_the_long_cluster(self, risk_manager):
        pf = make_portfolio(positions=correlated_longs(5))
        p = make_proposal(entry=50_000, stop=50_500, direction=Side.SELL)
        verdict = await evaluate(risk_manager, p, pf, corr=None)
        assert verdict.decision == RiskDecision.APPROVED

    @pytest.mark.asyncio
    async def test_uncorrelated_assets_are_independent(self, risk_manager):
        import numpy as np
        rng = np.random.default_rng(1)
        candles = {"BTCUSDT": candles_from_returns(rng.normal(0, 0.01, 200))}
        for i in range(5):
            candles[f"ALT{i}USDT"] = candles_from_returns(rng.normal(0, 0.01, 200))
        corr = CorrelationMatrix.from_candles(candles)
        pf = make_portfolio(positions=correlated_longs(5))
        verdict = await evaluate(risk_manager, make_proposal(), pf, corr=corr)
        assert verdict.decision == RiskDecision.APPROVED
        assert verdict.risk_metrics["correlated_with"] == []

    @pytest.mark.asyncio
    async def test_highly_correlated_assets_share_one_limit(self, risk_manager):
        import numpy as np
        rng = np.random.default_rng(2)
        beta = rng.normal(0, 0.01, 200)
        candles = {"BTCUSDT": candles_from_returns(beta)}
        for i in range(5):
            candles[f"ALT{i}USDT"] = candles_from_returns(beta + rng.normal(0, 0.002, 200))
        corr = CorrelationMatrix.from_candles(candles)
        pf = make_portfolio(positions=correlated_longs(5))
        verdict = await evaluate(risk_manager, make_proposal(), pf, corr=corr)
        assert any(r.startswith("CORRELATED_RISK_EXCEEDED") for r in verdict.rejection_reasons)
        assert len(verdict.risk_metrics["correlated_with"]) == 5
