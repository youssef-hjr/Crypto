"""
Backtester: fills on the next bar, every trade gated by the Risk Manager,
the circuit breaker, funding, and no look-ahead.
"""
from datetime import timedelta

import numpy as np
import pytest

from config.settings import RiskSettings
from research.backtester import Backtester, BacktestConfig
from research.regimes import PHASES
from research.simulator import CostModel, funding_times
from research.synthetic import start_plus, synthetic_history
from strategies import build_strategies
from tests.research_helpers import Scripted, at, flat, symbol_data

NO_SLIP = CostModel(base_slippage_bps=0.0, impact_bps_per_pct=0.0)


def bt(costs=NO_SLIP, risk=None, **cfg):
    return Backtester(BacktestConfig(costs=costs, **cfg), risk or RiskSettings())


@pytest.mark.asyncio
async def test_signal_fills_at_the_next_bar_open():
    closes = flat(300)
    opens = list(closes)
    opens[251] = 100.5  # Gap up after the signal bar
    data = {"BTCUSDT": symbol_data(closes, opens=opens)}
    res = await bt().run([Scripted(bars=(250,))], data)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.opened_at == at(251)  # Not at the close of bar 250
    assert t.entry == pytest.approx(100.5 * (1 + 0.0001))  # Half spread
    assert t.exit_reason == "end_of_test"


@pytest.mark.asyncio
async def test_size_comes_from_the_risk_budget():
    res = await bt().run([Scripted(bars=(250,))], {"BTCUSDT": symbol_data(flat(300))})
    t = res.trades[0]
    assert t.risk_usd == pytest.approx(10_000 * 0.0005, rel=0.01)  # 0.05% of equity
    assert t.details["risk_decision"] in ("approved", "reduced")


@pytest.mark.asyncio
async def test_risk_manager_can_veto_every_trade():
    res = await bt(risk=RiskSettings(min_opportunity_score=90)).run(
        [Scripted(bars=(100, 150, 200))], {"BTCUSDT": symbol_data(flat(300))})
    assert res.trades == []
    assert res.decisions["rejected"] == 3
    assert res.rejection_reasons["LOW_OPPORTUNITY_SCORE"] == 3


@pytest.mark.asyncio
async def test_entry_cancelled_when_the_open_gaps_through_the_stop():
    closes = flat(300)
    opens = list(closes)
    opens[101] = 97.0  # Below the 2% stop
    res = await bt().run([Scripted(bars=(100,))], {"BTCUSDT": symbol_data(closes, opens=opens)})
    assert res.trades == []
    assert res.cancelled_entries["gapped_through_stop"] == 1


def gap_crash_data():
    closes = [100.0] * 102 + [90.0] * 198
    opens = [100.0] * 102 + [90.0] * 198  # Bar 102 opens 10% lower
    return {"BTCUSDT": symbol_data(closes, opens=opens)}


@pytest.mark.asyncio
async def test_circuit_breaker_trips_and_blocks_trading():
    risk = RiskSettings(max_daily_drawdown_pct=0.001)  # 0.1% day loss trips it
    res = await bt(risk=risk, breaker_reset_daily=False).run(
        [Scripted(bars=(100, 104, 130))], gap_crash_data())
    assert [t.exit_reason for t in res.trades] == ["stop_gap"]
    assert len(res.breaker_trips) == 1
    assert res.breaker_trips[0]["reason"] == "DAILY_DRAWDOWN_LIMIT"
    assert res.rejection_reasons["CIRCUIT_BREAKER_ACTIVE"] == 2


@pytest.mark.asyncio
async def test_backtest_operator_resets_the_breaker_next_day():
    risk = RiskSettings(max_daily_drawdown_pct=0.001)
    res = await bt(risk=risk).run([Scripted(bars=(100, 104, 130))], gap_crash_data())
    assert res.rejection_reasons["CIRCUIT_BREAKER_ACTIVE"] == 1  # Bar 104, same day
    assert len(res.trades) == 2  # Bar 130 trades again the next day
    assert res.metrics()["breaker_trips"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("side,sign", [("buy", 1), ("sell", -1)])
async def test_funding_charged_at_each_settlement(side, sign):
    closes = flat(72)
    t0 = at(0)
    funding = {t0 + timedelta(hours=h): 0.001 for h in range(80)}
    res = await bt().run([Scripted(bars=(10,), side=side)],
                         {"BTCUSDT": symbol_data(closes, funding=funding)})
    t = res.trades[0]
    n = len(funding_times(t.opened_at, at(71)))
    assert n >= 3
    # Longs pay a positive rate, shorts receive it
    assert t.funding == pytest.approx(sign * n * t.quantity * 100.0 * 0.001)


@pytest.mark.asyncio
async def test_unknown_funding_is_charged_to_either_side():
    res = await bt().run([Scripted(bars=(10,), side="sell")],
                         {"BTCUSDT": symbol_data(flat(72))})
    assert res.trades[0].funding > 0


@pytest.mark.asyncio
async def test_future_bars_do_not_change_past_trades():
    """Cut the history short: every trade that closed before the cut is
    identical, so no decision used data from after it."""
    data = synthetic_history()
    cut = start_plus(200)
    short = {s: type(d)(d.symbol, d.candles[d.candles.index < cut], d.funding, d.rules)
             for s, d in data.items()}
    full = await bt(costs=CostModel()).run(build_strategies("all"), data)
    part = await bt(costs=CostModel()).run(build_strategies("all"), short)

    def key(t):
        return (t.symbol, t.strategy, t.opened_at, round(t.entry, 9), round(t.exit, 9),
                round(t.net_pnl, 9), t.exit_reason)
    early = [key(t) for t in full.trades if t.closed_at < cut - timedelta(days=1)]
    assert len(early) > 50
    assert early == [key(t) for t in part.trades if t.closed_at < cut - timedelta(days=1)]


@pytest.mark.asyncio
async def test_synthetic_run_covers_every_phase_and_is_repeatable():
    data = synthetic_history()
    a = await bt(costs=CostModel()).run(build_strategies("all"), data)
    b = await bt(costs=CostModel()).run(build_strategies("all"), synthetic_history())
    assert a.metrics() == b.metrics()
    phases = a.by_phase()
    assert list(phases) == PHASES
    assert all(phases[p]["trades"] > 0 for p in PHASES)
    m = a.metrics()
    for k in ("cagr", "sharpe", "sortino", "calmar", "profit_factor", "expectancy_r",
              "max_drawdown", "fees", "funding", "slippage", "turnover", "time_in_market"):
        assert k in m
    # Equity at the end = start + every trade's net P&L
    assert m["end_equity"] == pytest.approx(10_000 + sum(t.net_pnl for t in a.trades), abs=0.01)


@pytest.mark.asyncio
async def test_start_end_limit_trading_but_not_warm_up():
    data = synthetic_history()
    lo, hi = start_plus(100), start_plus(160)
    res = await bt(costs=CostModel()).run(build_strategies("trend"), data, lo, hi)
    assert res.trades
    assert all(lo <= t.opened_at < hi for t in res.trades)
    assert res.equity.index[0] > lo and res.equity.index[-1] <= hi


@pytest.mark.asyncio
async def test_correlated_symbols_share_the_correlation_cap():
    """Two symbols moving together, both signalling: the second long counts
    against the first's correlated risk."""
    rng = np.random.default_rng(0)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 300)))
    data = {"AAAUSDT": symbol_data(base, symbol="AAAUSDT"),
            "BBBUSDT": symbol_data(base * 2, symbol="BBBUSDT")}
    risk = RiskSettings(max_correlated_risk_pct=0.0006)  # Room for one trade, not two
    res = await bt(risk=risk).run([Scripted(bars=(250,))], data)
    assert len(res.trades) == 1
    assert res.rejection_reasons["CORRELATED_RISK_EXCEEDED"] == 1


@pytest.mark.asyncio
async def test_total_drawdown_trip_is_never_reset():
    """Hitting the total drawdown limit stops trading for good, even with
    the next-day reset on: the drawdown doesn't go away overnight."""
    risk = RiskSettings(max_total_drawdown_pct=0.001, max_daily_drawdown_pct=0.5)
    res = await bt(risk=risk).run([Scripted(bars=(100, 104, 130))], gap_crash_data())
    assert [t.exit_reason for t in res.trades] == ["stop_gap"]
    assert len(res.breaker_trips) == 1
    assert res.breaker_trips[0]["reason"] == "TOTAL_DRAWDOWN_LIMIT"
    assert res.rejection_reasons["CIRCUIT_BREAKER_ACTIVE"] == 2
    assert res.halted_at is not None and res.metrics()["halted_at"]
