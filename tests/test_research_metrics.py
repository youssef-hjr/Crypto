"""Performance metrics, market phase labels, Monte Carlo and walk-forward."""
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from config.settings import RiskSettings
from research import metrics as perf
from research.backtester import Backtester, BacktestConfig
from research.monte_carlo import monte_carlo
from research.regimes import PHASES, coverage, daily_phases, phases_at
from research.simulator import TradeRecord
from research.synthetic import start_plus, synthetic_history
from research.walk_forward import (
    GateThresholds, WalkForwardConfig, validation_gate, walk_forward, windows,
)
from strategies.breakout import Breakout
from tests.research_helpers import T0

BASE = TradeRecord(
    symbol="X", strategy="s", direction="buy", opened_at=T0, closed_at=T0, entry=100,
    exit=101, stop=99, target=None, quantity=1, notional=100, leverage=1, risk_usd=1,
    r_multiple=1, gross_pnl=1, fees=0, funding=0, net_pnl=1, mae_r=0, mfe_r=1,
    slippage_usd=0, exit_reason="target", bars_held=1, confidence=0.8,
    market_regime="sideways", market_phases=["bull"], equity_at_entry=1000)


def trades(pnls, phases=("bull",)):
    return [replace(BASE, net_pnl=p, r_multiple=p, closed_at=T0 + timedelta(hours=i),
                    market_phases=list(phases)) for i, p in enumerate(pnls)]


def test_trade_metrics():
    m = perf.trade_metrics(trades([2, -1, -1, -1, 3]))
    assert m["trades"] == 5 and m["win_rate"] == 0.4
    assert m["profit_factor"] == pytest.approx(5 / 3, abs=1e-3)
    assert m["expectancy_r"] == pytest.approx(0.4)
    assert m["max_consecutive_losses"] == 3
    assert perf.trade_metrics(trades([1, 2]))["profit_factor"] is None  # No losses


def test_equity_metrics():
    idx = pd.date_range("2025-01-01", periods=366, freq="D", tz="UTC")
    curve = pd.Series(np.linspace(100, 110, 366), index=idx)
    curve.iloc[100] = 95  # One bad day
    m = perf.equity_metrics(curve)
    assert m["total_return"] == pytest.approx(0.10)
    assert m["cagr"] == pytest.approx(0.10, abs=0.001)
    assert m["max_drawdown"] == pytest.approx((curve.iloc[99] - 95) / curve.iloc[99], abs=1e-5)
    assert m["worst_day"] < 0 and m["sharpe"] > 0 and m["calmar"] > 0


def test_by_phase_lists_every_phase_even_without_trades():
    out = perf.by_phase(trades([1, -1], phases=("bear", "crash")))
    assert list(out) == PHASES
    assert out["bear"]["trades"] == 2 and out["bull"]["trades"] == 0


def daily_series(values):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_phase_labels():
    up = list(np.linspace(100, 130, 40))
    crash = [130 * (1 - 0.06 * i) for i in range(1, 6)]  # -26% in 5 days
    recovery = list(np.linspace(crash[-1], 120, 30))
    labels = daily_phases(daily_series(up + crash + recovery))
    assert labels["bull"].iloc[39]
    assert labels["crash"].iloc[44]
    assert labels["recovery"].iloc[60:].any()
    assert not labels["recovery"].iloc[:44].any()
    assert sum(coverage(labels).values()) > 0


def test_a_trade_is_labelled_with_the_previous_day():
    labels = daily_phases(daily_series([100] * 30 + [70]))  # Crash on the last day
    last = labels.index[-1]
    assert "crash" not in phases_at(labels, last + timedelta(hours=12))
    assert "crash" in phases_at(labels, last + timedelta(days=1, hours=1))


def test_monte_carlo():
    t = trades([0.02, -0.01, -0.01, 0.03, -0.01] * 10)
    for x in t:
        x.equity_at_entry = 1.0
    mc = monte_carlo(t, runs=500)
    assert mc["runs"] == 500
    assert mc["total_return_p5"] <= mc["total_return_p50"] <= mc["total_return_p95"]
    assert 0 <= mc["max_drawdown_p50"] <= mc["max_drawdown_p95"]
    assert monte_carlo(t, runs=500) == mc  # Seeded
    assert monte_carlo(t[:5])["runs"] == 0


def test_walk_forward_windows():
    w = list(windows(start_plus(0), start_plus(300), WalkForwardConfig(120, 60)))
    assert [(a.day, b - a, c - b) for a, b, c in w][0][1:] == (timedelta(120), timedelta(60))
    assert len(w) == 3


@pytest.mark.asyncio
async def test_walk_forward_trades_only_out_of_sample():
    bt = Backtester(BacktestConfig(), RiskSettings())
    data = bt.prepare(synthetic_history())
    res = await walk_forward(bt, Breakout, data, start_plus(0), start_plus(367),
                             WalkForwardConfig(train_days=120, test_days=60))
    assert len(res.windows) == 4
    for w in res.windows:
        assert all(w.test_start <= t.opened_at < w.test_end for t in w.trades)
        assert set(w.params) == set(Breakout.DEFAULT_PARAMS)
    assert res.metrics["trades"] == len(res.trades) > 0
    assert set(res.gate) == {"enough_trades", "positive_expectancy", "profit_factor",
                             "max_drawdown", "edge_kept_out_of_sample", "profitable_windows"}
    strict = validation_gate(res.metrics, res.in_sample_expectancy_r, res.windows,
                             GateThresholds(min_trades=100_000))
    assert not strict["enough_trades"]["passed"]
