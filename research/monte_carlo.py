"""
Monte Carlo on trade results (PDF section 31).

Resamples the backtest's trades with replacement, in random order, many
times. The spread of outcomes shows how much of the result was the luck of
the order and the draw: if the 95th percentile drawdown is far worse than
the backtest's, plan for that one.

Each trade is replayed as a return on equity (net P&L / equity at entry),
so the resampled curves compound like the real one.
"""
from __future__ import annotations

import numpy as np

from research.simulator import TradeRecord


def monte_carlo(trades: list[TradeRecord], runs: int = 1000, seed: int = 1) -> dict:
    returns = np.array([t.return_on_equity for t in trades])
    if len(returns) < 10:
        return {"runs": 0, "note": "fewer than 10 trades; not enough to resample"}
    rng = np.random.default_rng(seed)
    samples = rng.choice(returns, size=(runs, len(returns)), replace=True)
    curves = np.cumprod(1 + samples, axis=1)
    finals = curves[:, -1] - 1
    peaks = np.maximum.accumulate(np.concatenate([np.ones((runs, 1)), curves], axis=1), axis=1)[:, 1:]
    drawdowns = ((peaks - curves) / peaks).max(axis=1)

    def pct(a, q):
        return round(float(np.percentile(a, q)), 6)

    return {
        "runs": runs,
        "trades_per_run": len(returns),
        "total_return_p5": pct(finals, 5),
        "total_return_p50": pct(finals, 50),
        "total_return_p95": pct(finals, 95),
        "max_drawdown_p50": pct(drawdowns, 50),
        "max_drawdown_p95": pct(drawdowns, 95),
        "prob_loss": round(float((finals < 0).mean()), 4),
    }
