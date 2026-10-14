"""
Performance metrics (PDF section 32). Return-based numbers come from the
daily equity curve; trade-based numbers from the closed trades.

At 0.05% risk per trade, dollar returns are small by design. Read the
R-based numbers (expectancy in R, average R) to judge the edge itself.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from research.regimes import PHASES
from research.simulator import TradeRecord


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((peak - equity) / peak).max())


def max_consecutive_losses(trades: list[TradeRecord]) -> int:
    worst = run = 0
    for t in sorted(trades, key=lambda t: t.closed_at):
        run = run + 1 if t.net_pnl < 0 else 0
        worst = max(worst, run)
    return worst


def trade_metrics(trades: list[TradeRecord]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    pnl = np.array([t.net_pnl for t in trades])
    r = np.array([t.r_multiple for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    return {
        "trades": n,
        "win_rate": round(len(wins) / n, 4),
        "avg_win": round(float(wins.mean()), 4) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 4) if len(losses) else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        "expectancy_usd": round(float(pnl.mean()), 4),
        "expectancy_r": round(float(r.mean()), 4),
        "avg_r": round(float(r.mean()), 4),
        "r_std": round(float(r.std(ddof=1)), 4) if n > 1 else 0.0,
        "net_pnl": round(float(pnl.sum()), 2),
        "max_consecutive_losses": max_consecutive_losses(trades),
        "fees": round(sum(t.fees for t in trades), 2),
        "funding": round(sum(t.funding for t in trades), 2),
        "slippage": round(sum(t.slippage_usd for t in trades), 2),
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 1),
        "avg_mfe_r": round(float(np.mean([t.mfe_r for t in trades])), 3),
        "avg_mae_r": round(float(np.mean([t.mae_r for t in trades])), 3),
    }


def equity_metrics(curve: pd.Series, gross_exposure: pd.Series | None = None) -> dict:
    """Metrics from an equity curve indexed by time."""
    if len(curve) < 2:
        return {}
    daily = curve.resample("1D").last().dropna()
    start, end = float(curve.iloc[0]), float(curve.iloc[-1])
    days = max((curve.index[-1] - curve.index[0]).total_seconds() / 86400, 1e-9)
    total = end / start - 1
    cagr = (end / start) ** (365 / days) - 1 if end > 0 else -1.0
    rets = daily.pct_change().dropna()
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    downside = rets[rets < 0]
    dstd = math.sqrt((downside ** 2).mean()) if len(downside) else 0.0
    mdd = max_drawdown(curve)
    weekly = daily.resample("W").last().pct_change().dropna()
    monthly = daily.resample("MS").last().pct_change().dropna()
    out = {
        "start_equity": round(start, 2),
        "end_equity": round(end, 2),
        "total_return": round(total, 6),
        "cagr": round(cagr, 6),
        "max_drawdown": round(mdd, 6),
        "sharpe": round(float(rets.mean() / std * math.sqrt(365)), 3) if std > 0 else 0.0,
        "sortino": round(float(rets.mean() / dstd * math.sqrt(365)), 3) if dstd > 0 else 0.0,
        "calmar": round(cagr / mdd, 3) if mdd > 0 else 0.0,
        "avg_daily_return": round(float(rets.mean()), 7) if len(rets) else 0.0,
        "worst_day": round(float(rets.min()), 6) if len(rets) else 0.0,
        "avg_weekly_return": round(float(weekly.mean()), 6) if len(weekly) else 0.0,
        "worst_week": round(float(weekly.min()), 6) if len(weekly) else 0.0,
        "avg_monthly_return": round(float(monthly.mean()), 6) if len(monthly) else 0.0,
        "worst_month": round(float(monthly.min()), 6) if len(monthly) else 0.0,
        "days": round(days, 1),
    }
    if gross_exposure is not None and len(gross_exposure):
        aligned = gross_exposure.reindex(curve.index).fillna(0.0)
        out["time_in_market"] = round(float((aligned > 0).mean()), 4)
        out["avg_gross_leverage"] = round(float((aligned / curve).mean()), 5)
    return out


def summarize(trades: list[TradeRecord], curve: pd.Series,
              gross_exposure: pd.Series | None = None) -> dict:
    m = {**equity_metrics(curve, gross_exposure), **trade_metrics(trades)}
    if trades and len(curve):
        traded = sum(t.notional + t.quantity * t.exit for t in trades)
        m["turnover"] = round(traded / float(curve.mean()), 3)  # x average equity
    return m


def by_group(trades: list[TradeRecord], key) -> dict[str, dict]:
    groups: dict[str, list[TradeRecord]] = {}
    for t in trades:
        for k in key(t):
            groups.setdefault(k, []).append(t)
    return {k: trade_metrics(v) for k, v in sorted(groups.items())}


def by_phase(trades: list[TradeRecord]) -> dict[str, dict]:
    """Trade metrics per market phase, in the PDF's order. Phases with no
    trades are listed with zero trades so gaps are visible."""
    grouped = by_group(trades, lambda t: t.market_phases)
    return {p: grouped.get(p, {"trades": 0}) for p in PHASES}


def by_regime(trades: list[TradeRecord]) -> dict[str, dict]:
    """Per scanner regime at entry (QuantEngine.detect_regime)."""
    return by_group(trades, lambda t: [t.market_regime or "unknown"])


def by_strategy(trades: Iterable[TradeRecord]) -> dict[str, dict]:
    return by_group(list(trades), lambda t: [t.strategy])
