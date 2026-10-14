"""
Walk-forward validation (PDF sections 30 and 31).

The history is cut into windows. In each one, the strategy's small
parameter grid is tried on the in-sample (training) part, the best setting
is picked, and then it is run once, untouched, on the out-of-sample part
that follows. Only the out-of-sample results count: they are what the
strategy would have done without knowing the future.

    |---- train 365d ----|-- test 90d --|
                 |---- train 365d ----|-- test 90d --|  ... and so on

The best setting is the one with the highest t-statistic of R per trade
(mean / std x sqrt(n)), with at least `min_trades` trades. That rewards a
steady edge over a few lucky trades.

The validation gate at the end is a set of plain checks on the
out-of-sample results. Passing means "worth paper trading", nothing more.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from research import metrics as perf
from research.backtester import Backtester, PreparedSymbol
from research.simulator import TradeRecord
from strategies.base import Strategy


@dataclass
class WalkForwardConfig:
    train_days: int = 365
    test_days: int = 90
    step_days: Optional[int] = None  # Defaults to test_days (windows don't overlap)
    min_trades: int = 20  # In-sample trades needed for a setting to count


@dataclass
class GateThresholds:
    min_trades: int = 30
    min_expectancy_r: float = 0.0  # After all costs
    min_profit_factor: float = 1.1
    max_drawdown: float = 0.10
    min_retained_edge: float = 0.5  # OOS expectancy vs in-sample
    min_profitable_windows: float = 0.5


@dataclass
class Window:
    train_start: datetime
    test_start: datetime
    test_end: datetime
    params: dict
    in_sample: dict
    out_sample: dict
    equity: pd.Series = field(repr=False, default=None)
    trades: list[TradeRecord] = field(repr=False, default_factory=list)


@dataclass
class WalkForwardResult:
    strategy: str
    windows: list[Window]
    trades: list[TradeRecord]
    equity: pd.Series
    metrics: dict
    in_sample_expectancy_r: Optional[float]
    gate: dict

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.gate.values())


def objective(m: dict, min_trades: int) -> Optional[float]:
    n = m.get("trades", 0)
    if n < min_trades:
        return None
    std = m.get("r_std") or 0.0
    if std <= 0:
        return None
    return m["expectancy_r"] / std * math.sqrt(n)


def windows(start: datetime, end: datetime, cfg: WalkForwardConfig):
    train, test = timedelta(days=cfg.train_days), timedelta(days=cfg.test_days)
    step = timedelta(days=cfg.step_days or cfg.test_days)
    t = start
    while t + train + test <= end:
        yield t, t + train, t + train + test
        t += step


async def walk_forward(
    backtester: Backtester,
    strategy_cls: type[Strategy],
    data: dict[str, PreparedSymbol],
    start: datetime,
    end: datetime,
    cfg: WalkForwardConfig = WalkForwardConfig(),
    gate: GateThresholds = GateThresholds(),
) -> WalkForwardResult:
    grid = strategy_cls.grid() or [{}]
    results: list[Window] = []
    for train_lo, test_lo, test_hi in windows(start, end, cfg):
        best_params, best_score, best_is = None, None, {"trades": 0}
        for params in grid:
            res = await backtester.run([strategy_cls(**params)], data, train_lo, test_lo)
            m = perf.trade_metrics(res.trades)
            score = objective(m, cfg.min_trades)
            if score is not None and (best_score is None or score > best_score):
                best_params, best_score, best_is = params, score, m
        if best_params is None:  # Nothing traded enough: fall back to the defaults
            best_params = {}
            res = await backtester.run([strategy_cls()], data, train_lo, test_lo)
            best_is = perf.trade_metrics(res.trades)
        oos = await backtester.run([strategy_cls(**best_params)], data, test_lo, test_hi)
        results.append(Window(
            train_start=train_lo, test_start=test_lo, test_end=test_hi,
            params={**strategy_cls.DEFAULT_PARAMS, **best_params},
            in_sample=best_is, out_sample=oos.metrics(),
            equity=oos.equity, trades=oos.trades,
        ))

    trades = [t for w in results for t in w.trades]
    equity = _stitch([w.equity for w in results], backtester.config.initial_equity)
    m = perf.summarize(trades, equity)
    is_trades = sum(w.in_sample.get("trades", 0) for w in results)
    is_exp = (sum(w.in_sample.get("expectancy_r", 0) * w.in_sample.get("trades", 0)
                  for w in results) / is_trades) if is_trades else None
    return WalkForwardResult(
        strategy=strategy_cls.name, windows=results, trades=trades, equity=equity,
        metrics=m, in_sample_expectancy_r=is_exp,
        gate=validation_gate(m, is_exp, results, gate),
    )


def validation_gate(m: dict, is_expectancy: Optional[float], results: list[Window],
                    g: GateThresholds) -> dict:
    n = m.get("trades", 0)
    exp = m.get("expectancy_r", 0.0) if n else 0.0
    pf = m.get("profit_factor") if n else 0.0
    pf = math.inf if pf is None and n else (pf or 0.0)  # None = no losing trades
    dd = m.get("max_drawdown", 0.0)
    profitable = sum(1 for w in results if w.out_sample.get("net_pnl", 0) > 0)
    share = profitable / len(results) if results else 0.0
    retained = (exp / is_expectancy) if is_expectancy and is_expectancy > 0 else None
    return {
        "enough_trades": {"value": n, "need": f">= {g.min_trades}",
                          "passed": n >= g.min_trades},
        "positive_expectancy": {"value": round(exp, 4), "need": f"> {g.min_expectancy_r} R",
                                "passed": n > 0 and exp > g.min_expectancy_r},
        "profit_factor": {"value": pf if pf != math.inf else "no losses",
                          "need": f">= {g.min_profit_factor}",
                          "passed": n > 0 and pf >= g.min_profit_factor},
        "max_drawdown": {"value": round(dd, 4), "need": f"<= {g.max_drawdown:.0%}",
                         "passed": dd <= g.max_drawdown},
        "edge_kept_out_of_sample": {
            "value": round(retained, 3) if retained is not None else "no in-sample edge",
            "need": f">= {g.min_retained_edge:.0%} of in-sample",
            "passed": retained is not None and retained >= g.min_retained_edge},
        "profitable_windows": {"value": f"{profitable}/{len(results)}",
                               "need": f">= {g.min_profitable_windows:.0%}",
                               "passed": bool(results) and share >= g.min_profitable_windows},
    }


def _stitch(curves: list[pd.Series], start_equity: float) -> pd.Series:
    """Chain the out-of-sample equity curves, compounding window to window."""
    level, parts = start_equity, []
    for c in curves:
        if c is None or c.empty:
            continue
        scaled = c / float(c.iloc[0]) * level
        parts.append(scaled)
        level = float(scaled.iloc[-1])
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")]
