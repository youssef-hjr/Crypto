"""
Plain-text reports for the command line.
"""
from __future__ import annotations

import csv
from typing import Optional

from research.regimes import PHASES
from research.simulator import TradeRecord


def _pct(v, digits=2) -> str:
    return "-" if v is None else f"{v * 100:.{digits}f}%"


def _num(v, digits=2) -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:,.{digits}f}"


def summary_lines(m: dict) -> list[str]:
    if not m:
        return ["No bars in the tested range."]
    lines = [
        f"Equity        {_num(m.get('start_equity'))} -> {_num(m.get('end_equity'))}"
        f"   total {_pct(m.get('total_return'), 3)}   CAGR {_pct(m.get('cagr'), 3)}"
        f"   over {_num(m.get('days'), 0)} days",
        f"Risk          max drawdown {_pct(m.get('max_drawdown'), 3)}"
        f"   Sharpe {_num(m.get('sharpe'))}   Sortino {_num(m.get('sortino'))}"
        f"   Calmar {_num(m.get('calmar'))}",
        f"Periods       avg day {_pct(m.get('avg_daily_return'), 4)}"
        f" (worst {_pct(m.get('worst_day'), 3)})   avg week {_pct(m.get('avg_weekly_return'), 3)}"
        f" (worst {_pct(m.get('worst_week'), 3)})   avg month {_pct(m.get('avg_monthly_return'), 3)}"
        f" (worst {_pct(m.get('worst_month'), 3)})",
    ]
    if m.get("trades"):
        lines += [
            f"Trades        {m['trades']}   win rate {_pct(m['win_rate'], 1)}"
            f"   expectancy {m['expectancy_r']:+.3f} R ({_num(m['expectancy_usd'])} USD)"
            f"   profit factor {_num(m['profit_factor'])}",
            f"Wins/losses   avg win {_num(m['avg_win'])}   avg loss {_num(m['avg_loss'])}"
            f"   max losing streak {m['max_consecutive_losses']}"
            f"   avg MFE {m['avg_mfe_r']:+.2f} R   avg MAE {m['avg_mae_r']:+.2f} R",
            f"Costs         fees {_num(m['fees'])}   funding {_num(m['funding'])}"
            f"   slippage+spread {_num(m['slippage'])}   turnover {_num(m.get('turnover'))}x equity",
        ]
    else:
        lines.append("Trades        none")
    if "time_in_market" in m:
        lines.append(f"Exposure      in the market {_pct(m['time_in_market'], 1)} of the time"
                     f"   avg gross leverage {m['avg_gross_leverage']:.4f}x")
    if "signals" in m:
        lines.append(f"Risk Manager  {m['signals']} signals: {m['approved']} approved,"
                     f" {m['rejected']} rejected; {m['entries_cancelled']} entries cancelled;"
                     f" circuit breaker trips {m['breaker_trips']}")
    if m.get("halted_at"):
        lines.append(f"HALTED        the total drawdown limit tripped the circuit breaker on"
                     f" {m['halted_at'][:10]}; no new trades after that (it is never reset"
                     " automatically). Results after that date are not the strategy's.")
    return lines


def table(rows: dict[str, dict], title: str, first_col: str) -> list[str]:
    head = (f"{first_col:<17}{'trades':>7}{'win%':>7}{'exp R':>8}{'PF':>7}"
            f"{'net USD':>10}{'avg MFE':>9}")
    out = [title, head, "-" * len(head)]
    for name, m in rows.items():
        if not m.get("trades"):
            out.append(f"{name:<17}{0:>7}      -       -      -         -        -")
            continue
        pf = m["profit_factor"]
        out.append(f"{name:<17}{m['trades']:>7}{m['win_rate'] * 100:>6.1f}%"
                   f"{m['expectancy_r']:>+8.3f}{(pf if pf is not None else float('inf')):>7.2f}"
                   f"{m['net_pnl']:>10.2f}{m['avg_mfe_r']:>+9.2f}")
    return out


def coverage_lines(coverage: dict[str, dict[str, int]]) -> list[str]:
    head = f"{'days in phase':<17}" + "".join(f"{p[:9]:>10}" for p in PHASES)
    out = [head]
    for sym, cov in coverage.items():
        out.append(f"{sym:<17}" + "".join(f"{cov.get(p, 0):>10}" for p in PHASES))
    missing = [p for p in PHASES if all(c.get(p, 0) == 0 for c in coverage.values())]
    if missing:
        out.append(f"Not covered by this history: {', '.join(missing)}."
                   " Download a longer period to test them.")
    return out


def monte_carlo_lines(mc: dict) -> list[str]:
    if not mc.get("runs"):
        return [f"Monte Carlo: {mc.get('note', 'skipped')}"]
    return [
        f"Monte Carlo   {mc['runs']} resamples of {mc['trades_per_run']} trades:"
        f" total return p5 {_pct(mc['total_return_p5'], 3)} / median"
        f" {_pct(mc['total_return_p50'], 3)} / p95 {_pct(mc['total_return_p95'], 3)}",
        f"              max drawdown median {_pct(mc['max_drawdown_p50'], 3)},"
        f" p95 {_pct(mc['max_drawdown_p95'], 3)};  chance of a loss {_pct(mc['prob_loss'], 1)}",
    ]


def gate_lines(gate: dict) -> list[str]:
    out = []
    for name, c in gate.items():
        out.append(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name.replace('_', ' ')}:"
                   f" {c['value']} (needs {c['need']})")
    return out


def write_trades_csv(path: str, trades: list[TradeRecord]) -> Optional[str]:
    if not trades:
        return None
    rows = [t.as_row() for t in trades]
    for r in rows:
        r["market_phases"] = "|".join(r["market_phases"])
        r.pop("details", None)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path
