"""
Portfolio exposure: notional, leverage, money at risk, and correlated risk.

All of these are computed from the open positions themselves. Risk is the
money lost from the current price if every stop is hit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from models import Position
from portfolio.correlation import CorrelationMatrix


@dataclass
class Exposure:
    symbol: str
    direction: int  # +1 long, -1 short
    notional: float
    risk: Optional[float]  # None = no stop, so the risk is unbounded


def exposures(positions: list[Position]) -> list[Exposure]:
    return [Exposure(p.symbol, p.direction, p.notional_value, p.risk_amount)
            for p in positions]


def gross_exposure(items: list[Exposure]) -> float:
    return sum(e.notional for e in items)


def net_exposure(items: list[Exposure]) -> float:
    return sum(e.notional * e.direction for e in items)


def open_risk(items: list[Exposure]) -> float:
    """Total money at risk across positions that have a stop."""
    return sum(e.risk for e in items if e.risk is not None)


def unprotected(items: list[Exposure]) -> list[str]:
    """Symbols with an open position and no stop loss."""
    return [e.symbol for e in items if e.risk is None]


def correlated_risk(
    symbol: str,
    direction: int,
    new_risk: float,
    items: list[Exposure],
    corr: Optional[CorrelationMatrix],
    threshold: float,
) -> tuple[float, list[str]]:
    """
    Risk of the group of positions that would move together with a new
    trade: same direction and correlation at or above `threshold`, plus the
    new trade itself. An unknown correlation counts as correlated, and
    without a matrix every same-direction position counts.

    Opposite-direction positions are not netted as hedges; a hedge that
    fails in a fast market doesn't help.
    """
    total = new_risk
    members = []
    for e in items:
        if e.direction != direction or e.risk is None:
            continue
        c = corr.get(symbol, e.symbol) if corr is not None else None
        if c is None or c >= threshold:
            total += e.risk
            members.append(e.symbol)
    return total, members


def correlation_adjusted_risk(
    items: list[Exposure], corr: Optional[CorrelationMatrix]
) -> float:
    """
    Portfolio risk if stop-outs are as correlated as returns:
    sqrt(sum_ij r_i r_j d_i d_j c_ij). Unknown correlations count as 1.
    Reported as a metric next to the plain sum of risks.
    """
    total = 0.0
    for a in items:
        for b in items:
            if a.risk is None or b.risk is None:
                continue
            c = corr.get(a.symbol, b.symbol) if corr is not None else None
            c = 1.0 if c is None else c
            total += a.risk * b.risk * a.direction * b.direction * c
    return math.sqrt(max(0.0, total))
