"""
Strategy interface (PDF section 15).

A strategy looks at closed bars and may propose a trade. It never sizes the
trade or places an order: the proposal goes to the Risk Manager, which
sizes, reduces or rejects it, in backtests, paper trading and (later) live.

Every strategy implements the PDF's five methods:

  generate_signal()           direction to trade, or None
  calculate_stop()            invalidation price
  calculate_target()          take-profit price, or None to ride a trailing stop
  calculate_confidence()      0..1, how much evidence lines up
  calculate_expected_value()  expected R of the setup

`evaluate()` puts them together into a `Signal`.
"""
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from models import Side
from strategies.features import FeatureView


@dataclass
class Signal:
    symbol: str
    strategy: str
    direction: Side
    entry: float  # Close of the signal bar; the fill happens later
    stop: float
    target: Optional[float]
    trail_distance: Optional[float]  # Trailing stop distance in price, if any
    max_hold_bars: int
    confidence: float
    expected_r: float
    reason: str
    params: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Opportunity score handed to the Risk Manager (0-100)."""
        return round(self.confidence * 100, 2)


class Strategy(ABC):
    name: str = "base"
    # Defaults, and the small grid walk-forward is allowed to search.
    # Keep grids small: more combinations means more chances to overfit.
    DEFAULT_PARAMS: dict[str, Any] = {}
    PARAM_GRID: dict[str, list] = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params = {**self.DEFAULT_PARAMS, **params}

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"

    @classmethod
    def grid(cls) -> list[dict]:
        keys = list(cls.PARAM_GRID)
        return [dict(zip(keys, combo))
                for combo in itertools.product(*(cls.PARAM_GRID[k] for k in keys))]

    def prepare(self, df: pd.DataFrame) -> None:
        """Add any parameter-specific columns (causal only) to the frame."""

    # ── The PDF interface ──────────────────────────────────────────────

    @abstractmethod
    def generate_signal(self, f: FeatureView) -> Optional[Side]:
        ...

    @abstractmethod
    def calculate_stop(self, f: FeatureView, side: Side) -> Optional[float]:
        ...

    def calculate_target(self, f: FeatureView, side: Side, stop: float) -> Optional[float]:
        r = self.params.get("target_r")
        if not r:
            return None
        entry = f["close"]
        return entry + (entry - stop) * r  # Works for both sides: stop is on the other side

    @abstractmethod
    def calculate_confidence(self, f: FeatureView, side: Side) -> float:
        ...

    def calculate_expected_value(
        self, entry: float, stop: float, target: Optional[float],
        win_rate: Optional[float] = None,
    ) -> float:
        """
        Expected R. With a measured win rate (from backtests of this
        strategy in this regime) it is p x reward - (1 - p) x 1R. Without
        one it is just the reward-to-risk of the setup, which says nothing
        about the odds: the backtest is what tells you those.
        """
        risk = abs(entry - stop)
        if risk <= 0:
            return 0.0
        reward = abs(target - entry) / risk if target is not None else self.params.get(
            "assumed_r", 2.0)
        if win_rate is None:
            return round(reward, 3)
        return round(win_rate * reward - (1 - win_rate), 3)

    def trail_distance(self, f: FeatureView) -> Optional[float]:
        k = self.params.get("trail_atr")
        atr = f["atr"]
        return k * atr if k and atr else None

    # ── Putting it together ────────────────────────────────────────────

    def evaluate(self, f: FeatureView) -> Optional[Signal]:
        side = self.generate_signal(f)
        if side is None:
            return None
        entry = f["close"]
        stop = self.calculate_stop(f, side)
        if not entry or stop is None or stop <= 0:
            return None
        if (side == Side.BUY and stop >= entry) or (side == Side.SELL and stop <= entry):
            return None
        target = self.calculate_target(f, side, stop)
        confidence = self.calculate_confidence(f, side)
        return Signal(
            symbol=f.symbol, strategy=self.name, direction=side, entry=entry,
            stop=stop, target=target, trail_distance=self.trail_distance(f),
            max_hold_bars=int(self.params.get("max_hold_bars", 48)),
            confidence=round(min(1.0, max(0.0, confidence)), 3),
            expected_r=self.calculate_expected_value(entry, stop, target),
            reason=self.describe(f, side), params=dict(self.params),
        )

    def describe(self, f: FeatureView, side: Side) -> str:
        return f"{self.name} {side.value}"


def atr_stop(f: FeatureView, side: Side, k: float) -> Optional[float]:
    atr = f["atr"]
    if not atr:
        return None
    return f["close"] - k * atr if side == Side.BUY else f["close"] + k * atr
