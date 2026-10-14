"""
Trend following: trade an EMA crossover in the direction of the long-term
trend, and let a trailing stop decide when the trend is over.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from models import Side
from strategies.base import Strategy, atr_stop
from strategies.features import FeatureView, add_ema


class TrendFollowing(Strategy):
    name = "trend"
    DEFAULT_PARAMS = {
        "fast": 20,          # EMA periods for the crossover
        "slow": 50,
        "filter": 200,       # Only trade in the direction of this EMA
        "stop_atr": 2.5,     # Initial stop, and trailing distance, in ATRs
        "max_hold_bars": 240,
        "assumed_r": 2.0,    # No fixed target; used for expected R only
    }
    PARAM_GRID = {"fast": [10, 20], "slow": [50, 100], "stop_atr": [2.0, 3.0]}

    def prepare(self, df: pd.DataFrame) -> None:
        for n in (self.params["fast"], self.params["slow"], self.params["filter"]):
            add_ema(df, n)

    def generate_signal(self, f: FeatureView) -> Optional[Side]:
        p = self.params
        fast, slow, flt = f"ema_{p['fast']}", f"ema_{p['slow']}", f"ema_{p['filter']}"
        if not f.has(fast, slow, flt, "macd_histogram", "atr"):
            return None
        pf, ps = f.get(fast, -1), f.get(slow, -1)
        if pf is None or ps is None:
            return None
        close, hist = f["close"], f["macd_histogram"]
        if pf <= ps and f[fast] > f[slow] and close > f[flt] and hist > 0:
            return Side.BUY
        if pf >= ps and f[fast] < f[slow] and close < f[flt] and hist < 0:
            return Side.SELL
        return None

    def calculate_stop(self, f: FeatureView, side: Side) -> Optional[float]:
        return atr_stop(f, side, self.params["stop_atr"])

    def calculate_target(self, f, side, stop):
        return None  # Ride the trend; the trailing stop exits

    def trail_distance(self, f: FeatureView) -> Optional[float]:
        atr = f["atr"]
        return self.params["stop_atr"] * atr if atr else None

    def calculate_confidence(self, f: FeatureView, side: Side) -> float:
        sign = 1 if side == Side.BUY else -1
        c = 0.70
        if (f["relative_volume"] or 0) > 1.2:
            c += 0.05
        rsi = f["rsi"]
        if rsi is not None and 50 < 50 + sign * (rsi - 50) < 70:
            c += 0.05  # Momentum with the trade, not stretched
        if sign * (f["close"] - f[f"ema_{self.params['filter']}"]) / f["close"] > 0.02:
            c += 0.05  # Well clear of the long-term average
        return c

    def describe(self, f, side):
        p = self.params
        return (f"EMA{p['fast']} crossed {'above' if side == Side.BUY else 'below'} "
                f"EMA{p['slow']}, price {'above' if side == Side.BUY else 'below'} EMA{p['filter']}")
