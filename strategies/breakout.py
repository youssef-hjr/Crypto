"""
Breakout: a close beyond the high or low of the previous N bars, on above
average volume, with a fixed reward-to-risk target.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from models import Side
from strategies.base import Strategy, atr_stop
from strategies.features import FeatureView, add_channel


class Breakout(Strategy):
    name = "breakout"
    DEFAULT_PARAMS = {
        "lookback": 20,
        "stop_atr": 2.0,
        "target_r": 2.0,
        "min_rel_volume": 1.2,
        "max_hold_bars": 48,
    }
    PARAM_GRID = {"lookback": [20, 55], "stop_atr": [1.5, 2.5], "target_r": [2.0, 3.0]}

    def prepare(self, df: pd.DataFrame) -> None:
        add_channel(df, self.params["lookback"])

    def generate_signal(self, f: FeatureView) -> Optional[Side]:
        n = self.params["lookback"]
        hi, lo = f"chan_high_{n}", f"chan_low_{n}"
        if not f.has(hi, lo, "atr", "relative_volume"):
            return None
        if f["relative_volume"] < self.params["min_rel_volume"]:
            return None
        close, prev_close = f["close"], f.get("close", -1)
        prev_hi, prev_lo = f.get(hi, -1), f.get(lo, -1)
        if prev_close is None or prev_hi is None or prev_lo is None:
            return None
        # A fresh break only: the previous bar was still inside its channel
        if close > f[hi] and prev_close <= prev_hi:
            return Side.BUY
        if close < f[lo] and prev_close >= prev_lo:
            return Side.SELL
        return None

    def calculate_stop(self, f: FeatureView, side: Side) -> Optional[float]:
        return atr_stop(f, side, self.params["stop_atr"])

    def calculate_confidence(self, f: FeatureView, side: Side) -> float:
        sign = 1 if side == Side.BUY else -1
        c = 0.70
        ema50 = f["ema_50"]
        if ema50 and sign * (f["close"] - ema50) > 0:
            c += 0.05  # Breaking out with the trend
        if (f["relative_volume"] or 0) > 2.0:
            c += 0.05
        return c

    def describe(self, f, side):
        n = self.params["lookback"]
        return (f"Close broke the {n}-bar {'high' if side == Side.BUY else 'low'} "
                f"on {f['relative_volume']:.1f}x volume")
