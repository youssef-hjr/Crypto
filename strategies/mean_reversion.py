"""
Mean reversion: in a market without a trend, fade a close outside the
Bollinger band when RSI is stretched, targeting the band's middle.
"""
from __future__ import annotations

from typing import Optional

from models import Side
from strategies.base import Strategy, atr_stop
from strategies.features import FeatureView


class MeanReversion(Strategy):
    name = "mean_reversion"
    DEFAULT_PARAMS = {
        "bb_k": 2.0,          # Band width in standard deviations
        "rsi_band": 30,       # Long below this RSI, short above 100 - this
        "stop_atr": 1.5,
        "max_trend": 0.01,    # Skip if EMA20 and EMA50 are further apart than 1%
        "max_hold_bars": 24,
    }
    PARAM_GRID = {"bb_k": [2.0, 2.5], "rsi_band": [30, 25], "stop_atr": [1.0, 2.0]}

    def generate_signal(self, f: FeatureView) -> Optional[Side]:
        if not f.has("bb_mid", "bb_std", "rsi", "atr", "ema_20", "ema_50"):
            return None
        if abs(f["ema_20"] - f["ema_50"]) / f["ema_50"] > self.params["max_trend"]:
            return None  # Trending: fading it is fighting it
        k, band = self.params["bb_k"], self.params["rsi_band"]
        close, mid, std, rsi = f["close"], f["bb_mid"], f["bb_std"], f["rsi"]
        if close < mid - k * std and rsi < band:
            return Side.BUY
        if close > mid + k * std and rsi > 100 - band:
            return Side.SELL
        return None

    def calculate_stop(self, f: FeatureView, side: Side) -> Optional[float]:
        return atr_stop(f, side, self.params["stop_atr"])

    def calculate_target(self, f: FeatureView, side: Side, stop: float) -> Optional[float]:
        return f["bb_mid"]  # Back to the average

    def calculate_confidence(self, f: FeatureView, side: Side) -> float:
        rsi = f["rsi"]
        stretch = (self.params["rsi_band"] - rsi) if side == Side.BUY else (
            rsi - (100 - self.params["rsi_band"]))
        c = 0.70
        if stretch > 5:
            c += 0.05
        if abs(f["ema_20"] - f["ema_50"]) / f["ema_50"] < 0.003:
            c += 0.05  # Flat market
        return c

    def describe(self, f, side):
        return (f"Close outside the {self.params['bb_k']}σ band with RSI "
                f"{f['rsi']:.0f}; target the 20-bar mean")
