"""Builders shared by the backtest and paper-trading tests."""
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from models import Side
from research.backtester import SymbolData
from strategies.base import Strategy
from strategies.features import FeatureView

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def frame(closes, t0=T0, minutes=60, spread=0.001, volume=1_000.0, opens=None,
          highs=None, lows=None):
    """OHLCV frame from closes. Open = previous close unless given."""
    closes = np.asarray(closes, dtype=float)
    o = np.concatenate([[closes[0]], closes[:-1]]) if opens is None else np.asarray(opens, float)
    h = np.maximum(o, closes) * (1 + spread) if highs is None else np.asarray(highs, float)
    l = np.minimum(o, closes) * (1 - spread) if lows is None else np.asarray(lows, float)
    idx = pd.date_range(t0, periods=len(closes), freq=f"{minutes}min", tz="UTC")
    vol = np.full(len(closes), volume)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": closes, "volume": vol,
                         "turnover": vol * closes}, index=idx)


def symbol_data(closes, symbol="BTCUSDT", funding=None, **kw) -> SymbolData:
    return SymbolData(symbol=symbol, candles=frame(closes, **kw), funding=funding or {})


class Scripted(Strategy):
    """Signals on chosen bars, with a stop and target a fixed % away."""
    name = "scripted"
    DEFAULT_PARAMS = {"bars": (), "side": "buy", "stop_pct": 0.02, "target_pct": None,
                      "max_hold_bars": 1000, "trail_pct": None, "score": 0.8}

    def generate_signal(self, f: FeatureView) -> Optional[Side]:
        return Side(self.params["side"]) if f.i in self.params["bars"] else None

    def calculate_stop(self, f, side):
        k = self.params["stop_pct"]
        return f["close"] * (1 - k) if side == Side.BUY else f["close"] * (1 + k)

    def calculate_target(self, f, side, stop):
        k = self.params["target_pct"]
        if k is None:
            return None
        return f["close"] * (1 + k) if side == Side.BUY else f["close"] * (1 - k)

    def trail_distance(self, f):
        k = self.params["trail_pct"]
        return f["close"] * k if k else None

    def calculate_confidence(self, f, side):
        return self.params["score"]


def flat(n=300, price=100.0):
    return [price] * n


def at(i, minutes=60, t0=T0):
    return t0 + timedelta(minutes=minutes * i)
