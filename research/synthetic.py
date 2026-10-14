"""
Made-up price history with known phases (bull, sideways, bear, crash,
recovery, quiet and wild stretches), for tests and for trying the
backtester before any real history is downloaded.

Results on this data say nothing about real markets.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from research.backtester import SymbolData

# (days, daily drift, daily volatility)
SEGMENTS = [
    (60, 0.000, 0.020),   # sideways
    (90, 0.006, 0.030),   # bull
    (45, 0.000, 0.012),   # quiet sideways
    (60, -0.006, 0.035),  # bear
    (7, -0.050, 0.080),   # crash
    (60, 0.005, 0.040),   # recovery
    (45, 0.000, 0.070),   # wild sideways
]


def synthetic_history(
    symbols: list[str] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    bar_minutes: int = 60,
    seed: int = 7,
    start: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc),
    segments: list[tuple[int, float, float]] = SEGMENTS,
) -> dict[str, SymbolData]:
    rng = np.random.default_rng(seed)
    bars_per_day = 1440 // bar_minutes
    drift = np.concatenate([np.full(d * bars_per_day, mu / bars_per_day) for d, mu, _ in segments])
    vol = np.concatenate([np.full(d * bars_per_day, s / np.sqrt(bars_per_day))
                          for d, _, s in segments])
    n = len(drift)
    market = rng.standard_normal(n)  # Shared factor: the coins move together
    times = pd.date_range(start, periods=n, freq=f"{bar_minutes}min", tz="UTC")
    out = {}
    for k, sym in enumerate(symbols):
        beta = 1.0 + 0.3 * k
        idio = rng.standard_normal(n)
        r = drift * beta + vol * beta * (0.8 * market + 0.6 * idio)
        close = (100.0 * 10 ** k) * np.exp(np.cumsum(r))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wiggle = np.abs(rng.standard_normal(n)) * vol * beta * 0.5
        high = np.maximum(open_, close) * (1 + wiggle)
        low = np.minimum(open_, close) * (1 - wiggle)
        volume = rng.lognormal(mean=10, sigma=0.4, size=n) * (1 + 20 * np.abs(r))
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                           "volume": volume, "turnover": volume * close}, index=times)
        df.index.name = "open_time"
        funding = {t.to_pydatetime(): 0.0001 for t in times if t.hour in (0, 8, 16)}
        out[sym] = SymbolData(symbol=sym, candles=df, funding=funding)
    return out


def total_days(segments=SEGMENTS) -> int:
    return sum(d for d, _, _ in segments)


def start_plus(days: int, start: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)):
    return start + timedelta(days=days)
