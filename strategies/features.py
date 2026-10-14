"""
Indicator columns for strategies, computed once over a whole candle series.

Every column is causal: the value on bar i uses bars 0..i only (EWMs,
trailing windows, and shift(1) for "the previous N bars"). That is what
makes a backtest free of look-ahead, and it is tested: computing on the
full history and on history cut at bar i gives the same row i.

The formulas match `agents.quant.QuantEngine`, so a strategy sees the same
numbers in a backtest, in paper trading and in the live scanner.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from models import Candle, TechnicalFeatures


def interval_minutes(interval: str) -> int:
    """Bybit kline interval ("1", "5", "60", "240", "D") in minutes."""
    if interval.isdigit():
        return int(interval)
    if interval == "D":
        return 1440
    if interval == "W":
        return 10080
    raise ValueError(f"Unsupported interval: {interval}")


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    """Candles (oldest first) to a frame indexed by open time (UTC)."""
    df = pd.DataFrame({
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "volume": [c.volume for c in candles],
        "turnover": [c.turnover for c in candles],
    }, index=pd.DatetimeIndex([c.timestamp for c in candles], name="open_time"))
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def compute_features(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Add the standard indicator columns to an OHLCV frame (copy)."""
    out = df.copy()
    close, high, low = out["close"], out["high"], out["low"]

    # Turnover is quote volume; some sources leave it at 0, so fall back
    # to volume x close.
    turnover = out["turnover"].where(out["turnover"] > 0, out["volume"] * close)
    out["turnover"] = turnover

    for n in (20, 50, 100, 200):
        ema = close.ewm(span=n, adjust=False).mean()
        ema[: n - 1] = np.nan  # Not enough bars yet (QuantEngine returns None)
        out[f"ema_{n}"] = ema

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(avg_loss != 0, np.where(avg_gain > 0, 100.0, 50.0))
    out["rsi"] = rsi.where(avg_gain.notna())

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_histogram"] = macd - signal
    out.loc[out.index[: 26 + 9 - 1], ["macd", "macd_signal", "macd_histogram"]] = np.nan

    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    atr[:14] = np.nan
    out["atr"] = atr

    log_ret = np.log(close / close.shift(1))
    out["realized_volatility"] = (log_ret.rolling(20).std()
                                  * math.sqrt(1440 / bar_minutes))
    prev_avg_vol = out["volume"].shift(1).rolling(20).mean()
    out["relative_volume"] = out["volume"] / prev_avg_vol.replace(0, np.nan)
    out["momentum"] = close / close.shift(10) - 1

    # Bollinger band inputs (20 bars, including the current one)
    out["bb_mid"] = close.rolling(20).mean()
    out["bb_std"] = close.rolling(20).std(ddof=0)

    # 24h turnover from the bars themselves, for the liquidity caps
    bars_per_day = max(1, round(1440 / bar_minutes))
    out["turnover_24h"] = turnover.rolling(bars_per_day, min_periods=1).sum()
    return out


def add_channel(df: pd.DataFrame, n: int) -> None:
    """Donchian channel of the PREVIOUS n bars (the current bar excluded,
    so a close above it is a breakout, not a tautology)."""
    hi, lo = f"chan_high_{n}", f"chan_low_{n}"
    if hi not in df:
        df[hi] = df["high"].shift(1).rolling(n).max()
        df[lo] = df["low"].shift(1).rolling(n).min()


def add_ema(df: pd.DataFrame, n: int) -> str:
    col = f"ema_{n}"
    if col not in df:
        ema = df["close"].ewm(span=n, adjust=False).mean()
        ema[: n - 1] = np.nan
        df[col] = ema
    return col


class FeatureView:
    """
    Read-only access to one symbol's feature frame at bar i. Strategies can
    look back (offset < 0) but never forward: asking for a future bar raises.
    """

    __slots__ = ("_cols", "i", "symbol", "time")

    def __init__(self, cols: dict[str, np.ndarray], i: int, symbol: str, time):
        self._cols = cols
        self.i = i
        self.symbol = symbol
        self.time = time  # Open time of bar i

    def __getitem__(self, name: str) -> Optional[float]:
        return self.get(name)

    def get(self, name: str, offset: int = 0) -> Optional[float]:
        if offset > 0:
            raise LookupError("Strategies cannot read future bars")
        j = self.i + offset
        if j < 0:
            return None
        v = self._cols[name][j]
        return None if v != v else float(v)  # NaN -> None

    def has(self, *names: str) -> bool:
        return all(self.get(n) is not None for n in names)

    def technical(self) -> TechnicalFeatures:
        """The row as TechnicalFeatures, for QuantEngine.detect_regime."""
        g = self.get
        return TechnicalFeatures(
            symbol=self.symbol, timestamp=self.time,
            ema_20=g("ema_20"), ema_50=g("ema_50"), ema_100=g("ema_100"),
            ema_200=g("ema_200"), rsi=g("rsi"), macd=g("macd"),
            macd_signal=g("macd_signal"), macd_histogram=g("macd_histogram"),
            atr=g("atr"), volume=g("volume"), relative_volume=g("relative_volume"),
            realized_volatility=g("realized_volatility"), momentum=g("momentum"),
        )


def frame_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Columns as float arrays, for fast per-bar access in the backtest loop."""
    return {c: df[c].to_numpy(dtype=float) for c in df.columns}
