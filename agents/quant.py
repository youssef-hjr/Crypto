"""
Technical / Quantitative Feature Engine.

Computes deterministic technical indicators from OHLCV data.
Indicators are FEATURES — they are NOT automatic trading signals.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from models import Candle, MarketRegime, TechnicalFeatures

logger = structlog.get_logger()

# Regime thresholds, as daily-scaled realized volatility (0.06 = 6% per day).
# Starting points for liquid crypto perps; calibrate on stored data.
HIGH_VOL_DAILY = 0.06
COMPRESSION_VOL_DAILY = 0.012


class QuantEngine:
    """
    Deterministic feature engine.
    Takes candle data, outputs structured TechnicalFeatures.
    """

    def __init__(self, min_candles: int = 50, bar_minutes: int = 5):
        self._min_candles = min_candles
        # Used to scale per-bar volatility to a daily figure, so the same
        # thresholds work for any candle interval.
        self._bar_minutes = bar_minutes

    def calculate_features(self, candles: list[Candle]) -> Optional[TechnicalFeatures]:
        """Calculate all technical features from a candle series."""
        if len(candles) < self._min_candles:
            logger.warning("insufficient_candles", count=len(candles),
                           required=self._min_candles)
            return None

        df = self._candles_to_df(candles)
        symbol = candles[-1].symbol

        try:
            ema_20 = self._ema(df["close"], 20)
            ema_50 = self._ema(df["close"], 50)
            ema_100 = self._ema(df["close"], 100) if len(df) >= 100 else None
            ema_200 = self._ema(df["close"], 200) if len(df) >= 200 else None

            rsi = self._rsi(df["close"], 14)
            macd_line, macd_signal, macd_hist = self._macd(df["close"])
            atr = self._atr(df, 14)
            vwap = self._vwap(df)
            vol = df["volume"].iloc[-1]
            rel_vol = self._relative_volume(df, 20)
            realized_vol = self._realized_volatility(df["close"], 20, self._bar_minutes)
            momentum = self._momentum(df["close"], 10)

            return TechnicalFeatures(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                ema_20=ema_20,
                ema_50=ema_50,
                ema_100=ema_100,
                ema_200=ema_200,
                rsi=rsi,
                macd=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_hist,
                atr=atr,
                vwap=vwap,
                volume=vol,
                relative_volume=rel_vol,
                realized_volatility=realized_vol,
                momentum=momentum,
            )
        except Exception as e:
            logger.error("feature_calculation_error", symbol=symbol, error=str(e))
            return None

    def detect_regime(self, features: TechnicalFeatures) -> MarketRegime:
        """Classify market regime from technical features."""
        if features.ema_20 is None or features.ema_50 is None or features.atr is None:
            return MarketRegime.SIDEWAYS

        trend_up = features.ema_20 > features.ema_50
        strong_rsi = (features.rsi or 50) > 60
        weak_rsi = (features.rsi or 50) < 40
        macd_positive = (features.macd_histogram or 0) > 0
        high_vol = (features.realized_volatility or 0) > HIGH_VOL_DAILY
        has_trend = abs((features.ema_20 - features.ema_50) / features.ema_50) > 0.005

        if high_vol and weak_rsi and (features.rsi or 50) < 25:
            return MarketRegime.PANIC

        if high_vol:
            return MarketRegime.HIGH_VOLATILITY

        # Check trend before compression — a clear trend with low vol
        # is trending, not compressing
        if has_trend and trend_up and strong_rsi and macd_positive:
            return MarketRegime.STRONG_BULLISH
        elif has_trend and trend_up:
            return MarketRegime.WEAK_BULLISH
        elif has_trend and not trend_up and weak_rsi and not macd_positive:
            return MarketRegime.STRONG_BEARISH
        elif has_trend and not trend_up:
            return MarketRegime.WEAK_BEARISH

        rv = features.realized_volatility
        if rv is not None and rv < COMPRESSION_VOL_DAILY:
            return MarketRegime.COMPRESSION

        return MarketRegime.SIDEWAYS

    # ── Indicator Implementations ──────────────────────────────────────

    @staticmethod
    def _candles_to_df(candles: list[Candle]) -> pd.DataFrame:
        data = [{
            "timestamp": c.timestamp,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
            "turnover": c.turnover,
        } for c in candles]
        return pd.DataFrame(data)

    @staticmethod
    def _ema(series: pd.Series, period: int) -> Optional[float]:
        if len(series) < period:
            return None
        return float(series.ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> Optional[float]:
        if len(series) < period + 1:
            return None
        delta = series.diff().dropna()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

        last_gain = avg_gain.iloc[-1]
        last_loss = avg_loss.iloc[-1]

        if last_loss == 0:
            return 100.0 if last_gain > 0 else 50.0
        rs = last_gain / last_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi) if not np.isnan(rsi) else None

    @staticmethod
    def _macd(
        series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if len(series) < slow + signal:
            return None, None, None
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1]),
        )

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
        if len(df) < period + 1:
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return float(atr.iloc[-1])

    @staticmethod
    def _vwap(df: pd.DataFrame) -> Optional[float]:
        if df["volume"].sum() == 0:
            return None
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)
        val = vwap_series.iloc[-1]
        return float(val) if not np.isnan(val) else None

    @staticmethod
    def _relative_volume(df: pd.DataFrame, period: int = 20) -> Optional[float]:
        if len(df) < period + 1:
            return None
        avg_vol = df["volume"].iloc[-(period + 1):-1].mean()
        if avg_vol == 0:
            return None
        return float(df["volume"].iloc[-1] / avg_vol)

    @staticmethod
    def _realized_volatility(
        series: pd.Series, period: int = 20, bar_minutes: int = 5
    ) -> Optional[float]:
        """Std of log returns over `period` bars, scaled to one day."""
        if len(series) < period + 1:
            return None
        returns = np.log(series / series.shift(1)).dropna()
        if len(returns) < period:
            return None
        per_bar = float(returns.iloc[-period:].std())
        return per_bar * math.sqrt(1440 / bar_minutes)

    @staticmethod
    def _momentum(series: pd.Series, period: int = 10) -> Optional[float]:
        if len(series) < period + 1:
            return None
        return float((series.iloc[-1] / series.iloc[-period - 1]) - 1)
