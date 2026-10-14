"""
Tests for the Quantitative Feature Engine.
Validates indicator calculations against known values.
"""
import pytest
from datetime import datetime, timedelta

from agents.quant import QuantEngine
from models import Candle, MarketRegime


@pytest.fixture
def quant():
    return QuantEngine(min_candles=20)


def make_candles(
    prices: list[float],
    base_volume: float = 1000.0,
    symbol: str = "BTCUSDT",
) -> list[Candle]:
    """Generate candles from a price series for testing."""
    candles = []
    base_time = datetime(2024, 1, 1)
    for i, price in enumerate(prices):
        noise = price * 0.002  # Small high/low spread
        candles.append(Candle(
            timestamp=base_time + timedelta(minutes=5 * i),
            open=price - noise,
            high=price + noise,
            low=price - noise * 1.5,
            close=price,
            volume=base_volume * (1 + (i % 3) * 0.1),
            symbol=symbol,
        ))
    return candles


def trending_up_prices(n=100, start=100, step=0.5) -> list[float]:
    return [start + i * step for i in range(n)]


def trending_down_prices(n=100, start=200, step=0.5) -> list[float]:
    return [start - i * step for i in range(n)]


def sideways_prices(n=100, center=100, amplitude=2) -> list[float]:
    import math
    return [center + amplitude * math.sin(i * 0.3) for i in range(n)]


class TestIndicators:

    def test_ema_trending_up(self, quant):
        candles = make_candles(trending_up_prices(60))
        features = quant.calculate_features(candles)
        assert features is not None
        # EMA20 should be above EMA50 in uptrend
        assert features.ema_20 > features.ema_50

    def test_ema_trending_down(self, quant):
        candles = make_candles(trending_down_prices(60))
        features = quant.calculate_features(candles)
        assert features is not None
        # EMA20 should be below EMA50 in downtrend
        assert features.ema_20 < features.ema_50

    def test_rsi_overbought_in_uptrend(self, quant):
        candles = make_candles(trending_up_prices(60, step=2.0))
        features = quant.calculate_features(candles)
        assert features is not None
        assert features.rsi is not None
        assert features.rsi > 50  # Should be elevated in strong uptrend

    def test_rsi_oversold_in_downtrend(self, quant):
        candles = make_candles(trending_down_prices(60, start=200, step=2.0))
        features = quant.calculate_features(candles)
        assert features is not None
        assert features.rsi is not None
        assert features.rsi < 50

    def test_atr_positive(self, quant):
        candles = make_candles(trending_up_prices(60))
        features = quant.calculate_features(candles)
        assert features is not None
        assert features.atr > 0

    def test_macd_positive_in_uptrend(self, quant):
        candles = make_candles(trending_up_prices(60))
        features = quant.calculate_features(candles)
        assert features is not None
        assert features.macd is not None
        assert features.macd > 0

    def test_momentum_positive_in_uptrend(self, quant):
        candles = make_candles(trending_up_prices(60))
        features = quant.calculate_features(candles)
        assert features is not None
        assert features.momentum > 0

    def test_momentum_negative_in_downtrend(self, quant):
        candles = make_candles(trending_down_prices(60))
        features = quant.calculate_features(candles)
        assert features is not None
        assert features.momentum < 0

    def test_vwap_calculated(self, quant):
        candles = make_candles(trending_up_prices(60))
        features = quant.calculate_features(candles)
        assert features is not None
        assert features.vwap > 0

    def test_insufficient_candles(self, quant):
        candles = make_candles([100, 101, 102])
        features = quant.calculate_features(candles)
        assert features is None


class TestRegimeDetection:

    def test_strong_bullish(self, quant):
        candles = make_candles(trending_up_prices(100, step=2.0))
        features = quant.calculate_features(candles)
        regime = quant.detect_regime(features)
        assert regime in (MarketRegime.STRONG_BULLISH, MarketRegime.WEAK_BULLISH)

    def test_strong_bearish(self, quant):
        candles = make_candles(trending_down_prices(100, start=300, step=2.0))
        features = quant.calculate_features(candles)
        regime = quant.detect_regime(features)
        assert regime in (MarketRegime.STRONG_BEARISH, MarketRegime.WEAK_BEARISH)


class TestRealizedVolatility:

    def _random_walk(self, per_bar_std, n=200, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        return list(60000 * np.exp(np.cumsum(rng.normal(0, per_bar_std, n))))

    def test_scaled_to_daily(self, quant):
        # 0.15% per 5-minute bar is a normal BTC day: about 2.5% daily.
        features = quant.calculate_features(make_candles(self._random_walk(0.0015)))
        assert 0.015 < features.realized_volatility < 0.04

    def test_normal_market_is_not_compression(self, quant):
        features = quant.calculate_features(make_candles(self._random_walk(0.0015)))
        assert quant.detect_regime(features) != MarketRegime.COMPRESSION

    def test_violent_market_is_high_volatility(self, quant):
        features = quant.calculate_features(make_candles(self._random_walk(0.006)))
        assert quant.detect_regime(features) in (
            MarketRegime.HIGH_VOLATILITY, MarketRegime.PANIC)

    def test_bar_minutes_changes_scaling(self):
        prices = self._random_walk(0.0015)
        five = QuantEngine(min_candles=20, bar_minutes=5).calculate_features(make_candles(prices))
        hourly = QuantEngine(min_candles=20, bar_minutes=60).calculate_features(make_candles(prices))
        assert five.realized_volatility > hourly.realized_volatility
