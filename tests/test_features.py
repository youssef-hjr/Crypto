"""
Indicator columns: causal (no look-ahead) and identical to the live
QuantEngine.
"""
import math

import numpy as np
import pytest

from agents.quant import QuantEngine
from models import Candle
from strategies import build_strategies
from strategies.features import (
    FeatureView, candles_to_frame, compute_features, frame_columns, interval_minutes,
)
from tests.research_helpers import frame


def random_walk(n=400, seed=3):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = frame(closes)
    df["volume"] = rng.lognormal(8, 0.5, n)
    df["turnover"] = df["volume"] * df["close"]
    return df


def with_strategy_columns(df):
    f = compute_features(df, 60)
    for s in build_strategies("all"):
        s.prepare(f)
    return f


@pytest.mark.parametrize("cut", [60, 150, 260, 399])
def test_no_look_ahead(cut):
    """Row i is the same whether or not the bars after i exist."""
    df = random_walk()
    full = with_strategy_columns(df)
    truncated = with_strategy_columns(df.iloc[: cut + 1])
    a, b = full.iloc[cut], truncated.iloc[cut]
    for col in full.columns:
        x, y = a[col], b[col]
        assert (math.isnan(x) and math.isnan(y)) or x == pytest.approx(y, rel=1e-9), col


def test_last_row_matches_live_quant_engine():
    df = random_walk(n=250)
    candles = [Candle(timestamp=t, open=r.open, high=r.high, low=r.low, close=r.close,
                      volume=r.volume, turnover=r.turnover, symbol="X")
               for t, r in df.iterrows()]
    live = QuantEngine(bar_minutes=60).calculate_features(candles)
    f = compute_features(candles_to_frame(candles), 60).iloc[-1]
    for name in ("ema_20", "ema_50", "ema_100", "ema_200", "rsi", "macd", "macd_signal",
                 "macd_histogram", "atr", "relative_volume", "realized_volatility", "momentum"):
        assert f[name] == pytest.approx(getattr(live, name), rel=1e-9), name


def test_warm_up_is_nan_not_a_guess():
    f = compute_features(random_walk(n=30), 60)
    assert f["ema_50"].isna().all()
    assert f["ema_20"].iloc[18] != f["ema_20"].iloc[18]  # NaN
    assert not math.isnan(f["ema_20"].iloc[19])


def test_feature_view_refuses_future_bars():
    f = compute_features(random_walk(n=50), 60)
    view = FeatureView(frame_columns(f), 10, "X", f.index[10])
    assert view.get("close", -1) == f["close"].iloc[9]
    with pytest.raises(LookupError):
        view.get("close", 1)


def test_turnover_falls_back_to_volume_times_close():
    df = frame([100.0] * 30)
    df["turnover"] = 0.0
    f = compute_features(df, 60)
    assert f["turnover"].iloc[-1] == pytest.approx(1_000 * 100)
    assert f["turnover_24h"].iloc[-1] == pytest.approx(24 * 1_000 * 100)


def test_interval_minutes():
    assert interval_minutes("60") == 60
    assert interval_minutes("D") == 1440
    with pytest.raises(ValueError):
        interval_minutes("1h")
