"""Strategy library: signals, stops on the right side, targets, the grid."""
import numpy as np
import pytest

from models import Side
from strategies import STRATEGIES, build_strategies
from strategies.breakout import Breakout
from strategies.features import FeatureView, compute_features, frame_columns
from strategies.mean_reversion import MeanReversion
from strategies.trend import TrendFollowing
from tests.research_helpers import frame


def view_at(strategy, df, i=None):
    f = compute_features(df, 60)
    strategy.prepare(f)
    i = len(f) - 1 if i is None else i
    return FeatureView(frame_columns(f), i, "X", f.index[i])


def signals(strategy, df):
    f = compute_features(df, 60)
    strategy.prepare(f)
    cols = frame_columns(f)
    return [(i, s) for i in range(len(f))
            if (s := strategy.evaluate(FeatureView(cols, i, "X", f.index[i]))) is not None]


def test_trend_long_after_a_turn_up_above_the_long_average():
    # Long uptrend, a dip that pulls EMA20 under EMA50, then a resumption
    closes = list(np.linspace(100, 200, 260)) + list(np.linspace(200, 185, 40)) \
        + list(np.linspace(185, 230, 60))
    found = signals(TrendFollowing(), frame(closes))
    assert found, "expected a crossover signal"
    i, sig = found[-1]
    assert sig.direction == Side.BUY
    assert sig.stop < sig.entry and sig.target is None
    assert sig.trail_distance == pytest.approx(2.5 * (sig.entry - sig.stop) / 2.5)
    assert sig.confidence >= 0.70 and sig.score == round(sig.confidence * 100, 2)


def test_trend_mirror_for_shorts():
    closes = list(np.linspace(200, 100, 260)) + list(np.linspace(100, 108, 40)) \
        + list(np.linspace(108, 80, 60))
    found = signals(TrendFollowing(), frame(closes))
    assert found and found[-1][1].direction == Side.SELL
    assert found[-1][1].stop > found[-1][1].entry


def test_breakout_needs_a_fresh_break_on_volume():
    closes = [100.0] * 60 + [100.05, 103.0, 104.0]
    df = frame(closes, spread=0.001)
    df.loc[df.index[-2], "volume"] = 5_000  # Big volume on the breakout bar
    found = signals(Breakout(), df)
    assert [i for i, _ in found] == [61]  # The next bar is not a fresh break
    sig = found[0][1]
    assert sig.direction == Side.BUY
    risk = sig.entry - sig.stop
    assert sig.target == pytest.approx(sig.entry + 2 * risk)
    assert sig.expected_r == pytest.approx(2.0)


def test_breakout_ignores_low_volume():
    df = frame([100.0] * 60 + [100.05, 103.0])
    assert signals(Breakout(), df) == []


def test_mean_reversion_fades_a_stretch_back_to_the_mean():
    rng = np.random.default_rng(1)
    closes = list(100 + rng.normal(0, 0.3, 80)) + [97.0, 95.5]
    found = signals(MeanReversion(), frame(closes))
    assert found and found[-1][1].direction == Side.BUY
    sig = found[-1][1]
    assert sig.stop < sig.entry < sig.target


def test_mean_reversion_skips_trending_markets():
    closes = list(np.linspace(100, 130, 80)) + [125.0]
    assert all(s.direction != Side.BUY for _, s in signals(MeanReversion(), frame(closes)))


def test_registry_and_grid():
    assert set(STRATEGIES) == {"trend", "breakout", "mean_reversion"}
    assert [s.name for s in build_strategies("all")] == list(STRATEGIES)
    for cls in STRATEGIES.values():
        assert 1 < len(cls.grid()) <= 12  # Small grids: less room to overfit
    with pytest.raises(ValueError):
        build_strategies("grid_martingale")
    with pytest.raises(ValueError):
        TrendFollowing(unknown=1)


def test_expected_value_with_a_win_rate():
    s = Breakout()
    assert s.calculate_expected_value(100, 98, 104) == pytest.approx(2.0)
    assert s.calculate_expected_value(100, 98, 104, win_rate=0.4) == pytest.approx(0.2)
