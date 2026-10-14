"""
Tests for opportunity scoring.
A short setup must be able to score as well as the mirrored long setup.
"""
from datetime import datetime, timezone

import pytest

from agents.market_scanner import MarketScanner
from config.settings import AppSettings
from models import (
    DerivativesFeatures, MarketRegime, OrderFlowFeatures, Side, TechnicalFeatures,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def scanner():
    return MarketScanner(None, None, None, None, AppSettings())


def tech(ema_20, ema_50, macd_hist, rsi, momentum):
    return TechnicalFeatures(
        symbol="XYZUSDT", timestamp=NOW, ema_20=ema_20, ema_50=ema_50,
        macd_histogram=macd_hist, rsi=rsi, momentum=momentum,
        relative_volume=1.6, realized_volatility=0.03,
    )


def flow(book_imb, trade_imb):
    return OrderFlowFeatures(
        symbol="XYZUSDT", timestamp=NOW, bid_ask_imbalance=book_imb,
        buy_sell_imbalance=trade_imb, spread_bps=3.0, book_depth_bid=200_000,
    )


def deriv(funding):
    return DerivativesFeatures(symbol="XYZUSDT", timestamp=NOW, funding_rate=funding)


def test_bullish_setup_is_long(scanner):
    opp = scanner._score_opportunity(
        "XYZUSDT", tech(101, 100, 0.5, 65, 0.02), flow(0.3, 0.3), deriv(0.00005),
        MarketRegime.STRONG_BULLISH,
    )
    assert opp is not None
    assert opp.direction == Side.BUY


def test_mirrored_bearish_setup_scores_the_same_as_bullish(scanner):
    long_opp = scanner._score_opportunity(
        "XYZUSDT", tech(101, 100, 0.5, 65, 0.02), flow(0.3, 0.3), deriv(-0.002),
        MarketRegime.STRONG_BULLISH,
    )
    short_opp = scanner._score_opportunity(
        "XYZUSDT", tech(99, 100, -0.5, 35, -0.02), flow(-0.3, -0.3), deriv(0.002),
        MarketRegime.STRONG_BEARISH,
    )
    assert short_opp.direction == Side.SELL
    assert short_opp.overall_score == pytest.approx(long_opp.overall_score)
    assert short_opp.trend_score == long_opp.trend_score
    assert short_opp.momentum_score == long_opp.momentum_score
    assert short_opp.order_flow_score == long_opp.order_flow_score
    assert short_opp.derivatives_score == long_opp.derivatives_score


def test_strong_short_ranks_above_weak_long(scanner):
    strong_short = scanner._score_opportunity(
        "XYZUSDT", tech(99, 100, -0.5, 35, -0.02), flow(-0.3, -0.3), deriv(0.00005),
        MarketRegime.STRONG_BEARISH,
    )
    weak_long = scanner._score_opportunity(
        "XYZUSDT", tech(101, 100, 0.1, 58, 0.0), flow(0.0, 0.0), deriv(0.00005),
        MarketRegime.WEAK_BULLISH,
    )
    assert strong_short.overall_score > weak_long.overall_score


def test_no_directional_edge_returns_none(scanner):
    # Trend up, momentum neutral, order flow selling: no 2-of-3 agreement.
    opp = scanner._score_opportunity(
        "XYZUSDT", tech(101, 100, 0.5, 50, 0.0), flow(-0.3, -0.3), deriv(None),
        MarketRegime.SIDEWAYS,
    )
    assert opp is None


def test_crowded_longs_penalise_longs_not_shorts(scanner):
    assert scanner._derivatives_score(deriv(0.002), Side.BUY) < 50
    assert scanner._derivatives_score(deriv(0.002), Side.SELL) > 50
