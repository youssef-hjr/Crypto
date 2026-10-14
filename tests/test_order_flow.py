"""
Tests for the Order Flow Agent.
"""
import pytest
from datetime import datetime, timezone

from agents.order_flow import OrderFlowAgent
from models import OrderBookLevel, OrderBookSnapshot, Side, Trade


@pytest.fixture
def agent():
    return OrderFlowAgent(large_trade_threshold_usd=10_000)


def make_book(
    bid_prices: list[float], bid_qtys: list[float],
    ask_prices: list[float], ask_qtys: list[float],
    symbol: str = "BTCUSDT",
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        bids=[OrderBookLevel(price=p, quantity=q) for p, q in zip(bid_prices, bid_qtys)],
        asks=[OrderBookLevel(price=p, quantity=q) for p, q in zip(ask_prices, ask_qtys)],
    )


def make_trades(buy_count: int, sell_count: int, price: float = 50000) -> list[Trade]:
    now = datetime.now(timezone.utc)
    trades = []
    for i in range(buy_count):
        trades.append(Trade(
            symbol="BTCUSDT", timestamp=now, price=price,
            quantity=0.1, side=Side.BUY,
        ))
    for i in range(sell_count):
        trades.append(Trade(
            symbol="BTCUSDT", timestamp=now, price=price,
            quantity=0.1, side=Side.SELL,
        ))
    return trades


class TestOrderFlowFeatures:

    def test_balanced_book(self, agent):
        book = make_book(
            [49999, 49998, 49997], [1.0, 1.0, 1.0],
            [50001, 50002, 50003], [1.0, 1.0, 1.0],
        )
        features = agent.calculate_features(book, [])
        assert abs(features.bid_ask_imbalance) < 0.01

    def test_bid_heavy_book(self, agent):
        book = make_book(
            [49999, 49998, 49997], [5.0, 5.0, 5.0],
            [50001, 50002, 50003], [1.0, 1.0, 1.0],
        )
        features = agent.calculate_features(book, [])
        assert features.bid_ask_imbalance > 0.3

    def test_ask_heavy_book(self, agent):
        book = make_book(
            [49999, 49998, 49997], [1.0, 1.0, 1.0],
            [50001, 50002, 50003], [5.0, 5.0, 5.0],
        )
        features = agent.calculate_features(book, [])
        assert features.bid_ask_imbalance < -0.3

    def test_buy_dominant_trades(self, agent):
        book = make_book(
            [49999], [1.0], [50001], [1.0],
        )
        trades = make_trades(buy_count=20, sell_count=5)
        features = agent.calculate_features(book, trades)
        assert features.buy_sell_imbalance > 0.3

    def test_microprice(self, agent):
        book = make_book(
            [49999], [2.0], [50001], [1.0],
        )
        features = agent.calculate_features(book, [])
        # Microprice should be weighted toward the side with more quantity
        # More bids -> microprice closer to ask
        assert features.microprice > 50000

    def test_spread_calculated(self, agent):
        book = make_book(
            [49999], [1.0], [50001], [1.0],
        )
        features = agent.calculate_features(book, [])
        assert features.spread == pytest.approx(2.0)

    def test_large_trade_detection(self, agent):
        book = make_book([49999], [1.0], [50001], [1.0])
        now = datetime.now(timezone.utc)
        trades = [
            Trade(symbol="BTCUSDT", timestamp=now, price=50000,
                  quantity=0.5, side=Side.BUY),  # $25,000 > threshold
        ]
        features = agent.calculate_features(book, trades)
        assert features.large_trade_count == 1
