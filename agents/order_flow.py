"""
Order Flow Agent.

Analyzes real-time market microstructure: order book imbalance,
trade flow, liquidity, aggression, and price impact.

A single visible order-book wall is NOT treated as a reliable signal.
Order flow is always combined with price action, structure, and context.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import structlog

from models import OrderBookSnapshot, OrderFlowFeatures, Side, Trade

logger = structlog.get_logger()


class OrderFlowAgent:
    """Computes order-flow features from order books and trade streams."""

    def __init__(self, large_trade_threshold_usd: float = 50_000):
        self._large_trade_threshold = large_trade_threshold_usd

    def calculate_features(
        self,
        orderbook: OrderBookSnapshot,
        recent_trades: list[Trade],
        window_seconds: int = 60,
    ) -> OrderFlowFeatures:
        """
        Build an OrderFlowFeatures snapshot from the current book and
        recent trade stream.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=window_seconds)

        # Filter trades within window
        window_trades = [
            t for t in recent_trades
            if t.timestamp.replace(tzinfo=timezone.utc) >= cutoff
        ]

        # ── Book features ──────────────────────────────────────────────
        bid_vol = sum(b.quantity for b in orderbook.bids[:20])
        ask_vol = sum(a.quantity for a in orderbook.asks[:20])
        total_book = bid_vol + ask_vol

        imbalance = 0.0
        if total_book > 0:
            imbalance = (bid_vol - ask_vol) / total_book

        depth_bid = sum(b.quantity * b.price for b in orderbook.bids[:20])
        depth_ask = sum(a.quantity * a.price for a in orderbook.asks[:20])

        # Microprice: volume-weighted mid
        microprice = 0.0
        if orderbook.bids and orderbook.asks:
            bb = orderbook.bids[0]
            ba = orderbook.asks[0]
            total_top = bb.quantity + ba.quantity
            if total_top > 0:
                microprice = (
                    bb.price * ba.quantity + ba.price * bb.quantity
                ) / total_top

        # ── Trade features ─────────────────────────────────────────────
        buy_vol = sum(t.quantity for t in window_trades if t.side == Side.BUY)
        sell_vol = sum(t.quantity for t in window_trades if t.side == Side.SELL)
        total_trade_vol = buy_vol + sell_vol
        buy_sell_imb = 0.0
        if total_trade_vol > 0:
            buy_sell_imb = (buy_vol - sell_vol) / total_trade_vol

        trade_intensity = len(window_trades) / max(window_seconds, 1)

        # Large trades
        large_trades = [
            t for t in window_trades
            if t.price * t.quantity >= self._large_trade_threshold
        ]

        # Aggressive buy ratio
        aggressive_buy_count = sum(1 for t in window_trades if t.side == Side.BUY)
        agg_ratio = 0.0
        if len(window_trades) > 0:
            agg_ratio = aggressive_buy_count / len(window_trades)

        return OrderFlowFeatures(
            symbol=orderbook.symbol,
            timestamp=now,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            bid_ask_imbalance=imbalance,
            spread=orderbook.spread or 0.0,
            spread_bps=orderbook.spread_bps or 0.0,
            book_depth_bid=depth_bid,
            book_depth_ask=depth_ask,
            trade_intensity=trade_intensity,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            buy_sell_imbalance=buy_sell_imb,
            microprice=microprice,
            large_trade_count=len(large_trades),
            aggressive_buy_ratio=agg_ratio,
        )
