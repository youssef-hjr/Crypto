"""
Liquidity checks on the live order book.
"""
from __future__ import annotations

from models import OrderBookSnapshot, Side


def depth_usd(book: OrderBookSnapshot, side: Side, band_bps: float) -> float:
    """
    USD resting on the side a market order would take (asks for a buy,
    bids for a sell), within `band_bps` of the mid price.
    """
    mid = book.mid_price
    if not mid:
        return 0.0
    band = mid * band_bps / 10_000
    if side == Side.BUY:
        levels = [l for l in book.asks if l.price <= mid + band]
    else:
        levels = [l for l in book.bids if l.price >= mid - band]
    return sum(l.price * l.quantity for l in levels)


def is_crossed(book: OrderBookSnapshot) -> bool:
    return (book.best_bid is not None and book.best_ask is not None
            and book.best_bid >= book.best_ask)
