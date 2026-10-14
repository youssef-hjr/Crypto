"""Shared builders for risk and portfolio tests."""
from datetime import datetime, timedelta, timezone

from models import (
    Candle, InstrumentRules, MarketContext, OrderBookLevel, OrderBookSnapshot,
    PortfolioState, Side, TradeProposal,
)


def book(symbol="BTCUSDT", mid=50_000.0, half_spread=None, qty=10.0, levels=5,
         age_s=0.0, step=None):
    half_spread = mid * 1e-5 if half_spread is None else half_spread  # 0.2 bps spread
    step = step or abs(half_spread) * 2
    ts = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    return OrderBookSnapshot(
        symbol=symbol, timestamp=ts,
        bids=[OrderBookLevel(price=mid - half_spread - i * step, quantity=qty)
              for i in range(levels)],
        asks=[OrderBookLevel(price=mid + half_spread + i * step, quantity=qty)
              for i in range(levels)],
    )


def market(symbol="BTCUSDT", mid=50_000.0, turnover=1e9, qty_step=0.001,
           min_qty=0.001, min_notional=5.0, **book_kw):
    return MarketContext(
        symbol=symbol,
        orderbook=book(symbol, mid, **book_kw),
        turnover_24h=turnover,
        instrument=InstrumentRules(symbol=symbol, qty_step=qty_step,
                                   min_qty=min_qty, min_notional=min_notional),
    )


def portfolio(equity=10_000.0, positions=None, **kw):
    return PortfolioState(equity=equity, available_balance=equity, unrealized_pnl=0.0,
                          realized_pnl_today=0.0, positions=positions or [], **kw)


def proposal(symbol="BTCUSDT", entry=50_000.0, stop=49_500.0, size=500.0,
             leverage=1.0, score=85.0, direction=Side.BUY):
    return TradeProposal(symbol=symbol, direction=direction, strategy="test_strategy",
                         entry_price=entry, stop_loss=stop, position_size_usd=size,
                         leverage=leverage, opportunity_score=score,
                         confidence=0.85, expected_r=3.0)


def candles_from_returns(returns, symbol="X", start=100.0,
                         t0=datetime(2026, 1, 1, tzinfo=timezone.utc)):
    price, out = start, []
    for i, r in enumerate([0.0] + list(returns)):
        price *= 1 + r
        out.append(Candle(timestamp=t0 + timedelta(minutes=5 * i), open=price,
                          high=price, low=price, close=price, volume=1, symbol=symbol))
    return out
