"""
The structured brief Claude reads (PDF section 17: structured market
information, never raw exchange access).

Built only from domain objects: the candidate, features, recent candles,
order book, portfolio and risk state. No settings object, adapter or client
is ever passed in, so there is no path for an API key into a prompt. As a
second line of defence, `find_secrets` checks the final text against the
keys in the environment before anything is sent.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

from config.settings import RiskSettings
from models import (
    Candle, DerivativesFeatures, OrderBookSnapshot, OrderFlowFeatures, PortfolioState, Side,
    TechnicalFeatures, TickerInfo,
)
from portfolio.correlation import CorrelationMatrix

RECENT_CANDLES = 24  # Closed bars shown in full
SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "BYBIT_API_KEY", "BYBIT_API_SECRET",
                   "BINANCE_API_KEY", "BINANCE_API_SECRET")


@dataclass
class Candidate:
    """A trade the deterministic side of the system found."""
    symbol: str
    direction: Side
    strategy: str  # Strategy name, or "scanner" for the market scanner
    entry: float
    stop: float
    target: Optional[float]
    score: float  # 0-100, what the Risk Manager's minimum score applies to
    confidence: float  # The strategy's own 0-1
    expected_r: float  # The strategy's own (reward-to-risk without a win rate)
    reason: str
    regime: str  # The scanner's regime label
    bar_time: datetime  # Open time of the bar the signal came from
    bar_minutes: int
    trail_distance: Optional[float] = None
    max_hold_bars: Optional[int] = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.symbol, self.direction.value, self.strategy

    def as_dict(self) -> dict:
        risk = abs(self.entry - self.stop)
        return {
            "symbol": self.symbol, "side": "long" if self.direction == Side.BUY else "short",
            "strategy": self.strategy, "signal_bar_open_utc": self.bar_time.isoformat(),
            "bar_minutes": self.bar_minutes, "entry": _r(self.entry), "stop": _r(self.stop),
            "stop_distance_pct": _pct(risk / self.entry) if self.entry else None,
            "target": _r(self.target),
            "reward_to_risk": _r(abs(self.target - self.entry) / risk, 2)
            if self.target is not None and risk > 0 else None,
            "trailing_stop_distance": _r(self.trail_distance),
            "max_hold_bars": self.max_hold_bars,
            "deterministic_score_0_100": _r(self.score, 1),
            "strategy_confidence_0_1": _r(self.confidence, 3),
            "strategy_reason": self.reason, "scanner_regime": self.regime,
        }


@dataclass
class MarketSnapshot:
    """Whatever is known about the symbol right now. Missing parts are None."""
    technical: Optional[TechnicalFeatures] = None
    candles: list[Candle] = field(default_factory=list)  # Closed bars, oldest first
    orderbook: Optional[OrderBookSnapshot] = None
    order_flow: Optional[OrderFlowFeatures] = None
    derivatives: Optional[DerivativesFeatures] = None
    ticker: Optional[TickerInfo] = None


def build_brief(
    candidate: Candidate,
    market: MarketSnapshot,
    portfolio: PortfolioState,
    risk: dict,
    correlations: Optional[CorrelationMatrix] = None,
    other_candidates: Iterable[Candidate] = (),
) -> dict:
    """Everything Claude gets for one candidate, as plain JSON-able data."""
    return {
        "candidate": candidate.as_dict(),
        "technical": _technical(market.technical, candidate.entry),
        "recent_candles": _candles(market.candles),
        "order_book": _book(market.orderbook),
        "order_flow": _flow(market.order_flow),
        "derivatives": _derivatives(market.derivatives, market.ticker),
        "ticker_24h": _ticker(market.ticker),
        "portfolio": _portfolio(portfolio, candidate.symbol, correlations),
        "risk_state": risk,
        "other_candidates_this_cycle": [
            {"symbol": c.symbol, "side": "long" if c.direction == Side.BUY else "short",
             "strategy": c.strategy, "score": _r(c.score, 1)}
            for c in other_candidates if c.key != candidate.key],
        "not_available": ["news", "liquidations", "long_short_ratio",
                          "strategy_track_record"],
    }


def risk_state(risk_manager, settings: RiskSettings) -> dict:
    """The Risk Manager's limits and current state, as the brief shows them."""
    dd = risk_manager.drawdown.state
    return {
        "risk_per_trade_pct": _pct(settings.risk_per_trade_pct),
        "max_portfolio_risk_pct": _pct(settings.max_portfolio_risk_pct),
        "max_positions": settings.max_positions,
        "daily_drawdown_pct": _pct(dd.daily_drawdown_pct) if dd else None,
        "daily_drawdown_limit_pct": _pct(settings.max_daily_drawdown_pct),
        "total_drawdown_pct": _pct(dd.total_drawdown_pct) if dd else None,
        "total_drawdown_limit_pct": _pct(settings.max_total_drawdown_pct),
        "losing_trades_in_a_row": risk_manager.loss_streak,
        "circuit_breaker_active": risk_manager.is_circuit_breaker_active,
    }


def to_json(brief: dict) -> str:
    """Stable text: sorted keys, so identical briefs are identical bytes."""
    return json.dumps(brief, sort_keys=True, separators=(",", ":"), default=str)


def find_secrets(text: str, extra: Iterable[str] = ()) -> list[str]:
    """Names of any secret from the environment that appears in `text`."""
    found = []
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "")
        if len(value) >= 8 and value in text:
            found.append(name)
    found += [f"extra[{i}]" for i, v in enumerate(extra) if v and len(v) >= 8 and v in text]
    return found


# ── Sections ───────────────────────────────────────────────────────────

def _technical(t: Optional[TechnicalFeatures], price: float) -> Optional[dict]:
    if t is None:
        return None
    out = {k: _r(getattr(t, k)) for k in (
        "ema_20", "ema_50", "ema_100", "ema_200", "rsi", "macd", "macd_signal",
        "macd_histogram", "atr", "vwap", "relative_volume", "realized_volatility", "momentum")}
    if t.atr and price:
        out["atr_pct_of_price"] = _pct(t.atr / price)
    for k in ("ema_20", "ema_50", "ema_200"):
        v = getattr(t, k)
        if v and price:
            out[f"price_vs_{k}_pct"] = _pct(price / v - 1)
    out["as_of_utc"] = t.timestamp.isoformat()
    return out


def _candles(candles: list[Candle]) -> Optional[dict]:
    if not candles:
        return None
    rows = candles[-RECENT_CANDLES:]
    closes = [c.close for c in candles]
    last = closes[-1]

    def change(n: int) -> Optional[float]:
        return _pct(last / closes[-1 - n] - 1) if len(closes) > n and closes[-1 - n] else None

    return {
        "columns": ["open_time_utc", "open", "high", "low", "close", "volume"],
        "rows": [[c.timestamp.isoformat(), _r(c.open), _r(c.high), _r(c.low), _r(c.close),
                  _r(c.volume)] for c in rows],
        "change_pct_over_bars": {"1": change(1), "6": change(6), "24": change(24),
                                 str(len(closes) - 1): change(len(closes) - 1)},
        "high_of_all_bars": _r(max(c.high for c in candles)),
        "low_of_all_bars": _r(min(c.low for c in candles)),
        "bars_available": len(candles),
    }


def _book(b: Optional[OrderBookSnapshot]) -> Optional[dict]:
    if b is None or b.mid_price is None:
        return None
    mid = b.mid_price
    band = mid * 0.005

    def depth(levels) -> float:
        return sum(l.price * l.quantity for l in levels if abs(l.price - mid) <= band)

    bid, ask = depth(b.bids), depth(b.asks)
    return {"mid": _r(mid), "spread_bps": _r(b.spread_bps, 2),
            "bid_depth_usd_within_50bps": _r(bid, 0), "ask_depth_usd_within_50bps": _r(ask, 0),
            "depth_imbalance": _r((bid - ask) / (bid + ask), 3) if bid + ask else None,
            "as_of_utc": b.timestamp.isoformat()}


def _flow(f: Optional[OrderFlowFeatures]) -> Optional[dict]:
    if f is None:
        return None
    return {k: _r(getattr(f, k), 4) for k in (
        "bid_ask_imbalance", "buy_sell_imbalance", "aggressive_buy_ratio", "trade_intensity",
        "large_trade_count", "buy_volume", "sell_volume")}


def _derivatives(d: Optional[DerivativesFeatures], t: Optional[TickerInfo]) -> Optional[dict]:
    funding = d.funding_rate if d and d.funding_rate is not None else (t.funding_rate if t else None)
    oi = d.open_interest if d and d.open_interest is not None else (t.open_interest if t else None)
    if funding is None and oi is None:
        return None
    return {"funding_rate_pct_per_settlement": _pct(funding) if funding is not None else None,
            "open_interest": _r(oi),
            "open_interest_change_pct": _r(d.oi_change_pct) if d else None}


def _ticker(t: Optional[TickerInfo]) -> Optional[dict]:
    if t is None:
        return None
    return {"last_price": _r(t.last_price), "turnover_usd": _r(t.turnover_24h, 0),
            "high": _r(t.high_24h), "low": _r(t.low_24h),
            "change_pct": _r(t.price_change_pct_24h, 3)}


def _portfolio(p: PortfolioState, symbol: str,
               correlations: Optional[CorrelationMatrix]) -> dict:
    equity = p.equity or 0.0
    positions, at_risk = [], 0.0
    for pos in p.positions:
        risk = pos.risk_amount
        at_risk += risk or 0.0
        rho = correlations.get(symbol, pos.symbol) if correlations else None
        positions.append({
            "symbol": pos.symbol, "side": pos.side.value, "strategy": pos.strategy or None,
            "notional_usd": _r(pos.notional_value, 2),
            "unrealized_pnl_usd": _r(pos.unrealized_pnl, 2),
            "risk_to_stop_pct_of_equity": _pct(risk / equity) if risk is not None and equity else None,
            "return_correlation_with_candidate": _r(rho, 2),
        })
    gross = sum(pos.notional_value for pos in p.positions)
    return {"equity_usd": _r(equity, 2), "source": p.source,
            "open_positions": positions,
            "risk_to_stops_pct_of_equity": _pct(at_risk / equity) if equity else None,
            "gross_leverage": _r(gross / equity, 3) if equity else None}


# ── Number formatting: short, finite, JSON-safe ────────────────────────

def _r(v, digits: int = 6):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if digits == 6 and v != 0:
        # Six significant digits: enough for any price, short in tokens
        return float(f"{v:.6g}")
    return round(v, digits) if digits else round(v)


def _pct(fraction: Optional[float]) -> Optional[float]:
    return _r(fraction * 100, 4) if fraction is not None else None
