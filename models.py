"""
Core domain models for the trading system.
These are shared data structures used across all modules.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── Enums ──────────────────────────────────────────────────────────────

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class MarketRegime(str, Enum):
    STRONG_BULLISH = "strong_bullish"
    WEAK_BULLISH = "weak_bullish"
    STRONG_BEARISH = "strong_bearish"
    WEAK_BEARISH = "weak_bearish"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    PANIC = "panic"
    LIQUIDATION = "liquidation"


class RiskDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REDUCED = "reduced"  # Approved but with reduced size


# ── Market Data Models ─────────────────────────────────────────────────

class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float = 0.0
    symbol: str = ""


class OrderBookLevel(BaseModel):
    price: float
    quantity: float


class OrderBookSnapshot(BaseModel):
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel]  # Sorted best (highest) first
    asks: list[OrderBookLevel]  # Sorted best (lowest) first
    sequence: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        if self.spread and self.mid_price and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10_000
        return None


class Trade(BaseModel):
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: Side
    trade_id: str = ""


class TickerInfo(BaseModel):
    symbol: str
    last_price: float
    volume_24h: float
    turnover_24h: float
    high_24h: float
    low_24h: float
    price_change_pct_24h: float = 0.0
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    timestamp: datetime = Field(default_factory=utc_now)


# ── Trading Models ─────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    """A trade proposal that must pass risk validation before execution."""
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # Required for limit orders
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    leverage: float = 1.0
    strategy: str = ""
    reason: str = ""
    confidence: float = 0.0
    expected_r: float = 0.0


class Position(BaseModel):
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    leverage: float = 1.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    mark_price: Optional[float] = None  # Current price; entry is used if unknown
    strategy: str = ""
    opened_at: datetime = Field(default_factory=utc_now)

    @property
    def direction(self) -> int:
        """+1 for long, -1 for short."""
        return 1 if self.side == PositionSide.LONG else -1

    @property
    def current_price(self) -> float:
        return self.mark_price if self.mark_price else self.entry_price

    @property
    def notional_value(self) -> float:
        return self.current_price * self.quantity

    @property
    def risk_amount(self) -> Optional[float]:
        """Money lost from the current price if the stop is hit, or None if
        there is no stop. Unrealized profit above a trailing stop counts as
        at risk, because it can still be given back."""
        if self.stop_loss is None:
            return None
        distance = (self.current_price - self.stop_loss) * self.direction
        return max(0.0, distance) * self.quantity


class PortfolioState(BaseModel):
    equity: float
    available_balance: float
    unrealized_pnl: float
    realized_pnl_today: float
    positions: list[Position] = []
    # Where the numbers came from: "exchange" (the Bybit account) or "paper"
    # (a configured starting equity, used when no API key is set).
    source: str = "exchange"
    # Drawdown and risk already in use are NOT taken from here: the Risk
    # Manager computes them itself from equity history and open positions.
    timestamp: datetime = Field(default_factory=utc_now)


# ── Opportunity / Signal Models ────────────────────────────────────────

class OpportunityScore(BaseModel):
    symbol: str
    overall_score: float
    trend_score: float = 0.0
    momentum_score: float = 0.0
    order_flow_score: float = 0.0
    liquidity_score: float = 0.0
    volume_score: float = 0.0
    volatility_score: float = 0.0
    derivatives_score: float = 0.0
    market_structure_score: float = 0.0
    news_score: float = 0.0
    expected_r: float = 0.0
    portfolio_fit_score: float = 0.0
    correlation_risk: float = 0.0
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
    confidence: float = 0.0
    strategy: str = ""
    direction: Side = Side.BUY
    timestamp: datetime = Field(default_factory=utc_now)


class TradeProposal(BaseModel):
    """Output of AI reasoning, input to Risk Manager."""
    symbol: str
    direction: Side
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    targets: list[float] = []
    position_size_usd: float
    leverage: float = 1.0
    confidence: float = 0.0
    expected_r: float = 0.0
    reasoning: str = ""
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
    opportunity_score: float = 0.0
    invalidation: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class InstrumentRules(BaseModel):
    """Exchange order-size rules for one contract."""
    symbol: str
    qty_step: float
    min_qty: float
    max_qty: Optional[float] = None
    min_notional: float = 0.0


class MarketContext(BaseModel):
    """Live market facts the Risk Manager checks a proposal against."""
    symbol: str
    orderbook: Optional[OrderBookSnapshot] = None
    turnover_24h: Optional[float] = None  # USD traded in the last 24h
    instrument: Optional[InstrumentRules] = None


class RiskVerdict(BaseModel):
    """Output of the Risk Manager."""
    decision: RiskDecision
    original_proposal: TradeProposal
    approved_size_usd: Optional[float] = None
    approved_quantity: Optional[float] = None
    approved_leverage: Optional[float] = None
    rejection_reasons: list[str] = []
    risk_metrics: dict = {}
    timestamp: datetime = Field(default_factory=utc_now)


# ── Feature Models ─────────────────────────────────────────────────────

class TechnicalFeatures(BaseModel):
    symbol: str
    timestamp: datetime
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_100: Optional[float] = None
    ema_200: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr: Optional[float] = None
    vwap: Optional[float] = None
    volume: Optional[float] = None
    relative_volume: Optional[float] = None
    realized_volatility: Optional[float] = None
    momentum: Optional[float] = None


class OrderFlowFeatures(BaseModel):
    symbol: str
    timestamp: datetime
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    bid_ask_imbalance: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    book_depth_bid: float = 0.0
    book_depth_ask: float = 0.0
    trade_intensity: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_sell_imbalance: float = 0.0
    microprice: float = 0.0
    large_trade_count: int = 0
    aggressive_buy_ratio: float = 0.0


class DerivativesFeatures(BaseModel):
    symbol: str
    timestamp: datetime
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    oi_change_pct: Optional[float] = None
    long_short_ratio: Optional[float] = None
    liquidation_volume_long: float = 0.0
    liquidation_volume_short: float = 0.0
