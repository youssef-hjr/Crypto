"""
Exchange Abstraction Layer.

The strategy engine MUST NOT contain exchange-specific logic.
All exchange interaction goes through this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from models import (
    Candle, InstrumentRules, OrderBookSnapshot, OrderRequest, OrderStatus,
    Position, Side, TickerInfo, Trade,
)


class ExchangeError(Exception):
    """An exchange call failed. Raised wherever a silent default (0 equity,
    no positions) could be mistaken for a real answer."""


class ExchangeAdapter(ABC):
    """
    Abstract base for all exchange integrations.
    New exchanges implement this interface; the rest of the system
    never knows which exchange it's talking to.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connections (REST auth, WebSocket handshake)."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close all connections."""

    # ── Account ────────────────────────────────────────────────────────

    @abstractmethod
    async def get_balance(self) -> dict:
        """Return {currency: available_balance} mapping."""

    @abstractmethod
    async def get_equity(self) -> float:
        """Total account equity in quote currency (USDT)."""

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """All open positions."""

    # ── Market Data ────────────────────────────────────────────────────

    @abstractmethod
    async def get_ticker(self, symbol: str) -> TickerInfo:
        """Current ticker for a symbol."""

    @abstractmethod
    async def get_tickers(self) -> list[TickerInfo]:
        """All available tickers (for universe scanning)."""

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 50) -> OrderBookSnapshot:
        """Current order book snapshot."""

    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        """Recent trades for a symbol."""

    @abstractmethod
    async def get_candles(
        self, symbol: str, interval: str = "5", limit: int = 200
    ) -> list[Candle]:
        """Historical candles. interval in minutes for consistency."""

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Current funding rate for perpetual contracts."""

    @abstractmethod
    async def get_open_interest(self, symbol: str) -> Optional[float]:
        """Current open interest."""

    # ── WebSocket Subscriptions ────────────────────────────────────────

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, depth: int = 50) -> None:
        """Subscribe to order book updates via WebSocket."""

    @abstractmethod
    async def subscribe_trades(self, symbol: str) -> None:
        """Subscribe to trade stream via WebSocket."""

    @abstractmethod
    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to ticker updates."""

    # ── Orders (disabled until execution is validated) ─────────────────

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> dict:
        """Submit an order. Returns exchange order info."""

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order. Returns success."""

    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        """Check order status."""

    # ── Info ───────────────────────────────────────────────────────────

    @abstractmethod
    async def get_exchange_info(self, symbol: str) -> dict:
        """Instrument info: tick size, lot size, min notional, etc."""

    @abstractmethod
    async def get_instrument_rules(self, symbol: str) -> InstrumentRules:
        """Order-size rules (lot step, min/max quantity, min notional)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Exchange name identifier."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the adapter is connected and healthy."""
