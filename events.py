"""
Event-driven architecture for the trading system.
All components communicate through events, making the system
decoupled, testable, and observable.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger()


class EventType(str, Enum):
    # Market data
    MARKET_DATA_UPDATED = "market_data_updated"
    ORDERBOOK_UPDATED = "orderbook_updated"
    TRADE_RECEIVED = "trade_received"
    TICKER_UPDATED = "ticker_updated"

    # Analysis
    FEATURES_CALCULATED = "features_calculated"
    ORDER_FLOW_UPDATED = "order_flow_updated"
    REGIME_DETECTED = "regime_detected"
    OPPORTUNITY_SCORED = "opportunity_scored"

    # Trading
    SIGNAL_GENERATED = "signal_generated"
    TRADE_PROPOSED = "trade_proposed"
    RISK_APPROVED = "risk_approved"
    RISK_REJECTED = "risk_rejected"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"

    # Position management
    POSITION_OPENED = "position_opened"
    POSITION_UPDATED = "position_updated"
    POSITION_CLOSED = "position_closed"
    STOP_MOVED = "stop_moved"
    PROFIT_LOCKED = "profit_locked"

    # System
    CIRCUIT_BREAKER_TRIGGERED = "circuit_breaker_triggered"
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
    ERROR_OCCURRED = "error_occurred"


class Event:
    """Base event with metadata."""

    def __init__(self, event_type: EventType, data: Any = None, source: str = ""):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now(timezone.utc)
        self.id = f"{event_type.value}_{self.timestamp.timestamp()}"

    def __repr__(self) -> str:
        return f"Event({self.event_type.value}, source={self.source})"


# Type alias for async event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Async event bus for decoupled component communication.
    Components subscribe to event types and receive events asynchronously.
    """

    def __init__(self):
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._event_log: list[Event] = []
        self._max_log_size = 10_000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)
        logger.info("event_subscribed", event_type=event_type.value,
                     handler=handler.__qualname__)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers. Errors in handlers are logged, not propagated."""
        self._log_event(event)

        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            return

        tasks = []
        for handler in handlers:
            tasks.append(self._safe_call(handler, event))

        await asyncio.gather(*tasks)

    async def _safe_call(self, handler: EventHandler, event: Event) -> None:
        try:
            await handler(event)
        except Exception as e:
            logger.error("event_handler_error",
                         event_type=event.event_type.value,
                         handler=handler.__qualname__,
                         error=str(e))

    def _log_event(self, event: Event) -> None:
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size // 2:]

    @property
    def recent_events(self) -> list[Event]:
        return self._event_log[-100:]
