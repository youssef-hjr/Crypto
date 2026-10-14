"""
Data Recorder.

Subscribes to market-data and scanner events and writes them to the
database, so the scanner itself never needs to know about storage.
"""
from __future__ import annotations

import structlog

from typing import Optional

from database.repository import MarketDataRepository
from database.risk_repository import RiskRepository
from events import Event, EventBus, EventType

logger = structlog.get_logger()


class DataRecorder:

    def __init__(
        self,
        repository: MarketDataRepository,
        exchange: str,
        risk_repository: Optional[RiskRepository] = None,
    ):
        self._repo = repository
        self._exchange = exchange
        self._risk_repo = risk_repository

    def attach(self, event_bus: EventBus) -> None:
        event_bus.subscribe(EventType.MARKET_DATA_UPDATED, self._on_candles)
        event_bus.subscribe(EventType.FEATURES_CALCULATED, self._on_features)
        event_bus.subscribe(EventType.OPPORTUNITY_SCORED, self._on_opportunity)
        if self._risk_repo is not None:
            event_bus.subscribe(EventType.RISK_APPROVED, self._on_risk_verdict)
            event_bus.subscribe(EventType.RISK_REJECTED, self._on_risk_verdict)

    async def _on_candles(self, event: Event) -> None:
        d = event.data
        await self._repo.save_candles(
            self._exchange, d["symbol"], d["interval"], d["candles"]
        )

    async def _on_features(self, event: Event) -> None:
        d = event.data
        await self._repo.save_features(
            self._exchange, d["technical"], d["order_flow"],
            d["derivatives"], d["regime"],
        )

    async def _on_opportunity(self, event: Event) -> None:
        await self._repo.save_opportunity(self._exchange, event.data)

    async def _on_risk_verdict(self, event: Event) -> None:
        await self._risk_repo.save_decision(event.data)
