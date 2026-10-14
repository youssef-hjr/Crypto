"""
Circuit Breaker.

A global kill switch. Once tripped it blocks every new trade until an
operator resets it by hand, and that state is persisted, so restarting the
process does not clear it. If the stored state cannot be read, the breaker
starts tripped: it fails closed.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog

from events import Event, EventBus, EventType
from risk.state import InMemoryRiskStore, RiskStateStore

logger = structlog.get_logger()

STATE_KEY = "circuit_breaker"


class CircuitBreaker:

    def __init__(
        self,
        store: Optional[RiskStateStore] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self._store = store or InMemoryRiskStore()
        self._event_bus = event_bus
        self._active = False
        self._reason: Optional[str] = None
        self._tripped_at: Optional[datetime] = None

    async def load(self) -> None:
        """Restore state from the store. Call once at startup."""
        try:
            state = await self._store.get_state(STATE_KEY)
        except Exception as e:
            self._active = True
            self._reason = "STATE_UNAVAILABLE"
            self._tripped_at = datetime.now(timezone.utc)
            logger.critical("circuit_breaker_state_unreadable", error=str(e))
            return
        if state and state.get("active"):
            self._active = True
            self._reason = state.get("reason")
            tripped = state.get("tripped_at")
            self._tripped_at = datetime.fromisoformat(tripped) if tripped else None
            logger.critical("circuit_breaker_still_active",
                            reason=self._reason, tripped_at=tripped)

    async def trip(self, reason: str, details: Optional[dict] = None) -> None:
        if self._active:
            return  # Keep the first reason; it's the one that matters
        self._active = True
        self._reason = reason
        self._tripped_at = datetime.now(timezone.utc)
        logger.critical("circuit_breaker_triggered", reason=reason, **(details or {}))
        await self._store.set_state(STATE_KEY, self._as_dict())
        await self._store.log_event("breaker_tripped", reason, details)
        if self._event_bus is not None:
            await self._event_bus.publish(Event(
                EventType.CIRCUIT_BREAKER_TRIGGERED,
                {"reason": reason, "details": details or {}},
                source="circuit_breaker",
            ))

    async def reset(self, operator: str, note: str) -> None:
        """Manual reset only. Requires a named operator and a note saying
        what was checked, both of which go into the risk event log."""
        if not operator.strip() or not note.strip():
            raise ValueError("A reset needs an operator name and a note")
        previous = self._reason
        self._active = False
        self._reason = None
        self._tripped_at = None
        await self._store.set_state(STATE_KEY, self._as_dict())
        await self._store.log_event("breaker_reset", previous or "none",
                                    {"operator": operator, "note": note})
        logger.warning("circuit_breaker_reset", operator=operator, note=note,
                       previous_reason=previous)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def _as_dict(self) -> dict:
        return {
            "active": self._active,
            "reason": self._reason,
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
        }
