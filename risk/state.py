"""
Where the risk engine keeps state that must survive a restart.

`RiskStateStore` is the interface. The PostgreSQL version lives in
`database/risk_repository.py`; `InMemoryRiskStore` is for tests and for
--no-db dry runs, where nothing survives a restart.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Protocol


class RiskStateStore(Protocol):

    async def get_state(self, key: str) -> Optional[dict]:
        ...

    async def set_state(self, key: str, value: dict) -> None:
        ...

    async def log_event(self, kind: str, reason: str, details: Optional[dict] = None) -> None:
        ...

    async def save_equity(self, at: datetime, equity: float, source: str) -> None:
        ...


class InMemoryRiskStore:
    persistent = False

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}
        self.events: list[dict[str, Any]] = []
        self.equity: list[tuple[datetime, float, str]] = []

    async def get_state(self, key: str) -> Optional[dict]:
        value = self.state.get(key)
        return dict(value) if value is not None else None

    async def set_state(self, key: str, value: dict) -> None:
        self.state[key] = dict(value)

    async def log_event(self, kind: str, reason: str, details: Optional[dict] = None) -> None:
        self.events.append({"kind": kind, "reason": reason, "details": details or {}})

    async def save_equity(self, at: datetime, equity: float, source: str) -> None:
        self.equity.append((at, equity, source))
