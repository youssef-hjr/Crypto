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


class PrefixedRiskStore:
    """
    Gives a separate risk engine (paper trading, a backtest) its own circuit
    breaker, drawdown marks and loss streak on a shared store, by prefixing
    every key. A paper-trading loss can then never trip the breaker of the
    real account, or the other way round.
    """

    def __init__(self, store: RiskStateStore, prefix: str):
        self._store = store
        self._prefix = prefix
        self.persistent = getattr(store, "persistent", False)

    async def get_state(self, key: str) -> Optional[dict]:
        return await self._store.get_state(self._prefix + key)

    async def set_state(self, key: str, value: dict) -> None:
        await self._store.set_state(self._prefix + key, value)

    async def log_event(self, kind: str, reason: str, details: Optional[dict] = None) -> None:
        await self._store.log_event(kind, f"{self._prefix}{reason}", details)

    async def save_equity(self, at: datetime, equity: float, source: str) -> None:
        await self._store.save_equity(at, equity, source)
