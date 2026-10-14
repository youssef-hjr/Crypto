"""
Storage for research: reading candle history back, funding rates,
instrument rules, backtest runs and the trade journal.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine

from database.models import candles, funding_rates, instruments, research_runs, trade_journal
from database.repository import MarketDataRepository
from models import InstrumentRules

CHUNK = 1000  # Rows per insert, well under PostgreSQL's parameter limit


class ResearchRepository:

    def __init__(self, engine: AsyncEngine):
        self._engine = engine
        self._insert = MarketDataRepository(engine)._insert

    # ── History ────────────────────────────────────────────────────────

    async def candle_range(self, exchange: str, symbol: str, interval: str
                           ) -> tuple[Optional[datetime], Optional[datetime], int]:
        """(first open time, last open time, count) of stored candles."""
        c = candles.c
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                sa.select(sa.func.min(c.open_time), sa.func.max(c.open_time), sa.func.count())
                .where(c.exchange == exchange, c.symbol == symbol, c.interval == interval)
            )).first()
        return _utc(row[0]), _utc(row[1]), int(row[2] or 0)

    async def load_candles(self, exchange: str, symbol: str, interval: str,
                           start: Optional[datetime] = None,
                           end: Optional[datetime] = None) -> pd.DataFrame:
        """OHLCV indexed by open time (UTC), oldest first."""
        c = candles.c
        q = (sa.select(c.open_time, c.open, c.high, c.low, c.close, c.volume, c.turnover)
             .where(c.exchange == exchange, c.symbol == symbol, c.interval == interval)
             .order_by(c.open_time))
        if start is not None:
            q = q.where(c.open_time >= start)
        if end is not None:
            q = q.where(c.open_time < end)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(q)).all()
        df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close",
                                         "volume", "turnover"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        return df.set_index("open_time")

    async def save_funding(self, exchange: str, symbol: str,
                           rows: list[tuple[datetime, float]]) -> None:
        for i in range(0, len(rows), CHUNK):
            values = [{"exchange": exchange, "symbol": symbol, "funding_time": t, "rate": r}
                      for t, r in rows[i:i + CHUNK]]
            stmt = self._insert(funding_rates).values(values).on_conflict_do_nothing(
                index_elements=["exchange", "symbol", "funding_time"])
            async with self._engine.begin() as conn:
                await conn.execute(stmt)

    async def load_funding(self, exchange: str, symbol: str) -> dict[datetime, float]:
        f = funding_rates.c
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                sa.select(f.funding_time, f.rate)
                .where(f.exchange == exchange, f.symbol == symbol)
            )).all()
        return {_utc(t): r for t, r in rows}

    async def save_instrument(self, exchange: str, rules: InstrumentRules) -> None:
        now = datetime.now(timezone.utc)
        value = rules.model_dump(mode="json")
        stmt = self._insert(instruments).values(
            exchange=exchange, symbol=rules.symbol, rules=value, updated_at=now
        ).on_conflict_do_update(index_elements=["exchange", "symbol"],
                                set_={"rules": value, "updated_at": now})
        async with self._engine.begin() as conn:
            await conn.execute(stmt)

    async def load_instrument(self, exchange: str, symbol: str) -> Optional[InstrumentRules]:
        i = instruments.c
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                sa.select(i.rules).where(i.exchange == exchange, i.symbol == symbol)
            )).first()
        return InstrumentRules(**row[0]) if row else None

    # ── Runs and the journal ───────────────────────────────────────────

    async def save_run(self, mode: str, strategies: list, symbols: list[str], interval: str,
                       start: Optional[datetime], end: Optional[datetime], config: dict,
                       metrics: Optional[dict] = None,
                       by_phase: Optional[dict] = None) -> int:
        async with self._engine.begin() as conn:
            result = await conn.execute(research_runs.insert().values(
                created_at=datetime.now(timezone.utc), mode=mode,
                strategies=_json(strategies), symbols=symbols, interval=interval,
                start=start, end=end, config=_json(config),
                metrics=_json(metrics), by_phase=_json(by_phase),
            ).returning(research_runs.c.id))
            return int(result.scalar_one())

    async def update_run(self, run_id: int, metrics: dict, by_phase: Optional[dict] = None,
                         end: Optional[datetime] = None) -> None:
        values = {"metrics": _json(metrics)}
        if by_phase is not None:
            values["by_phase"] = _json(by_phase)
        if end is not None:
            values["end"] = end
        async with self._engine.begin() as conn:
            await conn.execute(research_runs.update()
                               .where(research_runs.c.id == run_id).values(**values))

    async def save_trades(self, run_id: Optional[int], mode: str, exchange: str,
                          trades: list) -> None:
        rows = [{
            "run_id": run_id, "mode": mode, "exchange": exchange,
            "symbol": t.symbol, "strategy": t.strategy,
            "market_regime": t.market_regime, "market_phases": t.market_phases,
            "direction": t.direction, "opened_at": t.opened_at, "closed_at": t.closed_at,
            "entry": t.entry, "exit": t.exit, "stop": t.stop, "target": t.target,
            "quantity": t.quantity, "notional": t.notional, "leverage": t.leverage,
            "risk_usd": t.risk_usd, "r_multiple": t.r_multiple, "gross_pnl": t.gross_pnl,
            "fees": t.fees, "funding": t.funding, "net_pnl": t.net_pnl,
            "mae_r": t.mae_r, "mfe_r": t.mfe_r, "slippage_usd": t.slippage_usd,
            "exit_reason": t.exit_reason, "decision_confidence": t.confidence,
            "details": _json(t.details),
        } for t in trades]
        for i in range(0, len(rows), CHUNK):
            async with self._engine.begin() as conn:
                await conn.execute(trade_journal.insert(), rows[i:i + CHUNK])


def _utc(t: Optional[datetime]) -> Optional[datetime]:
    if t is None:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def _json(v):
    """Plain JSON: datetimes as text, NaN and infinity as null (PostgreSQL's
    JSONB refuses them)."""
    if v is None:
        return None
    return _clean(json.loads(json.dumps(v, default=str)))


def _clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v
