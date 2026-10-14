"""
History downloader: candles, funding rates and order rules from the
exchange into the database, for backtests.

Downloads are incremental. Candles already stored are skipped, so running
the same command again only fetches what is missing at either end, and an
interrupted download keeps everything saved so far.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from database.repository import MarketDataRepository
from database.research_repository import ResearchRepository
from exchanges.base import ExchangeAdapter
from research.backtester import SymbolData
from strategies.features import interval_minutes

logger = structlog.get_logger()

BARS_PER_REQUEST = 1000


class HistoryDownloader:

    def __init__(self, exchange: ExchangeAdapter, market_repo: MarketDataRepository,
                 research_repo: ResearchRepository):
        self._exchange = exchange
        self._market = market_repo
        self._research = research_repo

    async def download(self, symbol: str, interval: str, start: datetime,
                       end: Optional[datetime] = None) -> dict:
        end = end or datetime.now(timezone.utc)
        bar = timedelta(minutes=interval_minutes(interval))
        name = self._exchange.name
        first, last, _ = await self._research.candle_range(name, symbol, interval)

        ranges = []
        if first is None:
            ranges.append((start, end))
        else:
            if start < first:
                ranges.append((start, first))
            if last + bar < end:
                ranges.append((last + bar, end))

        fetched = 0
        for lo, hi in ranges:
            # Newest window first: if the contract is younger than `start`,
            # the first empty window means there is nothing older.
            windows = []
            t = lo
            while t < hi:
                windows.append((t, min(hi, t + bar * BARS_PER_REQUEST)))
                t += bar * BARS_PER_REQUEST
            for w_lo, w_hi in reversed(windows):
                rows = await self._exchange.get_candle_history(symbol, interval, w_lo, w_hi)
                if not rows:
                    logger.info("history_no_older_data", symbol=symbol, before=str(w_hi))
                    break
                await self._market.save_candles(name, symbol, interval, rows)
                fetched += len(rows)
                logger.info("history_saved", symbol=symbol, interval=interval,
                            from_=str(rows[0].timestamp), to=str(rows[-1].timestamp),
                            total=fetched)

        funding = await self._exchange.get_funding_history(symbol, start, end)
        await self._research.save_funding(name, symbol, funding)

        try:
            rules = await self._exchange.get_instrument_rules(symbol)
            await self._research.save_instrument(name, rules)
        except Exception as e:  # Rules are optional; backtests fall back to defaults
            logger.warning("instrument_rules_unavailable", symbol=symbol, error=str(e))

        first, last, count = await self._research.candle_range(name, symbol, interval)
        return {"symbol": symbol, "interval": interval, "new_candles": fetched,
                "stored_candles": count, "first": first, "last": last,
                "funding_rows": len(funding)}


async def load_history(research_repo: ResearchRepository, exchange: str,
                       symbols: list[str], interval: str,
                       start: Optional[datetime] = None,
                       end: Optional[datetime] = None) -> dict[str, SymbolData]:
    """Stored history for a backtest. Loads from 300 bars (at least 30 days)
    before `start` so indicators like EMA 200 are warm on the first bar."""
    warm = max(timedelta(days=30), timedelta(minutes=interval_minutes(interval) * 300))
    warm_start = start - warm if start is not None else None
    out = {}
    for sym in symbols:
        df = await research_repo.load_candles(exchange, sym, interval, warm_start, end)
        if df.empty:
            logger.warning("no_history", symbol=sym, interval=interval,
                           hint="run: python backtest.py download")
            continue
        out[sym] = SymbolData(
            symbol=sym, candles=df,
            funding=await research_repo.load_funding(exchange, sym),
            rules=await research_repo.load_instrument(exchange, sym),
        )
    return out
