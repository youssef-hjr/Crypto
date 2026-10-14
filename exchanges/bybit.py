"""
Bybit V5 API Adapter.

Uses the official Bybit V5 API for:
  - REST: account, market data snapshots, order management
  - WebSocket: real-time orderbook, trades, tickers

Public market data comes from mainnet (read-only, no key).
The account side targets Bybit Demo Trading by default.
"""
from __future__ import annotations

import asyncio
import json
import time
import hmac
import hashlib
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Optional

import aiohttp
import structlog

from exchanges.base import ExchangeAdapter, ExchangeError
from models import (
    Candle, InstrumentRules, OrderBookLevel, OrderBookSnapshot, OrderRequest, OrderStatus,
    Position, PositionSide, Side, TickerInfo, Trade,
)
from config.settings import BybitSettings
from events import EventBus, Event, EventType

logger = structlog.get_logger()


class BybitAdapter(ExchangeAdapter):
    """
    Bybit V5 API adapter.
    Handles REST calls, WebSocket connections, and order book reconstruction.
    """

    def __init__(self, settings: BybitSettings, event_bus: EventBus):
        self._settings = settings
        self._event_bus = event_bus
        self._urls = settings.get_urls()
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_public: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_tasks: list[asyncio.Task] = []
        self._connected = False

        # Local order book cache: symbol -> OrderBookSnapshot
        self._orderbooks: dict[str, OrderBookSnapshot] = {}
        # Trade buffer per symbol
        self._recent_trades: dict[str, list[Trade]] = {}
        # Ticker cache
        self._tickers: dict[str, TickerInfo] = {}

        # Callbacks for WebSocket data
        self._orderbook_callbacks: list[Callable] = []
        self._trade_callbacks: list[Callable] = []

    # ── Connection ─────────────────────────────────────────────────────

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        self._connected = True
        logger.info("bybit_connected",
                     account_env=self._settings.account_env.value,
                     market_data_url=self._urls["rest_public"],
                     account_url=self._urls["rest_private"])
        await self._event_bus.publish(Event(
            EventType.EXCHANGE_CONNECTED, {"exchange": "bybit"}, source="bybit"
        ))

    async def disconnect(self) -> None:
        for task in self._ws_tasks:
            task.cancel()
        if self._ws_public and not self._ws_public.closed:
            await self._ws_public.close()
        if self._session and not self._session.closed:
            await self._session.close()
        self._connected = False
        logger.info("bybit_disconnected")

    @property
    def name(self) -> str:
        return "bybit"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._session is not None and not self._session.closed

    # ── REST Helpers ───────────────────────────────────────────────────

    def _sign_request(self, params: dict) -> dict:
        """Generate authentication headers for private endpoints."""
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        param_str = timestamp + self._settings.api_key + recv_window

        if params:
            sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            param_str += sorted_params

        signature = hmac.new(
            self._settings.api_secret.encode("utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "X-BAPI-API-KEY": self._settings.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json",
        }

    async def _request(
        self, method: str, path: str, params: dict = None,
        signed: bool = False
    ) -> dict:
        """Make a REST API request."""
        if not self._session:
            raise RuntimeError("Not connected. Call connect() first.")

        # Signed (account) calls go to the account environment; public
        # market data always comes from mainnet.
        base = self._urls["rest_private"] if signed else self._urls["rest_public"]
        url = f"{base}{path}"
        params = params or {}
        headers = self._sign_request(params) if signed else {"Content-Type": "application/json"}

        try:
            if method == "GET":
                async with self._session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            else:
                async with self._session.post(url, json=params, headers=headers) as resp:
                    data = await resp.json()

            if data.get("retCode") != 0:
                logger.error("bybit_api_error", path=path, ret_code=data.get("retCode"),
                             ret_msg=data.get("retMsg"))
            return data

        except Exception as e:
            logger.error("bybit_request_error", path=path, error=str(e))
            return {"retCode": -1, "retMsg": str(e), "result": {}}

    # ── Account ────────────────────────────────────────────────────────

    async def get_balance(self) -> dict:
        data = await self._request("GET", "/v5/account/wallet-balance",
                                   {"accountType": "UNIFIED"}, signed=True)
        result = {}
        for account in data.get("result", {}).get("list", []):
            for coin in account.get("coin", []):
                symbol = coin.get("coin", "")
                available = float(coin.get("availableToWithdraw", 0))
                result[symbol] = available
        return result

    @staticmethod
    def _ok(data: dict, path: str) -> dict:
        if data.get("retCode") != 0:
            raise ExchangeError(f"{path}: {data.get('retCode')} {data.get('retMsg')}")
        return data

    async def get_equity(self) -> float:
        # Raises instead of returning 0.0: a failed call must not look like
        # an empty account, or drawdown would read 100%.
        path = "/v5/account/wallet-balance"
        data = self._ok(await self._request("GET", path, {"accountType": "UNIFIED"},
                                            signed=True), path)
        for account in data.get("result", {}).get("list", []):
            equity = account.get("totalEquity")
            if equity not in (None, ""):
                return float(equity)
        raise ExchangeError(f"{path}: no totalEquity in response")

    async def get_positions(self) -> list[Position]:
        # Raises instead of returning []: hiding open positions would let
        # the risk engine approve trades against exposure it can't see.
        path = "/v5/position/list"
        data = self._ok(await self._request("GET", path,
                                            {"category": "linear", "settleCoin": "USDT"},
                                            signed=True), path)
        positions = []
        for pos in data.get("result", {}).get("list", []):
            size = float(pos.get("size", 0))
            if size == 0:
                continue
            positions.append(Position(
                symbol=pos["symbol"],
                side=PositionSide.LONG if pos["side"] == "Buy" else PositionSide.SHORT,
                entry_price=float(pos.get("avgPrice", 0)),
                quantity=size,
                leverage=float(pos.get("leverage", 1)),
                unrealized_pnl=float(pos.get("unrealisedPnl", 0)),
                mark_price=float(mp) if (mp := pos.get("markPrice")) else None,
                stop_loss=float(sl) if (sl := pos.get("stopLoss")) and sl != "0" else None,
                take_profit=float(tp) if (tp := pos.get("takeProfit")) and tp != "0" else None,
            ))
        return positions

    # ── Market Data (REST) ─────────────────────────────────────────────

    async def get_ticker(self, symbol: str) -> TickerInfo:
        data = await self._request("GET", "/v5/market/tickers",
                                   {"category": "linear", "symbol": symbol})
        items = data.get("result", {}).get("list", [])
        if not items:
            raise ValueError(f"No ticker data for {symbol}")
        t = items[0]
        return self._parse_ticker(t)

    async def get_tickers(self) -> list[TickerInfo]:
        data = await self._request("GET", "/v5/market/tickers",
                                   {"category": "linear"})
        tickers = []
        for t in data.get("result", {}).get("list", []):
            try:
                tickers.append(self._parse_ticker(t))
            except (ValueError, KeyError):
                continue
        return tickers

    def _parse_ticker(self, t: dict) -> TickerInfo:
        return TickerInfo(
            symbol=t["symbol"],
            last_price=float(t.get("lastPrice", 0)),
            volume_24h=float(t.get("volume24h", 0)),
            turnover_24h=float(t.get("turnover24h", 0)),
            high_24h=float(t.get("highPrice24h", 0)),
            low_24h=float(t.get("lowPrice24h", 0)),
            price_change_pct_24h=float(t.get("price24hPcnt", 0)) * 100,
            funding_rate=float(fr) if (fr := t.get("fundingRate")) else None,
            open_interest=float(oi) if (oi := t.get("openInterest")) else None,
        )

    async def get_orderbook(self, symbol: str, depth: int = 50) -> OrderBookSnapshot:
        # Only the WebSocket keeps the cached book current. Use it while it's
        # fresh; otherwise take a new REST snapshot (and don't cache that,
        # or the same snapshot would be returned forever).
        cached = self._orderbooks.get(symbol)
        if cached is not None and (
            datetime.now(timezone.utc) - cached.timestamp).total_seconds() < 2:
            return cached

        data = await self._request("GET", "/v5/market/orderbook",
                                   {"category": "linear", "symbol": symbol, "limit": depth})
        result = data.get("result", {})
        bids = [OrderBookLevel(price=float(b[0]), quantity=float(b[1]))
                for b in result.get("b", [])]
        asks = [OrderBookLevel(price=float(a[0]), quantity=float(a[1]))
                for a in result.get("a", [])]

        snapshot = OrderBookSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
            sequence=int(result.get("seq", 0)),
        )
        return snapshot

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> list[Trade]:
        data = await self._request("GET", "/v5/market/recent-trading-records",
                                   {"category": "linear", "symbol": symbol, "limit": limit})
        trades = []
        for t in data.get("result", {}).get("list", []):
            trades.append(Trade(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(int(t["T"]) / 1000, tz=timezone.utc),
                price=float(t["p"]),
                quantity=float(t["v"]),
                side=Side.BUY if t["S"] == "Buy" else Side.SELL,
                trade_id=t.get("i", ""),
            ))
        return trades

    async def get_candles(
        self, symbol: str, interval: str = "5", limit: int = 200
    ) -> list[Candle]:
        data = await self._request("GET", "/v5/market/kline", {
            "category": "linear", "symbol": symbol,
            "interval": interval, "limit": limit
        })
        candles = []
        now_ms = int(time.time() * 1000)
        for c in data.get("result", {}).get("list", []):
            # Bybit includes the candle that is still forming. Indicators on
            # it repaint and relative volume reads artificially low, so only
            # closed candles are returned.
            if not self._is_closed(int(c[0]), interval, now_ms):
                continue
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5]),
                turnover=float(c[6]) if len(c) > 6 else 0.0,
                symbol=symbol,
            ))
        # API returns newest first; reverse to chronological
        candles.reverse()
        return candles

    @staticmethod
    def _is_closed(start_ms: int, interval: str, now_ms: int) -> bool:
        """Whether a kline starting at start_ms has finished by now_ms."""
        if interval.isdigit():
            minutes = int(interval)
        elif interval == "D":
            minutes = 1440
        elif interval == "W":
            minutes = 10080
        elif interval == "M":
            start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
            now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
            return (now.year, now.month) != (start.year, start.month)
        else:
            raise ValueError(f"Unknown kline interval: {interval}")
        return start_ms + minutes * 60_000 <= now_ms

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        data = await self._request("GET", "/v5/market/funding/history",
                                   {"category": "linear", "symbol": symbol, "limit": 1})
        items = data.get("result", {}).get("list", [])
        if items:
            return float(items[0].get("fundingRate", 0))
        return None

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        data = await self._request("GET", "/v5/market/open-interest",
                                   {"category": "linear", "symbol": symbol,
                                    "intervalTime": "5min", "limit": 1})
        items = data.get("result", {}).get("list", [])
        if items:
            return float(items[0].get("openInterest", 0))
        return None

    # ── WebSocket ──────────────────────────────────────────────────────

    async def start_public_ws(self) -> None:
        """Start the public WebSocket connection and message loop."""
        task = asyncio.create_task(self._ws_public_loop())
        self._ws_tasks.append(task)

    async def _ws_public_loop(self) -> None:
        """Main WebSocket loop with reconnection logic."""
        while self._connected:
            try:
                async with self._session.ws_connect(
                    self._urls["ws_public"], heartbeat=20
                ) as ws:
                    self._ws_public = ws
                    logger.info("bybit_ws_connected", url=self._urls["ws_public"])
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_ws_message(json.loads(msg.data))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning("bybit_ws_closed")
                            break
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("bybit_ws_error", error=str(e))

            if self._connected:
                logger.info("bybit_ws_reconnecting", delay=5)
                await asyncio.sleep(5)

    async def _handle_ws_message(self, data: dict) -> None:
        """Route incoming WebSocket messages."""
        topic = data.get("topic", "")

        if topic.startswith("orderbook"):
            await self._handle_orderbook_ws(data)
        elif topic.startswith("publicTrade"):
            await self._handle_trade_ws(data)
        elif topic.startswith("tickers"):
            await self._handle_ticker_ws(data)
        elif "pong" in str(data.get("op", "")):
            pass  # heartbeat response
        elif data.get("success") is not None:
            logger.debug("bybit_ws_sub_response", data=data)

    async def _ws_subscribe(self, topics: list[str]) -> None:
        """Subscribe to WebSocket topics."""
        if self._ws_public and not self._ws_public.closed:
            msg = {"op": "subscribe", "args": topics}
            await self._ws_public.send_json(msg)
            logger.info("bybit_ws_subscribed", topics=topics)

    async def subscribe_orderbook(self, symbol: str, depth: int = 50) -> None:
        await self._ws_subscribe([f"orderbook.{depth}.{symbol}"])

    async def subscribe_trades(self, symbol: str) -> None:
        await self._ws_subscribe([f"publicTrade.{symbol}"])

    async def subscribe_ticker(self, symbol: str) -> None:
        await self._ws_subscribe([f"tickers.{symbol}"])

    # ── WebSocket Handlers ─────────────────────────────────────────────

    async def _handle_orderbook_ws(self, data: dict) -> None:
        """
        Handle orderbook updates with proper snapshot/delta management.
        Type 'snapshot': full replacement.
        Type 'delta': incremental update.
        """
        msg_type = data.get("type", "")
        topic = data.get("topic", "")
        symbol = topic.split(".")[-1] if "." in topic else ""
        result = data.get("data", {})

        bids_raw = result.get("b", [])
        asks_raw = result.get("a", [])
        seq = int(result.get("seq", 0))

        if msg_type == "snapshot":
            bids = [OrderBookLevel(price=float(b[0]), quantity=float(b[1]))
                    for b in bids_raw]
            asks = [OrderBookLevel(price=float(a[0]), quantity=float(a[1]))
                    for a in asks_raw]
            self._orderbooks[symbol] = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bids=sorted(bids, key=lambda x: x.price, reverse=True),
                asks=sorted(asks, key=lambda x: x.price),
                sequence=seq,
            )
        elif msg_type == "delta":
            if symbol not in self._orderbooks:
                logger.warning("orderbook_delta_no_snapshot", symbol=symbol)
                return

            current = self._orderbooks[symbol]

            # Validate sequence
            if seq <= current.sequence:
                return  # stale

            # Apply delta
            bid_dict = {b.price: b.quantity for b in current.bids}
            for b in bids_raw:
                price, qty = float(b[0]), float(b[1])
                if qty == 0:
                    bid_dict.pop(price, None)
                else:
                    bid_dict[price] = qty

            ask_dict = {a.price: a.quantity for a in current.asks}
            for a in asks_raw:
                price, qty = float(a[0]), float(a[1])
                if qty == 0:
                    ask_dict.pop(price, None)
                else:
                    ask_dict[price] = qty

            new_bids = [OrderBookLevel(price=p, quantity=q)
                        for p, q in sorted(bid_dict.items(), reverse=True)]
            new_asks = [OrderBookLevel(price=p, quantity=q)
                        for p, q in sorted(ask_dict.items())]

            self._orderbooks[symbol] = OrderBookSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                bids=new_bids,
                asks=new_asks,
                sequence=seq,
            )

        # Publish event
        if symbol in self._orderbooks:
            await self._event_bus.publish(Event(
                EventType.ORDERBOOK_UPDATED,
                self._orderbooks[symbol],
                source="bybit",
            ))

    async def _handle_trade_ws(self, data: dict) -> None:
        """Handle incoming trade stream messages."""
        topic = data.get("topic", "")
        symbol = topic.split(".")[-1] if "." in topic else ""

        for t in data.get("data", []):
            trade = Trade(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(int(t["T"]) / 1000, tz=timezone.utc),
                price=float(t["p"]),
                quantity=float(t["v"]),
                side=Side.BUY if t["S"] == "Buy" else Side.SELL,
                trade_id=t.get("i", ""),
            )

            if symbol not in self._recent_trades:
                self._recent_trades[symbol] = []
            self._recent_trades[symbol].append(trade)
            # Keep buffer bounded
            if len(self._recent_trades[symbol]) > 5000:
                self._recent_trades[symbol] = self._recent_trades[symbol][-2500:]

            await self._event_bus.publish(Event(
                EventType.TRADE_RECEIVED, trade, source="bybit"
            ))

    async def _handle_ticker_ws(self, data: dict) -> None:
        """Handle ticker stream updates."""
        result = data.get("data", {})
        symbol = result.get("symbol", "")
        if not symbol:
            return

        existing = self._tickers.get(symbol)

        self._tickers[symbol] = TickerInfo(
            symbol=symbol,
            last_price=float(result.get("lastPrice", existing.last_price if existing else 0)),
            volume_24h=float(result.get("volume24h", existing.volume_24h if existing else 0)),
            turnover_24h=float(result.get("turnover24h", existing.turnover_24h if existing else 0)),
            high_24h=float(result.get("highPrice24h", existing.high_24h if existing else 0)),
            low_24h=float(result.get("lowPrice24h", existing.low_24h if existing else 0)),
            funding_rate=float(fr) if (fr := result.get("fundingRate")) else (existing.funding_rate if existing else None),
            open_interest=float(oi) if (oi := result.get("openInterest")) else (existing.open_interest if existing else None),
        )

        await self._event_bus.publish(Event(
            EventType.TICKER_UPDATED, self._tickers[symbol], source="bybit"
        ))

    # ── Orders (DISABLED for Milestone 1) ──────────────────────────────

    async def place_order(self, order: OrderRequest) -> dict:
        logger.warning("order_execution_disabled",
                        msg="Execution is disabled in current phase")
        return {"status": "disabled", "msg": "Execution not enabled"}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        logger.warning("order_cancellation_disabled")
        return False

    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        return OrderStatus.REJECTED

    # ── Instrument Info ────────────────────────────────────────────────

    async def get_exchange_info(self, symbol: str) -> dict:
        data = await self._request("GET", "/v5/market/instruments-info",
                                   {"category": "linear", "symbol": symbol})
        items = data.get("result", {}).get("list", [])
        return items[0] if items else {}

    async def get_instrument_rules(self, symbol: str) -> InstrumentRules:
        info = await self.get_exchange_info(symbol)
        lot = info.get("lotSizeFilter") or {}
        if not lot.get("qtyStep"):
            raise ExchangeError(f"No lot size rules for {symbol}")
        return InstrumentRules(
            symbol=symbol,
            qty_step=float(lot["qtyStep"]),
            min_qty=float(lot.get("minOrderQty", 0)),
            max_qty=float(lot["maxOrderQty"]) if lot.get("maxOrderQty") else None,
            min_notional=float(lot.get("minNotionalValue", 0) or 0),
        )

    # ── Accessors ──────────────────────────────────────────────────────

    def get_cached_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        return self._orderbooks.get(symbol)

    def get_cached_trades(self, symbol: str, limit: int = 500) -> list[Trade]:
        return self._recent_trades.get(symbol, [])[-limit:]

    def get_cached_ticker(self, symbol: str) -> Optional[TickerInfo]:
        return self._tickers.get(symbol)
