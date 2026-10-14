"""
Tests for the Bybit adapter's data handling (no network).
"""
import time

import pytest

from config.settings import AccountEnv, BybitSettings
from events import EventBus
from exchanges.bybit import BybitAdapter

MIN = 60_000


def test_market_data_always_from_mainnet():
    for env in AccountEnv:
        urls = BybitSettings(BYBIT_ACCOUNT_ENV=env.value).get_urls()
        assert urls["rest_public"] == "https://api.bybit.com"
        assert urls["ws_public"].startswith("wss://stream.bybit.com/")


def test_default_account_is_demo():
    urls = BybitSettings().get_urls()
    assert urls["rest_private"] == "https://api-demo.bybit.com"


def test_is_closed():
    now = 1_700_000_000_000
    assert BybitAdapter._is_closed(now - 5 * MIN, "5", now)
    assert not BybitAdapter._is_closed(now - 4 * MIN, "5", now)
    assert not BybitAdapter._is_closed(now - 60 * MIN, "D", now)


@pytest.mark.asyncio
async def test_get_candles_drops_forming_candle(monkeypatch):
    adapter = BybitAdapter(BybitSettings(), EventBus())
    now_ms = int(time.time() * 1000)
    current = now_ms - (now_ms % (5 * MIN))  # start of the forming 5m bar
    rows = [  # Bybit returns newest first
        [str(current), "10", "11", "9", "10.5", "1", "10"],
        [str(current - 5 * MIN), "9", "10", "8", "10", "100", "1000"],
        [str(current - 10 * MIN), "8", "9", "7", "9", "100", "900"],
    ]

    async def fake_request(method, path, params=None, signed=False):
        return {"retCode": 0, "result": {"list": rows}}

    monkeypatch.setattr(adapter, "_request", fake_request)
    candles = await adapter.get_candles("BTCUSDT", interval="5", limit=3)

    assert len(candles) == 2
    assert candles[-1].volume == 100  # last returned candle is the closed one
    assert candles[0].timestamp < candles[1].timestamp  # chronological


@pytest.mark.asyncio
async def test_account_errors_raise_instead_of_looking_empty(monkeypatch):
    from exchanges.base import ExchangeError
    adapter = BybitAdapter(BybitSettings(), EventBus())

    async def failing(method, path, params=None, signed=False):
        return {"retCode": -1, "retMsg": "timeout", "result": {}}

    monkeypatch.setattr(adapter, "_request", failing)
    with pytest.raises(ExchangeError):
        await adapter.get_equity()
    with pytest.raises(ExchangeError):
        await adapter.get_positions()


@pytest.mark.asyncio
async def test_instrument_rules(monkeypatch):
    adapter = BybitAdapter(BybitSettings(), EventBus())

    async def fake(method, path, params=None, signed=False):
        return {"retCode": 0, "result": {"list": [{"lotSizeFilter": {
            "qtyStep": "0.001", "minOrderQty": "0.001", "maxOrderQty": "1190",
            "minNotionalValue": "5"}}]}}

    monkeypatch.setattr(adapter, "_request", fake)
    rules = await adapter.get_instrument_rules("BTCUSDT")
    assert (rules.qty_step, rules.min_qty, rules.max_qty, rules.min_notional) == (
        0.001, 0.001, 1190.0, 5.0)


@pytest.mark.asyncio
async def test_rest_orderbook_is_not_cached_forever(monkeypatch):
    adapter = BybitAdapter(BybitSettings(), EventBus())
    calls = []

    async def fake(method, path, params=None, signed=False):
        calls.append(path)
        return {"retCode": 0, "result": {"b": [["99", "1"]], "a": [["101", "1"]], "seq": 1}}

    monkeypatch.setattr(adapter, "_request", fake)
    await adapter.get_orderbook("BTCUSDT")
    await adapter.get_orderbook("BTCUSDT")
    assert len(calls) == 2
