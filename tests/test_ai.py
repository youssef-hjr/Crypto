"""
Claude integration (Milestone 4), offline: a scripted stand-in replaces the
API, so these tests never call Anthropic and cost nothing.

What must hold: anything but a valid, confident TRADE is NO TRADE; Claude
can only remove trades, never change them; the Risk Manager still decides;
no key ever reaches a prompt; and every decision is stored.
"""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from ai import AIDecision, ClaudeAdvisor, LLMResponse, ScriptedLLM, Verdict
from ai.briefing import Candidate, MarketSnapshot, build_brief, find_secrets, risk_state, to_json
from ai.schema import DECISION_SCHEMA
from config.settings import AISettings, RiskSettings
from database import AIDecisionRepository, MarketDataRepository
from database.models import metadata
from events import EventBus
from models import (
    Candle, OrderBookLevel, OrderBookSnapshot, PortfolioState, Position, PositionSide, Side,
    TechnicalFeatures,
)
from risk.risk_manager import RiskManager
from tests.test_paper import FakeLive, make_trader

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)


def answer(decision="TRADE", confidence=0.7, **kw) -> str:
    d = {"decision": decision, "reasoning": "Trend and order flow agree; stop is 2 ATR.",
         "confidence": confidence, "expected_r": 0.3, "invalidation": "Close back under EMA 50",
         "market_regime": "weak_bullish", "portfolio_impact": "First position, no overlap",
         "conflicting_signals": [], "missing_information": ["news"]}
    d.update(kw)
    return json.dumps(d)


def cand(symbol="BTCUSDT", score=80.0, strategy="trend", side=Side.BUY) -> Candidate:
    return Candidate(symbol=symbol, direction=side, strategy=strategy, entry=100.0,
                     stop=98.0 if side == Side.BUY else 102.0, target=104.0, score=score,
                     confidence=0.8, expected_r=2.0, reason="EMA 20/50 cross up",
                     regime="weak_bullish", bar_time=NOW - timedelta(hours=1), bar_minutes=60)


def candles(n=60, price=100.0):
    t0 = NOW - timedelta(hours=n)
    return [Candle(timestamp=t0 + timedelta(hours=i), open=price, high=price * 1.01,
                   low=price * 0.99, close=price * (1 + i / 1000), volume=10.0, symbol="BTCUSDT")
            for i in range(n)]


def snapshot() -> MarketSnapshot:
    book = OrderBookSnapshot(
        symbol="BTCUSDT", timestamp=NOW,
        bids=[OrderBookLevel(price=99.99 - i * 0.01, quantity=100) for i in range(5)],
        asks=[OrderBookLevel(price=100.01 + i * 0.01, quantity=100) for i in range(5)])
    tech = TechnicalFeatures(symbol="BTCUSDT", timestamp=NOW, ema_20=99.5, ema_50=99.0,
                             ema_200=95.0, rsi=58.0, atr=1.0, momentum=0.01)
    return MarketSnapshot(technical=tech, candles=candles(), orderbook=book)


def portfolio(positions=()):
    return PortfolioState(equity=10_000, available_balance=10_000, unrealized_pnl=0,
                          realized_pnl_today=0, positions=list(positions), source="paper")


def advisor(llm, journal=None, **settings) -> ClaudeAdvisor:
    return ClaudeAdvisor(llm, AISettings(**settings), mode="paper", journal=journal)


async def review_one(adv, c=None, now=NOW):
    c = c or cand()
    briefs = {c.key: build_brief(c, snapshot(), portfolio(), {"circuit_breaker_active": False})}
    return (await adv.review([c], briefs, now=now))[0]


# ── Answers ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_valid_trade_answer_is_used_as_given():
    r = await review_one(advisor(ScriptedLLM([answer()])))
    assert r.approved and r.ai_said == Verdict.TRADE and r.error is None
    d = r.decision
    assert (d.confidence, d.expected_r, d.market_regime.value) == (0.7, 0.3, "weak_bullish")
    assert d.invalidation and d.portfolio_impact and d.missing_information == ["news"]


@pytest.mark.asyncio
async def test_no_trade_answer_blocks_the_trade():
    r = await review_one(advisor(ScriptedLLM([answer("NO_TRADE", 0.8)])))
    assert not r.approved and r.ai_said == Verdict.NO_TRADE and r.error is None


@pytest.mark.parametrize("bad", [
    "not json at all",
    answer(confidence=1.7),  # Out of range
    answer(decision="BUY"),  # Not an allowed answer
    answer(reasoning="   "),  # Empty reasoning
    answer(market_regime="moon"),
    json.dumps({"decision": "TRADE"}),  # Fields missing
])
@pytest.mark.asyncio
async def test_invalid_answers_become_no_trade(bad):
    r = await review_one(advisor(ScriptedLLM([bad])))
    assert not r.approved and r.ai_said is None and r.error.startswith("Invalid answer")


@pytest.mark.parametrize("stop_reason,expected", [
    ("refusal", "declined"), ("max_tokens", "cut off")])
@pytest.mark.asyncio
async def test_refusals_and_cut_off_answers_become_no_trade(stop_reason, expected):
    resp = LLMResponse(text=answer(), stop_reason=stop_reason, model="m")
    r = await review_one(advisor(ScriptedLLM([resp])))
    assert not r.approved and expected in r.error


@pytest.mark.asyncio
async def test_api_errors_become_no_trade_and_are_retried_next_cycle():
    llm = ScriptedLLM([ConnectionError("network down"), answer()])
    adv = advisor(llm)
    r = await review_one(adv)
    assert not r.approved and "network down" in r.error
    r = await review_one(adv, now=NOW + timedelta(seconds=30))  # Failures aren't cached
    assert r.approved and len(llm.calls) == 2


@pytest.mark.asyncio
async def test_a_slow_answer_times_out_as_no_trade():
    class Slow(ScriptedLLM):
        async def complete(self, *a):
            await asyncio.sleep(5)
    r = await review_one(advisor(Slow(), timeout_seconds=0.05))
    assert not r.approved and "No answer within" in r.error


@pytest.mark.asyncio
async def test_low_confidence_trade_is_downgraded():
    r = await review_one(advisor(ScriptedLLM([answer(confidence=0.4)]), min_confidence=0.55))
    assert not r.approved and r.ai_said == Verdict.TRADE
    assert "under the minimum" in r.decision.reasoning


# ── Cost control ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_same_candidate_is_not_asked_twice_within_the_cache_window():
    llm = ScriptedLLM(default=answer())
    adv = advisor(llm, cache_minutes=30)
    await review_one(adv)
    r = await review_one(adv, now=NOW + timedelta(minutes=10))
    assert r.cached and r.approved and len(llm.calls) == 1
    await review_one(adv, now=NOW + timedelta(minutes=31))
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_daily_call_limit_means_no_trade_without_asking():
    llm = ScriptedLLM(default=answer())
    adv = advisor(llm, max_calls_per_day=2, cache_minutes=0)
    for i in range(2):
        assert (await review_one(adv, now=NOW + timedelta(minutes=i))).approved
    r = await review_one(adv, now=NOW + timedelta(minutes=5))
    assert not r.approved and "Daily limit" in r.error and len(llm.calls) == 2
    r = await review_one(adv, now=NOW + timedelta(days=1))  # New UTC day
    assert r.approved and len(llm.calls) == 3


@pytest.mark.asyncio
async def test_only_the_best_candidates_are_sent():
    llm = ScriptedLLM(default=answer())
    adv = advisor(llm, max_candidates_per_cycle=2)
    cs = [cand("A", 75), cand("B", 90), cand("C", 85)]
    assert [c.symbol for c in adv.top(cs)] == ["B", "C"]
    briefs = {c.key: build_brief(c, snapshot(), portfolio(), {}) for c in adv.top(cs)}
    reviews = await adv.review(cs, briefs, now=NOW)
    assert {r.candidate.symbol: r.approved for r in reviews} == {"B": True, "C": True, "A": False}
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_nothing_is_sent_while_the_circuit_breaker_is_active():
    llm = ScriptedLLM(default=answer())
    c = cand()
    reviews = await advisor(llm).review([c], {c.key: {}}, breaker_active=True, now=NOW)
    assert not reviews[0].approved and llm.calls == []


# ── What Claude sees ───────────────────────────────────────────────────

def test_brief_has_every_section_and_is_plain_json():
    pos = Position(symbol="ETHUSDT", side=PositionSide.LONG, entry_price=2000, quantity=1,
                   stop_loss=1950, mark_price=2010)
    bus = EventBus()
    rm = RiskManager(RiskSettings(), bus)
    brief = build_brief(cand(), snapshot(), portfolio([pos]), risk_state(rm, RiskSettings()))
    text = to_json(brief)
    assert "NaN" not in text and "Infinity" not in text
    assert set(brief) >= {"candidate", "technical", "recent_candles", "order_book",
                          "portfolio", "risk_state", "not_available"}
    assert brief["candidate"]["side"] == "long" and brief["candidate"]["stop_distance_pct"] == 2.0
    assert len(brief["recent_candles"]["rows"]) == 24
    assert brief["order_book"]["spread_bps"] == pytest.approx(2.0)
    assert brief["portfolio"]["open_positions"][0]["symbol"] == "ETHUSDT"
    assert brief["risk_state"]["risk_per_trade_pct"] == 0.05
    assert brief["order_flow"] is None  # Missing data is shown as missing


@pytest.mark.asyncio
async def test_no_key_ever_reaches_the_prompt(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "bybit-key-abc123")
    monkeypatch.setenv("BYBIT_API_SECRET", "bybit-secret-xyz789")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-000111222")
    llm = ScriptedLLM(default=answer())
    await review_one(advisor(llm))
    sent = json.dumps(llm.calls)
    for secret in ("bybit-key-abc123", "bybit-secret-xyz789", "sk-ant-test-000111222"):
        assert secret not in sent

    # And if one ever did end up in the data, the brief is not sent at all
    leaky = cand()
    leaky.reason = "debug bybit-secret-xyz789"
    assert find_secrets(to_json(leaky.as_dict())) == ["BYBIT_API_SECRET"]
    r = await review_one(advisor(llm), leaky)
    assert not r.approved and "secret" in r.error and len(llm.calls) == 1


def test_schema_matches_the_answer_model():
    assert set(DECISION_SCHEMA["required"]) == set(AIDecision.model_fields)
    assert DECISION_SCHEMA["additionalProperties"] is False


# ── The real client, with the network replaced ─────────────────────────

@pytest.mark.asyncio
async def test_anthropic_request_shape(monkeypatch):
    import anthropic
    import httpx2

    from ai.llm import AnthropicLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-000111222")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta")
        return httpx2.Response(200, headers={"request-id": "req_1"}, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": answer()}], "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3000, "output_tokens": 1000,
                      "cache_read_input_tokens": 2000}})

    llm = AnthropicLLM(AISettings())
    llm._client = anthropic.AsyncAnthropic(
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    r = await review_one(advisor(llm))
    assert r.approved
    body = seen["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["fallbacks"] == "default" and "server-side-fallback" in seen["beta"]
    assert "thinking" not in body and "temperature" not in body
    assert r.response.request_id == "req_1"
    assert r.response.cost_usd() == pytest.approx((3000 * 4 + 2000 * 0.4 + 1000 * 20) / 1e6)


def test_missing_api_key_is_a_clear_error(monkeypatch):
    from ai.llm import AnthropicLLM, MissingAPIKey
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingAPIKey, match="ANTHROPIC_API_KEY"):
        AnthropicLLM(AISettings())


# ── Paper trading with Claude in the loop ──────────────────────────────

def paper_with(llm, **kw):
    ex = FakeLive()
    trader, _ = make_trader(ex, **kw)
    trader._advisor = advisor(llm)
    return ex, trader


@pytest.mark.asyncio
async def test_paper_no_trade_means_no_position_and_no_risk_check():
    llm = ScriptedLLM([answer("NO_TRADE", 0.9)])
    ex, trader = paper_with(llm)
    await trader.load()
    report = await trader.cycle(ex.now)
    assert report.opened == [] and report.verdicts == []
    assert [r.decision.decision for r in report.ai_reviews] == [Verdict.NO_TRADE]
    brief = json.loads(llm.calls[0]["user"].split("<brief>")[1].split("</brief>")[0])
    assert brief["candidate"]["strategy"] == "scripted"
    assert brief["order_flow"] is None  # The fake exchange has no trades feed


@pytest.mark.asyncio
async def test_paper_trade_answer_goes_through_the_risk_manager_unchanged():
    ex, trader = paper_with(ScriptedLLM([answer(confidence=0.9, expected_r=5.0)]))
    await trader.load()
    report = await trader.cycle(ex.now)
    assert report.opened == ["BTCUSDT"]
    pos = trader.account.positions["BTCUSDT"]
    assert pos.risk_usd == pytest.approx(5.0, rel=0.02)  # Still 0.05%: Claude can't size
    assert pos.stop == pytest.approx(98.0)  # The strategy's stop, not Claude's
    assert pos.details["ai"]["decision"] == "TRADE"
    assert "order flow" in pos.details["ai"]["reasoning"]
    assert "Claude:" in report.verdicts[0].original_proposal.reasoning
    assert ex.orders == 0


@pytest.mark.asyncio
async def test_risk_manager_still_rejects_what_claude_approves():
    ex, trader = paper_with(ScriptedLLM([answer()]),
                            risk_settings=RiskSettings(min_opportunity_score=90))
    await trader.load()
    report = await trader.cycle(ex.now)
    assert report.ai_reviews[0].approved
    assert report.opened == []
    assert "LOW_OPPORTUNITY_SCORE" in report.verdicts[0].rejection_reasons[0]


@pytest.mark.asyncio
async def test_paper_claude_outage_means_no_trades():
    ex, trader = paper_with(ScriptedLLM([RuntimeError("503 overloaded")]))
    await trader.load()
    report = await trader.cycle(ex.now)
    assert report.opened == [] and "overloaded" in report.ai_reviews[0].error


# ── Storage ────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def journal():
    url = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    r = MarketDataRepository.from_url(url)
    async with r.engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await r.create_schema()
    yield AIDecisionRepository(r.engine)
    async with r.engine.begin() as conn:
        await conn.run_sync(metadata.drop_all)
    await r.close()


@pytest.mark.asyncio
async def test_every_decision_is_stored_with_its_brief_and_risk_verdict(journal):
    ex = FakeLive()
    trader, _ = make_trader(ex)
    trader._advisor = advisor(ScriptedLLM([answer()]), journal=journal)
    await trader.load()
    await trader.cycle(ex.now)

    rows = await journal.recent(5)
    assert len(rows) == 1
    row = rows[0]
    assert (row["decision"], row["mode"], row["risk_decision"]) == ("TRADE", "paper", "approved")
    assert row["response"]["invalidation"] == "Close back under EMA 50"
    assert await journal.calls_since(ex.now - timedelta(days=1)) == 1
    s = await journal.summary(ex.now - timedelta(days=1))
    assert s["TRADE"] == 1 and s["cost_usd"] == 0.0  # The scripted model has no price

    # A restarted advisor picks up today's count for the daily limit
    again = advisor(ScriptedLLM(), journal=journal)
    await again.load(now=ex.now)
    assert again.calls_today == 1


# ── Scanner preview (main.py --ai) ─────────────────────────────────────

class StubScanner:
    """Just the accessors the preview reads."""

    def __init__(self, price=100.0):
        self._candles = candles(60, price)
        self.price = self._candles[-1].close

    def technical(self, symbol):
        return TechnicalFeatures(symbol=symbol, timestamp=NOW, atr=1.0, rsi=60.0)

    def candles(self, symbol):
        return self._candles

    def ticker(self, symbol):
        return None

    def order_flow(self, symbol):
        return None

    def derivatives(self, symbol):
        return None


@pytest.mark.asyncio
async def test_preview_sends_only_trade_answers_to_the_risk_manager():
    from agents.risk_preview import run_risk_preview
    from config.settings import AppSettings
    from models import OpportunityScore
    from portfolio.portfolio import PortfolioService

    settings = AppSettings()
    scanner = StubScanner()
    ex = FakeLive(price=scanner.price)
    ex.now = datetime.now(timezone.utc)  # The Risk Manager checks the book's age
    opps = [OpportunityScore(symbol=s, overall_score=score, direction=Side.BUY)
            for s, score in (("BTCUSDT", 90), ("ETHUSDT", 85))]
    llm = ScriptedLLM([answer(), answer("NO_TRADE", 0.8)])
    adv = ClaudeAdvisor(llm, AISettings(), mode="preview")
    risk = RiskManager(settings.risk, EventBus())
    verdicts, reviews = await run_risk_preview(
        ex, scanner, risk, PortfolioService(ex, False, 10_000), settings, opps, advisor=adv)
    assert {r.candidate.symbol: r.approved for r in reviews} == {"BTCUSDT": True,
                                                                 "ETHUSDT": False}
    assert [v.original_proposal.symbol for v in verdicts] == ["BTCUSDT"]
    assert reviews[0].candidate.strategy == "scanner"
    assert reviews[0].candidate.stop == pytest.approx(scanner.price - 2.0)  # 2 ATR placeholder


@pytest.mark.asyncio
async def test_a_claude_account_runs_beside_the_plain_one():
    from config.settings import AppSettings
    from paper.engine import PaperTrader
    from risk.circuit_breaker import CircuitBreaker
    from risk.drawdown import DrawdownTracker
    from risk.state import InMemoryRiskStore, PrefixedRiskStore
    from tests.research_helpers import Scripted

    base, ex = InMemoryRiskStore(), FakeLive()

    def trader(prefix, adv=None):  # As paper_trade.py builds it, with and without --account
        store = PrefixedRiskStore(base, prefix)
        bus, settings = EventBus(), AppSettings()
        risk = RiskManager(settings.risk, bus, breaker=CircuitBreaker(store, bus),
                           drawdown=DrawdownTracker(store, source="paper_trading"), store=store)
        return PaperTrader(ex, risk, [Scripted(bars=(299,))], ["BTCUSDT"], settings, store,
                           advisor=adv)

    plain = trader("paper_trading:")
    claude = trader("paper_trading:claude:", advisor(ScriptedLLM([answer("NO_TRADE", 0.9)])))
    for t in (plain, claude):
        await t.load()
        await t.cycle(ex.now)
    assert set(plain.account.positions) == {"BTCUSDT"}
    assert claude.account.positions == {}  # Same signal, Claude said no
    assert "paper_trading:claude:account" in base.state and "paper_trading:account" in base.state
