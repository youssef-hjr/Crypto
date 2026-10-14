"""
Claude as a trade reviewer (PDF Milestone 4).

    candidate (strategy or scanner)
        -> Claude: TRADE or NO TRADE, with reasoning
        -> Risk Manager: approve, reduce or reject (final, cannot be overridden)
        -> paper fill or preview (execution is disabled)

What Claude can and cannot do:
  - It can only remove trades. It never sees a trade the deterministic
    side didn't propose, and cannot change side, entry, stop, target, size
    or leverage. A TRADE answer just lets the proposal go on to the Risk
    Manager, unchanged.
  - It never gets an exchange adapter, a settings object or a key. It gets
    a JSON brief (ai/briefing.py), checked for secrets before sending.
  - Anything other than a valid, confident TRADE is NO TRADE: API errors,
    timeouts, refusals, cut-off answers, invalid JSON, out-of-range values,
    confidence under AI_MIN_CONFIDENCE, the daily call limit, a tripped
    circuit breaker.

Every question and answer is logged (and stored in `ai_decisions` when a
database is available), with the brief that was sent, the cost, and later
the Risk Manager's verdict.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from pydantic import ValidationError

from ai.briefing import Candidate, build_brief, find_secrets, to_json
from ai.llm import LLMClient, LLMResponse
from ai.prompts import SYSTEM_PROMPT, user_message
from ai.schema import DECISION_SCHEMA, AIDecision, Verdict
from config.settings import AISettings
from models import RiskVerdict

logger = structlog.get_logger()


@dataclass
class AIReview:
    candidate: Candidate
    decision: AIDecision  # After every check: this is what the system acts on
    ai_said: Optional[Verdict] = None  # Claude's own answer, if it gave a valid one
    error: Optional[str] = None
    cached: bool = False
    sent: bool = False  # True if the API was called for this review
    latency_ms: Optional[int] = None
    response: Optional[LLMResponse] = None
    row_id: Optional[int] = None  # ai_decisions.id

    @property
    def approved(self) -> bool:
        return self.decision.is_trade

    def summary(self) -> dict:
        """Short version for logs and the paper journal."""
        d = self.decision
        return {"decision": d.decision.value, "confidence": d.confidence,
                "expected_r": d.expected_r, "reasoning": d.reasoning,
                "invalidation": d.invalidation, "market_regime": d.market_regime.value,
                "portfolio_impact": d.portfolio_impact,
                "conflicting_signals": d.conflicting_signals,
                "missing_information": d.missing_information,
                "error": self.error, "cached": self.cached, "ai_decision_id": self.row_id}


@dataclass
class _Cached:
    decision: AIDecision
    ai_said: Optional[Verdict]
    at: datetime
    row_id: Optional[int]


@dataclass
class ClaudeAdvisor:
    llm: LLMClient
    settings: AISettings
    mode: str = "preview"  # preview | paper: labels the stored rows
    journal: Optional[object] = None  # database.AIDecisionRepository
    _cache: dict = field(default_factory=dict)
    _day: Optional[datetime] = None
    _calls_today: int = 0

    async def load(self, now: Optional[datetime] = None) -> None:
        """Pick up today's call count from the database, so a restart doesn't
        reset the daily limit."""
        now = now or datetime.now(timezone.utc)
        self._roll_day(now)
        if self.journal is not None:
            self._calls_today = await self.journal.calls_since(self._day)

    @property
    def calls_today(self) -> int:
        return self._calls_today

    # ── Reviewing ──────────────────────────────────────────────────────

    async def review(
        self,
        candidates: list[Candidate],
        briefs: dict[tuple, dict],
        breaker_active: bool = False,
        now: Optional[datetime] = None,
    ) -> list[AIReview]:
        """One review per candidate, best score first. `briefs` maps a
        candidate's key to its brief (ai.briefing.build_brief); only the top
        `max_candidates_per_cycle` need one (see `top`)."""
        now = now or datetime.now(timezone.utc)
        self._roll_day(now)
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        reviews = []
        for i, cand in enumerate(ranked):
            if breaker_active:
                # The Risk Manager will reject everything anyway: don't pay to ask
                reviews.append(self._skip(cand, "Circuit breaker active: not asked"))
            elif i >= self.settings.max_candidates_per_cycle:
                reviews.append(self._skip(
                    cand, f"Not in the top {self.settings.max_candidates_per_cycle} "
                          "candidates this cycle: not asked"))
            else:
                brief = briefs.get(cand.key)
                reviews.append(await self._review_one(cand, brief, now) if brief is not None
                               else self._skip(cand, "No market data brief: not asked"))
        return reviews

    def top(self, candidates: list[Candidate]) -> list[Candidate]:
        """The candidates that will actually be sent, so only they need a brief."""
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[:self.settings.max_candidates_per_cycle]

    async def _review_one(self, cand: Candidate, brief: dict, now: datetime) -> AIReview:
        hit = self._cache.get(cand.key)
        if hit and now - hit.at < timedelta(minutes=self.settings.cache_minutes):
            return AIReview(cand, hit.decision, ai_said=hit.ai_said, cached=True,
                            row_id=hit.row_id)

        if self._calls_today >= self.settings.max_calls_per_day:
            review = self._skip(cand, f"Daily limit of {self.settings.max_calls_per_day} "
                                      "Claude calls reached: not asked")
            await self._store(review, brief, now)
            return review

        text = to_json(brief)
        leaked = find_secrets(text)
        if leaked:
            # Should be impossible: the brief is built from market data only.
            review = self._skip(cand, f"Brief contained a secret ({', '.join(leaked)}): not sent")
            logger.critical("ai_brief_contains_secret", symbol=cand.symbol, which=leaked)
            await self._store(review, None, now)
            return review

        self._calls_today += 1
        started = time.monotonic()
        response, error = None, None
        try:
            response = await asyncio.wait_for(
                self.llm.complete(SYSTEM_PROMPT, user_message(text), DECISION_SCHEMA),
                timeout=self.settings.timeout_seconds)
        except asyncio.TimeoutError:
            error = f"No answer within {self.settings.timeout_seconds:.0f}s"
        except Exception as e:  # Network, auth, rate limit, bad request, ...
            error = f"{type(e).__name__}: {e}"
        latency = int((time.monotonic() - started) * 1000)

        ai_said = None
        if response is not None:
            decision, ai_said, error = self._check(response, cand)
        else:
            decision = AIDecision.no_trade(f"Claude unavailable ({error})")
        review = AIReview(cand, decision, ai_said=ai_said, error=error, sent=True,
                          latency_ms=latency, response=response)
        await self._store(review, brief, now)
        if error is None or ai_said is not None:
            # Cache real answers only; a failed call may work next cycle
            self._cache[cand.key] = _Cached(decision, ai_said, now, review.row_id)

        logger.info("AI_" + decision.decision.value, symbol=cand.symbol,
                    side=cand.direction.value, strategy=cand.strategy,
                    confidence=decision.confidence, expected_r=decision.expected_r,
                    reasoning=decision.reasoning[:300], error=error, latency_ms=latency,
                    cost_usd=response.cost_usd() if response else None,
                    calls_today=self._calls_today)
        return review

    def _check(self, response: LLMResponse, cand: Candidate
               ) -> tuple[AIDecision, Optional[Verdict], Optional[str]]:
        """Turn the raw answer into a decision the system can act on.
        Returns (decision, Claude's own verdict if valid, error)."""
        if response.stop_reason == "refusal":
            why = f"Claude declined to answer (category {response.refusal_category})"
            return AIDecision.no_trade(why), None, why
        if response.stop_reason == "max_tokens":
            why = "Answer was cut off (max_tokens)"
            return AIDecision.no_trade(why), None, why
        try:
            decision = AIDecision.model_validate(json.loads(response.text))
        except (json.JSONDecodeError, ValidationError) as e:
            why = f"Invalid answer: {str(e)[:300]}"
            return AIDecision.no_trade(why), None, why

        said = decision.decision
        if decision.is_trade and decision.confidence < self.settings.min_confidence:
            downgraded = decision.model_copy(update={
                "decision": Verdict.NO_TRADE,
                "reasoning": f"[Confidence {decision.confidence:.2f} is under the minimum "
                             f"{self.settings.min_confidence:.2f}] {decision.reasoning}"})
            return downgraded, said, None
        return decision, said, None

    def _skip(self, cand: Candidate, why: str) -> AIReview:
        return AIReview(cand, AIDecision.no_trade(why), error=why)

    # ── Risk verdicts and storage ──────────────────────────────────────

    async def record_verdicts(self, reviews: list[AIReview], verdicts: list[RiskVerdict]) -> None:
        """Attach the Risk Manager's verdict to the stored Claude decision."""
        if self.journal is None:
            return
        by_symbol = {v.original_proposal.symbol: v for v in verdicts}
        for r in reviews:
            v = by_symbol.get(r.candidate.symbol)
            if v is None or r.row_id is None or r.cached:
                continue
            try:
                await self.journal.set_risk_verdict(r.row_id, v.decision.value,
                                                    v.rejection_reasons)
            except Exception as e:
                logger.error("ai_journal_update_failed", error=str(e))

    async def _store(self, review: AIReview, brief: Optional[dict], now: datetime) -> None:
        if self.journal is None:
            return
        c, d, resp = review.candidate, review.decision, review.response
        row = {
            "created_at": now, "mode": self.mode, "symbol": c.symbol, "strategy": c.strategy,
            "direction": c.direction.value, "bar_time": c.bar_time, "candidate": c.as_dict(),
            "brief": brief, "decision": d.decision.value,
            "ai_decision": review.ai_said.value if review.ai_said else None,
            "confidence": d.confidence, "expected_r": d.expected_r,
            "response": d.model_dump(mode="json") if review.ai_said else None,
            "raw_text": resp.text if resp else None, "error": review.error,
            "latency_ms": review.latency_ms,
            "model": (resp.model if resp else self.settings.model) if review.sent else None,
            "request_id": resp.request_id if resp else None,
            "input_tokens": resp.input_tokens if resp else None,
            "output_tokens": resp.output_tokens if resp else None,
            "cache_read_tokens": resp.cache_read_tokens if resp else None,
            "cost_usd": resp.cost_usd() if resp else None,
        }
        try:
            review.row_id = await self.journal.save(row)
        except Exception as e:
            # Losing the audit row is bad but must not stop the review
            logger.error("ai_journal_write_failed", symbol=c.symbol, error=str(e))

    def _roll_day(self, now: datetime) -> None:
        day = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if self._day != day:
            self._day, self._calls_today = day, 0


def build_candidate_briefs(candidates, markets, portfolio, risk, correlations) -> dict:
    """{candidate.key: brief} for a list of candidates. `markets` maps symbol
    to ai.briefing.MarketSnapshot; `risk` is ai.briefing.risk_state(...)."""
    return {c.key: build_brief(c, markets[c.symbol], portfolio, risk, correlations, candidates)
            for c in candidates}
