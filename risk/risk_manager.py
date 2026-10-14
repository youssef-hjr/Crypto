"""
Risk Manager.

Has ABSOLUTE VETO AUTHORITY over all trade proposals.
The AI cannot bypass these controls.

NON-NEGOTIABLE:
  - No martingale
  - No revenge trading
  - No uncontrolled leverage
  - No trading on stale data
  - No treating correlated assets as independent risk
  - Position size from stop distance, not arbitrary %
  - Risk scales with current equity (controlled compounding)

The Risk Manager computes its own inputs. Drawdown comes from the equity
history it has observed, money at risk from the open positions' stops, and
liquidity from the live order book. Nothing the caller says about those is
trusted. If something can't be checked, the trade is rejected.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

import structlog

from config.settings import RiskSettings
from events import Event, EventBus, EventType
from models import (
    MarketContext, PortfolioState, RiskDecision, RiskVerdict, Side, TradeProposal,
)
from portfolio import exposure as ex
from portfolio.correlation import CorrelationMatrix
from portfolio.liquidity import depth_usd, is_crossed
from risk.circuit_breaker import CircuitBreaker
from risk.drawdown import DrawdownTracker
from risk.state import InMemoryRiskStore, RiskStateStore

logger = structlog.get_logger()

LOSS_STREAK_KEY = "loss_streak"


class RiskManager:
    """
    Deterministic risk engine. Every trade proposal must pass through here.
    The Risk Manager can approve, reduce, or reject — never be bypassed.
    """

    def __init__(
        self,
        settings: RiskSettings,
        event_bus: EventBus,
        breaker: Optional[CircuitBreaker] = None,
        drawdown: Optional[DrawdownTracker] = None,
        store: Optional[RiskStateStore] = None,
    ):
        self._settings = settings
        self._event_bus = event_bus
        self._store = store or InMemoryRiskStore()
        self._breaker = breaker or CircuitBreaker(self._store, event_bus)
        self._drawdown = drawdown or DrawdownTracker(self._store)
        self._recent_losses: int = 0
        self._last_observed: Optional[tuple[datetime, float]] = None

    async def load(self) -> None:
        """Restore persisted state (breaker, drawdown peaks, loss streak)."""
        await self._breaker.load()
        await self._drawdown.load()
        streak = await self._store.get_state(LOSS_STREAK_KEY)
        self._recent_losses = int(streak["count"]) if streak else 0

    # ── Portfolio observation ──────────────────────────────────────────

    async def observe(self, portfolio: PortfolioState) -> None:
        """
        Feed a fresh equity reading into the drawdown tracker and trip the
        breaker if a drawdown limit is hit. `evaluate` calls this itself;
        call it directly to keep drawdown current between proposals.
        """
        key = (portfolio.timestamp, portfolio.equity)
        if self._last_observed == key or portfolio.equity <= 0:
            return
        self._last_observed = key
        dd = await self._drawdown.update(portfolio.equity, portfolio.timestamp)
        if dd.daily_drawdown_pct >= self._settings.max_daily_drawdown_pct:
            await self._breaker.trip("DAILY_DRAWDOWN_LIMIT", {
                "daily_drawdown_pct": dd.daily_drawdown_pct,
                "equity": dd.equity, "day_peak_equity": dd.day_peak_equity})
        if dd.total_drawdown_pct >= self._settings.max_total_drawdown_pct:
            await self._breaker.trip("TOTAL_DRAWDOWN_LIMIT", {
                "total_drawdown_pct": dd.total_drawdown_pct,
                "equity": dd.equity, "peak_equity": dd.peak_equity})

    # ── Evaluation ─────────────────────────────────────────────────────

    async def evaluate(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioState,
        market: Optional[MarketContext] = None,
        correlations: Optional[CorrelationMatrix] = None,
        now: Optional[datetime] = None,
    ) -> RiskVerdict:
        """
        Evaluate a trade proposal against all risk rules.
        Returns a RiskVerdict with approval, reduction, or rejection.
        """
        verdict = await self._evaluate(proposal, portfolio, market, correlations,
                                       now or datetime.now(timezone.utc))

        event_type = (
            EventType.RISK_APPROVED
            if verdict.decision in (RiskDecision.APPROVED, RiskDecision.REDUCED)
            else EventType.RISK_REJECTED
        )
        await self._event_bus.publish(Event(event_type, verdict, source="risk_manager"))

        logger.info("risk_verdict",
                     symbol=proposal.symbol,
                     decision=verdict.decision.value,
                     reasons=verdict.rejection_reasons,
                     size=verdict.approved_size_usd)
        return verdict

    async def _evaluate(
        self,
        proposal: TradeProposal,
        portfolio: PortfolioState,
        market: Optional[MarketContext],
        correlations: Optional[CorrelationMatrix],
        now: datetime,
    ) -> RiskVerdict:
        s = self._settings
        reasons: list[str] = []
        m: dict = {}

        # ── Equity, drawdown and the circuit breaker ───────────────────
        equity = portfolio.equity
        if equity <= 0:
            return self._reject(proposal, ["ZERO_OR_NEGATIVE_EQUITY"])
        if _age(portfolio.timestamp, now) > s.max_data_age_seconds:
            return self._reject(proposal, ["STALE_PORTFOLIO_STATE"])

        await self.observe(portfolio)
        if self._breaker.is_active:
            return self._reject(proposal, ["CIRCUIT_BREAKER_ACTIVE"],
                                {"breaker_reason": self._breaker.reason})

        dd = self._drawdown.state
        risk_budget = equity * s.risk_per_trade_pct
        m.update(equity=equity, risk_budget=risk_budget,
                 daily_drawdown_pct=dd.daily_drawdown_pct,
                 total_drawdown_pct=dd.total_drawdown_pct,
                 daily_pnl=dd.daily_pnl)

        # ── Valid prices required ──────────────────────────────────────
        entry = proposal.entry_price
        if entry <= 0:
            return self._reject(proposal, ["INVALID_ENTRY_PRICE"], m)
        if proposal.stop_loss <= 0:
            return self._reject(proposal, ["NO_STOP_LOSS"], m)

        # ── Stop and target must sit on the correct side of entry ──────
        # A long's stop is below entry and its target above; a short's is
        # the reverse. A stop on the wrong side is not a stop at all.
        is_long = proposal.direction == Side.BUY
        direction = 1 if is_long else -1
        stop_ok = (proposal.stop_loss < entry if is_long else proposal.stop_loss > entry)
        if not stop_ok:
            reasons.append("STOP_ON_WRONG_SIDE")
        targets = list(proposal.targets)
        if proposal.take_profit is not None:
            targets.append(proposal.take_profit)
        for tp in targets:
            tp_ok = tp > entry if is_long else tp < entry
            if not tp_ok:
                reasons.append(f"TARGET_ON_WRONG_SIDE: {tp}")

        stop_distance_pct = abs(entry - proposal.stop_loss) / entry
        m["stop_distance_pct"] = stop_distance_pct
        if stop_distance_pct > s.max_stop_distance_pct:
            reasons.append("STOP_DISTANCE_TOO_WIDE")
        if reasons:
            return self._reject(proposal, reasons, m)

        # ── Live market data ───────────────────────────────────────────
        reasons += self._check_market(proposal, market, now, m)
        if reasons:
            return self._reject(proposal, reasons, m)
        book = market.orderbook

        # ── Size from the stop distance ────────────────────────────────
        max_size = risk_budget / stop_distance_pct
        # NO MARTINGALE: after a losing streak the size goes down, never up.
        if self._recent_losses >= s.consecutive_loss_limit:
            max_size *= s.loss_streak_size_factor
            m["loss_streak_reduction"] = s.loss_streak_size_factor
        m["max_position_usd"] = max_size

        # ── Liquidity caps ─────────────────────────────────────────────
        depth = depth_usd(book, proposal.direction, s.depth_band_bps)
        book_cap = depth * s.max_book_participation
        turnover_cap = market.turnover_24h * s.max_turnover_participation
        m.update(book_depth_usd=depth, book_cap_usd=book_cap, turnover_cap_usd=turnover_cap)

        size = min(proposal.position_size_usd, max_size, book_cap, turnover_cap)
        if size < min(proposal.position_size_usd, max_size):
            m["liquidity_capped"] = True

        # ── Exchange order rules: round down to the lot step ───────────
        rules = market.instrument
        qty = math.floor(size / entry / rules.qty_step + 1e-9) * rules.qty_step
        if rules.max_qty:
            qty = min(qty, rules.max_qty)
        qty = round(qty, 12)
        size = qty * entry
        if qty < rules.min_qty or size < rules.min_notional or qty <= 0:
            reasons.append("BELOW_MIN_ORDER_SIZE")
        new_risk = size * stop_distance_pct
        m.update(approved_quantity=qty, approved_size_usd=size, trade_risk_usd=new_risk)

        # ── Leverage ───────────────────────────────────────────────────
        if proposal.leverage > s.max_leverage:
            reasons.append(f"LEVERAGE_EXCEEDED: {proposal.leverage} > {s.max_leverage}")
        items = ex.exposures(portfolio.positions)
        gross_after = ex.gross_exposure(items) + size
        m["gross_leverage_after"] = gross_after / equity
        if gross_after / equity > s.max_gross_leverage:
            reasons.append(
                f"GROSS_LEVERAGE_EXCEEDED: {gross_after / equity:.2f}x > {s.max_gross_leverage}x")
        m["net_exposure_after"] = ex.net_exposure(items) + size * direction

        # ── Positions and concentration ────────────────────────────────
        if len(portfolio.positions) >= s.max_positions:
            reasons.append("MAX_POSITIONS_REACHED")
        same_asset = sum(e.notional for e in items if e.symbol == proposal.symbol)
        concentration = (same_asset + size) / equity
        if concentration > s.max_single_asset_pct:
            reasons.append(
                f"CONCENTRATION_EXCEEDED: {concentration:.1%} > {s.max_single_asset_pct:.0%}")

        # ── Money at risk across the portfolio ─────────────────────────
        naked = ex.unprotected(items)
        if naked:
            reasons.append(f"UNPROTECTED_POSITION: {', '.join(naked)}")
        open_risk = ex.open_risk(items)
        portfolio_risk_pct = (open_risk + new_risk) / equity
        m.update(open_risk_usd=open_risk, portfolio_risk_pct=portfolio_risk_pct)
        if portfolio_risk_pct > s.max_portfolio_risk_pct:
            reasons.append(
                f"PORTFOLIO_RISK_EXCEEDED: {portfolio_risk_pct:.2%} > {s.max_portfolio_risk_pct:.2%}")

        # If every stop is hit today, the day must still end inside the limit.
        worst_day = dd.daily_drawdown_pct + portfolio_risk_pct
        m["worst_case_daily_drawdown_pct"] = worst_day
        if worst_day > s.max_daily_drawdown_pct:
            reasons.append("DAILY_LOSS_HEADROOM")

        # ── Correlated exposure ────────────────────────────────────────
        cluster_risk, members = ex.correlated_risk(
            proposal.symbol, direction, new_risk, items, correlations,
            s.correlation_threshold)
        cluster_pct = cluster_risk / equity
        m.update(correlated_risk_pct=cluster_pct, correlated_with=members,
                 correlations_known=correlations is not None,
                 correlation_adjusted_risk_usd=ex.correlation_adjusted_risk(
                     items + [ex.Exposure(proposal.symbol, direction, size, new_risk)],
                     correlations))
        if cluster_pct > s.max_correlated_risk_pct:
            reasons.append(
                f"CORRELATED_RISK_EXCEEDED: {cluster_pct:.2%} > {s.max_correlated_risk_pct:.2%}")

        # ── Minimum opportunity score ──────────────────────────────────
        if proposal.opportunity_score < s.min_opportunity_score:
            reasons.append(
                f"LOW_OPPORTUNITY_SCORE: {proposal.opportunity_score} < {s.min_opportunity_score}")

        # ── Verdict ────────────────────────────────────────────────────
        if reasons:
            return self._reject(proposal, reasons, m)
        reduced = size < proposal.position_size_usd * 0.99
        if reduced:
            m.update(size_reduced=True, original_size=proposal.position_size_usd)
        return RiskVerdict(
            decision=RiskDecision.REDUCED if reduced else RiskDecision.APPROVED,
            original_proposal=proposal,
            approved_size_usd=size,
            approved_quantity=qty,
            approved_leverage=proposal.leverage,
            risk_metrics=m,
        )

    def _check_market(
        self,
        proposal: TradeProposal,
        market: Optional[MarketContext],
        now: datetime,
        m: dict,
    ) -> list[str]:
        """Market data must be present, fresh and sane, or nothing else can
        be checked."""
        s = self._settings
        if market is None or market.symbol != proposal.symbol:
            return ["MARKET_DATA_MISSING"]
        book = market.orderbook
        if book is None or not book.bids or not book.asks or not book.mid_price:
            return ["ORDERBOOK_MISSING"]
        if is_crossed(book):
            return ["ORDERBOOK_CROSSED"]
        age = _age(book.timestamp, now)
        m["orderbook_age_s"] = age
        if age > s.max_data_age_seconds:
            return [f"STALE_MARKET_DATA: {age:.1f}s"]
        if market.turnover_24h is None:
            return ["TURNOVER_UNKNOWN"]
        if market.instrument is None:
            return ["INSTRUMENT_RULES_MISSING"]

        reasons = []
        m["spread_bps"] = book.spread_bps
        if book.spread_bps > s.max_spread_bps:
            reasons.append(f"SPREAD_TOO_WIDE: {book.spread_bps:.1f}bps")
        deviation = abs(proposal.entry_price - book.mid_price) / book.mid_price
        m["entry_deviation_pct"] = deviation
        if deviation > s.max_entry_deviation_pct:
            reasons.append(f"ENTRY_FAR_FROM_MARKET: {deviation:.2%}")
        return reasons

    def _reject(
        self,
        proposal: TradeProposal,
        reasons: list[str],
        metrics: dict = None,
    ) -> RiskVerdict:
        return RiskVerdict(
            decision=RiskDecision.REJECTED,
            original_proposal=proposal,
            rejection_reasons=reasons,
            risk_metrics=metrics or {},
        )

    # ── State Management ───────────────────────────────────────────────

    async def record_trade_result(self, net_pnl: float) -> None:
        """Update the loss streak after a trade closes (P&L after fees and
        funding). The streak is persisted."""
        if net_pnl < 0:
            self._recent_losses += 1
        else:
            self._recent_losses = 0  # Reset consecutive loss counter on win
        await self._store.set_state(LOSS_STREAK_KEY, {"count": self._recent_losses})

    async def reset_circuit_breaker(self, operator: str, note: str) -> None:
        """Manual reset only — requires operator validation."""
        await self._breaker.reset(operator, note)

    @property
    def is_circuit_breaker_active(self) -> bool:
        return self._breaker.is_active

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    @property
    def drawdown(self) -> DrawdownTracker:
        return self._drawdown

    # ── Position Sizing Helper ─────────────────────────────────────────

    def calculate_position_size(
        self, equity: float, stop_distance_pct: float
    ) -> float:
        """
        Calculate position size from equity and stop distance.
        Position size = Risk budget / Stop distance
        NOT: Position size = arbitrary % of equity
        """
        if stop_distance_pct <= 0:
            return 0.0
        risk_budget = equity * self._settings.risk_per_trade_pct
        return risk_budget / stop_distance_pct


def _age(ts: datetime, now: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds()
