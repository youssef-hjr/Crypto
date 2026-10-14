"""
Paper trading engine (Milestone 3).

Runs the strategies on LIVE Bybit market data (mainnet public data, no API
key needed) and trades a simulated account. No order is ever sent: the
exchange adapter is only asked for candles, order books, tickers, funding
history and order rules.

Each cycle (every `poll_seconds`):

  1. Open positions are checked against the 1-minute candles closed since
     the last cycle: stop, target and gap rules from `research.simulator`,
     the same code the backtester uses. Funding is charged from Bybit's
     funding history.
  2. When a new strategy bar (1 hour by default) has closed, trailing stops
     and time exits are updated and the strategies look for signals.
  3. Signals become proposals and go through the PortfolioAllocator and the
     Risk Manager with the REAL order book, 24h turnover and order rules.
  4. Approved trades fill against the real order book, level by level,
     paying the taker fee. A book too thin for the order fills partly.

The paper account has its own circuit breaker, drawdown marks and loss
streak (keys prefixed "paper_trading:"), so it can never block or unblock
the real account's. A tripped paper breaker stays tripped until reset with
`python paper_trade.py --reset-breaker "note"`, exactly like live.

Everything that matters survives a restart: the account and its open
positions are saved after every cycle, and closed trades go to the
trade_journal table (mode "paper").
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog

from agents.quant import QuantEngine
from config.settings import AppSettings
from exchanges.base import ExchangeAdapter
from models import (
    Candle, InstrumentRules, MarketContext, PortfolioState, Position, PositionSide,
    RiskDecision, RiskVerdict, Side, TradeProposal,
)
from portfolio.allocation import PortfolioAllocator
from portfolio.correlation import CorrelationMatrix
from portfolio.portfolio import next_risk_budget
from research.simulator import (
    CostModel, SimPosition, TradeRecord, check_exits, close_bar, close_trade,
    fill_entry_from_book, funding_times, market_exit,
)
from risk.risk_manager import RiskManager
from risk.state import RiskStateStore
from strategies.base import Signal, Strategy
from strategies.features import (
    FeatureView, candles_to_frame, compute_features, frame_columns, interval_minutes,
)

logger = structlog.get_logger()

ACCOUNT_KEY = "account"
STRATEGY_BARS = 300  # Candles fetched for indicators (EMA 200 needs warm-up)
MAX_SIGNAL_AGE = timedelta(minutes=5)  # Ignore signals from bars closed longer ago


@dataclass
class PaperAccount:
    initial_equity: float
    balance: float
    positions: dict[str, SimPosition] = field(default_factory=dict)
    entry_equity: dict[str, float] = field(default_factory=dict)
    last_strategy_bar: dict[str, str] = field(default_factory=dict)  # Open time, ISO
    last_minute: dict[str, str] = field(default_factory=dict)  # 1m bars checked up to
    started_at: str = ""
    run_id: Optional[int] = None
    closed_trades: int = 0
    realized_net: float = 0.0

    def equity(self) -> float:
        return self.balance + sum(p.unrealized() for p in self.positions.values())

    def to_dict(self) -> dict:
        return {
            "initial_equity": self.initial_equity, "balance": self.balance,
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "entry_equity": self.entry_equity,
            "last_strategy_bar": self.last_strategy_bar, "last_minute": self.last_minute,
            "started_at": self.started_at, "run_id": self.run_id,
            "closed_trades": self.closed_trades, "realized_net": self.realized_net,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PaperAccount":
        d = dict(d)
        d["positions"] = {s: SimPosition.from_dict(p) for s, p in d.get("positions", {}).items()}
        return cls(**d)


@dataclass
class CycleReport:
    at: datetime
    equity: float
    opened: list[str] = field(default_factory=list)
    closed: list[TradeRecord] = field(default_factory=list)
    verdicts: list[RiskVerdict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PaperTrader:

    def __init__(
        self,
        exchange: ExchangeAdapter,
        risk: RiskManager,
        strategies: list[Strategy],
        symbols: list[str],
        settings: AppSettings,
        store: RiskStateStore,
        interval: str = "60",
        costs: Optional[CostModel] = None,
        journal=None,  # database.ResearchRepository, or None to keep nothing
        initial_equity: Optional[float] = None,
    ):
        self._exchange = exchange
        self._risk = risk
        self._allocator = PortfolioAllocator(risk)
        self._strategies = strategies
        self._symbols = symbols
        self._settings = settings
        self._store = store
        self._interval = interval
        self._bar = timedelta(minutes=interval_minutes(interval))
        self._costs = costs or CostModel()
        self._journal = journal
        self._initial = initial_equity or settings.paper_equity
        self._quant = QuantEngine(bar_minutes=interval_minutes(interval))
        self._rules: dict[str, InstrumentRules] = {}
        self.account: Optional[PaperAccount] = None

    # ── State ──────────────────────────────────────────────────────────

    async def load(self) -> PaperAccount:
        await self._risk.load()
        saved = await self._store.get_state(ACCOUNT_KEY)
        if saved:
            self.account = PaperAccount.from_dict(saved)
        else:
            now = datetime.now(timezone.utc)
            self.account = PaperAccount(initial_equity=self._initial, balance=self._initial,
                                        started_at=now.isoformat())
            if self._journal is not None:
                self.account.run_id = await self._journal.save_run(
                    "paper", [{"name": s.name, "params": s.params} for s in self._strategies],
                    self._symbols, self._interval, now, None,
                    {"costs": self._costs.as_dict(), "initial_equity": self._initial})
            await self._save()
        return self.account

    async def _save(self) -> None:
        await self._store.set_state(ACCOUNT_KEY, self.account.to_dict())

    # ── One cycle ──────────────────────────────────────────────────────

    async def cycle(self, now: Optional[datetime] = None) -> CycleReport:
        now = now or datetime.now(timezone.utc)
        acct = self.account
        report = CycleReport(at=now, equity=acct.equity())

        # 1. Stops and targets on the 1-minute bars since the last check
        for sym in list(acct.positions):
            try:
                await self._manage_position(sym, now, report)
            except Exception as e:
                report.errors.append(f"{sym}: {e}")
                logger.error("paper_position_check_failed", symbol=sym, error=str(e))

        # 2. New strategy bars: trailing stops, time exits, signals
        signals: list[tuple[Signal, FeatureView]] = []
        history: dict[str, list[Candle]] = {}
        for sym in self._symbols:
            try:
                candles = await self._exchange.get_candles(sym, self._interval, STRATEGY_BARS)
            except Exception as e:
                report.errors.append(f"{sym}: {e}")
                continue
            if not candles:
                report.errors.append(f"{sym}: no {self._interval}m candles")
                continue
            history[sym] = candles
            last_open = candles[-1].timestamp.isoformat()
            if acct.last_strategy_bar.get(sym) == last_open:
                continue
            first_time = sym not in acct.last_strategy_bar
            acct.last_strategy_bar[sym] = last_open
            pos = acct.positions.get(sym)
            if pos is not None and not first_time:
                exit_ = close_bar(pos, candles[-1].close, candles[-1].turnover, self._costs)
                if exit_ is not None:
                    await self._close(sym, exit_, now, report)
            if sym in acct.positions:
                continue
            # Act only on a bar that has just closed, as the backtest does.
            # After a restart, an hour-old signal is history, not a trade.
            if now - (candles[-1].timestamp + self._bar) > MAX_SIGNAL_AGE:
                continue
            sig, view = self._signal(sym, candles)
            if sig is not None:
                signals.append((sig, view))

        # 3 and 4. Risk Manager, then fills against the real book
        if signals:
            await self._trade(signals, history, now, report)

        await self._mark(now)
        report.equity = acct.equity()
        await self._risk.observe(self._portfolio(now))
        await self._save()
        return report

    async def _manage_position(self, sym: str, now: datetime, report: CycleReport) -> None:
        acct = self.account
        pos = acct.positions[sym]
        # Funding settled since the last check, at Bybit's actual rates
        since = pos.funding_until or pos.opened_at
        due = funding_times(since, now)
        if due:
            rates = dict(await self._exchange.get_funding_history(sym, since, now + timedelta(seconds=1)))
            for t in due:
                rate = rates.get(t)
                price = pos.last_price
                if rate is None:
                    amount = pos.quantity * price * abs(self._costs.default_funding_rate)
                else:
                    amount = pos.quantity * price * rate * pos.direction
                pos.funding += amount
                acct.balance -= amount
            pos.funding_until = due[-1]

        checked = datetime.fromisoformat(acct.last_minute.get(sym, pos.opened_at.isoformat()))
        if now - checked > timedelta(minutes=190):
            # Down for a while: replay every minute missed, not just the latest 200
            minutes = await self._exchange.get_candle_history(sym, "1", checked, now)
        else:
            minutes = await self._exchange.get_candles(sym, "1", 200)
        if not minutes:
            # The stop could not be checked this cycle; say so loudly
            report.errors.append(f"{sym}: no 1-minute candles, stop not checked")
            return
        for c in minutes:
            if c.timestamp < checked:
                continue
            exit_ = check_exits(pos, c.open, c.high, c.low, c.close, c.turnover, self._costs)
            acct.last_minute[sym] = (c.timestamp + timedelta(minutes=1)).isoformat()
            if exit_ is not None:
                await self._close(sym, exit_, c.timestamp + timedelta(minutes=1), report)
                return

    def _signal(self, sym: str, candles: list[Candle]):
        frame = compute_features(candles_to_frame(candles), interval_minutes(self._interval))
        for s in self._strategies:
            s.prepare(frame)
        view = FeatureView(frame_columns(frame), len(frame) - 1, sym, frame.index[-1])
        best = None
        for s in self._strategies:
            sig = s.evaluate(view)
            if sig is not None and (best is None or sig.confidence > best.confidence):
                best = sig
        return best, view

    async def _trade(self, signals, history, now: datetime, report: CycleReport) -> None:
        acct = self.account
        equity = acct.equity()
        proposals, markets, meta = [], {}, {}
        for sig, view in signals:
            sym = sig.symbol
            try:
                book = await self._exchange.get_orderbook(sym, depth=self._settings.orderbook_depth)
                ticker = await self._exchange.get_ticker(sym)
                if sym not in self._rules:
                    self._rules[sym] = await self._exchange.get_instrument_rules(sym)
            except Exception as e:
                report.errors.append(f"{sym}: {e}")
                continue
            regime = self._quant.detect_regime(view.technical()).value
            stop_pct = abs(sig.entry - sig.stop) / sig.entry
            proposals.append(TradeProposal(
                symbol=sym, direction=sig.direction, strategy=sig.strategy,
                entry_price=sig.entry, stop_loss=sig.stop, take_profit=sig.target,
                position_size_usd=next_risk_budget(
                    equity, self._settings.risk.risk_per_trade_pct) / stop_pct,
                confidence=sig.confidence, expected_r=sig.expected_r,
                reasoning=sig.reason, market_regime=regime, opportunity_score=sig.score,
            ))
            markets[sym] = MarketContext(symbol=sym, orderbook=book,
                                         turnover_24h=ticker.turnover_24h,
                                         instrument=self._rules[sym])
            meta[sym] = (sig, regime, view)
        if not proposals:
            return

        corr_input = {s: history[s] for s in set(markets) | set(acct.positions) if s in history}
        correlations = CorrelationMatrix.from_candles(
            corr_input, min_samples=self._settings.risk.min_correlation_samples)
        verdicts = await self._allocator.allocate(
            proposals, self._portfolio(now), markets, correlations, now=now)
        report.verdicts.extend(verdicts)

        for v in verdicts:
            if v.decision == RiskDecision.REJECTED:
                continue
            sym = v.original_proposal.symbol
            sig, regime, view = meta[sym]
            book = markets[sym].orderbook
            d = 1 if sig.direction == Side.BUY else -1
            if (book.mid_price - sig.stop) * d <= 0:
                continue  # Already through the stop
            fill = fill_entry_from_book(sig.direction, v.approved_quantity, book,
                                        self._costs, self._rules.get(sym))
            if fill is None:
                continue
            acct.positions[sym] = SimPosition(
                symbol=sym, strategy=sig.strategy, direction=d, quantity=fill.quantity,
                entry_price=fill.price, initial_stop=sig.stop, stop=sig.stop,
                target=sig.target, trail_distance=sig.trail_distance,
                max_hold_bars=sig.max_hold_bars, opened_at=now, signal_price=sig.entry,
                risk_usd=abs(fill.price - sig.stop) * fill.quantity,
                confidence=sig.confidence, regime=regime, phases=[],
                details={"reason": sig.reason, "params": sig.params, "score": sig.score,
                         "risk_decision": v.decision.value, "partial_fill": fill.partial,
                         "book_mid": book.mid_price},
                fees=fill.fee, slippage_usd=fill.slippage_usd, funding_until=now,
            )
            acct.entry_equity[sym] = equity
            acct.last_minute[sym] = now.replace(second=0, microsecond=0).isoformat()
            acct.balance -= fill.fee
            report.opened.append(sym)
            logger.info("PAPER_OPEN", symbol=sym, side=sig.direction.value,
                        strategy=sig.strategy, qty=fill.quantity, price=round(fill.price, 6),
                        stop=round(sig.stop, 6), target=sig.target,
                        risk_usd=round(abs(fill.price - sig.stop) * fill.quantity, 2))

    async def _close(self, sym: str, exit_, at: datetime, report: CycleReport) -> None:
        acct = self.account
        pos = acct.positions.pop(sym)
        gross = (exit_.price - pos.entry_price) * pos.quantity * pos.direction
        acct.balance += gross - exit_.fee
        rec = close_trade(pos, exit_, at, acct.entry_equity.pop(sym, acct.initial_equity))
        acct.last_minute.pop(sym, None)
        acct.closed_trades += 1
        acct.realized_net += rec.net_pnl
        report.closed.append(rec)
        await self._risk.record_trade_result(rec.net_pnl)
        logger.info("PAPER_CLOSE", symbol=sym, reason=rec.exit_reason,
                    net_pnl=round(rec.net_pnl, 2), r=round(rec.r_multiple, 2))
        if self._journal is not None:
            await self._journal.save_trades(acct.run_id, "paper", self._exchange.name, [rec])

    async def _mark(self, now: datetime) -> None:
        """Mark open positions at the order book mid."""
        for sym, pos in self.account.positions.items():
            try:
                book = await self._exchange.get_orderbook(sym, depth=1)
                if book.mid_price:
                    pos.last_price = book.mid_price
            except Exception as e:
                logger.warning("paper_mark_failed", symbol=sym, error=str(e))

    def _portfolio(self, now: datetime) -> PortfolioState:
        acct = self.account
        positions = [Position(
            symbol=p.symbol, side=PositionSide.LONG if p.direction > 0 else PositionSide.SHORT,
            entry_price=p.entry_price, quantity=p.quantity, stop_loss=p.stop,
            mark_price=p.last_price, strategy=p.strategy,
        ) for p in acct.positions.values()]
        unreal = sum(p.unrealized() for p in acct.positions.values())
        return PortfolioState(equity=acct.balance + unreal, available_balance=acct.balance,
                              unrealized_pnl=unreal, realized_pnl_today=0.0,
                              positions=positions, source="paper_trading", timestamp=now)

    async def close_all(self, reason: str = "manual") -> list[TradeRecord]:
        """Close every paper position at the current book (market exit)."""
        report = CycleReport(at=datetime.now(timezone.utc), equity=self.account.equity())
        for sym in list(self.account.positions):
            pos = self.account.positions[sym]
            book = await self._exchange.get_orderbook(sym, depth=1)
            # Spread and fee only: the slippage model needs bar turnover
            costs = replace(self._costs, base_slippage_bps=0.0, impact_bps_per_pct=0.0)
            exit_ = market_exit(pos, book.mid_price, reason, 0.0, costs)
            await self._close(sym, exit_, report.at, report)
        await self._save()
        return report.closed

    def stats(self) -> dict:
        return {"equity": round(self.account.equity(), 2),
                "balance": round(self.account.balance, 2),
                "open_positions": len(self.account.positions),
                "closed_trades": self.account.closed_trades,
                "realized_net": round(self.account.realized_net, 2)}
