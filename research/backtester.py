"""
Backtester.

Replays stored candles bar by bar through the strategies, the Milestone 2
Risk Manager and the simulated fills in `research.simulator`.

How a trade happens, and why it can't see the future:

  1. Bar i closes. Strategies read the feature frame up to bar i only
     (`FeatureView` refuses future bars; indicators are causal).
  2. Each signal becomes a TradeProposal sized at the 0.05% risk budget.
     The PortfolioAllocator runs them through the Risk Manager at the
     simulated time, against the simulated account. Rejected means no trade.
  3. Approved proposals fill at the OPEN of the next bar, with spread,
     slippage, fees and a volume cap (see the simulator for the cost model).
  4. Every later bar checks the stop and target, charges funding and marks
     the position to market. The Risk Manager sees the equity after every
     bar, so drawdown limits and the circuit breaker work as they do live.

The backtest has its own in-memory risk state. It never touches the
circuit breaker, drawdown marks or loss streak of the real account.

The Risk Manager needs a live order book, which history doesn't have. The
backtest builds one from the bar: mid at the close, the configured spread,
and depth of `depth_fraction` x 24h turnover within 50 bps. Liquidity caps
and spread checks run against that, which is an assumption, not data.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import structlog

from agents.quant import QuantEngine
from config.settings import RiskSettings
from events import EventBus
from models import (
    InstrumentRules, MarketContext, OrderBookLevel, OrderBookSnapshot, PortfolioState,
    Position, PositionSide, RiskDecision, Side, TradeProposal,
)
from portfolio.allocation import PortfolioAllocator
from portfolio.correlation import CorrelationMatrix
from portfolio.portfolio import next_risk_budget
from research.regimes import coverage, daily_phases, phases_at
from research.simulator import (
    CostModel, EntryFill, ExitFill, SimPosition, TradeRecord, close_trade,
    fill_entry, funding_times, market_exit, step_bar,
)
from research import metrics as perf
from risk.risk_manager import RiskManager
from risk.state import InMemoryRiskStore
from strategies.base import Signal, Strategy
from strategies.features import (
    FeatureView, compute_features, frame_columns, interval_minutes,
)

logger = structlog.get_logger()


@dataclass
class BacktestConfig:
    interval: str = "60"
    initial_equity: float = 10_000.0
    costs: CostModel = field(default_factory=CostModel)
    # A tripped breaker blocks trading until an operator resets it. A
    # backtest has no operator, so by default it resets at the next UTC
    # day and every trip is counted in the report. Set False to keep it
    # tripped for the rest of the test, as it would be live.
    breaker_reset_daily: bool = True
    entry_delay_bars: int = 0  # Extra latency: fill this many bars later
    correlation_bars: int = 200
    exchange: str = "bybit"

    def as_dict(self) -> dict:
        return {"interval": self.interval, "initial_equity": self.initial_equity,
                "costs": self.costs.as_dict(),
                "breaker_reset_daily": self.breaker_reset_daily,
                "entry_delay_bars": self.entry_delay_bars,
                "correlation_bars": self.correlation_bars, "exchange": self.exchange}


@dataclass
class SymbolData:
    """History for one symbol: OHLCV indexed by open time (UTC), funding
    settlements, and the exchange's order rules."""
    symbol: str
    candles: pd.DataFrame
    funding: dict[datetime, float] = field(default_factory=dict)
    rules: Optional[InstrumentRules] = None


class PreparedSymbol:
    """Features and phase labels for one symbol, computed once and reused
    across runs (walk-forward runs many windows on the same history)."""

    def __init__(self, data: SymbolData, bar_minutes: int):
        self.data = data
        self.symbol = data.symbol
        self.frame = compute_features(data.candles, bar_minutes)
        self.phases = daily_phases(self.frame["close"])
        self.times = self.frame.index
        self.times_ns = self.frame.index.as_unit("ns").asi8
        self._cols: Optional[dict[str, np.ndarray]] = None
        self._ncols = -1

    def prepare(self, strategies: list[Strategy]) -> None:
        for s in strategies:
            s.prepare(self.frame)

    @property
    def cols(self) -> dict[str, np.ndarray]:
        if self._cols is None or self._ncols != len(self.frame.columns):
            self._cols = frame_columns(self.frame)
            self._ncols = len(self.frame.columns)
        return self._cols


@dataclass
class _Pending:
    signal: Signal
    quantity: float
    delay: int
    equity: float
    regime: str
    phases: list[str]
    details: dict


@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    equity: pd.Series
    gross_exposure: pd.Series
    decisions: Counter
    rejection_reasons: Counter
    cancelled_entries: Counter
    breaker_trips: list[dict]
    signals: int
    config: BacktestConfig
    strategies: list[dict]
    symbols: list[str]
    start: Optional[datetime]
    end: Optional[datetime]
    coverage: dict[str, dict[str, int]]

    def metrics(self) -> dict:
        m = perf.summarize(self.trades, self.equity, self.gross_exposure)
        m.update(signals=self.signals, breaker_trips=len(self.breaker_trips),
                 approved=self.decisions.get("approved", 0) + self.decisions.get("reduced", 0),
                 rejected=self.decisions.get("rejected", 0),
                 entries_cancelled=sum(self.cancelled_entries.values()))
        return m

    def by_phase(self) -> dict:
        return perf.by_phase(self.trades)


def default_rules(symbol: str, price: float) -> InstrumentRules:
    """Stand-in order rules when none were stored: a lot step about 1e-7 of
    the price and Bybit's 5 USDT minimum order value."""
    step = 10 ** (math.floor(math.log10(price)) - 7) if price > 0 else 1e-8
    return InstrumentRules(symbol=symbol, qty_step=step, min_qty=step, min_notional=5.0)


def synthetic_market(symbol: str, mid: float, turnover_24h: float, at: datetime,
                     costs: CostModel) -> MarketContext:
    """An order book built from a bar, for the Risk Manager's liquidity and
    spread checks. Depth is spread evenly over 5 levels within 40 bps."""
    half = mid * costs.spread_bps / 20_000
    depth = max(turnover_24h, 0.0) * costs.depth_fraction
    step = mid * 0.001  # 10 bps between levels
    qty = depth / 5 / mid if mid > 0 else 0.0
    book = OrderBookSnapshot(
        symbol=symbol, timestamp=at,
        bids=[OrderBookLevel(price=mid - half - i * step, quantity=qty) for i in range(5)],
        asks=[OrderBookLevel(price=mid + half + i * step, quantity=qty) for i in range(5)],
    )
    return MarketContext(symbol=symbol, orderbook=book, turnover_24h=turnover_24h)


class Backtester:

    def __init__(self, config: BacktestConfig, risk_settings: RiskSettings):
        self.config = config
        self.risk_settings = risk_settings
        self.bar_minutes = interval_minutes(config.interval)
        self._quant = QuantEngine(bar_minutes=self.bar_minutes)

    def prepare(self, data: dict[str, SymbolData]) -> dict[str, PreparedSymbol]:
        return {s: PreparedSymbol(d, self.bar_minutes) for s, d in data.items()
                if len(d.candles) > 0}

    async def run(
        self,
        strategies: list[Strategy],
        data: dict[str, SymbolData] | dict[str, PreparedSymbol],
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> BacktestResult:
        """Trade bars with open time in [start, end). Bars before `start`
        only warm up the indicators."""
        cfg, costs = self.config, self.config.costs
        prepared = {s: (d if isinstance(d, PreparedSymbol) else PreparedSymbol(d, self.bar_minutes))
                    for s, d in data.items()}
        for p in prepared.values():
            p.prepare(strategies)

        bar_ns = self.bar_minutes * 60 * 10**9
        start_ns = pd.Timestamp(start).value if start is not None else None
        end_ns = pd.Timestamp(end).value if end is not None else None

        # Which symbols have a bar at each open time
        bars_at: dict[int, list[tuple[str, int]]] = {}
        for sym, p in prepared.items():
            lo = 0 if start_ns is None else int(np.searchsorted(p.times_ns, start_ns))
            hi = len(p.times_ns) if end_ns is None else int(np.searchsorted(p.times_ns, end_ns))
            for i in range(lo, hi):
                bars_at.setdefault(int(p.times_ns[i]), []).append((sym, i))
        timeline = sorted(bars_at)

        store = InMemoryRiskStore()
        risk = RiskManager(self.risk_settings, EventBus(), store=store)
        allocator = PortfolioAllocator(risk)

        balance = cfg.initial_equity
        positions: dict[str, SimPosition] = {}
        pending: dict[str, _Pending] = {}
        entry_equity: dict[str, float] = {}
        last_bar: dict[str, int] = {}  # Latest bar index seen per symbol
        trades: list[TradeRecord] = []
        eq_times, eq_values, gross_values = [], [], []
        decisions, reasons, cancelled = Counter(), Counter(), Counter()
        signals_seen = 0
        current_day = None

        def close(sym: str, exit_: ExitFill, at: datetime) -> TradeRecord:
            nonlocal balance
            pos = positions.pop(sym)
            gross = (exit_.price - pos.entry_price) * pos.quantity * pos.direction
            balance += gross - exit_.fee
            rec = close_trade(pos, exit_, at, entry_equity.pop(sym, cfg.initial_equity))
            trades.append(rec)
            return rec

        for ts in timeline:
            open_time = pd.Timestamp(ts, tz="UTC")
            close_time = pd.Timestamp(ts + bar_ns, tz="UTC")

            for sym, i in bars_at[ts]:
                p = prepared[sym]
                c = p.cols
                o, h, l, cl = c["open"][i], c["high"][i], c["low"][i], c["close"][i]
                vol, turnover = c["volume"][i], c["turnover"][i]
                last_bar[sym] = i

                # Funding settlements up to this bar's open
                pos = positions.get(sym)
                if pos is not None:
                    since = pos.funding_until or pos.opened_at
                    for ft in funding_times(since, open_time):
                        rate = p.data.funding.get(ft)
                        if rate is None:
                            amount = pos.quantity * o * abs(costs.default_funding_rate)
                        else:
                            amount = pos.quantity * o * rate * pos.direction
                        pos.funding += amount
                        balance -= amount
                    pos.funding_until = open_time

                # Fill an entry approved at the previous close
                pend = pending.get(sym)
                if pend is not None:
                    if pend.delay > 0:
                        pend.delay -= 1
                    else:
                        del pending[sym]
                        fill = self._try_entry(pend, o, vol, turnover, p, cancelled)
                        if fill is not None:
                            sig = pend.signal
                            positions[sym] = SimPosition(
                                symbol=sym, strategy=sig.strategy,
                                direction=1 if sig.direction == Side.BUY else -1,
                                quantity=fill.quantity, entry_price=fill.price,
                                initial_stop=sig.stop, stop=sig.stop, target=sig.target,
                                trail_distance=sig.trail_distance,
                                max_hold_bars=sig.max_hold_bars, opened_at=open_time,
                                signal_price=sig.entry,
                                risk_usd=abs(fill.price - sig.stop) * fill.quantity,
                                confidence=sig.confidence, regime=pend.regime,
                                phases=pend.phases, details=pend.details,
                                fees=fill.fee, slippage_usd=fill.slippage_usd,
                            )
                            positions[sym].details["partial_fill"] = fill.partial
                            entry_equity[sym] = pend.equity
                            balance -= fill.fee

                pos = positions.get(sym)
                if pos is not None:
                    exit_ = step_bar(pos, o, h, l, cl, turnover, costs)
                    if exit_ is not None:
                        rec = close(sym, exit_, close_time)
                        await risk.record_trade_result(rec.net_pnl)

            # Mark to market
            unreal = sum(pos.unrealized() for pos in positions.values())
            equity = balance + unreal
            gross = sum(pos.quantity * pos.last_price for pos in positions.values())
            eq_times.append(close_time)
            eq_values.append(equity)
            gross_values.append(gross)

            # A new UTC day: the backtest's stand-in operator resets the breaker
            day = close_time.date()
            if day != current_day:
                current_day = day
                if cfg.breaker_reset_daily and risk.is_circuit_breaker_active:
                    await risk.reset_circuit_breaker(
                        "backtest", "Automatic next-day reset (backtest only)")

            if equity <= 0:
                logger.warning("backtest_account_wiped_out", at=str(close_time))
                break
            await risk.observe(self._portfolio(equity, unreal, {}, {}, close_time))

            # Signals on the bar that just closed
            signals: list[Signal] = []
            for sym, i in bars_at[ts]:
                if sym in positions or sym in pending:
                    continue
                p = prepared[sym]
                view = FeatureView(p.cols, i, sym, open_time)
                best = None
                for strat in strategies:
                    sig = strat.evaluate(view)
                    if sig is not None and (best is None or sig.confidence > best.confidence):
                        best = sig
                if best is not None:
                    signals.append(best)
            if not signals:
                continue
            signals_seen += len(signals)

            proposals, markets, meta = [], {}, {}
            for sig in signals:
                p = prepared[sig.symbol]
                i = last_bar[sig.symbol]
                view = FeatureView(p.cols, i, sig.symbol, open_time)
                regime = self._quant.detect_regime(view.technical()).value
                stop_pct = abs(sig.entry - sig.stop) / sig.entry
                proposals.append(TradeProposal(
                    symbol=sig.symbol, direction=sig.direction, strategy=sig.strategy,
                    entry_price=sig.entry, stop_loss=sig.stop, take_profit=sig.target,
                    position_size_usd=next_risk_budget(
                        equity, self.risk_settings.risk_per_trade_pct) / stop_pct,
                    confidence=sig.confidence, expected_r=sig.expected_r,
                    reasoning=sig.reason, market_regime=regime,
                    opportunity_score=sig.score, timestamp=close_time,
                ))
                market = synthetic_market(sig.symbol, sig.entry, p.cols["turnover_24h"][i],
                                          close_time, costs)
                market.instrument = p.data.rules or default_rules(sig.symbol, sig.entry)
                markets[sig.symbol] = market
                meta[sig.symbol] = (sig, regime, phases_at(p.phases, close_time), view)

            portfolio = self._portfolio(equity, unreal, positions, pending, close_time)
            correlations = self._correlations(prepared, last_bar,
                                              set(markets) | set(positions) | set(pending))
            verdicts = await allocator.allocate(proposals, portfolio, markets,
                                                correlations, now=close_time)
            for v in verdicts:
                decisions[v.decision.value] += 1
                for r in v.rejection_reasons:
                    reasons[r.split(":")[0]] += 1
                if v.decision == RiskDecision.REJECTED:
                    continue
                sig, regime, phases, view = meta[v.original_proposal.symbol]
                m = v.risk_metrics
                pending[sig.symbol] = _Pending(
                    signal=sig, quantity=v.approved_quantity, delay=cfg.entry_delay_bars,
                    equity=equity, regime=regime, phases=phases,
                    details={
                        "reason": sig.reason, "params": sig.params,
                        "expected_r": sig.expected_r, "score": sig.score,
                        "risk_decision": v.decision.value,
                        "risk": {k: m.get(k) for k in (
                            "risk_budget", "trade_risk_usd", "portfolio_risk_pct",
                            "correlated_risk_pct", "correlated_with",
                            "liquidity_capped", "loss_streak_reduction")},
                        "technical": {k: view.get(k) for k in (
                            "ema_20", "ema_50", "ema_200", "rsi", "atr",
                            "macd_histogram", "realized_volatility", "relative_volume")},
                    })

        # Close whatever is still open at the last close of the test
        if positions and eq_times:
            for sym in list(positions):
                c = prepared[sym].cols
                i = last_bar[sym]
                close(sym, market_exit(positions[sym], c["close"][i], "end_of_test",
                                       c["turnover"][i], costs), eq_times[-1])
            eq_values[-1] = balance

        trips = [e for e in store.events if e["kind"] == "breaker_tripped"]
        cov_start = start or (min(p.times[0] for p in prepared.values()) if prepared else None)
        return BacktestResult(
            trades=trades,
            equity=pd.Series(eq_values, index=pd.DatetimeIndex(eq_times), dtype=float),
            gross_exposure=pd.Series(gross_values, index=pd.DatetimeIndex(eq_times), dtype=float),
            decisions=decisions, rejection_reasons=reasons, cancelled_entries=cancelled,
            breaker_trips=trips, signals=signals_seen, config=cfg,
            strategies=[{"name": s.name, "params": s.params} for s in strategies],
            symbols=sorted(prepared), start=start, end=end,
            coverage={s: coverage(p.phases, cov_start, end) for s, p in prepared.items()},
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _try_entry(self, pend: _Pending, o: float, vol: float, turnover: float,
                   p: PreparedSymbol, cancelled: Counter) -> Optional[EntryFill]:
        sig = pend.signal
        d = 1 if sig.direction == Side.BUY else -1
        if (o - sig.stop) * d <= 0:
            cancelled["gapped_through_stop"] += 1
            return None
        if sig.target is not None and (o - sig.target) * d >= 0:
            cancelled["target_already_reached"] += 1
            return None
        rules = p.data.rules or default_rules(sig.symbol, sig.entry)
        fill = fill_entry(sig.direction, pend.quantity, o, vol, turnover,
                          self.config.costs, rules)
        if fill is None:
            cancelled["no_liquidity"] += 1
        return fill

    @staticmethod
    def _portfolio(equity: float, unreal: float, positions: dict[str, SimPosition],
                   pending: dict[str, _Pending], at: datetime) -> PortfolioState:
        held = [Position(
            symbol=pos.symbol,
            side=PositionSide.LONG if pos.direction > 0 else PositionSide.SHORT,
            entry_price=pos.entry_price, quantity=pos.quantity, stop_loss=pos.stop,
            mark_price=pos.last_price, strategy=pos.strategy,
        ) for pos in positions.values()]
        # Approved entries waiting for the next open already use risk budget
        held += [Position(
            symbol=sym,
            side=PositionSide.LONG if pe.signal.direction == Side.BUY else PositionSide.SHORT,
            entry_price=pe.signal.entry, quantity=pe.quantity, stop_loss=pe.signal.stop,
            strategy=pe.signal.strategy,
        ) for sym, pe in pending.items()]
        return PortfolioState(equity=equity, available_balance=equity, unrealized_pnl=unreal,
                              realized_pnl_today=0.0, positions=held, source="backtest",
                              timestamp=at)

    def _correlations(self, prepared: dict[str, PreparedSymbol], last_bar: dict[str, int],
                      symbols: set[str]) -> Optional[CorrelationMatrix]:
        if len(symbols) < 2:
            return None
        n = self.config.correlation_bars
        closes = {}
        for sym in symbols:
            p = prepared[sym]
            i = last_bar.get(sym)
            if i is None:
                continue
            lo = max(0, i - n + 1)
            closes[sym] = dict(zip(p.times_ns[lo:i + 1].tolist(),
                                   p.cols["close"][lo:i + 1].tolist()))
        return CorrelationMatrix.from_closes(
            closes, min_samples=self.risk_settings.min_correlation_samples)
