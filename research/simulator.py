"""
Simulated fills and position handling, shared by the backtester and the
paper trader so both follow exactly the same rules.

Cost model (PDF section 31: fees, funding, spread, slippage, latency,
liquidity, partial fills, order type behaviour):

  Entry      Market order at the open of the bar AFTER the signal bar (the
             signal is only known once that bar has closed). Pays half the
             spread, slippage and the taker fee. If the open has already
             gapped through the stop, the entry is cancelled.
  Partial    At most `max_bar_participation` of the fill bar's volume; the
             rest is cancelled, like an IOC market order. Below the
             exchange's minimum order size, nothing fills.
  Stop       Stop-market order. Fills at the stop, or at the open if the bar
             gapped through it, minus half the spread and double slippage;
             taker fee.
  Target     Resting limit order. Fills only if price trades THROUGH the
             target; maker fee, no slippage.
  Both hit   If one bar touches both the stop and the target, the stop is
             assumed to have come first (the pessimistic order).
  Time exit  Market order at the close after `max_hold_bars`.
  Trailing   Moves only in the trade's favour, once per bar close, never
             wider than before.
  Funding    Charged at each settlement (00:00, 08:00, 16:00 UTC) the
             position is open through. Longs pay a positive rate. Where the
             history has no rate, `default_funding_rate` is charged against
             the position whichever side it is on.
  Slippage   base_slippage_bps + impact_bps_per_pct x (order notional as a
             % of the fill bar's turnover).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from models import InstrumentRules, Side


@dataclass
class CostModel:
    taker_fee: float = 0.00055  # Bybit linear perpetuals, base tier
    maker_fee: float = 0.0002
    spread_bps: float = 2.0  # Full spread; half is paid on each market fill
    base_slippage_bps: float = 1.0
    impact_bps_per_pct: float = 5.0  # Extra bps per 1% of the bar's turnover
    stop_slippage_mult: float = 2.0  # Stops fill in fast markets
    max_bar_participation: float = 0.05  # Max share of a bar's volume one order takes
    default_funding_rate: float = 0.0001  # 0.01% per 8h, charged when unknown
    depth_fraction: float = 0.002  # Synthetic book: 0.2% of 24h turnover within 50 bps

    def slippage(self, notional: float, bar_turnover: float, stop: bool = False) -> float:
        """Slippage as a fraction of price (excluding the half spread)."""
        pct_of_bar = 100 * notional / bar_turnover if bar_turnover > 0 else 100.0
        bps = self.base_slippage_bps + self.impact_bps_per_pct * pct_of_bar
        if stop:
            bps *= self.stop_slippage_mult
        return bps / 10_000

    def as_dict(self) -> dict:
        return asdict(self)


FUNDING_HOURS = (0, 8, 16)


def funding_times(start: datetime, end: datetime) -> list[datetime]:
    """Funding settlements in (start, end]."""
    t = start.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    out = []
    while t <= end:
        if t > start and t.hour in FUNDING_HOURS:
            out.append(t)
        t += timedelta(hours=1)
    return out


def round_qty(qty: float, rules: Optional[InstrumentRules]) -> float:
    if rules is None or rules.qty_step <= 0:
        return qty
    q = math.floor(qty / rules.qty_step + 1e-9) * rules.qty_step
    if rules.max_qty:
        q = min(q, rules.max_qty)
    return round(q, 12)


def meets_minimum(qty: float, price: float, rules: Optional[InstrumentRules]) -> bool:
    if qty <= 0:
        return False
    if rules is None:
        return True
    return qty >= rules.min_qty and qty * price >= rules.min_notional


@dataclass
class SimPosition:
    """An open simulated position."""
    symbol: str
    strategy: str
    direction: int  # +1 long, -1 short
    quantity: float
    entry_price: float  # Actual fill, after spread and slippage
    initial_stop: float
    stop: float
    target: Optional[float]
    trail_distance: Optional[float]
    max_hold_bars: int
    opened_at: datetime
    signal_price: float  # Close of the signal bar, where the risk was judged
    risk_usd: float  # Loss at the initial stop from the actual fill
    leverage: float = 1.0
    confidence: float = 0.0
    regime: str = ""
    phases: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)
    bars_held: int = 0
    fees: float = 0.0
    funding: float = 0.0
    slippage_usd: float = 0.0
    best_price: float = 0.0  # For MFE
    worst_price: float = 0.0  # For MAE
    last_price: float = 0.0  # Latest mark
    funding_until: Optional[datetime] = None  # Settlements charged up to here

    def __post_init__(self):
        self.best_price = self.best_price or self.entry_price
        self.worst_price = self.worst_price or self.entry_price
        self.last_price = self.last_price or self.entry_price

    @property
    def side(self) -> Side:
        return Side.BUY if self.direction > 0 else Side.SELL

    def unrealized(self, price: Optional[float] = None) -> float:
        p = self.last_price if price is None else price
        return (p - self.entry_price) * self.quantity * self.direction

    def to_dict(self) -> dict:
        d = asdict(self)
        d["opened_at"] = self.opened_at.isoformat()
        d["funding_until"] = self.funding_until.isoformat() if self.funding_until else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SimPosition":
        d = dict(d)
        d["opened_at"] = datetime.fromisoformat(d["opened_at"])
        if d.get("funding_until"):
            d["funding_until"] = datetime.fromisoformat(d["funding_until"])
        return cls(**d)


@dataclass
class TradeRecord:
    """A closed simulated trade: the PDF's journal fields that exist so far."""
    symbol: str
    strategy: str
    direction: str
    opened_at: datetime
    closed_at: datetime
    entry: float
    exit: float
    stop: float
    target: Optional[float]
    quantity: float
    notional: float
    leverage: float
    risk_usd: float
    r_multiple: float
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    mae_r: float
    mfe_r: float
    slippage_usd: float
    exit_reason: str
    bars_held: int
    confidence: float
    market_regime: str
    market_phases: list[str]
    equity_at_entry: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def return_on_equity(self) -> float:
        return self.net_pnl / self.equity_at_entry if self.equity_at_entry else 0.0

    def as_row(self) -> dict:
        d = asdict(self)
        d["opened_at"] = self.opened_at.isoformat()
        d["closed_at"] = self.closed_at.isoformat()
        return d


@dataclass
class EntryFill:
    quantity: float
    price: float
    fee: float
    slippage_usd: float
    partial: bool


def fill_entry(
    side: Side, qty: float, open_price: float, bar_volume: float, bar_turnover: float,
    costs: CostModel, rules: Optional[InstrumentRules],
) -> Optional[EntryFill]:
    """Market entry at the bar's open. None if nothing can fill."""
    cap = bar_volume * costs.max_bar_participation
    fill_qty = round_qty(min(qty, cap), rules)
    if not meets_minimum(fill_qty, open_price, rules):
        return None
    d = 1 if side == Side.BUY else -1
    slip = costs.slippage(fill_qty * open_price, bar_turnover)
    price = open_price * (1 + d * (costs.spread_bps / 20_000 + slip))
    notional = fill_qty * price
    return EntryFill(
        quantity=fill_qty, price=price, fee=notional * costs.taker_fee,
        slippage_usd=abs(price - open_price) * fill_qty,
        partial=fill_qty < round_qty(qty, rules) - 1e-12,
    )


def fill_entry_from_book(side: Side, qty: float, book, costs: CostModel,
                         rules: Optional[InstrumentRules]) -> Optional[EntryFill]:
    """Paper trading: walk the real order book like a market order would.
    Fills what the visible book holds; the rest is cancelled."""
    levels = book.asks if side == Side.BUY else book.bids
    available = sum(l.quantity for l in levels)
    filled = round_qty(min(qty, available), rules)
    if filled <= 0 or not book.mid_price:
        return None
    remaining, cost = filled, 0.0
    for lvl in levels:
        take = min(remaining, lvl.quantity)
        cost += take * lvl.price
        remaining -= take
        if remaining <= 1e-12:
            break
    price = cost / filled
    if not meets_minimum(filled, price, rules):
        return None
    notional = filled * price
    return EntryFill(
        quantity=filled, price=price, fee=notional * costs.taker_fee,
        slippage_usd=abs(price - book.mid_price) * filled,
        partial=filled < round_qty(qty, rules) - 1e-12,
    )


@dataclass
class ExitFill:
    price: float
    reason: str
    fee: float
    slippage_usd: float


def market_exit(pos: SimPosition, raw: float, reason: str, bar_turnover: float,
                costs: CostModel, stop: bool = False) -> ExitFill:
    """Market (or triggered stop-market) exit at `raw`, after costs."""
    d = pos.direction
    slip = costs.slippage(pos.quantity * pos.entry_price, bar_turnover, stop=stop)
    price = raw * (1 - d * (costs.spread_bps / 20_000 + slip))
    return ExitFill(price, reason, pos.quantity * price * costs.taker_fee,
                    abs(raw - price) * pos.quantity)


def check_exits(pos: SimPosition, o: float, h: float, l: float, c: float,
                bar_turnover: float, costs: CostModel) -> Optional[ExitFill]:
    """
    Gaps at the open, then the stop and the target inside the bar. Updates
    the excursions and the mark. Returns the exit fill if the position
    closed in this bar.
    """
    d = pos.direction

    def limit_exit(price: float) -> ExitFill:
        return ExitFill(price, "target", pos.quantity * price * costs.maker_fee, 0.0)

    trailed = pos.stop != pos.initial_stop
    exit_: Optional[ExitFill] = None
    if (o - pos.stop) * d <= 0:
        exit_ = market_exit(pos, o, "trail_gap" if trailed else "stop_gap",
                            bar_turnover, costs, stop=True)
    elif pos.target is not None and (o - pos.target) * d > 0:
        exit_ = limit_exit(o)  # The resting limit fills at the better open
    else:
        adverse = l if d > 0 else h
        favourable = h if d > 0 else l
        if (adverse - pos.stop) * d <= 0:  # Pessimistic: stop before target
            exit_ = market_exit(pos, pos.stop, "trailing_stop" if trailed else "stop",
                                bar_turnover, costs, stop=True)
        elif pos.target is not None and (favourable - pos.target) * d > 0:
            exit_ = limit_exit(pos.target)

    # Excursions, capped at the exit price when the trade closed in this bar
    hi, lo = h, l
    if exit_ is not None:
        hi, lo = min(h, max(o, exit_.price)), max(l, min(o, exit_.price))
    if d > 0:
        pos.best_price, pos.worst_price = max(pos.best_price, hi), min(pos.worst_price, lo)
    else:
        pos.best_price, pos.worst_price = min(pos.best_price, lo), max(pos.worst_price, hi)
    pos.last_price = exit_.price if exit_ is not None else c
    return exit_


def close_bar(pos: SimPosition, c: float, bar_turnover: float,
              costs: CostModel) -> Optional[ExitFill]:
    """At the close of a strategy bar: count it, move the trailing stop
    (only ever tighter), and exit at market once `max_hold_bars` is up."""
    d = pos.direction
    pos.last_price = c
    pos.bars_held += 1
    if pos.trail_distance:
        candidate = c - d * pos.trail_distance
        if (candidate - pos.stop) * d > 0:
            pos.stop = candidate
    if pos.bars_held >= pos.max_hold_bars:
        return market_exit(pos, c, "time", bar_turnover, costs)
    return None


def step_bar(pos: SimPosition, o: float, h: float, l: float, c: float,
             bar_turnover: float, costs: CostModel) -> Optional[ExitFill]:
    """One strategy bar: intrabar exits, then the close."""
    return (check_exits(pos, o, h, l, c, bar_turnover, costs)
            or close_bar(pos, c, bar_turnover, costs))


def close_trade(pos: SimPosition, exit_: ExitFill, at: datetime,
                equity_at_entry: float) -> TradeRecord:
    gross = (exit_.price - pos.entry_price) * pos.quantity * pos.direction
    fees = pos.fees + exit_.fee
    net = gross - fees - pos.funding
    risk = pos.risk_usd if pos.risk_usd > 0 else float("nan")
    unit = abs(pos.entry_price - pos.initial_stop)
    mfe = (pos.best_price - pos.entry_price) * pos.direction / unit if unit else 0.0
    mae = (pos.worst_price - pos.entry_price) * pos.direction / unit if unit else 0.0
    return TradeRecord(
        symbol=pos.symbol, strategy=pos.strategy, direction=pos.side.value,
        opened_at=pos.opened_at, closed_at=at, entry=pos.entry_price, exit=exit_.price,
        stop=pos.initial_stop, target=pos.target, quantity=pos.quantity,
        notional=pos.quantity * pos.entry_price, leverage=pos.leverage,
        risk_usd=pos.risk_usd, r_multiple=net / risk if risk == risk else 0.0,
        gross_pnl=gross, fees=fees, funding=pos.funding, net_pnl=net,
        mae_r=round(mae, 4), mfe_r=round(mfe, 4),
        slippage_usd=pos.slippage_usd + exit_.slippage_usd,
        exit_reason=exit_.reason, bars_held=pos.bars_held, confidence=pos.confidence,
        market_regime=pos.regime, market_phases=list(pos.phases),
        equity_at_entry=equity_at_entry, details=pos.details,
    )
