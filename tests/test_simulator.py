"""Simulated fills: spread, slippage, fees, partial fills, stops, targets,
trailing stops, time exits and funding."""
from datetime import datetime, timezone

import pytest

from models import InstrumentRules, OrderBookLevel, OrderBookSnapshot, Side
from research.simulator import (
    CostModel, SimPosition, check_exits, close_bar, close_trade, fill_entry,
    fill_entry_from_book, funding_times, step_bar,
)

T = datetime(2025, 1, 1, tzinfo=timezone.utc)
NO_SLIP = CostModel(base_slippage_bps=0.0, impact_bps_per_pct=0.0)


def long_pos(entry=100.0, stop=98.0, target=None, trail=None, max_hold=100, qty=1.0):
    return SimPosition(symbol="X", strategy="t", direction=1, quantity=qty, entry_price=entry,
                       initial_stop=stop, stop=stop, target=target, trail_distance=trail,
                       max_hold_bars=max_hold, opened_at=T, signal_price=entry,
                       risk_usd=(entry - stop) * qty)


def test_entry_pays_half_spread_slippage_and_taker_fee():
    c = CostModel(spread_bps=2.0, base_slippage_bps=1.0, impact_bps_per_pct=0.0)
    f = fill_entry(Side.BUY, 1.0, 100.0, bar_volume=1_000, bar_turnover=1e5, costs=c, rules=None)
    assert f.price == pytest.approx(100 * (1 + 0.0001 + 0.0001))
    assert f.fee == pytest.approx(f.price * 0.00055)
    s = fill_entry(Side.SELL, 1.0, 100.0, 1_000, 1e5, c, None)
    assert s.price < 100.0


def test_entry_capped_at_bar_participation_is_partial():
    rules = InstrumentRules(symbol="X", qty_step=0.1, min_qty=0.1, min_notional=5)
    f = fill_entry(Side.BUY, 100.0, 100.0, bar_volume=200, bar_turnover=2e4,
                   costs=NO_SLIP, rules=rules)
    assert f.quantity == pytest.approx(10.0)  # 5% of 200
    assert f.partial


def test_entry_below_exchange_minimum_does_not_fill():
    rules = InstrumentRules(symbol="X", qty_step=1, min_qty=1, min_notional=5)
    assert fill_entry(Side.BUY, 0.5, 100.0, 1_000, 1e5, NO_SLIP, rules) is None


def test_slippage_grows_with_size_relative_to_the_bar():
    c = CostModel()
    assert c.slippage(10_000, 1e6) > c.slippage(100, 1e6)
    assert c.slippage(100, 1e6, stop=True) == pytest.approx(2 * c.slippage(100, 1e6))


def test_stop_inside_the_bar_fills_at_the_stop_minus_costs():
    p = long_pos()
    e = check_exits(p, 100, 101, 97, 99, 1e6, NO_SLIP)
    assert e.reason == "stop"
    assert e.price == pytest.approx(98 * (1 - 0.0001))  # Half the 2 bps spread


def test_gap_through_the_stop_fills_at_the_open():
    p = long_pos()
    e = check_exits(p, 95, 96, 94, 95.5, 1e6, NO_SLIP)
    assert e.reason == "stop_gap"
    assert e.price == pytest.approx(95 * (1 - 0.0001))


def test_target_needs_price_to_trade_through_and_pays_maker():
    p = long_pos(target=104)
    assert check_exits(p, 100, 104, 99, 103, 1e6, NO_SLIP) is None  # Touch only
    e = check_exits(p, 103, 104.5, 102, 104, 1e6, NO_SLIP)
    assert e.reason == "target" and e.price == 104
    assert e.fee == pytest.approx(104 * 0.0002)


def test_stop_assumed_first_when_both_are_hit():
    p = long_pos(target=104)
    assert check_exits(p, 100, 105, 97, 101, 1e6, NO_SLIP).reason == "stop"


def test_trailing_stop_only_tightens():
    p = long_pos(trail=2.0)
    close_bar(p, 103, 1e6, NO_SLIP)
    assert p.stop == pytest.approx(101)
    close_bar(p, 102, 1e6, NO_SLIP)
    assert p.stop == pytest.approx(101)  # Never back down
    e = check_exits(p, 102, 102.5, 100.5, 101, 1e6, NO_SLIP)
    assert e.reason == "trailing_stop"


def test_time_exit_after_max_hold_bars():
    p = long_pos(max_hold=2)
    assert step_bar(p, 100, 100.5, 99.5, 100, 1e6, NO_SLIP) is None
    e = step_bar(p, 100, 100.5, 99.5, 100.2, 1e6, NO_SLIP)
    assert e.reason == "time" and p.bars_held == 2


def test_close_trade_accounts_for_every_cost_and_r():
    p = long_pos(qty=2.0)
    p.fees, p.funding = 0.11, 0.05
    check_exits(p, 100, 103, 99.5, 102, 1e6, NO_SLIP)
    e = check_exits(p, 102, 102, 97.5, 98, 1e6, NO_SLIP)
    t = close_trade(p, e, T, equity_at_entry=10_000)
    assert t.gross_pnl == pytest.approx((e.price - 100) * 2)
    assert t.net_pnl == pytest.approx(t.gross_pnl - 0.11 - e.fee - 0.05)
    assert t.r_multiple == pytest.approx(t.net_pnl / 4.0)
    assert t.mfe_r == pytest.approx(1.5)  # High of 103 = +1.5R
    assert t.mae_r < -1.0


def test_funding_settlements_at_0_8_16_utc():
    a = datetime(2025, 1, 1, 7, 30, tzinfo=timezone.utc)
    b = datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert [t.hour for t in funding_times(a, b)] == [8, 16, 0]
    assert funding_times(b, b) == []


def test_book_walk_fill_averages_levels_and_fills_what_is_there():
    book = OrderBookSnapshot(symbol="X", timestamp=T,
                             bids=[OrderBookLevel(price=99.9, quantity=1)],
                             asks=[OrderBookLevel(price=100.1, quantity=1),
                                   OrderBookLevel(price=100.3, quantity=1)])
    f = fill_entry_from_book(Side.BUY, 1.5, book, NO_SLIP, None)
    assert f.price == pytest.approx((100.1 + 0.5 * 100.3) / 1.5)
    assert not f.partial
    f = fill_entry_from_book(Side.BUY, 5, book, NO_SLIP, None)
    assert f.quantity == 2 and f.partial


def test_position_round_trips_through_a_dict():
    p = long_pos(target=104, trail=1.5)
    p.funding_until = T
    assert SimPosition.from_dict(p.to_dict()) == p
