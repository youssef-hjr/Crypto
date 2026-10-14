"""
Database tables for market data and scanner output.

Everything the scanner sees and decides is stored, so later milestones
(backtesting, research, post-trade analysis) have history to work from.
Schema changes after this first version should go through Alembic.
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData()

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
JSONType = sa.JSON().with_variant(JSONB(), "postgresql")


candles = sa.Table(
    "candles", metadata,
    sa.Column("exchange", sa.String(32), primary_key=True),
    sa.Column("symbol", sa.String(64), primary_key=True),
    sa.Column("interval", sa.String(8), primary_key=True),
    sa.Column("open_time", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("open", sa.Float, nullable=False),
    sa.Column("high", sa.Float, nullable=False),
    sa.Column("low", sa.Float, nullable=False),
    sa.Column("close", sa.Float, nullable=False),
    sa.Column("volume", sa.Float, nullable=False),
    sa.Column("turnover", sa.Float, nullable=False, default=0.0),
)

feature_snapshots = sa.Table(
    "feature_snapshots", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              primary_key=True, autoincrement=True),
    sa.Column("exchange", sa.String(32), nullable=False),
    sa.Column("symbol", sa.String(64), nullable=False),
    sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("market_regime", sa.String(32)),
    sa.Column("technical", JSONType),
    sa.Column("order_flow", JSONType),
    sa.Column("derivatives", JSONType),
    sa.Index("ix_feature_snapshots_symbol_time", "symbol", "captured_at"),
)

opportunities = sa.Table(
    "opportunities", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              primary_key=True, autoincrement=True),
    sa.Column("exchange", sa.String(32), nullable=False),
    sa.Column("symbol", sa.String(64), nullable=False),
    sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("direction", sa.String(8), nullable=False),
    sa.Column("overall_score", sa.Float, nullable=False),
    sa.Column("market_regime", sa.String(32)),
    sa.Column("scores", JSONType),  # Full OpportunityScore
    sa.Index("ix_opportunities_symbol_time", "symbol", "scored_at"),
)

# ── Risk engine ────────────────────────────────────────────────────────

# Small key/value store for state the Risk Manager must not forget on
# restart: the circuit breaker, equity high-water marks, the loss streak.
risk_state = sa.Table(
    "risk_state", metadata,
    sa.Column("key", sa.String(64), primary_key=True),
    sa.Column("value", JSONType, nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

# Append-only log of breaker trips, resets and other risk incidents.
risk_events = sa.Table(
    "risk_events", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              primary_key=True, autoincrement=True),
    sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("kind", sa.String(32), nullable=False),
    sa.Column("reason", sa.String(128), nullable=False),
    sa.Column("details", JSONType),
    sa.Index("ix_risk_events_time", "occurred_at"),
)

# Every verdict, with the full proposal and the numbers behind it, so any
# decision can be reconstructed later.
risk_decisions = sa.Table(
    "risk_decisions", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              primary_key=True, autoincrement=True),
    sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("symbol", sa.String(64), nullable=False),
    sa.Column("direction", sa.String(8), nullable=False),
    sa.Column("decision", sa.String(16), nullable=False),
    sa.Column("approved_size_usd", sa.Float),
    sa.Column("reasons", JSONType),
    sa.Column("proposal", JSONType),
    sa.Column("metrics", JSONType),
    sa.Index("ix_risk_decisions_symbol_time", "symbol", "decided_at"),
)

equity_snapshots = sa.Table(
    "equity_snapshots", metadata,
    sa.Column("captured_at", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("equity", sa.Float, nullable=False),
    sa.Column("source", sa.String(16), nullable=False),
)

# ── Research: history, backtests and the trade journal ─────────────────

# Funding rate history for perpetuals (Bybit settles every 8 hours).
funding_rates = sa.Table(
    "funding_rates", metadata,
    sa.Column("exchange", sa.String(32), primary_key=True),
    sa.Column("symbol", sa.String(64), primary_key=True),
    sa.Column("funding_time", sa.DateTime(timezone=True), primary_key=True),
    sa.Column("rate", sa.Float, nullable=False),
)

# Order-size rules per contract, saved when history is downloaded so a
# backtest rounds quantities the way the exchange would.
instruments = sa.Table(
    "instruments", metadata,
    sa.Column("exchange", sa.String(32), primary_key=True),
    sa.Column("symbol", sa.String(64), primary_key=True),
    sa.Column("rules", JSONType, nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

# One row per backtest, walk-forward test or paper-trading session.
research_runs = sa.Table(
    "research_runs", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              primary_key=True, autoincrement=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("mode", sa.String(16), nullable=False),  # backtest | walk_forward | paper
    sa.Column("strategies", JSONType, nullable=False),
    sa.Column("symbols", JSONType, nullable=False),
    sa.Column("interval", sa.String(8), nullable=False),
    sa.Column("start", sa.DateTime(timezone=True)),
    sa.Column("end", sa.DateTime(timezone=True)),
    sa.Column("config", JSONType),
    sa.Column("metrics", JSONType),
    sa.Column("by_phase", JSONType),
)

# Every simulated trade, backtest or paper, with the PDF's journal fields.
# Fields with no source yet (news, AI reasoning) stay empty until later
# milestones fill them.
trade_journal = sa.Table(
    "trade_journal", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              sa.ForeignKey("research_runs.id", ondelete="CASCADE")),
    sa.Column("mode", sa.String(16), nullable=False),
    sa.Column("exchange", sa.String(32), nullable=False),
    sa.Column("symbol", sa.String(64), nullable=False),
    sa.Column("strategy", sa.String(64), nullable=False),
    sa.Column("market_regime", sa.String(32)),
    sa.Column("market_phases", JSONType),
    sa.Column("direction", sa.String(8), nullable=False),
    sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("closed_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("entry", sa.Float, nullable=False),
    sa.Column("exit", sa.Float, nullable=False),
    sa.Column("stop", sa.Float, nullable=False),
    sa.Column("target", sa.Float),
    sa.Column("quantity", sa.Float, nullable=False),
    sa.Column("notional", sa.Float, nullable=False),
    sa.Column("leverage", sa.Float, nullable=False),
    sa.Column("risk_usd", sa.Float, nullable=False),
    sa.Column("r_multiple", sa.Float, nullable=False),
    sa.Column("gross_pnl", sa.Float, nullable=False),
    sa.Column("fees", sa.Float, nullable=False),
    sa.Column("funding", sa.Float, nullable=False),
    sa.Column("net_pnl", sa.Float, nullable=False),
    sa.Column("mae_r", sa.Float),
    sa.Column("mfe_r", sa.Float),
    sa.Column("slippage_usd", sa.Float),
    sa.Column("exit_reason", sa.String(32), nullable=False),
    sa.Column("decision_confidence", sa.Float),
    sa.Column("details", JSONType),  # technical state, reasoning, risk metrics
    sa.Index("ix_trade_journal_run", "run_id"),
    sa.Index("ix_trade_journal_symbol_time", "symbol", "opened_at"),
)

# ── AI decisions (Milestone 4) ──────────────────────────────────────────

# Every question put to Claude and its answer, so any trade (or skipped
# trade) can be reconstructed: the brief it saw, what it said, what it cost,
# and what the Risk Manager then decided (PDF section 42).
ai_decisions = sa.Table(
    "ai_decisions", metadata,
    sa.Column("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"),
              primary_key=True, autoincrement=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("mode", sa.String(16), nullable=False),  # preview | paper
    sa.Column("symbol", sa.String(64), nullable=False),
    sa.Column("strategy", sa.String(64), nullable=False),
    sa.Column("direction", sa.String(8), nullable=False),
    sa.Column("bar_time", sa.DateTime(timezone=True)),
    sa.Column("candidate", JSONType),  # Entry, stop, target, score, ...
    sa.Column("brief", JSONType),  # Exactly what Claude was sent
    sa.Column("model", sa.String(64)),
    sa.Column("request_id", sa.String(128)),
    sa.Column("decision", sa.String(16), nullable=False),  # TRADE | NO_TRADE, after checks
    sa.Column("ai_decision", sa.String(16)),  # What Claude itself answered, if anything
    sa.Column("confidence", sa.Float),
    sa.Column("expected_r", sa.Float),
    sa.Column("response", JSONType),  # The validated answer
    sa.Column("raw_text", sa.Text),  # The answer as received
    sa.Column("error", sa.Text),  # Why it fell back to NO TRADE, if it did
    sa.Column("latency_ms", sa.Integer),
    sa.Column("input_tokens", sa.Integer),
    sa.Column("output_tokens", sa.Integer),
    sa.Column("cache_read_tokens", sa.Integer),
    sa.Column("cost_usd", sa.Float),
    sa.Column("risk_decision", sa.String(16)),  # approved | reduced | rejected
    sa.Column("risk_reasons", JSONType),
    sa.Index("ix_ai_decisions_time", "created_at"),
    sa.Index("ix_ai_decisions_symbol_time", "symbol", "created_at"),
)
