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
