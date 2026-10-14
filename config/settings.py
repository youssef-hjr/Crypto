"""
Configuration management for the crypto trading system.
All sensitive values come from environment variables.
API keys are NEVER hardcoded or passed to AI components.
"""
from __future__ import annotations

from enum import Enum
from pydantic import Field
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    DEMO = "demo"
    PAPER = "paper"
    LIVE = "live"


class ExchangeName(str, Enum):
    BYBIT = "bybit"
    BINANCE = "binance"


class AccountEnv(str, Enum):
    DEMO = "demo"        # Bybit Demo Trading: real prices, simulated money
    TESTNET = "testnet"  # Bybit testnet: separate, thin, artificial market
    MAINNET = "mainnet"  # Real money. Never allowed in Milestone 1.


class BybitSettings(BaseSettings):
    """
    Bybit exchange configuration. Credentials from env vars only.

    Market data and the account are configured separately:
      - Public market data always comes from MAINNET. It needs no API key,
        and Bybit Demo Trading has no public streams of its own (its docs say
        demo public data is identical to mainnet's).
      - The account (balances, positions, orders) uses `account_env`,
        which defaults to Demo Trading.
    """
    api_key: str = Field(default="", alias="BYBIT_API_KEY")
    api_secret: str = Field(default="", alias="BYBIT_API_SECRET")
    account_env: AccountEnv = Field(default=AccountEnv.DEMO, alias="BYBIT_ACCOUNT_ENV")

    class Config:
        env_prefix = "BYBIT_"

    def get_urls(self) -> dict:
        urls = {
            # Public market data: always mainnet, read-only.
            "rest_public": "https://api.bybit.com",
            "ws_public": "wss://stream.bybit.com/v5/public/linear",
        }
        if self.account_env == AccountEnv.DEMO:
            urls["rest_private"] = "https://api-demo.bybit.com"
            urls["ws_private"] = "wss://stream-demo.bybit.com/v5/private"
        elif self.account_env == AccountEnv.TESTNET:
            urls["rest_private"] = "https://api-testnet.bybit.com"
            urls["ws_private"] = "wss://stream-testnet.bybit.com/v5/private"
        else:
            urls["rest_private"] = "https://api.bybit.com"
            urls["ws_private"] = "wss://stream.bybit.com/v5/private"
        return urls


class RiskSettings(BaseSettings):
    """Risk management parameters. These are deterministic and non-negotiable.
    Every value can be overridden with a RISK_* environment variable."""
    # Per trade
    risk_per_trade_pct: float = 0.0005  # 0.05% of current equity per trade
    max_stop_distance_pct: float = 0.10  # Stops wider than 10% are rejected
    max_leverage: float = 5.0  # Max leverage setting on any one position
    min_opportunity_score: float = 70.0  # Minimum score to consider a trade

    # Portfolio
    max_positions: int = 10  # Max concurrent positions
    max_single_asset_pct: float = 0.20  # Max 20% of equity in one asset
    max_portfolio_risk_pct: float = 0.02  # Max 2% of equity at risk across all stops
    max_gross_leverage: float = 3.0  # Total notional / equity, all positions together

    # Correlation: assets whose returns correlate at or above the threshold
    # are treated as one bet, and their combined risk (same direction) is capped.
    correlation_threshold: float = 0.70
    max_correlated_risk_pct: float = 0.0025  # 0.25% = five 0.05% trades
    min_correlation_samples: int = 100  # Fewer overlapping bars = assume correlated

    # Drawdown (computed by the Risk Manager from equity history)
    max_daily_drawdown_pct: float = 0.05  # From the day's high, UTC day; trips the breaker
    max_total_drawdown_pct: float = 0.15  # From the all-time high; trips the breaker

    # Liquidity and market data
    max_spread_bps: float = 10.0  # Reject if the spread is wider than this
    depth_band_bps: float = 50.0  # Book depth counted within this distance of mid
    max_book_participation: float = 0.10  # Max 10% of that depth
    max_turnover_participation: float = 0.001  # Max 0.1% of 24h turnover
    max_entry_deviation_pct: float = 0.005  # Entry must be within 0.5% of mid
    max_data_age_seconds: float = 15.0  # Order book / portfolio older than this is stale

    # No martingale: after this many losses in a row, the size is cut, never raised
    consecutive_loss_limit: int = 3
    loss_streak_size_factor: float = 0.5
    no_martingale: bool = True  # NEVER override this

    class Config:
        env_prefix = "RISK_"


class DatabaseSettings(BaseSettings):
    # Store candles, features and scored opportunities. Turn off only for
    # quick local dry runs (or pass --no-db to main.py).
    enabled: bool = Field(default=True, alias="DATABASE_ENABLED")
    url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/crypto_agent",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    class Config:
        env_prefix = "DB_"


class AppSettings(BaseSettings):
    """Top-level application settings."""
    environment: Environment = Environment.DEMO
    log_level: str = "INFO"
    trading_enabled: bool = False  # Must be explicitly enabled
    execution_enabled: bool = False  # Separate from trading_enabled

    # Asset universe defaults
    default_quote_currency: str = "USDT"
    min_24h_volume_usd: float = 1_000_000.0
    scan_interval_seconds: int = 30
    orderbook_depth: int = 50

    # Equity used for the risk preview when no Bybit API key is set
    paper_equity: float = 10_000.0

    # Sub-configs
    bybit: BybitSettings = BybitSettings()
    risk: RiskSettings = RiskSettings()
    database: DatabaseSettings = DatabaseSettings()

    class Config:
        env_prefix = "APP_"


def get_settings() -> AppSettings:
    """Factory to build settings from environment."""
    return AppSettings()
