"""
Market Scanner Agent.

Continuously evaluates the eligible asset universe, filtering by
liquidity, volatility, and market quality, then scoring opportunities.

The universe is dynamically generated from exchange data — never hardcoded.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from agents.order_flow import OrderFlowAgent
from agents.quant import QuantEngine
from config.settings import AppSettings
from events import Event, EventBus, EventType
from exchanges.base import ExchangeAdapter
from models import (
    Candle, DerivativesFeatures, MarketRegime, OpportunityScore,
    OrderFlowFeatures, Side, TechnicalFeatures, TickerInfo,
)

logger = structlog.get_logger()

# Candle interval the scanner works on, in minutes (Bybit kline interval).
CANDLE_INTERVAL = "5"

# Composite score weights. Configurable and testable; must sum to 1.0.
SCORE_WEIGHTS = {
    "trend": 0.20,
    "momentum": 0.15,
    "order_flow": 0.20,
    "liquidity": 0.15,
    "volume": 0.10,
    "volatility": 0.10,
    "derivatives": 0.10,
}

# Volatility bands, as daily-scaled realized volatility (0.03 = 3% per day).
# Starting points for liquid crypto perps; calibrate on stored data.
VOL_TOO_LOW = 0.01
VOL_SWEET_LOW = 0.015
VOL_SWEET_HIGH = 0.06
VOL_TOO_HIGH = 0.10


class MarketScanner:
    """
    Scans the full tradable universe, filters for quality,
    and produces ranked OpportunityScores.
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        event_bus: EventBus,
        quant: QuantEngine,
        order_flow: OrderFlowAgent,
        settings: AppSettings,
    ):
        self._exchange = exchange
        self._event_bus = event_bus
        self._quant = quant
        self._order_flow = order_flow
        self._settings = settings

        # Cached analysis results per symbol
        self._technical: dict[str, TechnicalFeatures] = {}
        self._flow: dict[str, OrderFlowFeatures] = {}
        self._derivatives: dict[str, DerivativesFeatures] = {}
        self._opportunities: dict[str, OpportunityScore] = {}
        # Raw inputs, kept for the risk engine (correlation, liquidity)
        self._candles: dict[str, list[Candle]] = {}
        self._tickers: dict[str, TickerInfo] = {}

    # ── Universe Construction ──────────────────────────────────────────

    async def build_universe(self) -> list[str]:
        """
        Dynamically build the tradable universe from exchange data.
        Filters by liquidity, spread, and availability.
        """
        tickers = await self._exchange.get_tickers()
        eligible = []

        for t in tickers:
            if not t.symbol.endswith(self._settings.default_quote_currency):
                continue
            if t.turnover_24h < self._settings.min_24h_volume_usd:
                continue
            if t.last_price <= 0:
                continue
            eligible.append(t)
            self._tickers[t.symbol] = t

        # Sort by volume descending
        eligible.sort(key=lambda x: x.turnover_24h, reverse=True)
        symbols = [t.symbol for t in eligible]

        logger.info("universe_built", count=len(symbols),
                     top_5=symbols[:5])
        return symbols

    # ── Full Scan Cycle ────────────────────────────────────────────────

    async def scan(self) -> list[OpportunityScore]:
        """
        Run a complete scan cycle:
        1. Build universe
        2. Fetch candles + orderbook + trades for each symbol
        3. Calculate features
        4. Score opportunities
        5. Rank and return
        """
        symbols = await self.build_universe()
        opportunities = []

        # Process symbols in parallel batches to avoid rate limits
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [self._analyze_symbol(sym) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error("scan_error", symbol=sym, error=str(result))
                elif result is not None:
                    opportunities.append(result)

            # Rate limiting between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.5)

        # Rank by overall score
        opportunities.sort(key=lambda x: x.overall_score, reverse=True)

        # Publish scored opportunities
        for opp in opportunities:
            self._opportunities[opp.symbol] = opp
            await self._event_bus.publish(Event(
                EventType.OPPORTUNITY_SCORED, opp, source="scanner"
            ))

        logger.info("scan_complete",
                     total_analyzed=len(symbols),
                     opportunities=len(opportunities),
                     top_score=opportunities[0].overall_score if opportunities else 0)

        return opportunities

    async def _analyze_symbol(self, symbol: str) -> Optional[OpportunityScore]:
        """Analyze a single symbol: technicals + order flow + derivatives."""
        try:
            # Fetch data
            candles = await self._exchange.get_candles(
                symbol, interval=CANDLE_INTERVAL, limit=200
            )
            await self._event_bus.publish(Event(
                EventType.MARKET_DATA_UPDATED,
                {"symbol": symbol, "interval": CANDLE_INTERVAL, "candles": candles},
                source="scanner",
            ))
            self._candles[symbol] = candles
            if len(candles) < 50:
                return None

            # Technical features
            tech = self._quant.calculate_features(candles)
            if tech is None:
                return None
            self._technical[symbol] = tech

            # Order book + trades for order flow
            orderbook = await self._exchange.get_orderbook(symbol, depth=50)
            recent_trades = await self._exchange.get_recent_trades(symbol, limit=200)

            flow = self._order_flow.calculate_features(orderbook, recent_trades)
            self._flow[symbol] = flow

            # Derivatives data
            funding = await self._exchange.get_funding_rate(symbol)
            oi = await self._exchange.get_open_interest(symbol)
            deriv = DerivativesFeatures(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                funding_rate=funding,
                open_interest=oi,
            )
            self._derivatives[symbol] = deriv

            # Market regime
            regime = self._quant.detect_regime(tech)

            await self._event_bus.publish(Event(
                EventType.FEATURES_CALCULATED,
                {"symbol": symbol, "technical": tech, "order_flow": flow,
                 "derivatives": deriv, "regime": regime},
                source="scanner",
            ))

            # Score the opportunity
            score = self._score_opportunity(symbol, tech, flow, deriv, regime)
            return score

        except Exception as e:
            logger.error("analyze_symbol_error", symbol=symbol, error=str(e))
            return None

    # ── Scoring ────────────────────────────────────────────────────────

    def _score_opportunity(
        self,
        symbol: str,
        tech: TechnicalFeatures,
        flow: OrderFlowFeatures,
        deriv: DerivativesFeatures,
        regime: MarketRegime,
    ) -> Optional[OpportunityScore]:
        """
        Score an opportunity based on multiple factors.
        Score = Positive Evidence - Risk Penalties
        Weights are configurable and testable.

        Directional sub-scores (trend, momentum, order flow, derivatives)
        measure evidence FOR the chosen direction, so a strong short setup
        scores as high as an equally strong long setup. Returns None when
        neither direction has a clear edge: no trade is a valid outcome.
        """
        # ── Direction-neutral scores ───────────────────────────────────

        # Liquidity score
        liq_score = 50.0
        if flow.spread_bps < 5:
            liq_score += 30
        elif flow.spread_bps < 15:
            liq_score += 15
        elif flow.spread_bps > 30:
            liq_score -= 20
        if flow.book_depth_bid > 100_000:
            liq_score += 10
        liq_score = max(0, min(100, liq_score))

        # Volume score
        vol_score = 50.0
        rel_vol = tech.relative_volume if tech.relative_volume is not None else 1.0
        if rel_vol > 2.0:
            vol_score += 30
        elif rel_vol > 1.5:
            vol_score += 15
        elif rel_vol < 0.5:
            vol_score -= 20
        vol_score = max(0, min(100, vol_score))

        # Volatility score (moderate is best).
        # realized_volatility is scaled to one day (see QuantEngine).
        volatility_score = 50.0
        rv = tech.realized_volatility
        if rv is not None:
            if VOL_SWEET_LOW < rv < VOL_SWEET_HIGH:
                volatility_score += 20  # Sweet spot
            elif rv > VOL_TOO_HIGH:
                volatility_score -= 15  # Too volatile
            elif rv < VOL_TOO_LOW:
                volatility_score -= 10  # Too quiet
        volatility_score = max(0, min(100, volatility_score))

        # ── Directional scores, computed for both sides ────────────────
        sides = {}
        for side in (Side.BUY, Side.SELL):
            trend = self._trend_score(tech, side)
            momentum = self._momentum_score(tech, side)
            order_flow = self._order_flow_score(flow, side)
            derivatives = self._derivatives_score(deriv, side)
            agreeing = sum([trend > 60, momentum > 60, order_flow > 60])
            sides[side] = (trend, momentum, order_flow, derivatives, agreeing)

        # ── Direction determination ────────────────────────────────────
        # Require at least 2 of 3 directional scores to agree.
        buy_agree = sides[Side.BUY][4]
        sell_agree = sides[Side.SELL][4]
        if buy_agree >= 2 and buy_agree > sell_agree:
            direction = Side.BUY
        elif sell_agree >= 2 and sell_agree > buy_agree:
            direction = Side.SELL
        else:
            logger.debug("no_directional_edge", symbol=symbol,
                         buy_agree=buy_agree, sell_agree=sell_agree)
            return None

        trend_score, momentum_score, of_score, deriv_score, _ = sides[direction]

        # ── Overall composite score ────────────────────────────────────
        weights = SCORE_WEIGHTS

        overall = (
            trend_score * weights["trend"]
            + momentum_score * weights["momentum"]
            + of_score * weights["order_flow"]
            + liq_score * weights["liquidity"]
            + vol_score * weights["volume"]
            + volatility_score * weights["volatility"]
            + deriv_score * weights["derivatives"]
        )

        return OpportunityScore(
            symbol=symbol,
            overall_score=round(overall, 2),
            trend_score=round(trend_score, 2),
            momentum_score=round(momentum_score, 2),
            order_flow_score=round(of_score, 2),
            liquidity_score=round(liq_score, 2),
            volume_score=round(vol_score, 2),
            volatility_score=round(volatility_score, 2),
            derivatives_score=round(deriv_score, 2),
            market_regime=regime,
            direction=direction,
            confidence=round(overall / 100, 2),
            timestamp=datetime.now(timezone.utc),
        )

    # ── Directional sub-scores (0-100, 50 = neutral) ───────────────────

    @staticmethod
    def _trend_score(tech: TechnicalFeatures, side: Side) -> float:
        sign = 1 if side == Side.BUY else -1
        score = 50.0
        if tech.ema_20 is not None and tech.ema_50 is not None:
            if sign * (tech.ema_20 - tech.ema_50) > 0:
                score += 25
            else:
                score -= 25
        if tech.macd_histogram:
            if sign * tech.macd_histogram > 0:
                score += 15
            else:
                score -= 15
        return max(0, min(100, score))

    @staticmethod
    def _momentum_score(tech: TechnicalFeatures, side: Side) -> float:
        # Mirror RSI and momentum for shorts so both sides use the same rules.
        rsi = tech.rsi if tech.rsi is not None else 50.0
        mom = tech.momentum or 0.0
        if side == Side.SELL:
            rsi = 100 - rsi
            mom = -mom
        score = 50.0
        if 55 < rsi < 75:
            score += 25  # Momentum with the trade, not stretched
        elif rsi >= 75:
            score -= 10  # Stretched: exhaustion risk
        elif rsi < 45:
            score -= 15  # Momentum against the trade
        if mom > 0.01:
            score += 15
        return max(0, min(100, score))

    @staticmethod
    def _order_flow_score(flow: OrderFlowFeatures, side: Side) -> float:
        sign = 1 if side == Side.BUY else -1
        score = 50.0
        if sign * flow.bid_ask_imbalance > 0.2:
            score += 20
        elif sign * flow.bid_ask_imbalance < -0.2:
            score -= 20
        if sign * flow.buy_sell_imbalance > 0.2:
            score += 15
        elif sign * flow.buy_sell_imbalance < -0.2:
            score -= 15
        if flow.large_trade_count > 0:
            score += 10
        return max(0, min(100, score))

    @staticmethod
    def _derivatives_score(deriv: DerivativesFeatures, side: Side) -> float:
        score = 50.0
        fr = deriv.funding_rate
        if fr is None:
            return score
        if abs(fr) < 0.0001:
            score += 10  # Neutral funding
        elif fr > 0.001:
            # Crowded longs: bad for new longs, supportive for shorts
            score += -15 if side == Side.BUY else 10
        elif fr < -0.001:
            # Crowded shorts: squeeze risk for shorts, supportive for longs
            score += 10 if side == Side.BUY else -15
        return max(0, min(100, score))

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def opportunities(self) -> dict[str, OpportunityScore]:
        return self._opportunities

    def technical(self, symbol: str) -> Optional[TechnicalFeatures]:
        return self._technical.get(symbol)

    def order_flow(self, symbol: str) -> Optional[OrderFlowFeatures]:
        return self._flow.get(symbol)

    def derivatives(self, symbol: str) -> Optional[DerivativesFeatures]:
        return self._derivatives.get(symbol)

    def candles(self, symbol: str) -> list[Candle]:
        return self._candles.get(symbol, [])

    def ticker(self, symbol: str) -> Optional[TickerInfo]:
        return self._tickers.get(symbol)

    def get_top_opportunities(self, n: int = 10) -> list[OpportunityScore]:
        ranked = sorted(
            self._opportunities.values(),
            key=lambda x: x.overall_score,
            reverse=True,
        )
        return ranked[:n]
