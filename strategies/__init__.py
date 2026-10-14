"""Strategy library. Grid trading and martingale-style averaging down are
deliberately absent: the PDF's non-negotiables rule them out."""
from strategies.base import Signal, Strategy
from strategies.breakout import Breakout
from strategies.mean_reversion import MeanReversion
from strategies.trend import TrendFollowing

STRATEGIES: dict[str, type[Strategy]] = {
    TrendFollowing.name: TrendFollowing,
    Breakout.name: Breakout,
    MeanReversion.name: MeanReversion,
}


def build_strategies(names: str | list[str]) -> list[Strategy]:
    """'trend,breakout' or 'all' to strategy instances with default params."""
    if isinstance(names, str):
        names = list(STRATEGIES) if names == "all" else [n.strip() for n in names.split(",")]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies {unknown}; choose from {sorted(STRATEGIES)}")
    return [STRATEGIES[n]() for n in names]


__all__ = ["STRATEGIES", "Signal", "Strategy", "build_strategies"]
