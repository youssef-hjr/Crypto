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


def parse_params(text: str | None) -> dict:
    """'fast=10,stop_atr=2.0' to {'fast': 10, 'stop_atr': 2.0}."""
    params: dict = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"Bad parameter {item!r}; write it as name=value, e.g. fast=10")
        value = value.strip()
        try:
            params[key.strip()] = int(value)
        except ValueError:
            try:
                params[key.strip()] = float(value)
            except ValueError:
                raise ValueError(f"Parameter {key.strip()} needs a number, not {value!r}")
    return params


def build_strategies(names: str | list[str], params: dict | None = None) -> list[Strategy]:
    """'trend,breakout' or 'all' to strategy instances. params override the
    defaults and need exactly one strategy, so it's clear which they belong to."""
    if isinstance(names, str):
        names = list(STRATEGIES) if names == "all" else [n.strip() for n in names.split(",")]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategies {unknown}; choose from {sorted(STRATEGIES)}")
    if params:
        if len(names) != 1:
            raise ValueError("--params needs a single --strategy, e.g. --strategy trend")
        return [STRATEGIES[names[0]](**params)]
    return [STRATEGIES[n]() for n in names]


__all__ = ["STRATEGIES", "Signal", "Strategy", "build_strategies", "parse_params"]
