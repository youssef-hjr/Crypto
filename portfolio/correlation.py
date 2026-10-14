"""
Return correlation between assets.

BTC, ETH, SOL, AVAX and LINK longs are often one big crypto-beta bet, so the
Risk Manager needs to know which assets move together. Correlations are
computed from log returns of closed candles, aligned on candle open time.

Where there isn't enough overlapping history, the correlation is unknown
and the Risk Manager treats the pair as correlated (the conservative side).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from models import Candle


class CorrelationMatrix:

    def __init__(self, values: dict[tuple[str, str], float], samples: dict[tuple[str, str], int]):
        self._values = values
        self._samples = samples

    @classmethod
    def from_candles(
        cls, candles: dict[str, list[Candle]], min_samples: int = 100
    ) -> "CorrelationMatrix":
        closes = {
            sym: {c.timestamp: c.close for c in rows if c.close > 0}
            for sym, rows in candles.items()
        }
        symbols = sorted(closes)
        values: dict[tuple[str, str], float] = {}
        samples: dict[tuple[str, str], int] = {}
        for i, a in enumerate(symbols):
            for b in symbols[i + 1:]:
                shared = sorted(closes[a].keys() & closes[b].keys())
                if len(shared) < 3:
                    continue
                ra = np.diff(np.log([closes[a][t] for t in shared]))
                rb = np.diff(np.log([closes[b][t] for t in shared]))
                key = _key(a, b)
                samples[key] = len(ra)
                if len(ra) < min_samples or ra.std() == 0 or rb.std() == 0:
                    continue
                values[key] = float(np.corrcoef(ra, rb)[0, 1])
        return cls(values, samples)

    def get(self, a: str, b: str) -> Optional[float]:
        """Correlation of returns, or None if it isn't known."""
        if a == b:
            return 1.0
        return self._values.get(_key(a, b))

    def samples(self, a: str, b: str) -> int:
        return self._samples.get(_key(a, b), 0)

    @property
    def symbols(self) -> set[str]:
        return {s for pair in self._values for s in pair}

    def pairs(self) -> Iterable[tuple[str, str, float]]:
        for (a, b), v in self._values.items():
            yield a, b, v


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)
