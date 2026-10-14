"""
Market phases for the PDF's Milestone 3 test matrix: bull, bear, sideways,
high volatility, low volatility, crashes and recovery periods.

These label the history so results can be split by phase. They are for
reporting only: no strategy reads them. They are worked out from each
symbol's own daily closes, looking only backwards (except the volatility
cut-offs, which are percentiles of the whole tested sample so "high" and
"low" mean the same thing across the report).

A bar can carry several labels, for example bear + high_volatility + crash.

  bull / bear       30-day return of at least +15% / at most -15%
  sideways          neither
  high / low vol    30-day realized volatility in the top / bottom quarter
  crash             at least 20% below the highest close of the last 14 days
  recovery          up at least 15% from the lowest close of the last 30 days,
                    within 90 days of a crash, and still below the pre-crash high
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = ["bull", "bear", "sideways", "high_volatility", "low_volatility",
          "crash", "recovery"]

TREND_DAYS = 30
TREND_THRESHOLD = 0.15
VOL_DAYS = 30
CRASH_LOOKBACK_DAYS = 14
CRASH_DROP = 0.20
RECOVERY_LOOKBACK_DAYS = 30
RECOVERY_RISE = 0.15
RECOVERY_WINDOW_DAYS = 90


def daily_phases(close: pd.Series) -> pd.DataFrame:
    """Boolean phase columns per UTC day, from a close series of any interval."""
    daily = close.resample("1D").last().dropna()
    ret_30 = daily / daily.shift(TREND_DAYS) - 1
    vol = np.log(daily / daily.shift(1)).rolling(VOL_DAYS).std()
    lo_cut, hi_cut = vol.quantile(0.25), vol.quantile(0.75)

    out = pd.DataFrame(index=daily.index)
    out["bull"] = ret_30 >= TREND_THRESHOLD
    out["bear"] = ret_30 <= -TREND_THRESHOLD
    out["sideways"] = ret_30.notna() & ~out["bull"] & ~out["bear"]
    out["high_volatility"] = vol >= hi_cut
    out["low_volatility"] = vol <= lo_cut

    peak_14 = daily.rolling(CRASH_LOOKBACK_DAYS, min_periods=1).max()
    out["crash"] = daily <= peak_14 * (1 - CRASH_DROP)

    low_30 = daily.rolling(RECOVERY_LOOKBACK_DAYS, min_periods=1).min()
    crash_days = out["crash"].astype(float)
    recent_crash = crash_days.rolling(RECOVERY_WINDOW_DAYS, min_periods=1).max() > 0
    pre_crash_high = daily.rolling(RECOVERY_WINDOW_DAYS + CRASH_LOOKBACK_DAYS,
                                   min_periods=1).max()
    out["recovery"] = (recent_crash & ~out["crash"]
                       & (daily >= low_30 * (1 + RECOVERY_RISE))
                       & (daily < pre_crash_high))
    return out.fillna(False).astype(bool)


def phases_at(daily: pd.DataFrame, when) -> list[str]:
    """Labels in force at `when`: those of the last COMPLETED day, so a trade
    is never labelled with information from the rest of its own day."""
    day = pd.Timestamp(when).tz_convert("UTC").normalize() - pd.Timedelta(days=1)
    if day not in daily.index:
        idx = daily.index.searchsorted(day, side="right") - 1
        if idx < 0:
            return []
        day = daily.index[idx]
    row = daily.loc[day]
    return [p for p in PHASES if bool(row[p])]


def coverage(daily: pd.DataFrame, start=None, end=None) -> dict[str, int]:
    """Days spent in each phase within [start, end)."""
    d = daily
    if start is not None:
        d = d[d.index >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.index < pd.Timestamp(end)]
    return {p: int(d[p].sum()) for p in PHASES}
