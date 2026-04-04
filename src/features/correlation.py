"""Rolling correlation and breakdown-event features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_avg_correlation(
    sector_returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Average pairwise rolling correlation across sectors.

    Uses ``min_periods=window`` — early rows are NaN (safe).
    """
    n_cols = sector_returns.shape[1]
    if n_cols < 2:
        return pd.Series(np.nan, index=sector_returns.index, name="avg_corr")

    corr_series: list[pd.Series] = []
    cols = sector_returns.columns.tolist()
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            pair = sector_returns[[cols[i], cols[j]]].rolling(
                window=window, min_periods=window
            ).corr()
            # rolling().corr() returns a multi-index DF; extract cross-correlation
            idx = pair.index.get_level_values(1)
            cross = pair.loc[idx == cols[j], cols[i]]
            cross.index = cross.index.droplevel(1)
            corr_series.append(cross)

    avg = pd.concat(corr_series, axis=1).mean(axis=1)
    avg.name = "avg_corr"
    return avg


def correlation_breakdown(
    avg_corr: pd.Series,
    window: int = 63,
    threshold_std: float = 2.0,
) -> pd.Series:
    """Binary flag: 1 when rolling correlation drops more than *threshold_std*
    standard deviations below its rolling mean (correlation breakdown event)."""
    roll_mean = avg_corr.rolling(window=window, min_periods=window).mean()
    roll_std = avg_corr.rolling(window=window, min_periods=window).std()
    flag = ((avg_corr - roll_mean) < -threshold_std * roll_std).astype(int)
    flag.name = "corr_breakdown"
    return flag
