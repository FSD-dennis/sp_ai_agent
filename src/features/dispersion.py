"""Cross-sector return dispersion features."""

from __future__ import annotations

import pandas as pd


def return_dispersion(sector_returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional standard deviation of sector daily returns.

    Computed row-wise — no look-ahead.
    """
    disp = sector_returns.std(axis=1)
    disp.name = "return_dispersion"
    return disp


def dispersion_zscore(
    dispersion: pd.Series,
    window: int = 63,
) -> pd.Series:
    """Rolling z-score of dispersion — uses backward-looking stats only."""
    roll_mean = dispersion.rolling(window=window, min_periods=window).mean()
    roll_std = dispersion.rolling(window=window, min_periods=window).std()
    zscore = (dispersion - roll_mean) / roll_std
    zscore.name = "dispersion_zscore"
    return zscore
