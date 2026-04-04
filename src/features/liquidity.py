"""Liquidity-shock and illiquidity features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def volume_shock(
    volume: pd.DataFrame | pd.Series,
    window: int = 21,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """Z-score of daily volume relative to its rolling mean.

    Returns a DataFrame with columns ``<ticker>_vol_zscore`` and a binary
    ``<ticker>_vol_shock`` flag (1 when |z| > threshold).
    """
    if isinstance(volume, pd.Series):
        volume = volume.to_frame()

    roll_mean = volume.rolling(window=window, min_periods=window).mean()
    roll_std = volume.rolling(window=window, min_periods=window).std()
    zscore = (volume - roll_mean) / roll_std

    shock = (zscore.abs() > threshold).astype(int)

    zscore.columns = [f"{c}_vol_zscore" for c in volume.columns]
    shock.columns = [f"{c}_vol_shock" for c in volume.columns]

    return pd.concat([zscore, shock], axis=1)


def amihud_illiquidity(
    returns: pd.DataFrame | pd.Series,
    volume: pd.DataFrame | pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    """Amihud illiquidity ratio: rolling mean of |return| / volume.

    Higher values indicate lower liquidity.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    if isinstance(volume, pd.Series):
        volume = volume.to_frame()

    # Align columns — only compute for common tickers
    common = returns.columns.intersection(volume.columns)
    ratio = returns[common].abs() / volume[common].replace(0, np.nan)
    illiq = ratio.rolling(window=window, min_periods=window).mean()
    illiq.columns = [f"{c}_amihud" for c in common]
    return illiq
