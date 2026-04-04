"""Rolling volatility features — purely backward-looking."""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION_FACTOR = np.sqrt(252)


def rolling_volatility(
    returns: pd.DataFrame | pd.Series,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Annualized rolling standard deviation of returns.

    All windows use ``min_periods=window`` so early rows are NaN (dropped later
    by the feature builder), guaranteeing no partial-window leakage.
    """
    if windows is None:
        windows = [5, 21, 63]

    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    parts: list[pd.DataFrame] = []
    for w in windows:
        vol = returns.rolling(window=w, min_periods=w).std() * ANNUALIZATION_FACTOR
        vol.columns = [f"{c}_vol_{w}d" for c in returns.columns]
        parts.append(vol)

    return pd.concat(parts, axis=1)


def ewma_volatility(
    returns: pd.DataFrame | pd.Series,
    span: int = 21,
) -> pd.DataFrame:
    """Exponentially-weighted moving average volatility (lightweight GARCH proxy)."""
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    ewma = returns.ewm(span=span, min_periods=span).std() * ANNUALIZATION_FACTOR
    ewma.columns = [f"{c}_ewma_vol" for c in returns.columns]
    return ewma
