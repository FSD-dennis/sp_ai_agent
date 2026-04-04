"""Macro-proxy features derived from market data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vix_proxy(spy_returns: pd.Series, window: int = 21) -> pd.Series:
    """Realized volatility of SPY as a VIX proxy (annualized)."""
    vol = spy_returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)
    vol.name = "vix_proxy"
    return vol


def yield_curve_proxy(
    tlt_prices: pd.Series,
    shy_prices: pd.Series,
) -> pd.Series:
    """TLT/SHY price ratio as a yield-slope proxy.

    Rising ratio → steepening curve; falling → flattening / inversion.
    The ratio is purely point-in-time (no look-ahead).
    """
    ratio = tlt_prices / shy_prices
    ratio.name = "yield_curve_proxy"
    return ratio
