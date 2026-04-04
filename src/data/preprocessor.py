"""Clean raw price data and compute returns with leakage-safe transforms."""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def clean_prices(
    prices: pd.DataFrame,
    ffill_limit: int = 5,
    max_missing_pct: float = 0.20,
) -> pd.DataFrame:
    """Forward-fill gaps (up to *ffill_limit* days) and drop bad columns.

    Columns with more than *max_missing_pct* of values still missing after
    forward-filling are dropped entirely.
    """
    df = prices.copy()

    # Forward-fill then back-fill first row only
    df = df.ffill(limit=ffill_limit)
    df = df.bfill(limit=1)

    # Drop columns that are still too sparse
    missing_frac = df.isna().mean()
    bad_cols = missing_frac[missing_frac > max_missing_pct].index.tolist()
    if bad_cols:
        log.warning(
            "Dropping columns with >%.0f%% missing: %s",
            max_missing_pct * 100,
            bad_cols,
        )
        df = df.drop(columns=bad_cols)

    # Drop any remaining rows with NaN (warm-up rows)
    df = df.dropna()

    log.info(
        "Cleaned prices: %d rows × %d columns (%s → %s)",
        len(df),
        len(df.columns),
        df.index.min().date(),
        df.index.max().date(),
    )
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns.  Uses .pct_change() which is purely backward-looking."""
    returns = prices.pct_change()
    returns = returns.iloc[1:]  # drop first NaN row
    return returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log daily returns.  Uses shift(1) — no look-ahead."""
    import numpy as np

    log_ret = np.log(prices / prices.shift(1))
    log_ret = log_ret.iloc[1:]
    return log_ret
