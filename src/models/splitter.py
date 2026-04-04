"""Strictly chronological train/test splitting — no shuffling."""

from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* by time: first *train_ratio* rows → train, rest → test.

    Asserts that ``train.index.max() < test.index.min()`` (no overlap).
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    assert train.index.max() < test.index.min(), "Train/test overlap detected!"

    log.info(
        "Split %d rows → train %d (%s–%s) | test %d (%s–%s)",
        n,
        len(train),
        train.index.min().date(),
        train.index.max().date(),
        len(test),
        test.index.min().date(),
        test.index.max().date(),
    )
    return train, test


def expanding_window_splits(
    df: pd.DataFrame,
    min_train: int = 252,
    step: int = 63,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward (expanding window) splits.

    Each split keeps all data up to a cut-off as training and the next
    *step* rows as test.  The first split starts after *min_train* rows.
    """
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    n = len(df)
    cut = min_train
    while cut + step <= n:
        train = df.iloc[:cut]
        test = df.iloc[cut : cut + step]
        splits.append((train, test))
        cut += step

    log.info("Created %d expanding-window splits (min_train=%d, step=%d)", len(splits), min_train, step)
    return splits
