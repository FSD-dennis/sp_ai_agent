"""Tests for chronological splitter — strict temporal ordering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.splitter import chronological_split, expanding_window_splits


@pytest.fixture
def time_series_df():
    dates = pd.bdate_range("2020-01-02", periods=500)
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"feat1": rng.normal(size=500), "feat2": rng.normal(size=500)},
        index=dates,
    )


# ── chronological_split ──────────────────────────────────────────


def test_split_ratio(time_series_df):
    train, test = chronological_split(time_series_df, train_ratio=0.7)
    assert len(train) == 350
    assert len(test) == 150


def test_no_overlap(time_series_df):
    train, test = chronological_split(time_series_df, train_ratio=0.7)
    assert train.index.max() < test.index.min()


def test_covers_all_rows(time_series_df):
    train, test = chronological_split(time_series_df, train_ratio=0.7)
    assert len(train) + len(test) == len(time_series_df)


def test_temporal_order(time_series_df):
    train, test = chronological_split(time_series_df, train_ratio=0.7)
    assert (train.index == sorted(train.index)).all()
    assert (test.index == sorted(test.index)).all()


# ── expanding_window_splits ──────────────────────────────────────


def test_expanding_splits_count(time_series_df):
    splits = expanding_window_splits(time_series_df, min_train=252, step=63)
    expected = (500 - 252) // 63
    assert len(splits) == expected


def test_expanding_no_overlap(time_series_df):
    splits = expanding_window_splits(time_series_df, min_train=252, step=63)
    for train, test in splits:
        assert train.index.max() < test.index.min()


def test_expanding_train_grows(time_series_df):
    splits = expanding_window_splits(time_series_df, min_train=252, step=63)
    train_sizes = [len(tr) for tr, _ in splits]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] == 252
