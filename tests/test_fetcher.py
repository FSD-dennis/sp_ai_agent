"""Tests for src.data.fetcher — uses mocking to avoid network calls."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

from src.data.fetcher import fetch_prices


@pytest.fixture
def mock_price_df():
    dates = pd.bdate_range("2023-01-02", periods=100)
    return pd.DataFrame(
        {"Close": np.random.default_rng(42).normal(100, 5, 100)},
        index=dates,
    )


@patch("src.data.fetcher.yf.download")
def test_fetch_returns_dataframe(mock_download, mock_price_df):
    mock_download.return_value = mock_price_df
    result = fetch_prices(["SPY"], start="2023-01-01", sleep_between=0)
    assert isinstance(result, pd.DataFrame)
    assert "SPY" in result.columns
    assert len(result) == 100


@patch("src.data.fetcher.yf.download")
def test_fetch_multiple_tickers(mock_download, mock_price_df):
    mock_download.return_value = mock_price_df
    result = fetch_prices(["SPY", "XLK"], start="2023-01-01", sleep_between=0)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 2


@patch("src.data.fetcher.yf.download")
def test_fetch_empty_raises(mock_download):
    mock_download.return_value = pd.DataFrame()
    with pytest.raises(RuntimeError, match="No price data"):
        fetch_prices(["INVALIDTICKER"], start="2023-01-01", sleep_between=0)


@patch("src.data.fetcher.yf.download")
def test_fetch_caches_to_parquet(mock_download, mock_price_df, tmp_path):
    mock_download.return_value = mock_price_df
    result = fetch_prices(["SPY"], start="2023-01-01", cache_dir=tmp_path, sleep_between=0)
    assert (tmp_path / "prices.parquet").exists()
    assert len(result) == 100

    # Second call should hit cache (download not called again with fresh mock)
    mock_download.reset_mock()
    mock_download.return_value = pd.DataFrame()  # would fail if called
    result2 = fetch_prices(["SPY"], start="2023-01-01", cache_dir=tmp_path, sleep_between=0)
    assert len(result2) == 100
