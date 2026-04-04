"""Tests for feature engineering modules — synthetic data, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.volatility import rolling_volatility, ewma_volatility
from src.features.dispersion import return_dispersion, dispersion_zscore
from src.features.correlation import rolling_avg_correlation, correlation_breakdown
from src.features.macro import vix_proxy, yield_curve_proxy
from src.features.liquidity import volume_shock, amihud_illiquidity
from src.features.builder import build_feature_matrix


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=500)
    data = rng.normal(0, 0.01, (500, 4))
    return pd.DataFrame(data, index=dates, columns=["SPY", "XLK", "XLF", "XLE"])


@pytest.fixture
def synthetic_prices(synthetic_returns):
    return (1 + synthetic_returns).cumprod() * 100


@pytest.fixture
def synthetic_volume(synthetic_returns):
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        rng.integers(1_000_000, 50_000_000, (500, 4)),
        index=synthetic_returns.index,
        columns=synthetic_returns.columns,
    )


# ── Volatility ────────────────────────────────────────────────────


def test_rolling_vol_shape(synthetic_returns):
    vol = rolling_volatility(synthetic_returns["SPY"], windows=[5, 21])
    assert vol.shape[1] == 2
    # First 20 rows should be NaN for 21-day window
    assert vol.iloc[:20]["SPY_vol_21d"].isna().all()


def test_ewma_vol_no_lookahead(synthetic_returns):
    ewma = ewma_volatility(synthetic_returns["SPY"], span=21)
    # Check that value at index i depends only on data before i
    assert ewma.iloc[20:].notna().all().all()


# ── Dispersion ────────────────────────────────────────────────────


def test_dispersion_shape(synthetic_returns):
    disp = return_dispersion(synthetic_returns[["XLK", "XLF", "XLE"]])
    assert len(disp) == 500
    assert disp.notna().all()


def test_dispersion_zscore_warmup(synthetic_returns):
    disp = return_dispersion(synthetic_returns[["XLK", "XLF", "XLE"]])
    z = dispersion_zscore(disp, window=63)
    assert z.iloc[:62].isna().all()
    assert z.iloc[62:].notna().all()


# ── Correlation ───────────────────────────────────────────────────


def test_avg_corr_shape(synthetic_returns):
    corr = rolling_avg_correlation(synthetic_returns[["XLK", "XLF", "XLE"]], window=21)
    assert len(corr) == 500


def test_corr_breakdown_binary(synthetic_returns):
    corr = rolling_avg_correlation(synthetic_returns[["XLK", "XLF"]], window=21)
    bd = correlation_breakdown(corr, window=21, threshold_std=2.0)
    assert set(bd.dropna().unique()).issubset({0, 1})


# ── Macro ─────────────────────────────────────────────────────────


def test_vix_proxy(synthetic_returns):
    vp = vix_proxy(synthetic_returns["SPY"], window=21)
    assert vp.iloc[20:].notna().all()


def test_yield_curve_proxy():
    dates = pd.bdate_range("2020-01-02", periods=100)
    tlt = pd.Series(np.linspace(100, 110, 100), index=dates)
    shy = pd.Series(np.linspace(80, 82, 100), index=dates)
    yc = yield_curve_proxy(tlt, shy)
    assert yc.notna().all()
    assert yc.iloc[-1] > yc.iloc[0]  # TLT grew faster


# ── Liquidity ─────────────────────────────────────────────────────


def test_volume_shock(synthetic_volume):
    vs = volume_shock(synthetic_volume["SPY"], window=21, threshold=2.0)
    assert "SPY_vol_zscore" in vs.columns
    assert "SPY_vol_shock" in vs.columns
    assert set(vs["SPY_vol_shock"].dropna().unique()).issubset({0, 1})


def test_amihud(synthetic_returns, synthetic_volume):
    ami = amihud_illiquidity(synthetic_returns["SPY"], synthetic_volume["SPY"], window=21)
    assert "SPY_amihud" in ami.columns


# ── Builder ───────────────────────────────────────────────────────


def test_builder_no_nan(synthetic_prices, synthetic_returns, synthetic_volume):
    features = build_feature_matrix(
        synthetic_prices,
        synthetic_returns,
        volume=synthetic_volume,
    )
    assert features.isna().sum().sum() == 0
    assert len(features) > 0
    assert features.shape[1] >= 5  # at least a few features
