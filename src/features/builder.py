"""Orchestrate feature construction from all sub-modules."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.volatility import rolling_volatility, ewma_volatility
from src.features.dispersion import return_dispersion, dispersion_zscore
from src.features.correlation import rolling_avg_correlation, correlation_breakdown
from src.features.macro import vix_proxy, yield_curve_proxy
from src.features.liquidity import volume_shock, amihud_illiquidity
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_feature_matrix(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    macro_prices: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a unified feature matrix from price/return/volume data.

    Every feature is computed with backward-looking windows only (no
    look-ahead).  Rows with NaN (from warm-up periods) are dropped at the
    end so downstream models receive a clean matrix.
    """
    if cfg is None:
        cfg = {}
    feat_cfg = cfg.get("features", {})

    parts: list[pd.DataFrame] = []

    # ── 1. Volatility ──────────────────────────────────────────────
    spy_col = "SPY" if "SPY" in returns.columns else returns.columns[0]
    spy_ret = returns[spy_col]

    vol = rolling_volatility(
        spy_ret,
        windows=feat_cfg.get("volatility_windows", [5, 21, 63]),
    )
    parts.append(vol)

    ewma = ewma_volatility(spy_ret, span=feat_cfg.get("ewma_span", 21))
    parts.append(ewma)

    # ── 2. Dispersion (needs >1 sector column) ────────────────────
    sector_cols = [c for c in returns.columns if c != spy_col]
    if len(sector_cols) >= 2:
        disp = return_dispersion(returns[sector_cols])
        parts.append(disp.to_frame())
        disp_z = dispersion_zscore(
            disp,
            window=feat_cfg.get("dispersion_zscore_window", 63),
        )
        parts.append(disp_z.to_frame())

    # ── 3. Correlation ────────────────────────────────────────────
    if len(sector_cols) >= 2:
        corr_win = feat_cfg.get("correlation_window", 63)
        avg_corr = rolling_avg_correlation(returns[sector_cols], window=corr_win)
        parts.append(avg_corr.to_frame())
        breakdown = correlation_breakdown(
            avg_corr,
            window=corr_win,
            threshold_std=feat_cfg.get("correlation_breakdown_std", 2.0),
        )
        parts.append(breakdown.to_frame())

    # ── 4. Macro proxies ──────────────────────────────────────────
    vix = vix_proxy(spy_ret, window=feat_cfg.get("vix_proxy_window", 21))
    parts.append(vix.to_frame())

    if macro_prices is not None and "TLT" in macro_prices.columns and "SHY" in macro_prices.columns:
        yc = yield_curve_proxy(macro_prices["TLT"], macro_prices["SHY"])
        parts.append(yc.to_frame())

    # ── 5. Liquidity ──────────────────────────────────────────────
    if volume is not None and spy_col in volume.columns:
        vs = volume_shock(
            volume[spy_col],
            window=feat_cfg.get("volume_shock_window", 21),
            threshold=feat_cfg.get("volume_shock_threshold", 2.0),
        )
        parts.append(vs)

        amihud = amihud_illiquidity(
            returns[spy_col],
            volume[spy_col],
            window=feat_cfg.get("amihud_window", 21),
        )
        parts.append(amihud)

    # ── Combine & clean ──────────────────────────────────────────
    features = pd.concat(parts, axis=1)
    features = features.loc[returns.index.intersection(features.index)]
    n_before = len(features)
    features = features.dropna()
    n_after = len(features)
    log.info(
        "Feature matrix: %d features × %d rows (dropped %d warm-up rows)",
        features.shape[1],
        n_after,
        n_before - n_after,
    )
    return features
