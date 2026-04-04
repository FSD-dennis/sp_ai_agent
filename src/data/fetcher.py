"""Fetch daily price data from Yahoo Finance with local parquet caching."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

log = get_logger(__name__)


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path | None = None,
    sleep_between: float = 0.5,
) -> pd.DataFrame:
    """Download adjusted-close prices for *tickers* from Yahoo Finance.

    If *cache_dir* is provided and a cached parquet file already covers the
    requested date range, the cached copy is returned instead.  Otherwise
    fresh data is downloaded and the cache is updated.

    Returns a DataFrame with DatetimeIndex rows × ticker columns.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "prices.parquet"
        cached = _try_load_cache(cache_path, tickers, start, end)
        if cached is not None:
            log.info("Loaded %d rows from cache (%s)", len(cached), cache_path)
            return cached
    else:
        cache_path = None

    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        log.info("Fetching %s …", ticker)
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                log.warning("No data returned for %s – skipping", ticker)
                continue
            # yf.download may return MultiIndex columns; flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames[ticker] = df["Close"]
        except Exception:
            log.exception("Failed to fetch %s", ticker)
        time.sleep(sleep_between)

    if not frames:
        raise RuntimeError("No price data could be fetched for any ticker")

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)

    if cache_path is not None:
        prices.to_parquet(cache_path)
        log.info("Cached %d rows → %s", len(prices), cache_path)

    return prices


def fetch_volume(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path | None = None,
    sleep_between: float = 0.5,
) -> pd.DataFrame:
    """Download daily volume data for *tickers* from Yahoo Finance."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "volume.parquet"
        if cache_path.exists():
            vol = pd.read_parquet(cache_path)
            log.info("Loaded volume cache (%d rows)", len(vol))
            return vol
    else:
        cache_path = None

    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        log.info("Fetching volume for %s …", ticker)
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames[ticker] = df["Volume"]
        except Exception:
            log.exception("Failed to fetch volume for %s", ticker)
        time.sleep(sleep_between)

    if not frames:
        raise RuntimeError("No volume data could be fetched")

    volume = pd.DataFrame(frames)
    volume.index = pd.to_datetime(volume.index)
    volume.sort_index(inplace=True)

    if cache_path is not None:
        volume.to_parquet(cache_path)

    return volume


# ── helpers ───────────────────────────────────────────────────────────


def _try_load_cache(
    path: Path, tickers: list[str], start: str, end: str | None
) -> pd.DataFrame | None:
    """Return cached DataFrame if it exists and covers the requested range."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Check ticker coverage
    missing = set(tickers) - set(df.columns)
    if missing:
        log.info("Cache missing tickers %s – re-fetching", missing)
        return None
    # Check date coverage (allow 5 days tolerance for weekends/holidays)
    if pd.Timestamp(start) < df.index.min() - pd.Timedelta(days=5):
        log.info("Cache starts after requested start – re-fetching")
        return None
    if end is not None and pd.Timestamp(end) > df.index.max() + pd.Timedelta(days=3):
        log.info("Cache ends before requested end – re-fetching")
        return None
    return df[tickers]
