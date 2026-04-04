#!/usr/bin/env python3
"""Incremental daily update — fetches latest data, re-runs models,
generates a new daily report.

Usage::

    python scripts/run_daily.py
    python scripts/run_daily.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.agent.graph import run_pipeline

log = get_logger("run_daily")


def main() -> None:
    parser = argparse.ArgumentParser(description="S&P 500 Regime Agent — Daily Update")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))

    # Force end_date to today and clear cache to get fresh data
    cfg["data"]["end_date"] = datetime.now().strftime("%Y-%m-%d")
    cache_dir = cfg["data"].get("cache_dir")
    if cache_dir:
        # Remove cached prices so yfinance fetches the latest bar
        cache_path = Path(cache_dir) / "prices.parquet"
        if cache_path.exists():
            cache_path.unlink()
            log.info("Cleared price cache for fresh fetch")
        vol_cache = Path(cache_dir) / "volume.parquet"
        if vol_cache.exists():
            vol_cache.unlink()

    log.info("Running daily update for %s", cfg["data"]["end_date"])

    state = run_pipeline(cfg)

    log.info("─" * 60)
    log.info("Daily update complete")
    log.info("  Report : %s", state.get("report_path", "N/A"))
    log.info("─" * 60)


if __name__ == "__main__":
    main()
