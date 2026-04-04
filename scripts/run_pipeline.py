#!/usr/bin/env python3
"""Full pipeline entry point — fetches data, builds features, fits models,
evaluates, generates plots and report.

Usage::

    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config config/settings.yaml --seed 123
    python scripts/run_pipeline.py --start-date 2015-01-01 --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so ``src.*`` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger
from src.agent.graph import run_pipeline

log = get_logger("run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="S&P 500 Regime Detection Agent — Full Pipeline")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--start-date", default=None, help="Override data start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Override data end date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Load & override config
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.start_date:
        cfg["data"]["start_date"] = args.start_date
    if args.end_date:
        cfg["data"]["end_date"] = args.end_date

    set_global_seed(cfg.get("seed", 42))
    log.info("Starting full pipeline (seed=%d)", cfg.get("seed", 42))

    state = run_pipeline(cfg)

    log.info("─" * 60)
    log.info("Pipeline finished successfully")
    log.info("  Report : %s", state.get("report_path", "N/A"))
    log.info("  Plots  : %s", list(state.get("plot_paths", {}).values()))
    log.info("─" * 60)


if __name__ == "__main__":
    main()
