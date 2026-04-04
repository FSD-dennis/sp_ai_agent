"""Visualization helpers — all plots save to disk and close figures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logger import get_logger

log = get_logger(__name__)

REGIME_COLORS = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"]


def _save(fig: plt.Figure, directory: str | Path, name: str) -> str:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    path = directory / f"{name}_{stamp}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Return a relative path (safe for sharing / reports)
    rel = str(path)
    log.info("Saved plot → %s", rel)
    return rel


# ── 1. Regime timeline ───────────────────────────────────────────


def plot_regime_timeline(
    dates: pd.DatetimeIndex,
    regimes: np.ndarray,
    prices: pd.Series,
    output_dir: str | Path,
    title: str = "Regime Timeline",
) -> str:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(dates[: len(prices)], prices.values[: len(dates)], color="black", lw=0.8, label="Price")

    # Color background by regime
    n = min(len(dates), len(regimes))
    for i in range(n - 1):
        color = REGIME_COLORS[regimes[i] % len(REGIME_COLORS)]
        ax.axvspan(dates[i], dates[i + 1], alpha=0.25, color=color, lw=0)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    return _save(fig, output_dir, "regime_timeline")


# ── 2. Feature correlation heatmap ───────────────────────────────


def plot_feature_heatmap(
    features: pd.DataFrame,
    output_dir: str | Path,
) -> str:
    corr = features.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation", fontsize=13)
    return _save(fig, output_dir, "feature_heatmap")


# ── 3. Regime performance bars ───────────────────────────────────


def plot_regime_performance(
    regime_stats: dict[int, dict[str, float]],
    output_dir: str | Path,
) -> str:
    regimes = sorted(regime_stats.keys())
    metrics = ["ann_return", "ann_vol", "sharpe"]
    data = {m: [regime_stats[r][m] for r in regimes] for m in metrics}

    x = np.arange(len(regimes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, data[m], width, label=m)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Regime {r}" for r in regimes])
    ax.set_title("Per-Regime Performance", fontsize=13)
    ax.legend()
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    return _save(fig, output_dir, "regime_performance")


# ── 4. Transition matrix heatmap ─────────────────────────────────


def plot_transition_matrix(
    trans_mat: np.ndarray,
    output_dir: str | Path,
) -> str:
    n = trans_mat.shape[0]
    labels = [f"Regime {i}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        trans_mat,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Transition Probabilities", fontsize=13)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    return _save(fig, output_dir, "transition_matrix")


# ── 5. Model comparison ─────────────────────────────────────────


def plot_model_comparison(
    hmm_stats: dict[int, dict[str, float]],
    cluster_stats: dict[int, dict[str, float]],
    output_dir: str | Path,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, (label, stats) in zip(axes, [("HMM", hmm_stats), ("Clustering", cluster_stats)]):
        regimes = sorted(stats.keys())
        sharpes = [stats[r]["sharpe"] for r in regimes]
        colors = [REGIME_COLORS[r % len(REGIME_COLORS)] for r in regimes]
        ax.bar([f"R{r}" for r in regimes], sharpes, color=colors)
        ax.set_title(f"{label} — Sharpe by Regime", fontsize=12)
        ax.axhline(0, color="grey", lw=0.5, ls="--")

    fig.suptitle("HMM vs Clustering Comparison", fontsize=13)
    return _save(fig, output_dir, "model_comparison")
