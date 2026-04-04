"""Regime evaluation metrics and model comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.utils.logger import get_logger

log = get_logger(__name__)


def regime_statistics(
    returns: pd.Series,
    regimes: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Per-regime summary: mean return, vol, Sharpe, max-drawdown, avg duration."""
    df = pd.DataFrame({"ret": returns.values[: len(regimes)], "regime": regimes})
    stats: dict[int, dict[str, float]] = {}

    for regime_id, grp in df.groupby("regime"):
        ann_ret = grp["ret"].mean() * 252
        ann_vol = grp["ret"].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cum = (1 + grp["ret"]).cumprod()
        drawdown = (cum / cum.cummax() - 1).min()

        # Average consecutive run length
        runs = _run_lengths(regimes, regime_id)
        avg_dur = float(np.mean(runs)) if runs else 0.0

        stats[int(regime_id)] = {
            "ann_return": round(ann_ret, 4),
            "ann_vol": round(ann_vol, 4),
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(drawdown, 4),
            "avg_duration_days": round(avg_dur, 1),
            "pct_time": round(len(grp) / len(df), 4),
        }
    return stats


def transition_matrix(regimes: np.ndarray) -> np.ndarray:
    """Empirical transition-probability matrix."""
    n = int(regimes.max()) + 1
    mat = np.zeros((n, n))
    for i in range(len(regimes) - 1):
        mat[regimes[i], regimes[i + 1]] += 1
    # Normalise rows
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div-by-zero
    return np.round(mat / row_sums, 4)


def regime_stability(regimes: np.ndarray) -> dict[str, float]:
    """Overall stability metrics: avg regime duration and switches per year."""
    switches = int(np.sum(np.diff(regimes) != 0))
    n_days = len(regimes)
    years = n_days / 252
    return {
        "total_switches": switches,
        "switches_per_year": round(switches / years, 2) if years > 0 else 0.0,
        "avg_regime_duration_days": round(n_days / (switches + 1), 1),
    }


def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score for clustering quality."""
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels >= len(X):
        return 0.0
    return round(float(silhouette_score(X, labels)), 4)


def compare_models(
    hmm_regimes: np.ndarray,
    cluster_regimes: np.ndarray,
    returns: pd.Series,
    X: np.ndarray | None = None,
) -> dict[str, dict]:
    """Side-by-side comparison of HMM and clustering results."""
    result: dict[str, dict] = {
        "hmm": {
            "regime_stats": regime_statistics(returns, hmm_regimes),
            "transition_matrix": transition_matrix(hmm_regimes).tolist(),
            "stability": regime_stability(hmm_regimes),
        },
        "clustering": {
            "regime_stats": regime_statistics(returns, cluster_regimes),
            "transition_matrix": transition_matrix(cluster_regimes).tolist(),
            "stability": regime_stability(cluster_regimes),
        },
    }
    if X is not None:
        result["clustering"]["silhouette"] = compute_silhouette(X, cluster_regimes)
    return result


# ── Helpers ───────────────────────────────────────────────────────


def _run_lengths(regimes: np.ndarray, target: int) -> list[int]:
    """Return list of consecutive-run lengths for *target* regime."""
    runs: list[int] = []
    count = 0
    for r in regimes:
        if r == target:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs
