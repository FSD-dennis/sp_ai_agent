"""Tests for HMM and clustering models on synthetic regime data."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.hmm_model import fit_hmm, predict_regimes
from src.models.clustering_model import fit_kmeans, fit_gmm, predict_clusters
from src.models.evaluation import (
    regime_statistics,
    transition_matrix,
    regime_stability,
    compute_silhouette,
)
from src.utils.seed import set_global_seed
import pandas as pd


@pytest.fixture
def two_regime_data():
    """Synthetic data with two clearly separable regimes."""
    set_global_seed(42)
    rng = np.random.default_rng(42)
    # Regime 0: low vol, positive drift (clearly separable)
    r0 = rng.normal(0.01, 0.005, (200, 3))
    # Regime 1: high vol, negative drift (clearly separable)
    r1 = rng.normal(-0.02, 0.03, (200, 3))
    X = np.vstack([r0, r1])
    labels_true = np.array([0] * 200 + [1] * 200)
    dates = pd.bdate_range("2020-01-02", periods=400)
    returns = pd.Series(X[:, 0], index=dates)
    return X, labels_true, returns


# ── HMM ──────────────────────────────────────────────────────────


def test_hmm_fit_predict(two_regime_data):
    X, true_labels, _ = two_regime_data
    model, scaler = fit_hmm(X, n_regimes=2, seed=42)
    preds = predict_regimes(model, X, scaler)
    assert preds.shape == (400,)
    assert len(set(preds)) == 2


def test_hmm_separates_regimes(two_regime_data):
    X, true_labels, _ = two_regime_data
    model, scaler = fit_hmm(X, n_regimes=2, seed=42)
    preds = predict_regimes(model, X, scaler)
    # Labels may be swapped, so check both mappings
    match_a = np.mean(preds == true_labels)
    match_b = np.mean(preds == (1 - true_labels))
    assert max(match_a, match_b) > 0.80


# ── KMeans ────────────────────────────────────────────────────────


def test_kmeans_fit_predict(two_regime_data):
    X, _, _ = two_regime_data
    model, scaler = fit_kmeans(X, n_clusters=2, seed=42)
    preds = predict_clusters(model, X, scaler)
    assert preds.shape == (400,)
    assert len(set(preds)) == 2


def test_kmeans_separates_regimes(two_regime_data):
    X, true_labels, _ = two_regime_data
    model, scaler = fit_kmeans(X, n_clusters=2, seed=42)
    preds = predict_clusters(model, X, scaler)
    match_a = np.mean(preds == true_labels)
    match_b = np.mean(preds == (1 - true_labels))
    assert max(match_a, match_b) > 0.70


# ── GMM ───────────────────────────────────────────────────────────


def test_gmm_fit_predict(two_regime_data):
    X, _, _ = two_regime_data
    model, scaler = fit_gmm(X, n_components=2, seed=42)
    preds = predict_clusters(model, X, scaler)
    assert preds.shape == (400,)
    assert len(set(preds)) == 2


# ── Evaluation ────────────────────────────────────────────────────


def test_regime_statistics(two_regime_data):
    _, true_labels, returns = two_regime_data
    stats = regime_statistics(returns, true_labels)
    assert 0 in stats and 1 in stats
    assert stats[0]["ann_vol"] < stats[1]["ann_vol"]  # regime 0 is low-vol


def test_transition_matrix(two_regime_data):
    _, true_labels, _ = two_regime_data
    mat = transition_matrix(true_labels)
    assert mat.shape == (2, 2)
    assert np.allclose(mat.sum(axis=1), 1.0)


def test_regime_stability(two_regime_data):
    _, true_labels, _ = two_regime_data
    stab = regime_stability(true_labels)
    assert stab["total_switches"] == 1  # one switch from R0→R1


def test_silhouette(two_regime_data):
    X, true_labels, _ = two_regime_data
    sil = compute_silhouette(X, true_labels)
    assert 0.0 < sil <= 1.0
