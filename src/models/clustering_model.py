"""KMeans and GMM clustering-based regime detection."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

log = get_logger(__name__)


# ── KMeans ────────────────────────────────────────────────────────


def fit_kmeans(
    X_train: np.ndarray,
    n_clusters: int = 3,
    n_init: int = 10,
    seed: int = 42,
) -> tuple[KMeans, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=seed,
    )
    model.fit(X_scaled)
    log.info("KMeans inertia=%.2f (k=%d)", model.inertia_, n_clusters)
    return model, scaler


# ── GMM ───────────────────────────────────────────────────────────


def fit_gmm(
    X_train: np.ndarray,
    n_components: int = 3,
    n_init: int = 10,
    seed: int = 42,
) -> tuple[GaussianMixture, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = GaussianMixture(
        n_components=n_components,
        n_init=n_init,
        random_state=seed,
    )
    model.fit(X_scaled)
    log.info("GMM converged=%s, BIC=%.2f", model.converged_, model.bic(X_scaled))
    return model, scaler


# ── Predict ───────────────────────────────────────────────────────


def predict_clusters(
    model: KMeans | GaussianMixture,
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


# ── Persistence ───────────────────────────────────────────────────


def save_model(model: KMeans | GaussianMixture, scaler: StandardScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    log.info("Clustering model saved → %s", path)


def load_model(path: str | Path) -> tuple[KMeans | GaussianMixture, StandardScaler]:
    data = joblib.load(path)
    return data["model"], data["scaler"]
