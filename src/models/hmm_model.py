"""Gaussian HMM regime detection using hmmlearn."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

log = get_logger(__name__)


def fit_hmm(
    X_train: np.ndarray,
    n_regimes: int = 3,
    n_iter: int = 200,
    covariance_type: str = "full",
    seed: int = 42,
) -> tuple[GaussianHMM, StandardScaler]:
    """Fit a Gaussian HMM on *X_train* (already a 2-D array).

    Returns the fitted model **and** the scaler (fit on train only).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=seed,
        verbose=False,
    )
    model.fit(X_scaled)
    log.info(
        "HMM converged=%s after %d iters (n_regimes=%d)",
        model.monitor_.converged,
        model.monitor_.iter,
        n_regimes,
    )
    return model, scaler


def predict_regimes(
    model: GaussianHMM,
    X: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    """Predict most-likely hidden-state sequence for *X*."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def save_model(model: GaussianHMM, scaler: StandardScaler, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, path)
    log.info("HMM model saved → %s", path)


def load_model(path: str | Path) -> tuple[GaussianHMM, StandardScaler]:
    data = joblib.load(path)
    return data["model"], data["scaler"]
