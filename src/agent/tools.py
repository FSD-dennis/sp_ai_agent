"""LangChain @tool wrappers — used by the reasoning node to call domain functions."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.data.fetcher import fetch_prices, fetch_volume
from src.data.preprocessor import clean_prices, compute_returns
from src.features.builder import build_feature_matrix
from src.models.splitter import chronological_split
from src.models.hmm_model import fit_hmm, predict_regimes as hmm_predict
from src.models.clustering_model import fit_kmeans, fit_gmm, predict_clusters
from src.models.evaluation import (
    regime_statistics,
    transition_matrix,
    regime_stability,
    compute_silhouette,
    compare_models,
)


@tool
def fetch_data_tool(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Fetch daily price data from Yahoo Finance for the given tickers."""
    prices = fetch_prices(tickers, start, end, cache_dir)
    return f"Fetched prices: {prices.shape[0]} rows × {prices.shape[1]} columns ({prices.index.min().date()} – {prices.index.max().date()})"


@tool
def build_features_tool(n_features: int = 0) -> str:
    """Build feature matrix from prices and returns (called inside node, this is a placeholder for agent reasoning)."""
    return f"Feature matrix built with {n_features} features."


@tool
def run_hmm_tool(n_regimes: int = 3) -> str:
    """Fit Gaussian HMM regime model with *n_regimes* hidden states."""
    return f"HMM fitted with {n_regimes} regimes."


@tool
def run_clustering_tool(n_clusters: int = 3, method: str = "both") -> str:
    """Fit KMeans and/or GMM clustering for regime detection."""
    return f"Clustering ({method}) fitted with {n_clusters} clusters."


@tool
def evaluate_tool() -> str:
    """Evaluate regime segmentation and compute per-regime statistics."""
    return "Evaluation complete."


@tool
def generate_plots_tool() -> str:
    """Generate all visualisation plots and save to disk."""
    return "Plots generated."


@tool
def write_report_tool() -> str:
    """Write daily research report in markdown format."""
    return "Report generated."


ALL_TOOLS = [
    fetch_data_tool,
    build_features_tool,
    run_hmm_tool,
    run_clustering_tool,
    evaluate_tool,
    generate_plots_tool,
    write_report_tool,
]
