"""Pipeline node functions that update AgentState."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from src.agent.state import AgentState
from src.data.fetcher import fetch_prices, fetch_volume
from src.data.preprocessor import clean_prices, compute_returns
from src.features.builder import build_feature_matrix
from src.models.splitter import chronological_split
from src.models.hmm_model import fit_hmm, predict_regimes as hmm_predict, save_model as hmm_save
from src.models.clustering_model import (
    fit_kmeans,
    fit_gmm,
    predict_clusters,
    save_model as cluster_save,
)
from src.models.evaluation import (
    regime_statistics,
    transition_matrix,
    regime_stability,
    compute_silhouette,
    compare_models,
)
from src.visualization.plots import (
    plot_regime_timeline,
    plot_feature_heatmap,
    plot_regime_performance,
    plot_transition_matrix,
    plot_model_comparison,
)
from src.reporting.generator import generate_report, save_report
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── 1. Data node ──────────────────────────────────────────────────


def data_node(state: AgentState) -> dict[str, Any]:
    """Fetch, clean, and compute returns."""
    cfg = state["config"]
    data_cfg = cfg["data"]
    preproc = cfg.get("preprocessing", {})

    tickers = data_cfg["tickers"]
    macro_tickers = data_cfg.get("macro_tickers", [])
    start = data_cfg["start_date"]
    end = data_cfg.get("end_date")
    cache = data_cfg.get("cache_dir")

    all_tickers = tickers + macro_tickers

    prices = fetch_prices(all_tickers, start, end, cache)
    volume = fetch_volume(tickers, start, end, cache)

    prices = clean_prices(
        prices,
        ffill_limit=preproc.get("ffill_limit", 5),
        max_missing_pct=preproc.get("max_missing_pct", 0.20),
    )
    volume = clean_prices(volume, ffill_limit=preproc.get("ffill_limit", 5))

    # Separate macro prices before computing returns
    macro_cols = [c for c in macro_tickers if c in prices.columns]
    main_cols = [c for c in tickers if c in prices.columns]
    macro_prices = prices[macro_cols] if macro_cols else None
    main_prices = prices[main_cols]

    returns = compute_returns(main_prices)

    log.info("Data node complete: %d rows, %d tickers", len(returns), len(returns.columns))
    return {
        "prices": main_prices,
        "volume": volume,
        "returns": returns,
        "macro_prices": macro_prices,
        "current_step": "data_done",
    }


# ── 2. Feature node ──────────────────────────────────────────────


def feature_node(state: AgentState) -> dict[str, Any]:
    """Build feature matrix from data node outputs."""
    features = build_feature_matrix(
        prices=state["prices"],
        returns=state["returns"],
        volume=state.get("volume"),
        macro_prices=state.get("macro_prices"),
        cfg=state["config"],
    )
    log.info("Feature node complete: %d features × %d rows", features.shape[1], features.shape[0])
    return {"features": features, "current_step": "features_done"}


# ── 3. Model node ────────────────────────────────────────────────


def model_node(state: AgentState) -> dict[str, Any]:
    """Fit HMM and clustering models (train-only scaler, predict both splits)."""
    cfg = state["config"]
    model_cfg = cfg.get("models", {})
    split_cfg = cfg.get("split", {})
    seed = cfg.get("seed", 42)
    outputs_cfg = cfg.get("outputs", {})

    features = state["features"]
    returns = state["returns"]

    # Align returns to feature index
    common_idx = features.index.intersection(returns.index)
    features = features.loc[common_idx]
    returns = returns.loc[common_idx]

    spy_col = "SPY" if "SPY" in returns.columns else returns.columns[0]
    spy_returns = returns[spy_col]

    # Chronological split
    train_feat, test_feat = chronological_split(features, split_cfg.get("train_ratio", 0.70))
    split_index = len(train_feat)

    train_ret = spy_returns.loc[train_feat.index]
    test_ret = spy_returns.loc[test_feat.index]

    X_train = train_feat.values
    X_test = test_feat.values
    n_regimes = model_cfg.get("n_regimes", 3)
    hmm_cfg = model_cfg.get("hmm", {})

    # ── HMM ───────────────────────────────────────────────────
    hmm_model, hmm_scaler = fit_hmm(
        X_train,
        n_regimes=n_regimes,
        n_iter=hmm_cfg.get("n_iter", 200),
        covariance_type=hmm_cfg.get("covariance_type", "full"),
        seed=seed,
    )
    hmm_regimes_train = hmm_predict(hmm_model, X_train, hmm_scaler)
    hmm_regimes_test = hmm_predict(hmm_model, X_test, hmm_scaler)
    hmm_regimes = np.concatenate([hmm_regimes_train, hmm_regimes_test])

    models_dir = outputs_cfg.get("models_dir", "outputs/models")
    hmm_save(hmm_model, hmm_scaler, f"{models_dir}/hmm_model.joblib")

    # ── Clustering ────────────────────────────────────────────
    clust_cfg = model_cfg.get("clustering", {})
    method = clust_cfg.get("method", "both")
    n_init = clust_cfg.get("n_init", 10)

    if method in ("kmeans", "both"):
        km_model, km_scaler = fit_kmeans(X_train, n_clusters=n_regimes, n_init=n_init, seed=seed)
        cluster_save(km_model, km_scaler, f"{models_dir}/kmeans_model.joblib")
        cluster_regimes_train = predict_clusters(km_model, X_train, km_scaler)
        cluster_regimes_test = predict_clusters(km_model, X_test, km_scaler)
    if method in ("gmm", "both"):
        gmm_model, gmm_scaler = fit_gmm(X_train, n_components=n_regimes, n_init=n_init, seed=seed)
        cluster_save(gmm_model, gmm_scaler, f"{models_dir}/gmm_model.joblib")
        if method == "gmm":
            cluster_regimes_train = predict_clusters(gmm_model, X_train, gmm_scaler)
            cluster_regimes_test = predict_clusters(gmm_model, X_test, gmm_scaler)

    cluster_regimes = np.concatenate([cluster_regimes_train, cluster_regimes_test])

    log.info("Model node complete: HMM + clustering fitted (%d regimes)", n_regimes)
    return {
        "train_features": X_train,
        "test_features": X_test,
        "train_returns": train_ret,
        "test_returns": test_ret,
        "split_index": split_index,
        "hmm_regimes": hmm_regimes,
        "hmm_regimes_test": hmm_regimes_test,
        "cluster_regimes": cluster_regimes,
        "cluster_regimes_test": cluster_regimes_test,
        "current_step": "model_done",
    }


# ── 4. Eval node ─────────────────────────────────────────────────


def eval_node(state: AgentState) -> dict[str, Any]:
    """Compute regime statistics and model comparison."""
    returns = state["returns"]
    spy_col = "SPY" if "SPY" in returns.columns else returns.columns[0]
    spy_returns = returns[spy_col]

    # Align to feature index
    features = state["features"]
    common_idx = features.index.intersection(spy_returns.index)
    spy_returns = spy_returns.loc[common_idx]

    hmm_regimes = state["hmm_regimes"]
    cluster_regimes = state["cluster_regimes"]

    hmm_stats = regime_statistics(spy_returns, hmm_regimes)
    cluster_stats = regime_statistics(spy_returns, cluster_regimes)

    hmm_trans = transition_matrix(hmm_regimes)
    cluster_trans = transition_matrix(cluster_regimes)

    hmm_stab = regime_stability(hmm_regimes)
    cluster_stab = regime_stability(cluster_regimes)

    X_full = features.values
    sil = compute_silhouette(X_full, cluster_regimes)

    comparison = compare_models(hmm_regimes, cluster_regimes, spy_returns, X_full)

    log.info("Eval node complete")
    return {
        "hmm_stats": hmm_stats,
        "cluster_stats": cluster_stats,
        "hmm_transition": hmm_trans,
        "cluster_transition": cluster_trans,
        "hmm_stability": hmm_stab,
        "cluster_stability": cluster_stab,
        "silhouette": sil,
        "comparison": comparison,
        "current_step": "eval_done",
    }


# ── 5. Viz node ──────────────────────────────────────────────────


def viz_node(state: AgentState) -> dict[str, Any]:
    """Generate all plots."""
    cfg = state["config"]
    plots_dir = cfg.get("outputs", {}).get("plots_dir", "outputs/plots")

    features = state["features"]
    prices = state["prices"]
    spy_col = "SPY" if "SPY" in prices.columns else prices.columns[0]

    common_idx = features.index.intersection(prices.index)

    paths: dict[str, str] = {}
    paths["regime_timeline"] = plot_regime_timeline(
        dates=common_idx,
        regimes=state["hmm_regimes"],
        prices=prices.loc[common_idx, spy_col],
        output_dir=plots_dir,
        title="HMM Regime Timeline — SPY",
    )
    paths["feature_heatmap"] = plot_feature_heatmap(features, plots_dir)
    paths["regime_performance"] = plot_regime_performance(state["hmm_stats"], plots_dir)
    paths["transition_matrix"] = plot_transition_matrix(state["hmm_transition"], plots_dir)
    paths["model_comparison"] = plot_model_comparison(
        state["hmm_stats"],
        state["cluster_stats"],
        plots_dir,
    )

    log.info("Viz node complete: %d plots saved", len(paths))
    return {"plot_paths": paths, "current_step": "viz_done"}


# ── 6. Report node ───────────────────────────────────────────────


def report_node(state: AgentState) -> dict[str, Any]:
    """Render and save the markdown research report."""
    cfg = state["config"]
    reports_dir = cfg.get("outputs", {}).get("reports_dir", "outputs/reports")

    hmm_regimes = state["hmm_regimes"]
    cluster_regimes = state["cluster_regimes"]

    context = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "tickers": cfg["data"]["tickers"],
        "data_start": state["returns"].index.min().strftime("%Y-%m-%d"),
        "data_end": state["returns"].index.max().strftime("%Y-%m-%d"),
        "executive_summary": state.get("executive_summary", "_See reasoning node output below._"),
        "hmm_current_regime": int(hmm_regimes[-1]),
        "cluster_current_regime": int(cluster_regimes[-1]),
        "hmm_stats": state["hmm_stats"],
        "cluster_stats": state["cluster_stats"],
        "hmm_transition_matrix": state["hmm_transition"],
        "hmm_stability": state["hmm_stability"],
        "cluster_stability": state["cluster_stability"],
        "silhouette": state.get("silhouette"),
        "plot_paths": state.get("plot_paths", {}),
        "n_regimes": cfg.get("models", {}).get("n_regimes", 3),
    }

    content = generate_report(context)
    report_path = save_report(content, reports_dir)

    log.info("Report node complete → %s", report_path)
    return {"report_path": report_path, "current_step": "report_done"}


# ── 7. Reasoning node (LLM) ─────────────────────────────────────


def reasoning_node(state: AgentState) -> dict[str, Any]:
    """Call GPT-4o to interpret results and produce executive summary.

    Respects the max_llm_calls budget.  If no API key is configured the
    node falls back to a deterministic summary.
    """
    cfg = state["config"]
    agent_cfg = cfg.get("agent", {})
    api_key = agent_cfg.get("openai_api_key", "")
    llm_calls = state.get("llm_calls_made", 0)
    max_calls = agent_cfg.get("max_llm_calls", 3)

    if not api_key or llm_calls >= max_calls:
        summary = _deterministic_summary(state)
        return {
            "executive_summary": summary,
            "llm_calls_made": llm_calls,
            "current_step": "reasoning_done",
        }

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=agent_cfg.get("model_name", "gpt-4o"),
        temperature=agent_cfg.get("temperature", 0.2),
        api_key=api_key,
    )

    prompt = _build_interpretation_prompt(state)
    response = llm.invoke([
        SystemMessage(content="You are a senior quantitative researcher. Write a concise executive summary for a daily regime-detection research note."),
        HumanMessage(content=prompt),
    ])

    summary = response.content
    log.info("Reasoning node: LLM call %d/%d complete", llm_calls + 1, max_calls)
    return {
        "executive_summary": summary,
        "llm_calls_made": llm_calls + 1,
        "current_step": "reasoning_done",
    }


# ── Helpers ──────────────────────────────────────────────────────


def _deterministic_summary(state: AgentState) -> str:
    """Fallback summary when no LLM is available."""
    hmm_stats = state.get("hmm_stats", {})
    hmm_regimes = state.get("hmm_regimes", np.array([]))
    stability = state.get("hmm_stability", {})

    if len(hmm_regimes) == 0:
        return "No regime data available."

    current = int(hmm_regimes[-1])
    current_info = hmm_stats.get(current, {})

    lines = [
        f"The market is currently in **Regime {current}**.",
        f"This regime is characterised by an annualised return of {current_info.get('ann_return', 0):.2%} "
        f"and volatility of {current_info.get('ann_vol', 0):.2%} (Sharpe {current_info.get('sharpe', 0):.2f}).",
        f"The model identified {len(hmm_stats)} distinct regimes with "
        f"{stability.get('switches_per_year', 0):.1f} regime switches per year on average.",
    ]
    return " ".join(lines)


def _build_interpretation_prompt(state: AgentState) -> str:
    hmm_stats = state.get("hmm_stats", {})
    cluster_stats = state.get("cluster_stats", {})
    hmm_regimes = state.get("hmm_regimes", np.array([]))
    stability = state.get("hmm_stability", {})
    sil = state.get("silhouette", 0.0)

    current_hmm = int(hmm_regimes[-1]) if len(hmm_regimes) > 0 else "N/A"
    return (
        f"HMM current regime: {current_hmm}\n"
        f"HMM regime stats: {hmm_stats}\n"
        f"HMM stability: {stability}\n"
        f"Clustering regime stats: {cluster_stats}\n"
        f"Clustering silhouette: {sil}\n"
        "Please write a 3-5 sentence executive summary covering: "
        "the current regime, key risk signals, model agreement, "
        "and one actionable insight."
    )
