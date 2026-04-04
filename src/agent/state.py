"""Agent state schema for the LangGraph pipeline."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Typed state flowing through the LangGraph StateGraph.

    All fields are optional (total=False) so nodes can progressively
    populate them.
    """

    # ── Configuration ─────────────────────────────────────────────
    config: dict[str, Any]

    # ── Raw data ──────────────────────────────────────────────────
    prices: Any          # pd.DataFrame (DatetimeIndex × tickers)
    volume: Any          # pd.DataFrame
    returns: Any         # pd.DataFrame (simple returns)
    macro_prices: Any    # pd.DataFrame (TLT, SHY, …)

    # ── Features ──────────────────────────────────────────────────
    features: Any        # pd.DataFrame — unified feature matrix

    # ── Train / Test ──────────────────────────────────────────────
    train_features: Any  # np.ndarray
    test_features: Any   # np.ndarray
    train_returns: Any   # pd.Series
    test_returns: Any    # pd.Series
    split_index: int     # row index where test begins

    # ── Model outputs ─────────────────────────────────────────────
    hmm_regimes: Any          # np.ndarray (full-sample decoded states)
    hmm_regimes_test: Any     # np.ndarray (test-only)
    cluster_regimes: Any      # np.ndarray
    cluster_regimes_test: Any # np.ndarray

    # ── Evaluation ────────────────────────────────────────────────
    hmm_stats: dict[int, dict[str, float]]
    cluster_stats: dict[int, dict[str, float]]
    hmm_transition: Any  # np.ndarray
    cluster_transition: Any
    hmm_stability: dict[str, float]
    cluster_stability: dict[str, float]
    silhouette: float
    comparison: dict[str, dict]

    # ── Outputs ───────────────────────────────────────────────────
    plot_paths: dict[str, str]
    report_path: str
    executive_summary: str

    # ── Control ───────────────────────────────────────────────────
    current_step: str
    error: str | None
    messages: list[dict[str, str]]
    llm_calls_made: int
