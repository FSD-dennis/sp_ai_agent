"""LangGraph StateGraph definition — wires all pipeline nodes."""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState
from src.agent.nodes import (
    data_node,
    feature_node,
    model_node,
    eval_node,
    viz_node,
    report_node,
    reasoning_node,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_graph() -> StateGraph:
    """Construct and return the compiled StateGraph.

    Flow::

        START → data → features → model → eval → reasoning → viz → report → END

    The reasoning node runs *before* viz/report so its executive summary
    is available for the report template.  If the reasoning node wants to
    retry the model with different parameters it can set ``current_step``
    to ``"retry_model"`` (optional extension — the default graph runs
    linearly).
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("data", data_node)
    graph.add_node("features", feature_node)
    graph.add_node("model", model_node)
    graph.add_node("eval", eval_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("viz", viz_node)
    graph.add_node("report", report_node)

    # Edges
    graph.add_edge(START, "data")
    graph.add_edge("data", "features")
    graph.add_edge("features", "model")
    graph.add_edge("model", "eval")
    graph.add_edge("eval", "reasoning")

    # After reasoning: optional retry or continue
    graph.add_conditional_edges(
        "reasoning",
        _after_reasoning,
        {"continue": "viz", "retry_model": "model"},
    )
    graph.add_edge("viz", "report")
    graph.add_edge("report", END)

    return graph.compile()


def _after_reasoning(state: AgentState) -> str:
    """Route after reasoning node: retry model or continue."""
    if state.get("current_step") == "retry_model":
        log.info("Reasoning requested model retry")
        return "retry_model"
    return "continue"


def run_pipeline(config: dict[str, Any]) -> AgentState:
    """Convenience wrapper: build graph, invoke with initial state, return final state."""
    graph = build_graph()
    initial_state: AgentState = {
        "config": config,
        "current_step": "start",
        "messages": [],
        "llm_calls_made": 0,
        "error": None,
    }
    final_state = graph.invoke(initial_state)
    log.info("Pipeline complete — step: %s", final_state.get("current_step"))
    return final_state
