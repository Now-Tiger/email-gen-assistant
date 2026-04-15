#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.evaluator.nodes import (
    aggregate_node,
    fact_coverage_node,
    generate_email_node,
    tone_alignment_node,
    writing_quality_node,
)
from src.state import EvaluatorState


def build_evaluator_graph() -> CompiledStateGraph:
    """
    Builds and compiles the evaluator LangGraph.

    Graph topology:

        START → generate_email
                    ├── fact_coverage   ─┐
                    ├── tone_alignment   ├── aggregate → END
                    └── writing_quality ─┘

    The three metric nodes fan out from generate_email in parallel — LangGraph
    runs them concurrently and waits for all three before advancing to the
    aggregate node (fan-in). Each metric node touches a different state key
    so there are no write conflicts during the merge.
    """
    graph: StateGraph = StateGraph(EvaluatorState)

    graph.add_node("generate_email", generate_email_node)
    graph.add_node("fact_coverage", fact_coverage_node)
    graph.add_node("tone_alignment", tone_alignment_node)
    graph.add_node("writing_quality", writing_quality_node)
    graph.add_node("aggregate", aggregate_node)

    # Entry point
    graph.add_edge(START, "generate_email")

    # Fan-out: generate_email → three metric nodes in parallel
    graph.add_edge("generate_email", "fact_coverage")
    graph.add_edge("generate_email", "tone_alignment")
    graph.add_edge("generate_email", "writing_quality")

    # Fan-in: all three metric nodes → aggregate
    graph.add_edge("fact_coverage", "aggregate")
    graph.add_edge("tone_alignment", "aggregate")
    graph.add_edge("writing_quality", "aggregate")

    graph.add_edge("aggregate", END)

    return graph.compile()


# Module-level singleton — compiled once, reused for every scenario invocation.
evaluator_graph: CompiledStateGraph = build_evaluator_graph()
