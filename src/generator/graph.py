#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.generator.nodes import cot_reasoning_node, parse_output_node
from src.state import GeneratorState


def build_generator_graph() -> CompiledStateGraph:
    """
    Builds and compiles the email generator LangGraph.

    Graph topology:
        START → cot_reasoning → parse_output → END

    cot_reasoning_node: single LLM call that produces 2-3 sentences of
    structural reasoning (CoT) followed by the full email starting with
    'Subject:'. Stores both in state.

    parse_output_node: splits raw_output into subject and body fields.
    Pure string processing — no LLM call.
    """
    graph: StateGraph = StateGraph(GeneratorState)

    graph.add_node("cot_reasoning", cot_reasoning_node)
    graph.add_node("parse_output", parse_output_node)

    graph.add_edge(START, "cot_reasoning")
    graph.add_edge("cot_reasoning", "parse_output")
    graph.add_edge("parse_output", END)

    return graph.compile()


# Module-level singleton — compile once, reuse across all invocations.
generator_graph: CompiledStateGraph = build_generator_graph()
