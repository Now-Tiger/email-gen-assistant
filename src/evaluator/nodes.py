#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from statistics import mean

from src.state import EvaluatorState
from src.generator.graph import generator_graph
from src.evaluator.metrics import fact_coverage_score, tone_alignment_score, writing_quality_score


logger = logging.getLogger(__name__)


def generate_email_node(state: EvaluatorState) -> dict:
    """
    Runs the generator graph for this scenario.

    Returns only the keys it updates — LangGraph merges these into state.
    Runs before the parallel fan-out so it can return the full delta safely.
    """
    result = generator_graph.invoke({
        "intent": state["intent"],
        "facts": state["facts"],
        "tone": state["tone"],
        "model_name": state["model_name"],
        "reasoning": None,
        "raw_output": None,
        "subject": None,
        "body": None,
        "error": None,
    })

    if result.get("error"):
        logger.warning(
            "Generator error for scenario %d: %s",
            state["scenario_id"],
            result["error"],
        )

    return {
        "generated_subject": result.get("subject") or "",
        "generated_body": result.get("body") or "",
        "error": result.get("error"),
    }


def fact_coverage_node(state: EvaluatorState) -> dict:
    """
    Metric 1: semantic fact coverage via sentence-transformers cosine similarity.

    Parallel branch — returns ONLY its metric key to avoid concurrent write
    conflicts on shared state fields (LangGraph raises InvalidUpdateError if
    multiple parallel branches write to the same key in the same step).
    """
    body = state.get("generated_body") or ""
    score = fact_coverage_score(state["facts"], body)
    logger.debug("Scenario %d | fact_coverage=%.4f", state["scenario_id"], score)
    return {"metric_fact_coverage": score}


def tone_alignment_node(state: EvaluatorState) -> dict:
    """
    Metric 2: LLM-as-Judge tone alignment via OpenRouter.

    Parallel branch — returns only its two keys (score + reason).
    The reason string is preserved so it appears in the output CSV.
    """
    body = state.get("generated_body") or ""
    score, reason = tone_alignment_score(body, state["tone"])

    logger.debug(
        "Scenario %d | tone_alignment=%.4f | %s",
        state["scenario_id"],
        score,
        reason[:80],
    )

    return {"metric_tone_alignment": score, "tone_alignment_reason": reason}


def writing_quality_node(state: EvaluatorState) -> dict:
    """
    Metric 3: professional writing quality — grammar tool + structural checks.

    Parallel branch — returns only its metric key.
    """
    body = state.get("generated_body") or ""
    score = writing_quality_score(body)
    logger.debug("Scenario %d | writing_quality=%.4f", state["scenario_id"], score)
    return {"metric_writing_quality": score}


def aggregate_node(state: EvaluatorState) -> dict:
    """
    Fan-in node — waits for all three metric branches, then computes the
    composite score as the mean of the three metrics.

    A metric that failed or was skipped contributes 0.0.
    """
    scores = [
        state.get("metric_fact_coverage") or 0.0,
        state.get("metric_tone_alignment") or 0.0,
        state.get("metric_writing_quality") or 0.0,
    ]

    composite = round(mean(scores), 4)

    logger.info(
        "Scenario %d | composite=%.4f  (fc=%.4f  ta=%.4f  wq=%.4f)",
        state["scenario_id"],
        composite,
        scores[0],
        scores[1],
        scores[2],
    )

    return {"composite_score": composite}
