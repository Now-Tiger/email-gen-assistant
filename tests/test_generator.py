#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

from src.generator.graph import generator_graph  # noqa: E402  (after load_dotenv)


def _make_state(intent: str, facts: list[str], tone: str, model_name: str | None = None) -> dict:
    return {
        "intent": intent,
        "facts": facts,
        "tone": tone,
        "model_name": model_name,
        "reasoning": None,
        "raw_output": None,
        "subject": None,
        "body": None,
        "error": None,
    }


def test_generator_returns_subject_and_body():
    result = generator_graph.invoke(
        _make_state(
            intent="Follow up after a job interview",
            facts=[
                "Interview was on Monday",
                "Role is Senior ML Engineer",
                "Team uses PyTorch",
            ],
            tone="Formal, grateful",
        )
    )
    assert result["subject"], "Subject must not be empty"
    assert result["body"], "Body must not be empty"
    assert result["error"] is None, f"Unexpected error: {result['error']}"

    print("\n--- Subject ---")
    print(result["subject"])
    print("\n--- Reasoning ---")
    print(result["reasoning"] or "(none captured)")
    print("\n--- Body (first 300 chars) ---")
    print(result["body"][:300])


def test_generator_includes_facts_in_body():
    """Verifies that at least one key fact keyword appears in the email body."""
    result = generator_graph.invoke(
        _make_state(
            intent="Pitch a product to a new client",
            facts=[
                "Product is called DocAI",
                "Reduces document review time by 60%",
                "Offering a free 2-week pilot",
            ],
            tone="Persuasive, confident",
        )
    )
    assert result["error"] is None, f"Generator error: {result['error']}"
    body_lower = result["body"].lower()
    # At least one specific fact keyword should appear
    fact_keywords = ["docai", "60%", "pilot"]
    matched = [kw for kw in fact_keywords if kw in body_lower]
    assert matched, (
        f"None of the expected fact keywords {fact_keywords} found in body.\n"
        f"Body: {result['body'][:400]}"
    )


def test_generator_tone_casual():
    """Smoke test with a casual tone — ensures the graph handles non-formal tones."""
    result = generator_graph.invoke(
        _make_state(
            intent="Catch up with a former colleague",
            facts=[
                "Have not spoken in 6 months",
                "Both working in AI now",
            ],
            tone="Warm, casual",
        )
    )
    assert result["subject"], "Subject must not be empty"
    assert result["body"], "Body must not be empty"
    assert result["error"] is None, f"Unexpected error: {result['error']}"
