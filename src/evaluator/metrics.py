#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import re

import language_tool_python
from sentence_transformers import SentenceTransformer, util

from src.prompts import TONE_JUDGE_PROMPT
from src.utils import get_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model instances — loaded once at first use, reused for every call.
# Loading SentenceTransformer and LanguageTool is expensive (seconds each);
# keeping them as module-level singletons avoids that cost per scenario.
# ---------------------------------------------------------------------------
_sentence_model: SentenceTransformer | None = None
_grammar_tool: language_tool_python.LanguageTool | None = None


def _get_sentence_model() -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def _get_grammar_tool() -> language_tool_python.LanguageTool:
    global _grammar_tool
    if _grammar_tool is None:
        _grammar_tool = language_tool_python.LanguageTool("en-US")
    return _grammar_tool


# ---------------------------------------------------------------------------
# Metric 1: Fact Coverage Score
# ---------------------------------------------------------------------------
# Definition: Measures what percentage of the key input facts are semantically
# represented in the generated email body.
#
# Logic: For each fact, compute cosine similarity between that fact's embedding
# and the full email body embedding using all-MiniLM-L6-v2. Similarity ≥ 0.45
# is treated as the fact being "present" (score = 1.0). Values below 0.45 are
# scaled proportionally. Final score = mean across all facts in [0.0, 1.0].
#
# Why 0.45 threshold: empirically, sentence-transformers cosine similarity
# between a short factual claim and a paragraph containing it typically lands
# between 0.45–0.75. Absent claims typically score below 0.30.
# ---------------------------------------------------------------------------
_FACT_COVERAGE_THRESHOLD = 0.45


def fact_coverage_score(facts: list[str], email_body: str) -> float:
    """
    Returns a float in [0.0, 1.0] representing the proportion of input facts
    that are semantically present in the generated email body.
    """
    if not facts or not email_body:
        return 0.0

    model = _get_sentence_model()
    body_embedding = model.encode(email_body, convert_to_tensor=True)

    per_fact_scores: list[float] = []

    for fact in facts:
        fact_embedding = model.encode(fact, convert_to_tensor=True)
        similarity = float(util.cos_sim(fact_embedding, body_embedding).item())
        normalised = min(similarity / _FACT_COVERAGE_THRESHOLD, 1.0)
        per_fact_scores.append(max(normalised, 0.0))

    return round(sum(per_fact_scores) / len(per_fact_scores), 4)


# ---------------------------------------------------------------------------
# Metric 2: Tone Alignment Score
# ---------------------------------------------------------------------------
# Definition: Uses an LLM judge via OpenRouter to score how accurately the
# generated email's tone matches the requested tone label on a 0–10 scale,
# normalised to [0.0, 1.0].
#
# Logic: The judge receives the requested tone string and the full email body,
# then returns {"score": int, "reason": str}. Score is divided by 10.
# The reason field is preserved for explainability in the evaluation CSV.
#
# Why LLM-as-Judge: semantic similarity cannot distinguish "formal" from
# "urgent" — both may use similar vocabulary. Only a language model that
# understands register, emotional loading, and sentence rhythm can reliably
# evaluate tone accuracy.
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def tone_alignment_score(email_body: str, requested_tone: str, judge_model: str | None = None) -> tuple[float, str]:
    """
    Returns (score: float in [0.0, 1.0], reason: str).

    `judge_model` overrides the default LLM_MODEL env var — pass a cheaper
    model here to reduce evaluation cost (the judging task is simpler than
    generation).
    """
    if not email_body or not requested_tone:
        return 0.0, "Missing input"

    prompt = TONE_JUDGE_PROMPT.format(requested_tone=requested_tone, email=email_body)

    llm = get_llm(model=judge_model, temperature=0.0)
    raw_content = ""

    try:
        response = llm.invoke(prompt)
        raw_content = response.content.strip()  # type: ignore[union-attr]
        # Strip markdown code fences the model may wrap JSON in
        cleaned = _JSON_FENCE_RE.sub("", raw_content).strip()
        data: dict = json.loads(cleaned)
        score = float(data.get("score", 0)) / 10.0
        reason = str(data.get("reason", ""))
        return round(min(max(score, 0.0), 1.0), 4), reason

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Tone judge parse error: %s | raw response: %.200s", exc, raw_content)
        return 0.0, f"Parse error: {exc}"

    except Exception as exc:
        logger.error("Tone judge LLM call failed: %s", exc)
        return 0.0, f"LLM error: {exc}"


# ---------------------------------------------------------------------------
# Metric 3: Professional Writing Quality Score
# ---------------------------------------------------------------------------
# Definition: Composite of two equally-weighted sub-scores:
#   (a) Grammar/spelling correctness via LanguageTool    (weight: 0.5)
#   (b) Structural completeness via regex checks         (weight: 0.5)
# Final score = 0.5 * grammar_score + 0.5 * structure_score
#
# Grammar logic: count LanguageTool errors, compute error-rate per word.
# An error rate of 0.1 (1 per 10 words) maps to score 0.0; rate of 0.0
# maps to score 1.0. Linear interpolation between the two.
#
# Structure logic: 4 equally-weighted binary checks:
#   1. Has a "Subject:" line
#   2. Has a greeting (Dear / Hi / Hello / Good morning|afternoon|evening)
#   3. Has a professional closing (Regards / Best / Thanks / Sincerely / Yours)
#   4. Has ≥4 non-empty lines (indicates a multi-paragraph body)
#
# Why hybrid: grammar tools catch lexical errors that LLMs may generate;
# structural checks catch missing components that are invisible to grammar
# tools but would make an email unusable in a real business context.
# ---------------------------------------------------------------------------
_GRAMMAR_ERROR_RATE_MAX = 0.1  # 1 error per 10 words → grammar_score = 0.0

_STRUCTURE_PATTERNS: list[tuple[str, str]] = [
    (r"Subject:", "has_subject_line"),
    (r"(Dear|Hi\b|Hello\b|Good\s+(?:morning|afternoon|evening))", "has_greeting"),
    (
        r"(Sincerely|Best\s+regards|Kind\s+regards|Regards|Thanks|"
        r"Warm\s+regards|Thank\s+you|Yours)",
        "has_closing",
    ),
]


def _grammar_score(text: str) -> float:
    tool = _get_grammar_tool()
    matches = tool.check(text)
    word_count = max(len(text.split()), 1)
    error_rate = len(matches) / word_count
    return max(0.0, 1.0 - (error_rate / _GRAMMAR_ERROR_RATE_MAX))


def _structure_score(email: str) -> float:
    checks: list[bool] = []

    for pattern, _ in _STRUCTURE_PATTERNS:
        checks.append(bool(re.search(pattern, email, re.IGNORECASE)))

    # Multi-paragraph check: at least 4 non-empty lines
    non_empty_lines = [ln for ln in email.split("\n") if ln.strip()]
    checks.append(len(non_empty_lines) >= 4)
    return sum(checks) / len(checks)


def writing_quality_score(email: str) -> float:
    """
    Returns a float in [0.0, 1.0].
    Composite of grammar correctness (50%) and structural completeness (50%).
    """
    if not email:
        return 0.0
    return round(0.5 * _grammar_score(email) + 0.5 * _structure_score(email), 4)
