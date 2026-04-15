#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

from src.evaluator.metrics import fact_coverage_score, tone_alignment_score, writing_quality_score


# Shared fixtures
SAMPLE_EMAIL = """\
Subject: Following Up — Senior ML Engineer Interview

Dear Sarah,

Thank you for the interview on Monday. I enjoyed discussing PyTorch and the
Recommendations team's work.

I look forward to hearing about the next steps.

Best regards,
[Your Name]
"""

SAMPLE_FACTS = [
    "Interview was held on Monday",
    "Role is Senior ML Engineer on the Recommendations team",
    "Discussed experience with PyTorch",
]


# Metric 1 — Fact Coverage Score

def test_fact_coverage_high_when_facts_present():
    score = fact_coverage_score(SAMPLE_FACTS, SAMPLE_EMAIL)
    assert score >= 0.6, f"Expected >= 0.6 for facts that are clearly present, got {score:.3f}"


def test_fact_coverage_low_when_facts_absent():
    score = fact_coverage_score(
        ["discussion about quantum computing budgets in 1987"],
        SAMPLE_EMAIL,
    )
    assert score < 0.5, f"Expected < 0.5 for absent facts, got {score:.3f}"


def test_fact_coverage_empty_inputs():
    assert fact_coverage_score([], SAMPLE_EMAIL) == 0.0
    assert fact_coverage_score(SAMPLE_FACTS, "") == 0.0


def test_fact_coverage_score_in_range():
    score = fact_coverage_score(SAMPLE_FACTS, SAMPLE_EMAIL)
    assert 0.0 <= score <= 1.0, f"Score out of [0, 1] range: {score}"


# Metric 2 — Tone Alignment Score

def test_tone_alignment_returns_tuple():
    result = tone_alignment_score(SAMPLE_EMAIL, "Formal, grateful")
    assert isinstance(result, tuple) and len(result) == 2


def test_tone_alignment_score_in_range():
    score, _ = tone_alignment_score(SAMPLE_EMAIL, "Formal, grateful")
    assert 0.0 <= score <= 1.0, f"Score out of [0, 1] range: {score}"


def test_tone_alignment_reason_is_non_empty_string():
    _, reason = tone_alignment_score(SAMPLE_EMAIL, "Formal, grateful")
    assert isinstance(reason, str) and len(reason) > 0, "Reason must be a non-empty string"


def test_tone_alignment_missing_inputs():
    score, reason = tone_alignment_score("", "Formal")
    assert score == 0.0
    assert "Missing" in reason

    score, reason = tone_alignment_score(SAMPLE_EMAIL, "")
    assert score == 0.0


# Metric 3 — Writing Quality Score

def test_writing_quality_well_formed_email():
    score = writing_quality_score(SAMPLE_EMAIL)
    assert score >= 0.7, f"Expected >= 0.7 for a well-formed email, got {score:.3f}"


def test_writing_quality_empty_string():
    assert writing_quality_score("") == 0.0


def test_writing_quality_score_in_range():
    score = writing_quality_score(SAMPLE_EMAIL)
    assert 0.0 <= score <= 1.0, f"Score out of [0, 1] range: {score}"


def test_writing_quality_penalises_missing_structure():
    """An email with no Subject, greeting, or closing should score lower."""
    bare_text = "Please find attached the report you requested. Let me know if you need anything else."
    score = writing_quality_score(bare_text)
    assert score < writing_quality_score(SAMPLE_EMAIL), (
        "Well-formed email should score higher than bare text with no structure"
    )
