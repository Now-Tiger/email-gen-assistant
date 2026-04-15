from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel


class GeneratorState(TypedDict):
    # inputs
    intent: str
    facts: list[str]
    tone: str
    model_name: str | None
    # intermediate
    reasoning: str | None       # CoT reasoning extracted before the email
    raw_output: str | None      # full LLM response text
    # outputs
    subject: str | None
    body: str | None
    error: str | None


class EvaluatorState(TypedDict):
    # inputs
    scenario_id: int
    intent: str
    facts: list[str]
    tone: str
    model_name: str
    human_reference: str
    # intermediate — populated by generate_email_node
    generated_subject: str | None
    generated_body: str | None
    # metric outputs — populated by parallel metric nodes
    metric_fact_coverage: float | None
    metric_tone_alignment: float | None
    metric_writing_quality: float | None
    tone_alignment_reason: str | None
    # final — populated by aggregate_node
    composite_score: float | None
    error: str | None


class Scenario(BaseModel):
    id: int
    intent: str
    facts: list[str]
    tone: str
    domain: str
    human_reference: str
