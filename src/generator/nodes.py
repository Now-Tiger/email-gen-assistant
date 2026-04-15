#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from src.state import GeneratorState
from src.utils import get_llm

logger = logging.getLogger(__name__)


def cot_reasoning_node(state: GeneratorState) -> GeneratorState:
    """
    Single LLM call that combines Chain-of-Thought reasoning with email
    generation.

    The USER_TEMPLATE instructs the model to first write 2-3 sentences of
    structural reasoning (audience, CTA, tone strategy), then immediately
    produce the full email starting with 'Subject:'. This single-call CoT
    approach is cheaper than a two-call approach while still forcing the
    model to plan before generating.

    The raw response is stored in `raw_output`; the leading reasoning block
    is split off and stored in `reasoning`. `parse_output_node` handles the
    final extraction of subject and body.
    """
    facts_bulleted = "\n".join(f"- {f}" for f in state["facts"])

    user_content = USER_TEMPLATE.format(
        intent=state["intent"],
        facts_bulleted=facts_bulleted,
        tone=state["tone"],
    )

    llm = get_llm(model=state["model_name"], temperature=0.7)

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return {**state, "reasoning": None, "raw_output": None, "error": str(exc)}

    raw: str = response.content  # type: ignore[assignment]

    # Split the response at the first occurrence of "Subject:" to separate
    # the CoT reasoning preamble from the actual email.
    parts = re.split(r"(?=Subject:)", raw, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        reasoning = parts[0].strip()
        email_raw = parts[1].strip()
    else:
        # Model skipped the reasoning step — treat full output as the email.
        reasoning = ""
        email_raw = raw.strip()

    logger.debug("CoT reasoning: %s", reasoning[:120])

    return {
        **state,
        "reasoning": reasoning,
        "raw_output": email_raw,
        "error": None,
    }


def parse_output_node(state: GeneratorState) -> GeneratorState:
    """
    Extracts the subject line and body from the raw email text stored in
    `raw_output`.

    Expected format (enforced by USER_TEMPLATE):
        Subject: <subject line>
        <blank line>
        <email body>

    Sets `error` if either field is empty so callers can detect failures
    without raising exceptions.
    """
    raw = (state.get("raw_output") or "").strip()
    if not raw:
        return {**state, "subject": "", "body": "", "error": "raw_output is empty"}

    subject_match = re.search(r"^Subject:\s*(.+)", raw, re.IGNORECASE | re.MULTILINE)
    subject = subject_match.group(1).strip() if subject_match else ""

    # Remove the subject line (and optional blank line after it) to get body.
    body = re.sub(r"(?i)^Subject:.*\n?", "", raw, count=1).strip()

    if not subject:
        logger.warning("Could not parse subject from output: %s", raw[:100])
        return {
            **state,
            "subject": "",
            "body": body,
            "error": "Subject line not found in LLM output",
        }

    if not body:
        return {
            **state,
            "subject": subject,
            "body": "",
            "error": "Email body is empty after parsing",
        }

    return {**state, "subject": subject, "body": body, "error": None}
