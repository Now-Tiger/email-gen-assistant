#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv, find_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

from src.state import Scenario

_ = load_dotenv(find_dotenv())

logger = logging.getLogger(__name__)


def get_llm(model: str | None = None, temperature: float = 0.7) -> ChatOpenAI:
    """
    Returns a ChatOpenAI client pointed at OpenRouter.
    Falls back to BASE_URL env var, then the hardcoded OpenRouter endpoint.
    Model defaults to LLM_MODEL env var, then openai/gpt-4o-mini.
    """
    base_url = (os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("BASE_URL", "") or "https://openrouter.ai/api/v1")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    resolved_model = model or os.environ.get("LLM_MODEL", "") or "openai/gpt-oss-120b:free"

    return ChatOpenAI(
        base_url=base_url,
        api_key=SecretStr(api_key),
        model=resolved_model,
        temperature=temperature,
    )


def load_scenarios(path: str = "data/scenarios.json") -> list[Scenario]:
    """Load and validate test scenarios from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Scenario(**s) for s in raw]


def save_results(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to CSV, creating parent directories if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Results saved to %s", path)
