# Architecture Reference

> Canonical reference document. Every other phase doc derives from this.
> Last updated: 2026-04-15

---

## 1. Project Overview

An AI email-generation assistant that:
1. **Generates** professional emails from (intent, facts, tone) inputs using a LangGraph agent.
2. **Evaluates** output quality with 3 custom metrics across 10 test scenarios.
3. **Compares** two OpenRouter models (A/B) and produces a structured CSV report.

---

## 2. LLM Provider — OpenRouter via OpenAI SDK

All LLM calls go through [OpenRouter](https://openrouter.ai) using the standard OpenAI SDK.
This means a single `ChatOpenAI` client works for every model (GPT, Mistral, Llama, etc.)
by just swapping the `model` string.

```python
from langchain_openai import ChatOpenAI
from os import getenv

def get_llm(model: str | None = None, temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=getenv("OPENROUTER_API_KEY"),
        model=model or getenv("LLM_MODEL", "openai/gpt-4o-mini"),
        temperature=temperature,
    )
```

Required environment variables:

| Variable | Example value |
|---|---|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` |
| `MODEL_A` | `openai/gpt-4o-mini` |
| `MODEL_B` | `arcee-ai/trinity-large-preview:free` |

---

## 3. Repository Layout

```
email-gen-assistant/
├── src/
│   ├── __init__.py
│   ├── state.py            # All TypedDict + Pydantic state definitions
│   ├── prompts.py          # System prompt, user template, few-shot examples, judge prompt
│   ├── utils.py            # get_llm(), load_scenarios(), save_results()
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── nodes.py        # cot_reasoning_node, generate_email_node, parse_output_node
│   │   └── graph.py        # build_generator_graph() -> CompiledGraph
│   └── evaluator/
│       ├── __init__.py
│       ├── metrics.py      # fact_coverage_score(), tone_alignment_score(), writing_quality_score()
│       ├── nodes.py        # one node per metric + aggregate_node
│       └── graph.py        # build_evaluator_graph() -> CompiledGraph
├── data/
│   ├── scenarios.json      # 10 test scenarios + human reference emails
│   └── results/            # auto-generated output directory
│       ├── model_a.csv
│       ├── model_b.csv
│       └── comparison.csv
├── report/
│   └── analysis.md         # 1-page comparative analysis (Phase 6)
├── tests/
│   ├── test_generator.py
│   └── test_metrics.py
├── docs/                   # all phase docs
├── run_eval.py             # CLI entrypoint
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 4. State Models (`src/state.py`)

### 4.1 Generator Graph State

```python
from typing import TypedDict

class GeneratorState(TypedDict):
    # inputs
    intent: str
    facts: list[str]
    tone: str
    model_name: str
    # intermediate
    reasoning: str | None       # CoT output from cot_reasoning_node
    raw_output: str | None      # full LLM text from generate_email_node
    # outputs
    subject: str | None
    body: str | None
    error: str | None
```

### 4.2 Evaluator Graph State

```python
class EvaluatorState(TypedDict):
    # inputs
    scenario_id: int
    intent: str
    facts: list[str]
    tone: str
    model_name: str
    human_reference: str
    # intermediate
    generated_subject: str | None
    generated_body: str | None
    # metric outputs (populated by parallel metric nodes)
    metric_fact_coverage: float | None
    metric_tone_alignment: float | None
    metric_writing_quality: float | None
    # final
    composite_score: float | None
    error: str | None
```

### 4.3 Scenario Pydantic Model

```python
from pydantic import BaseModel

class Scenario(BaseModel):
    id: int
    intent: str
    facts: list[str]
    tone: str
    domain: str
    human_reference: str
```

---

## 5. Graph Wiring

### 5.1 Generator Graph

```
START
  └─► cot_reasoning_node       # CoT: think about audience, CTA, structure
        └─► generate_email_node # produce full email conditioned on reasoning
              └─► parse_output_node  # extract Subject: / body
                    └─► END
```

### 5.2 Evaluator Graph

```
START
  └─► generate_email_node      # calls GeneratorGraph internally
        ├─► fact_coverage_node     # sentence-transformers semantic similarity
        ├─► tone_alignment_node    # LLM-as-Judge via OpenRouter
        └─► writing_quality_node   # grammar tool + structural regex checks
              └─► (all three) ─► aggregate_node
                                      └─► END
```

The three metric nodes run as parallel branches using LangGraph's `Send` API or
conditional edges fanning out from `generate_email_node`.

---

## 6. Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "langgraph>=0.2",
    "langchain>=0.3",
    "langchain-openai>=0.3",
    "openai>=1.0",
    "pydantic>=2.0",
    "sentence-transformers>=3.0",
    "language-tool-python>=2.8",
    "pandas>=2.0",
    "python-dotenv>=1.0",
]
```

---

## 7. Prompting Strategy

Three techniques combined (documented fully in `phase-1-generator.md`):

| Technique | Where applied |
|---|---|
| **Role-playing** | System prompt — model acts as "senior executive communications specialist" |
| **Few-shot examples** | System prompt — 3 worked (intent, facts, tone) → email pairs |
| **Chain-of-Thought** | Separate `cot_reasoning_node` produces a think block; `generate_email_node` conditions on it |

---

## 8. Metrics Summary

| # | Name | Method | Score range |
|---|---|---|---|
| 1 | Fact Coverage | Semantic cosine similarity (sentence-transformers) | 0.0–1.0 |
| 2 | Tone Alignment | LLM-as-Judge via OpenRouter (0–10 → normalised) | 0.0–1.0 |
| 3 | Writing Quality | Grammar tool (0.5) + structural regex checks (0.5) | 0.0–1.0 |

Composite score = mean of all three metrics.

---

## 9. Model Comparison

| | Model A | Model B |
|---|---|---|
| Model | `openai/gpt-4o-mini` | `arcee-ai/trinity-large-preview:free` |
| Prompt | Full few-shot + CoT | Identical |
| Provider | OpenRouter | OpenRouter |
| Hypothesis | Higher quality | Free tier, potential quality trade-off |

---

## 10. Output Schema (`data/results/model_a.csv`)

| Column | Type | Description |
|---|---|---|
| `scenario_id` | int | 1–10 |
| `model` | str | OpenRouter model string |
| `intent` | str | Input intent |
| `tone` | str | Input tone |
| `generated_subject` | str | Parsed subject line |
| `generated_body` | str | Parsed email body |
| `metric_fact_coverage` | float | 0.0–1.0 |
| `metric_tone_alignment` | float | 0.0–1.0 |
| `metric_writing_quality` | float | 0.0–1.0 |
| `composite_score` | float | Mean of 3 metrics |
