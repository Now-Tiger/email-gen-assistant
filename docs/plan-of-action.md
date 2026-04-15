# Email Generation Assistant — Plan of Action

> Generated: 2026-04-15  
> LLM Provider: OpenRouter via OpenAI SDK (`langchain-openai` + `openai`)  
> Frameworks: LangGraph, Pydantic v2, sentence-transformers, language-tool-python

---

## Assignment Summary

Three-part AI engineering assessment:

1. **Email Generator** — LangGraph agent with advanced prompt engineering (Role-play + Few-shot + CoT)
2. **Evaluation System** — 3 custom metrics, 10 test scenarios, structured CSV report
3. **Model Comparison** — A/B test two OpenRouter models, write a 1-page analysis

---

## Architecture Overview

```
GeneratorGraph
  START → cot_reasoning_node → parse_output_node → END

EvaluatorGraph (per scenario)
  START → generate_email_node
            ├── fact_coverage_node
            ├── tone_alignment_node
            └── writing_quality_node
                  └── aggregate_node → END
```

All LLM calls use `ChatOpenAI` pointed at `https://openrouter.ai/api/v1` with
`OPENROUTER_API_KEY`. Swapping models requires only changing a model name string.

---

## Project Structure

```
email-gen-assistant/
├── src/
│   ├── state.py              # GeneratorState, EvaluatorState, Scenario (TypedDict + Pydantic)
│   ├── prompts.py            # SYSTEM_PROMPT, USER_TEMPLATE, TONE_JUDGE_PROMPT
│   ├── utils.py              # get_llm(), load_scenarios(), save_results()
│   ├── generator/
│   │   ├── nodes.py          # cot_reasoning_node, parse_output_node
│   │   └── graph.py          # generator_graph
│   └── evaluator/
│       ├── metrics.py        # fact_coverage_score, tone_alignment_score, writing_quality_score
│       ├── nodes.py          # generate_email, fact_coverage, tone_alignment, writing_quality, aggregate
│       └── graph.py          # evaluator_graph
├── data/
│   ├── scenarios.json        # 10 scenarios + human reference emails
│   └── results/
│       ├── model_a.csv
│       ├── model_b.csv
│       └── comparison.csv
├── report/
│   └── analysis.md
├── tests/
│   ├── test_generator.py
│   └── test_metrics.py
├── docs/                     # all phase docs (this folder)
├── run_eval.py               # CLI: python run_eval.py --model <model> --output <path>
├── compare_results.py        # CLI: python compare_results.py
├── .env.example
├── pyproject.toml
└── README.md
```

---

## LLM Provider — OpenRouter

```python
# src/utils.py
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

`.env` variables:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_A=openai/gpt-4o-mini
MODEL_B=arcee-ai/trinity-large-preview:free
LLM_MODEL=openai/gpt-4o-mini
```

---

## Dependencies

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

## Prompting Strategy

Three techniques combined in one prompt flow:

| Technique | Implementation |
|---|---|
| **Role-playing** | System prompt casts model as "senior executive communications specialist" |
| **Few-shot** | 3 worked (intent + facts + tone → email) examples in system prompt |
| **Chain-of-Thought** | `USER_TEMPLATE` asks for 2–3 sentence structural reasoning before the email |

The CoT reasoning and the email generation happen in the same LLM call (one node).
`parse_output_node` splits the response into `reasoning` and `body` fields.

---

## Evaluation Metrics

| # | Name | Technique | Score |
|---|---|---|---|
| 1 | **Fact Coverage** | `sentence-transformers` cosine similarity per fact vs email body | 0.0–1.0 |
| 2 | **Tone Alignment** | LLM-as-Judge (OpenRouter, returns `{"score": 0-10, "reason": "..."}`) | 0.0–1.0 |
| 3 | **Writing Quality** | `language-tool-python` grammar (50%) + regex structural checks (50%) | 0.0–1.0 |

Composite score = `mean(metric_1, metric_2, metric_3)`

---

## Model Comparison

| | Model A | Model B |
|---|---|---|
| Model | `openai/gpt-4o-mini` | `arcee-ai/trinity-large-preview:free` |
| Cost | Low | Free |
| Prompt | Full few-shot + CoT | Identical |
| Hypothesis | Quality baseline | Cost/quality trade-off |

---

## Phase Timeline

| Phase | Task | Doc | Time |
|---|---|---|---|
| 0 | Repo setup, deps, stubs, env | `phase-0-setup.md` | 30 min |
| 1 | LangGraph generator + prompts | `phase-1-generator.md` | 2–3 hr |
| 2 | 10 scenarios + human references | `phase-2-scenarios.md` | 2–3 hr |
| 3 | 3 custom metrics | `phase-3-metrics.md` | 3–4 hr |
| 4 | Evaluator graph + `run_eval.py` | `phase-4-evaluation.md` | 2 hr |
| 5 | Run Model B + comparison CSV | `phase-5-comparison.md` | 1–2 hr |
| 6 | Analysis + README + report | `phase-6-report.md` | 2 hr |
| **Total** | | | **~13–17 hr** |

---

## How to Use These Docs

Each phase doc in this folder is **fully self-contained**:
- Lists its own prerequisites (what must already exist)
- Provides complete code templates with type annotations
- Ends with a completion checklist an agent can verify

An AI coding agent can be dropped into any phase by reading:
1. `docs/00-architecture.md` — for the full system context
2. The relevant `docs/phase-N-*.md` file — for exact implementation instructions

Phases must be executed in order (0 → 1 → 2 → 3 → 4 → 5 → 6) because each phase
depends on the stubs and data produced by the previous phase.
