#!/usr/bin/env python3
"""
CLI entrypoint for the email generation evaluation pipeline.

Runs all 10 scenarios through the evaluator graph for a given OpenRouter model,
scores each email on 3 custom metrics, and writes a CSV with per-scenario scores
plus an AVERAGE summary row.

Usage:
    python run_eval.py --model openrouter/elephant-alpha --output data/results/model_a.csv
    python run_eval.py --model minimax/minimax-m2.5:free  --output data/results/model_b.csv

    # With retry fallback models (tried in order when primary rate-limits):
    python run_eval.py \\
        --model openrouter/elephant-alpha \\
        --fallback-models "minimax/minimax-m2.5:free,z-ai/glm-4.5-air:free" \\
        --output data/results/model_a.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from os import getenv

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.evaluator.graph import evaluator_graph  # noqa: E402  (after load_dotenv)
from src.utils import load_scenarios, save_results  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_METRIC_COLS = [
    "metric_fact_coverage",
    "metric_tone_alignment",
    "metric_writing_quality",
    "composite_score",
]

_STATE_TEMPLATE = {
    "generated_subject": None,
    "generated_body": None,
    "metric_fact_coverage": None,
    "metric_tone_alignment": None,
    "metric_writing_quality": None,
    "tone_alignment_reason": None,
    "composite_score": None,
    "error": None,
}


def _invoke_scenario(scenario_id: int, intent: str, facts: list[str], tone: str,
                     human_reference: str, model_name: str) -> dict:
    """Single graph invocation for one scenario with one model."""
    return evaluator_graph.invoke({
        "scenario_id": scenario_id,
        "intent": intent,
        "facts": facts,
        "tone": tone,
        "model_name": model_name,
        "human_reference": human_reference,
        **_STATE_TEMPLATE,
    })


def _run_scenario_with_retry(
    scenario,
    primary_model: str,
    fallback_models: list[str],
    max_retries: int,
    retry_base_wait: float,
) -> tuple[dict, str]:
    """
    Tries `primary_model` for `max_retries + 1` attempts with exponential
    backoff on failure, then falls through each model in `fallback_models`
    using the same retry logic.

    Returns (result_dict, model_actually_used).
    """
    model_pool = [primary_model] + fallback_models

    for model in model_pool:
        for attempt in range(max_retries + 1):
            result = _invoke_scenario(
                scenario.id, scenario.intent, scenario.facts,
                scenario.tone, scenario.human_reference, model,
            )
            error = result.get("error") or ""
            if not error:
                return result, model

            wait = min(retry_base_wait * (2 ** attempt), 60.0)
            if attempt < max_retries:
                logger.warning(
                    "Scenario %d | model=%s | attempt %d/%d failed (%s) — "
                    "retrying in %.0fs",
                    scenario.id, model, attempt + 1, max_retries + 1,
                    error[:60], wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Scenario %d | model=%s | all %d attempts exhausted (%s)",
                    scenario.id, model, max_retries + 1, error[:60],
                )

        if model != model_pool[-1]:
            logger.info(
                "Scenario %d | switching to next model in pool",
                scenario.id,
            )

    # All models exhausted — return last failed result
    return result, model_pool[-1]


def run_evaluation(
    model_name: str,
    output_path: str,
    fallback_models: list[str] | None = None,
    max_retries: int = 3,
    inter_scenario_delay: float = 5.0,
    retry_base_wait: float = 10.0,
) -> pd.DataFrame:
    """
    Runs the full evaluation pipeline for `model_name` against all 10 scenarios.

    On 429 / generation errors, retries with exponential backoff, then falls
    through `fallback_models` in order before recording the error.

    Returns a DataFrame with 11 rows: one per scenario + AVERAGE.
    """
    scenarios = load_scenarios("data/scenarios.json")
    fallback_models = fallback_models or []
    rows: list[dict] = []

    for i, scenario in enumerate(scenarios, start=1):
        logger.info(
            "Scenario %2d/10 | %s | tone: %s",
            i, scenario.intent[:55], scenario.tone,
        )

        result, model_used = _run_scenario_with_retry(
            scenario, model_name, fallback_models,
            max_retries, retry_base_wait,
        )

        if model_used != model_name:
            logger.info("Scenario %d | used fallback model: %s", scenario.id, model_used)

        rows.append({
            "scenario_id": result["scenario_id"],
            "model": model_name,          # always the primary label for CSV grouping
            "model_used": model_used,     # actual model that produced the email
            "domain": scenario.domain,
            "intent": scenario.intent,
            "tone": scenario.tone,
            "generated_subject": result.get("generated_subject") or "",
            "generated_body": result.get("generated_body") or "",
            "metric_fact_coverage": result.get("metric_fact_coverage"),
            "metric_tone_alignment": result.get("metric_tone_alignment"),
            "metric_writing_quality": result.get("metric_writing_quality"),
            "composite_score": result.get("composite_score"),
            "tone_alignment_reason": result.get("tone_alignment_reason") or "",
            "error": result.get("error") or "",
        })

        if i < len(scenarios) and inter_scenario_delay > 0:
            time.sleep(inter_scenario_delay)

    df = pd.DataFrame(rows)

    # AVERAGE row (excludes errored scenarios from metric averages)
    clean_df = df[df["error"].fillna("").astype(str).str.strip() == ""]
    summary: dict = {
        "scenario_id": "AVERAGE",
        "model": model_name,
        "model_used": "",
        "domain": "",
        "intent": "",
        "tone": "",
        "generated_subject": "",
        "generated_body": "",
        "tone_alignment_reason": "",
        "error": f"{len(df) - len(clean_df)} error(s)" if len(clean_df) < len(df) else "",
    }
    for col in _METRIC_COLS:
        vals = clean_df[col].dropna()
        summary[col] = round(float(vals.mean()), 4) if len(vals) > 0 else 0.0

    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    save_results(df, output_path)
    return df


def _print_summary(df: pd.DataFrame, model_name: str) -> None:
    avg = df[df["scenario_id"] == "AVERAGE"].iloc[0]
    err_mask = df["error"].fillna("").astype(str).str.strip().ne("")
    error_count = len(df[(df["scenario_id"] != "AVERAGE") & err_mask])
    clean_count = 10 - error_count

    print()
    print("=" * 56)
    print("  Evaluation Summary")
    print("=" * 56)
    print(f"  Model:           {model_name}")
    print(f"  Scenarios OK:    {clean_count}/10")
    if error_count:
        print(f"  Errors:          {error_count} scenario(s) — excluded from averages")
    print(f"  Fact Coverage:   {float(avg['metric_fact_coverage']):.3f}  (avg over {clean_count} clean)")
    print(f"  Tone Alignment:  {float(avg['metric_tone_alignment']):.3f}")
    print(f"  Writing Quality: {float(avg['metric_writing_quality']):.3f}")
    print(f"  Composite Score: {float(avg['composite_score']):.3f}")
    print("=" * 56)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the email generation evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=getenv("MODEL_A") or getenv("LLM_MODEL") or "openai/gpt-4o-mini",
        help="Primary OpenRouter model string",
    )
    parser.add_argument(
        "--fallback-models",
        type=str,
        default="",
        help="Comma-separated fallback models to try when primary rate-limits",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/model_a.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries per scenario per model (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(getenv("EVAL_DELAY_SECONDS", "5")),
        help="Seconds between scenarios (default: 5)",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=10.0,
        help="Base wait seconds for retry backoff (default: 10, doubles each attempt)",
    )
    args = parser.parse_args()

    fallback_models = [m.strip() for m in args.fallback_models.split(",") if m.strip()]

    logger.info(
        "Starting evaluation | model=%s | fallbacks=%s | output=%s",
        args.model, fallback_models or "none", args.output,
    )

    try:
        df = run_evaluation(
            model_name=args.model,
            output_path=args.output,
            fallback_models=fallback_models,
            max_retries=args.retries,
            inter_scenario_delay=args.delay,
            retry_base_wait=args.retry_wait,
        )
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        sys.exit(1)

    _print_summary(df, args.model)
    logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
