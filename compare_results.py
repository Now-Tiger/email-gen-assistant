#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge model_a.csv and model_b.csv into a side-by-side comparison CSV.

Prints:
  1. Per-scenario score table
  2. Aggregate metric comparison (with win/loss tally)
  3. Worst-3 scenarios for each model
  4. Failure mode analysis

Usage:
    python compare_results.py
    python compare_results.py --model-a data/results/model_a.csv \\
                              --model-b data/results/model_b.csv \\
                              --output  data/results/comparison.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METRICS = [
    "metric_fact_coverage",
    "metric_tone_alignment",
    "metric_writing_quality",
    "composite_score",
]

METRIC_LABELS = {
    "metric_fact_coverage":   "Fact Coverage",
    "metric_tone_alignment":  "Tone Alignment",
    "metric_writing_quality": "Writing Quality",
    "composite_score":        "Composite Score",
}

_COL_W = 24  # label column width
_MOD_W = 12  # model score column width


def _short(name: str, width: int = _MOD_W) -> str:
    """Truncate/pad a model name for header display."""
    name = name.split("/")[-1]       # strip provider prefix
    name = name.replace(":free", "") # strip tier suffix
    return name[:width]


def _is_error(series: pd.Series) -> pd.Series:
    """True for rows that have a non-empty error string."""
    return series.fillna("").astype(str).str.strip().ne("")


def load_results(path: str) -> pd.DataFrame:
    """Load a results CSV, dropping the AVERAGE summary row."""
    df = pd.read_csv(path)
    # Normalise error column: NaN → "" so comparisons work uniformly
    if "error" in df.columns:
        df["error"] = df["error"].fillna("").astype(str).str.strip()
    return df[df["scenario_id"].astype(str) != "AVERAGE"].copy()


def build_comparison(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """
    Join on scenario_id, compute per-metric deltas (a – b),
    and add a 'winner' column per scenario.
    """
    keep = ["scenario_id", "intent", "tone", "domain", "error"] + METRICS

    def _safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
        return [c for c in cols if c in df.columns]

    merged = df_a[_safe_cols(df_a, keep)].merge(
        df_b[_safe_cols(df_b, keep)],
        on="scenario_id",
        suffixes=("_a", "_b"),
    )

    for m in METRICS:
        merged[f"{m}_delta"] = (
            merged[f"{m}_a"].fillna(0) - merged[f"{m}_b"].fillna(0)
        )

    merged["winner"] = merged["composite_score_delta"].apply(
        lambda d: "A" if d > 0.02 else ("B" if d < -0.02 else "tie")
    )
    return merged


def _print_scenario_table(merged: pd.DataFrame, name_a: str, name_b: str) -> None:
    ha = _short(name_a, 10)
    hb = _short(name_b, 10)
    header = f"{'Scenario':<10} {'Domain':<16} {'Tone':<12} {ha:>10} {hb:>10} {'Δ(A-B)':>8}  Win"
    print(header)
    print("-" * len(header))
    for _, row in merged.iterrows():
        sid = str(row["scenario_id"])
        domain = str(row.get("domain_a", row.get("domain", ""))).replace("_", " ")[:15]
        tone = str(row.get("tone_a", row.get("tone", "")))[:11]
        ca = row.get("composite_score_a", 0.0)
        cb = row.get("composite_score_b", 0.0)
        delta = row.get("composite_score_delta", 0.0)
        winner = row.get("winner", "?")
        print(f"  {sid:<8} {domain:<16} {tone:<12} {ca:>10.3f} {cb:>10.3f} {delta:>+8.3f}  [{winner}]")


def _print_aggregate_table(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> None:
    clean_a = df_a[~_is_error(df_a["error"])]
    clean_b = df_b[~_is_error(df_b["error"])]
    ha = _short(name_a)
    hb = _short(name_b)

    print(f"\n{'Metric':<{_COL_W}} {ha:>{_MOD_W}} {hb:>{_MOD_W}} {'Delta':>8}  Win")
    print(f"{'(avg over clean rows)':<{_COL_W}} {'n=' + str(len(clean_a)):>{_MOD_W}} {'n=' + str(len(clean_b)):>{_MOD_W}} {'(A-B)':>8}")
    print("-" * (_COL_W + _MOD_W * 2 + 12))

    wins_a = wins_b = ties = 0
    for m in METRICS:
        avg_a = clean_a[m].mean() if len(clean_a) else 0.0
        avg_b = clean_b[m].mean() if len(clean_b) else 0.0
        delta = avg_a - avg_b
        if delta > 0.02:
            winner, wins_a = "A", wins_a + 1
        elif delta < -0.02:
            winner, wins_b = "B", wins_b + 1
        else:
            winner, ties = "=", ties + 1
        label = METRIC_LABELS[m]
        print(f"{label:<{_COL_W}} {avg_a:>{_MOD_W}.3f} {avg_b:>{_MOD_W}.3f} {delta:>+8.3f}  [{winner}]")

    print("-" * (_COL_W + _MOD_W * 2 + 12))
    print(f"{'Wins':<{_COL_W}} {wins_a:>{_MOD_W}} {wins_b:>{_MOD_W}}")
    if ties:
        print(f"{'  (ties)':>{_COL_W + _MOD_W * 2 + 8}}  {ties}")


def _print_worst_scenarios(df: pd.DataFrame, model_name: str, n: int = 3) -> None:
    clean = df[~_is_error(df["error"])]
    if len(clean) == 0:
        print(f"  No successful scenarios for {model_name}")
        return
    worst = clean.nsmallest(n, "composite_score")
    for _, row in worst.iterrows():
        intent = str(row.get("intent", ""))[:52]
        domain = str(row.get("domain", "")).replace("_", " ")
        print(
            f"  Scenario {row['scenario_id']:>2} | {domain:<15} | "
            f"composite={row['composite_score']:.3f}  |  {intent}"
        )


def _print_failure_mode_analysis(merged: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """
    Look for patterns in the scenarios where B underperforms A by > 0.05.
    """
    losers_b = merged[merged["composite_score_delta"] > 0.05].copy()
    if losers_b.empty:
        print("  Model B had no scenarios where it underperformed A by > 0.05.")
        return

    tone_col = "tone_a" if "tone_a" in losers_b.columns else "tone"
    domain_col = "domain_a" if "domain_a" in losers_b.columns else "domain"

    tone_counts  = losers_b[tone_col].value_counts() if tone_col in losers_b.columns else pd.Series()
    domain_counts = losers_b[domain_col].value_counts() if domain_col in losers_b.columns else pd.Series()

    print(f"  B underperformed A by >0.05 in {len(losers_b)} scenario(s):")
    for _, row in losers_b.iterrows():
        intent = str(row.get("intent_a", row.get("intent", "")))[:52]
        print(
            f"    Scenario {row['scenario_id']:>2} | "
            f"composite Δ={row['composite_score_delta']:+.3f}  |  {intent}"
        )
    if not tone_counts.empty:
        print(f"  Most common tone in B's weak scenarios:   {tone_counts.index[0]}")
    if not domain_counts.empty:
        domain_clean = str(domain_counts.index[0]).replace("_", " ")
        print(f"  Most common domain in B's weak scenarios: {domain_clean}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two model evaluation CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-a", default="data/results/model_a.csv")
    parser.add_argument("--model-b", default="data/results/model_b.csv")
    parser.add_argument("--output",  default="data/results/comparison.csv")
    args = parser.parse_args()

    for p in (args.model_a, args.model_b):
        if not Path(p).exists():
            raise FileNotFoundError(f"Results file not found: {p}")

    df_a = load_results(args.model_a)
    df_b = load_results(args.model_b)

    name_a = str(df_a["model"].iloc[0]) if "model" in df_a.columns else "Model A"
    name_b = str(df_b["model"].iloc[0]) if "model" in df_b.columns else "Model B"

    merged = build_comparison(df_a, df_b)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Comparison CSV saved → {args.output}\n")

    # ---------- Per-scenario table ----------
    print("=" * 74)
    print("  Per-Scenario Composite Score Comparison")
    print("=" * 74)
    _print_scenario_table(merged, name_a, name_b)

    # ---------- Aggregate table ----------
    print("\n" + "=" * 74)
    print("  Aggregate Metric Comparison  (excluding errored scenarios)")
    print("=" * 74)
    _print_aggregate_table(df_a, df_b, name_a, name_b)

    # ---------- Worst scenarios ----------
    print("\n" + "=" * 74)
    print(f"  Worst 3 Scenarios — Model A ({_short(name_a)})")
    print("=" * 74)
    _print_worst_scenarios(df_a, name_a)

    print(f"\n  Worst 3 Scenarios — Model B ({_short(name_b)})")
    print("-" * 74)
    _print_worst_scenarios(df_b, name_b)

    # ---------- Failure mode analysis ----------
    print("\n" + "=" * 74)
    print("  Failure Mode Analysis — Where does B underperform A?")
    print("=" * 74)
    _print_failure_mode_analysis(merged, df_b)

    # ---------- Error summary ----------
    err_a = df_a[_is_error(df_a["error"])]
    err_b = df_b[_is_error(df_b["error"])]
    if len(err_a) or len(err_b):
        print("\n" + "=" * 74)
        print("  Error Summary")
        print("=" * 74)
        if len(err_a):
            print(f"  Model A errors ({len(err_a)}): scenarios "
                  f"{list(err_a['scenario_id'].astype(str))}")
        if len(err_b):
            print(f"  Model B errors ({len(err_b)}): scenarios "
                  f"{list(err_b['scenario_id'].astype(str))}")

    print()


if __name__ == "__main__":
    main()
