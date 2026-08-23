#!/usr/bin/env python3
"""Compare Experiment 006 with the seed-42 Experiment 005 control."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"
SOURCES = {
    "uniform_intermediate_control": REPO_ROOT
    / "experiments/005_anytime_depth/results/multiseed/"
    / "random_r8_24_intermediate_seed42_dense_eval.json",
    "hard_token_g05": RESULTS_DIR / "hard_token_g05_eval.json",
}


def load_rows(variant: str, path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {variant} evaluation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for result in payload["results"]:
        rows.append(
            {
                "variant": variant,
                "eval_loops": int(result["loops"]),
                "loss": float(result["loss"]),
                "perplexity": float(result["perplexity"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = [row for variant, path in SOURCES.items() for row in load_rows(variant, path)]
    write_csv(RESULTS_DIR / "summary.csv", rows)

    metrics = []
    for variant in SOURCES:
        variant_rows = [row for row in rows if row["variant"] == variant]
        variant_rows.sort(key=lambda row: row["eval_loops"])
        best = min(variant_rows, key=lambda row: row["perplexity"])
        by_depth = {row["eval_loops"]: row["perplexity"] for row in variant_rows}
        violations = sum(
            right["perplexity"] > left["perplexity"]
            for left, right in zip(variant_rows, variant_rows[1:])
        )
        metrics.append(
            {
                "variant": variant,
                "best_depth": best["eval_loops"],
                "best_perplexity": best["perplexity"],
                "ppl_at_8": by_depth.get(8),
                "ppl_at_11": by_depth.get(11),
                "ppl_at_16": by_depth.get(16),
                "ppl_at_24": by_depth.get(24),
                "ppl_at_32": by_depth.get(32),
                "regret_at_32_percent": 100.0
                * (by_depth[32] / best["perplexity"] - 1.0),
                "adjacent_ppl_increases": violations,
            }
        )
    write_csv(RESULTS_DIR / "metrics.csv", metrics)
    print(f"Wrote {RESULTS_DIR / 'summary.csv'} ({len(rows)} rows)")
    print(f"Wrote {RESULTS_DIR / 'metrics.csv'}")
    for metric in metrics:
        print(metric)


if __name__ == "__main__":
    main()
