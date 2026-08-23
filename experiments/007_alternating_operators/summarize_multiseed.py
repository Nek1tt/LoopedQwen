#!/usr/bin/env python3
"""Rebuild Experiment 007 multiseed tables from the committed evaluation JSON."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results" / "multiseed"
CONTROL_DIR = REPO_ROOT / "experiments/005_anytime_depth/results/multiseed"
SEEDS = (42, 43, 44)
LOOPS = tuple(range(4, 33))


def load_curve(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    curve = {int(row["loops"]): float(row["perplexity"]) for row in payload["results"]}
    if tuple(curve) != LOOPS:
        raise ValueError(f"Expected depths 4..32 in {path}")
    return curve


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(variant: str, seed: int | str, curve: dict[int, float]) -> dict:
    best_depth = min(curve, key=curve.get)
    best_ppl = curve[best_depth]
    return {
        "variant": variant,
        "seed": seed,
        "best_depth": best_depth,
        "best_perplexity": best_ppl,
        "mean_ppl_8_24": statistics.mean(curve[loop] for loop in range(8, 25)),
        "ppl_at_8": curve[8],
        "ppl_at_16": curve[16],
        "ppl_at_24": curve[24],
        "ppl_at_32": curve[32],
        "regret_at_32_percent": 100.0 * (curve[32] / best_ppl - 1.0),
        "adjacent_ppl_increases": sum(curve[d + 1] > curve[d] for d in range(4, 32)),
    }


def main() -> None:
    curves: dict[tuple[str, int], dict[int, float]] = {}
    for seed in SEEDS:
        curves[("fixed_am_control", seed)] = load_curve(
            CONTROL_DIR / f"random_r8_24_intermediate_seed{seed}_dense_eval.json"
        )
        curves[("fixed_ma", seed)] = load_curve(
            RESULTS_DIR / f"fixed_ma_seed{seed}_dense_eval.json"
        )

    by_seed_rows = []
    paired_rows = []
    metric_rows = []
    for seed in SEEDS:
        control = curves[("fixed_am_control", seed)]
        method = curves[("fixed_ma", seed)]
        metric_rows.extend(
            (metric_row("fixed_am_control", seed, control), metric_row("fixed_ma", seed, method))
        )
        for loop in LOOPS:
            by_seed_rows.extend(
                {
                    "variant": variant,
                    "seed": seed,
                    "eval_loops": loop,
                    "perplexity": curve[loop],
                }
                for variant, curve in (("fixed_am_control", control), ("fixed_ma", method))
            )
            paired_rows.append(
                {
                    "seed": seed,
                    "eval_loops": loop,
                    "fixed_am_perplexity": control[loop],
                    "fixed_ma_perplexity": method[loop],
                    "absolute_delta": method[loop] - control[loop],
                    "relative_delta_percent": 100.0 * (method[loop] / control[loop] - 1.0),
                }
            )

    aggregate_rows = []
    aggregate_curves = {}
    for variant in ("fixed_am_control", "fixed_ma"):
        aggregate_curves[variant] = {}
        for loop in LOOPS:
            values = [curves[(variant, seed)][loop] for seed in SEEDS]
            aggregate_curves[variant][loop] = statistics.mean(values)
            aggregate_rows.append(
                {
                    "variant": variant,
                    "eval_loops": loop,
                    "seeds": len(values),
                    "mean_perplexity": statistics.mean(values),
                    "std_perplexity": statistics.stdev(values),
                    "min_perplexity": min(values),
                    "max_perplexity": max(values),
                }
            )
        metric_rows.append(metric_row(variant, "mean", aggregate_curves[variant]))

    paired_seed_rows = []
    for seed in SEEDS:
        deltas = [
            100.0
            * (curves[("fixed_ma", seed)][loop] / curves[("fixed_am_control", seed)][loop] - 1.0)
            for loop in range(8, 25)
        ]
        paired_seed_rows.append(
            {
                "seed": seed,
                "mean_relative_delta_percent_8_24": statistics.mean(deltas),
                "fixed_ma_better_depths_4_32": sum(
                    curves[("fixed_ma", seed)][loop]
                    < curves[("fixed_am_control", seed)][loop]
                    for loop in LOOPS
                ),
            }
        )

    matrix = {
        ("fixed_am", "attention_mlp"): curves[("fixed_am_control", 42)],
        ("fixed_ma", "mlp_attention"): curves[("fixed_ma", 42)],
        ("fixed_ma", "attention_mlp"): load_curve(
            RESULTS_DIR / "fixed_ma_seed42_force_am_eval.json"
        ),
    }
    matrix_rows = [
        {
            "trained_schedule": trained,
            "evaluated_schedule": evaluated,
            "eval_loops": loop,
            "perplexity": curve[loop],
        }
        for (trained, evaluated), curve in matrix.items()
        for loop in LOOPS
    ]

    write_csv(RESULTS_DIR / "summary_by_seed.csv", by_seed_rows)
    write_csv(RESULTS_DIR / "summary_aggregate.csv", aggregate_rows)
    write_csv(RESULTS_DIR / "paired_comparison.csv", paired_rows)
    write_csv(RESULTS_DIR / "paired_seed_summary.csv", paired_seed_rows)
    write_csv(RESULTS_DIR / "metrics.csv", metric_rows)
    write_csv(RESULTS_DIR / "cross_order_matrix.csv", matrix_rows)
    print("Rebuilt Experiment 007 multiseed tables")


if __name__ == "__main__":
    main()
