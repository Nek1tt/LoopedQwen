#!/usr/bin/env python3
"""Build performance, intervention, and operator-diagnostic tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"
CONTROL = (
    REPO_ROOT
    / "experiments/005_anytime_depth/results/multiseed/"
    / "random_r8_24_intermediate_seed42_dense_eval.json"
)
EVALUATIONS = {
    "fixed_am_control": CONTROL,
    "fixed_ma": RESULTS_DIR / "fixed_ma_eval.json",
    "alternating_am_ma": RESULTS_DIR / "alternating_am_ma_eval.json",
    "alternating_force_am": RESULTS_DIR / "alternating_am_ma_force_am_eval.json",
    "alternating_force_ma": RESULTS_DIR / "alternating_am_ma_force_ma_eval.json",
    "alternating_reverse_parity": (
        RESULTS_DIR / "alternating_am_ma_reverse_parity_eval.json"
    ),
}


def read_results(path: Path) -> tuple[dict, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"No results in {path}")
    return payload, results


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    payloads = {}
    result_sets = {}
    summary_rows = []
    for variant, path in EVALUATIONS.items():
        payload, results = read_results(path)
        payloads[variant] = payload
        result_sets[variant] = {int(row["loops"]): row for row in results}
        for row in results:
            summary_rows.append(
                {
                    "variant": variant,
                    "eval_loops": int(row["loops"]),
                    "loss": float(row["loss"]),
                    "perplexity": float(row["perplexity"]),
                }
            )
    write_csv(RESULTS_DIR / "summary.csv", summary_rows)

    metric_rows = []
    for variant, by_depth in result_sets.items():
        depths = sorted(by_depth)
        best_depth = min(depths, key=lambda depth: by_depth[depth]["perplexity"])
        best_ppl = float(by_depth[best_depth]["perplexity"])
        increases = sum(
            by_depth[right]["perplexity"] > by_depth[left]["perplexity"]
            for left, right in zip(depths, depths[1:])
        )
        range_depths = [depth for depth in depths if 8 <= depth <= 24]
        metric_rows.append(
            {
                "variant": variant,
                "best_depth": best_depth,
                "best_perplexity": best_ppl,
                "mean_ppl_8_24": sum(
                    by_depth[depth]["perplexity"] for depth in range_depths
                )
                / len(range_depths),
                "ppl_at_8": by_depth[8]["perplexity"],
                "ppl_at_16": by_depth[16]["perplexity"],
                "ppl_at_24": by_depth[24]["perplexity"],
                "ppl_at_32": by_depth[32]["perplexity"],
                "regret_at_32_percent": 100.0
                * (by_depth[32]["perplexity"] / best_ppl - 1.0),
                "adjacent_ppl_increases": increases,
            }
        )
    write_csv(RESULTS_DIR / "metrics.csv", metric_rows)

    native = result_sets["alternating_am_ma"]
    intervention_rows = []
    for variant in (
        "alternating_force_am",
        "alternating_force_ma",
        "alternating_reverse_parity",
    ):
        for depth, row in result_sets[variant].items():
            native_ppl = float(native[depth]["perplexity"])
            intervention_ppl = float(row["perplexity"])
            intervention_rows.append(
                {
                    "intervention": variant,
                    "eval_loops": depth,
                    "native_perplexity": native_ppl,
                    "intervention_perplexity": intervention_ppl,
                    "delta_ppl": intervention_ppl - native_ppl,
                    "delta_percent": 100.0 * (intervention_ppl / native_ppl - 1.0),
                }
            )
    write_csv(RESULTS_DIR / "intervention_effects.csv", intervention_rows)

    diagnostic_rows = []
    for variant in ("fixed_ma", "alternating_am_ma"):
        row = result_sets[variant][32]
        fields = (
            row.get("attention_relative_update_by_loop"),
            row.get("mlp_relative_update_by_loop"),
            row.get("operator_defect_by_loop"),
        )
        if any(values is None for values in fields):
            raise ValueError(f"Missing operator diagnostics for {variant}")
        for index in range(32):
            diagnostic_rows.append(
                {
                    "variant": variant,
                    "loop": index + 1,
                    "operator_order": row["operator_order_by_loop"][index],
                    "attention_relative_update": fields[0][index],
                    "mlp_relative_update": fields[1][index],
                    "operator_defect": fields[2][index],
                    "full_relative_update": row["relative_update_by_loop"][index],
                    "update_cosine_to_previous_update": row[
                        "update_cosine_to_previous_update_by_loop"
                    ][index],
                    "directional_diversity": row["directional_diversity_by_loop"][index],
                }
            )
    write_csv(RESULTS_DIR / "operator_diagnostics.csv", diagnostic_rows)

    print(f"Wrote {len(summary_rows)} rows to results/summary.csv")
    print(f"Wrote {len(metric_rows)} rows to results/metrics.csv")
    print(f"Wrote {len(intervention_rows)} rows to results/intervention_effects.csv")
    print(f"Wrote {len(diagnostic_rows)} rows to results/operator_diagnostics.csv")


if __name__ == "__main__":
    main()
