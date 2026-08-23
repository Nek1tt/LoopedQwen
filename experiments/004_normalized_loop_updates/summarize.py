#!/usr/bin/env python3
"""Combine Experiment 004 evaluation JSON files into summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
VARIANT_ORDER = {
    "control_r16": 0,
    "projected_fixed_a025_r16": 1,
    "projected_learned_a025_r16": 2,
}
FIELDNAMES = (
    "variant",
    "metric_source",
    "loop_update_mode",
    "loop_update_alpha_config",
    "loop_update_start_loop",
    "learned_schedule_slope",
    "eval_loops",
    "loss",
    "perplexity",
    "last_loop_update_alpha",
    "last_hidden_norm",
    "last_relative_update",
    "last_cosine_to_previous",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Experiment 004")
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_DIR / "results")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"No evaluation results in {path}")
    variant = path.name.removesuffix("_eval.json")
    rows = []
    for result in results:
        alphas = result.get("loop_update_alpha_by_loop", [])
        rows.append(
            {
                "variant": variant,
                "metric_source": "evaluation",
                "loop_update_mode": payload.get("loop_update_mode", "full"),
                "loop_update_alpha_config": payload.get("loop_update_alpha_config"),
                "loop_update_start_loop": payload.get("loop_update_start_loop"),
                "learned_schedule_slope": payload.get("loop_update_schedule_slope_learned"),
                "eval_loops": result["loops"],
                "loss": result["loss"],
                "perplexity": result["perplexity"],
                "last_loop_update_alpha": alphas[-1] if alphas else None,
                "last_hidden_norm": result["hidden_norm_by_loop"][-1],
                "last_relative_update": result["relative_update_by_loop"][-1],
                "last_cosine_to_previous": result["cosine_to_previous_by_loop"][-1],
            }
        )
    return rows


def load_training_validation(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "variant": payload["variant"],
        "metric_source": "training_validation",
        "loop_update_mode": payload["loop_update_mode"],
        "loop_update_alpha_config": payload["loop_update_alpha_config"],
        "loop_update_start_loop": payload["loop_update_start_loop"],
        "learned_schedule_slope": payload.get("learned_schedule_slope"),
        "eval_loops": payload["loops"],
        "loss": payload["loss"],
        "perplexity": payload["perplexity"],
        "last_loop_update_alpha": payload.get("last_loop_update_alpha"),
        "last_hidden_norm": None,
        "last_relative_update": None,
        "last_cosine_to_previous": None,
    }


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    paths = list(results_dir.glob("*_eval.json"))
    if not paths:
        raise FileNotFoundError(f"No *_eval.json files found in {results_dir}")
    rows = [row for path in paths for row in load_rows(path)]
    rows.extend(
        load_training_validation(path)
        for path in results_dir.glob("*_train_validation.json")
    )
    rows.sort(
        key=lambda row: (
            VARIANT_ORDER.get(str(row["variant"]), len(VARIANT_ORDER)),
            str(row["variant"]),
            int(row["eval_loops"]),
        )
    )
    output = (args.output or results_dir / "summary.csv").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
