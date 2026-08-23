#!/usr/bin/env python3
"""Combine experiment 003 evaluation JSON files into summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
VARIANT_ORDER = {
    "control_r16": 0,
    "relative_s003_r16": 1,
    "spherical_s003_r16": 2,
}
FIELDNAMES = (
    "variant",
    "noise_mode",
    "noise_std",
    "noise_warmup_steps",
    "eval_loops",
    "loss",
    "perplexity",
    "last_hidden_norm",
    "last_relative_update",
    "last_cosine_to_previous",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment 003")
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
        rows.append(
            {
                "variant": variant,
                "noise_mode": payload.get("loop_noise_mode", "relative"),
                "noise_std": payload.get("loop_noise_std", 0.0),
                "noise_warmup_steps": payload.get("loop_noise_warmup_steps", 0),
                "eval_loops": result["loops"],
                "loss": result["loss"],
                "perplexity": result["perplexity"],
                "last_hidden_norm": result["hidden_norm_by_loop"][-1],
                "last_relative_update": result["relative_update_by_loop"][-1],
                "last_cosine_to_previous": result["cosine_to_previous_by_loop"][-1],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    paths = list(results_dir.glob("*_eval.json"))
    if not paths:
        raise FileNotFoundError(f"No *_eval.json files found in {results_dir}")
    rows = [row for path in paths for row in load_rows(path)]
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
