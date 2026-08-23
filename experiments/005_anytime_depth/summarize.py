#!/usr/bin/env python3
"""Combine Experiment 005 evaluation JSON files into summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
VARIANT_ORDER = {
    "fixed_r16_final": 0,
    "random_r8_24_final": 1,
    "random_r8_24_intermediate": 2,
}
FIELDNAMES = (
    "variant",
    "eval_loops",
    "loss",
    "perplexity",
    "last_loop_update_alpha",
    "last_hidden_norm",
    "last_relative_update",
    "last_cosine_to_previous",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Experiment 005")
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
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
