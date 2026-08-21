#!/usr/bin/env python3
"""Combine experiment evaluation JSON files into summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
VARIANT_ORDER = {
    "baseline_r8": 0,
    "fixed_05_r8": 1,
    "learned_shared_r8": 2,
}
FIELDNAMES = (
    "variant",
    "train_mode",
    "learned_alpha",
    "eval_loops",
    "loss",
    "perplexity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path; defaults to <results-dir>/summary.csv",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variant = path.name.removesuffix("_eval.json")
    mode = payload["loop_update_mode"]
    alpha = payload["loop_update_alpha"]
    results = payload["results"]
    if not isinstance(results, list) or not results:
        raise ValueError(f"No evaluation results in {path}")

    rows = []
    for result in results:
        rows.append(
            {
                "variant": variant,
                "train_mode": mode,
                "learned_alpha": alpha,
                "eval_loops": result["loops"],
                "loss": result["loss"],
                "perplexity": result["perplexity"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    paths = list(results_dir.glob("*_eval.json"))
    if not paths:
        raise FileNotFoundError(f"No *_eval.json files found in {results_dir}")

    rows = []
    for path in paths:
        rows.extend(load_rows(path))
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
