#!/usr/bin/env python3
"""Create a compact Experiment 007 result archive without model weights."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
VARIANTS = ("fixed_ma", "alternating_am_ma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Experiment 007 results")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiment_007_results.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = [
        EXPERIMENT_DIR / "README.md",
        *sorted((EXPERIMENT_DIR / "configs").glob("*.yaml")),
        *sorted(path for path in (EXPERIMENT_DIR / "results").rglob("*") if path.is_file()),
        REPO_ROOT
        / "experiments/005_anytime_depth/results/multiseed/"
        / "random_r8_24_intermediate_seed42_dense_eval.json",
    ]
    for variant in VARIANTS:
        run_dir = REPO_ROOT / "outputs" / "experiments" / "007" / variant
        candidates.extend(
            (
                run_dir / "metrics.jsonl",
                run_dir / "run_config.json",
                run_dir / "best_hf" / "training_summary.json",
                run_dir / "best_hf" / "config.json",
            )
        )
    missing = [path for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Experiment 007 result files: " + ", ".join(map(str, missing))
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "variants": VARIANTS,
        "checkpoints_included": False,
        "notebook_included": False,
    }
    written = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        for path in candidates:
            if path not in written:
                archive.write(path, path.relative_to(REPO_ROOT))
                written.add(path)
    print(f"Wrote {output} ({output.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()
