#!/usr/bin/env python3
"""Create a compact Experiment 005 result bundle without checkpoint weights."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
VARIANTS = (
    "fixed_r16_final",
    "random_r8_24_final",
    "random_r8_24_intermediate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Experiment 005 results")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "experiment_005_results.zip")
    parser.add_argument("--include-checkpoints", action="store_true")
    return parser.parse_args()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": git_revision(),
        "variants": VARIANTS,
        "checkpoints_included": args.include_checkpoints,
    }
    candidates = [
        EXPERIMENT_DIR / "README.md",
        *sorted((EXPERIMENT_DIR / "configs").glob("*.yaml")),
        *sorted(path for path in (EXPERIMENT_DIR / "results").rglob("*") if path.is_file()),
    ]
    for variant in VARIANTS:
        run_dir = REPO_ROOT / "outputs" / "experiments" / "005" / variant
        candidates.extend(
            run_dir / name
            for name in (
                "metrics.jsonl",
                "run_config.json",
                "best_hf/training_summary.json",
                "best_hf/config.json",
            )
        )
        if args.include_checkpoints:
            candidates.extend((run_dir / "best_hf").glob("*"))
    runtime_info = REPO_ROOT / "runtime_info.txt"
    if runtime_info.is_file():
        candidates.append(runtime_info)

    written = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        for path in candidates:
            if path.is_file() and path not in written:
                archive.write(path, path.relative_to(REPO_ROOT))
                written.add(path)
    print(f"Wrote {output} ({output.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    main()
