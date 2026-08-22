#!/usr/bin/env python3
"""Train and evaluate one variant of experiment 003."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CONFIGS = {
    name: EXPERIMENT_DIR / "configs" / f"{name}.yaml"
    for name in ("control_r16", "relative_s003_r16", "spherical_s003_r16")
}
DEFAULT_LOOPS = (1, 2, 4, 8, 12, 16, 20, 24, 32)
TOKEN_LIMIT = 10_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run loop-state-noise experiment")
    parser.add_argument("--variant", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--loops", type=int, nargs="+", default=DEFAULT_LOOPS)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_DIR / "results")
    return parser.parse_args()


def validate_token_budget(config: dict) -> None:
    model = config["model"]
    training = config["training"]
    tokens_per_step = (
        training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
        * model["max_position_embeddings"]
    )
    steps = training["max_train_tokens"] // tokens_per_step
    actual_tokens = steps * tokens_per_step
    if training["max_train_tokens"] > TOKEN_LIMIT or actual_tokens > TOKEN_LIMIT:
        raise ValueError(f"Token budget exceeds {TOKEN_LIMIT:,}: {actual_tokens:,}")
    print(f"Token budget: {steps} steps, {actual_tokens:,} processed tokens", flush=True)


def main() -> None:
    args = parse_args()
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be positive")
    if not args.loops or any(loop < 1 for loop in args.loops):
        raise ValueError("--loops must contain positive integers")

    config_path = CONFIGS[args.variant]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_token_budget(config)
    training = config["training"]
    output_dir = REPO_ROOT / training["output_dir"]
    best_checkpoint = output_dir / "best_hf"
    resume_checkpoint = output_dir / "last_state.pt"

    if not args.eval_only:
        command = [sys.executable, "scripts/train.py", "--config", str(config_path)]
        if args.resume and resume_checkpoint.is_file():
            command.extend(["--resume", str(resume_checkpoint)])
            print(f"Resuming from {resume_checkpoint}", flush=True)
        elif args.resume:
            print("Resume checkpoint not found; starting from scratch", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    if not best_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_checkpoint}. "
            "Run training first or remove --eval-only."
        )

    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{args.variant}_eval.json"
    command = [
        sys.executable,
        "scripts/eval.py",
        "--checkpoint",
        str(best_checkpoint),
        "--val-file",
        training["val_file"],
        "--loops",
        *(str(loop) for loop in args.loops),
        "--batches",
        str(args.eval_batches),
        "--batch-size",
        str(training["micro_batch_size"]),
        "--output",
        str(result_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
