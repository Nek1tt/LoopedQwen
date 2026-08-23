#!/usr/bin/env python3
"""Train, densely evaluate, and intervene on Experiment 007 variants."""

from __future__ import annotations

import argparse
import gc
import os
import runpy
import subprocess
import sys
from pathlib import Path

import torch
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
VARIANTS = ("fixed_ma", "alternating_am_ma")
CONFIGS = {name: EXPERIMENT_DIR / "configs" / f"{name}.yaml" for name in VARIANTS}
DEFAULT_LOOPS = tuple(range(4, 33))
TOKEN_LIMIT = 10_000_000
INTERVENTIONS = {
    "force_am": "attention_mlp",
    "force_ma": "mlp_attention",
    "reverse_parity": "alternating_mlp_attention",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alternating-operator experiment")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--operator-diagnostic-batches", type=int, default=5)
    parser.add_argument("--loops", type=int, nargs="+", default=DEFAULT_LOOPS)
    parser.add_argument("--results-dir", type=Path, default=EXPERIMENT_DIR / "results")
    parser.add_argument("--skip-interventions", action="store_true")
    parser.add_argument("--skip-existing-eval", action="store_true")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run in the active Python kernel so tqdm remains visible in Colab",
    )
    return parser.parse_args()


def validate_plan(config: dict) -> None:
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
    if training.get("depth_sampling") != "uniform":
        raise ValueError("Experiment 007 requires uniform random depth")
    if not training.get("intermediate_lm_losses"):
        raise ValueError("Experiment 007 requires intermediate LM losses")
    print(
        f"Plan: {steps} optimizer steps (indices 0..{steps - 1}), "
        f"{actual_tokens:,} processed tokens; depth=uniform "
        f"{training['min_train_loops']}–{training['max_train_loops']}; "
        f"operator_schedule={model['loop_operator_schedule']}",
        flush=True,
    )


def run_command(arguments: list[str], in_process: bool) -> None:
    if not in_process:
        subprocess.run([sys.executable, *arguments], cwd=REPO_ROOT, check=True)
        return

    script = REPO_ROOT / arguments[0]
    previous_argv = sys.argv[:]
    previous_sys_path = sys.path[:]
    previous_cwd = Path.cwd()
    print(f"\n▶ {' '.join(arguments)}", flush=True)
    try:
        os.chdir(REPO_ROOT)
        source_dir = str(REPO_ROOT / "src")
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        sys.argv = [str(script), *arguments[1:]]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
        os.chdir(previous_cwd)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate(
    args: argparse.Namespace,
    checkpoint: Path,
    val_file: str,
    batch_size: int,
    output: Path,
    operator_schedule: str | None = None,
    diagnostic_batches: int = 0,
) -> None:
    if args.skip_existing_eval and output.is_file():
        print(f"✓ Existing evaluation retained: {output}", flush=True)
        return
    command = [
        "scripts/eval.py",
        "--checkpoint",
        str(checkpoint),
        "--val-file",
        val_file,
        "--loops",
        *(str(loop) for loop in args.loops),
        "--batches",
        str(args.eval_batches),
        "--batch-size",
        str(batch_size),
        "--operator-diagnostic-batches",
        str(diagnostic_batches),
        "--output",
        str(output),
    ]
    if operator_schedule is not None:
        command.extend(["--operator-schedule", operator_schedule])
    run_command(command, args.in_process)
    print(f"✓ Result saved to {output}", flush=True)


def main() -> None:
    args = parse_args()
    if args.eval_batches < 1:
        raise ValueError("--eval-batches must be positive")
    if not 0 <= args.operator_diagnostic_batches <= args.eval_batches:
        raise ValueError("Require 0 <= diagnostic batches <= eval batches")
    if not args.loops or any(loop < 1 for loop in args.loops):
        raise ValueError("--loops must contain positive integers")

    config_path = CONFIGS[args.variant]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_plan(config)
    training = config["training"]
    output_dir = REPO_ROOT / training["output_dir"]
    best_checkpoint = output_dir / "best_hf"
    resume_checkpoint = output_dir / "last_state.pt"

    if not args.eval_only:
        command = ["scripts/train.py", "--config", str(config_path)]
        if args.resume and resume_checkpoint.is_file():
            command.extend(["--resume", str(resume_checkpoint)])
            print(f"Resuming from {resume_checkpoint}", flush=True)
        elif args.resume:
            print("Resume checkpoint not found; starting from scratch", flush=True)
        run_command(command, args.in_process)

    if not best_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Best checkpoint not found: {best_checkpoint}. "
            "Run training first or remove --eval-only."
        )

    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluate(
        args,
        best_checkpoint,
        training["val_file"],
        training["micro_batch_size"],
        results_dir / f"{args.variant}_eval.json",
        diagnostic_batches=args.operator_diagnostic_batches,
    )

    if args.variant == "alternating_am_ma" and not args.skip_interventions:
        for label, schedule in INTERVENTIONS.items():
            evaluate(
                args,
                best_checkpoint,
                training["val_file"],
                training["micro_batch_size"],
                results_dir / f"{args.variant}_{label}_eval.json",
                operator_schedule=schedule,
            )


if __name__ == "__main__":
    main()
