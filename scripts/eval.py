#!/usr/bin/env python3
import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm.auto import tqdm

from looped_lm import LoopedQwen3ForCausalLM
from looped_lm.data import BinaryTokenDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint at several loop counts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-file", default="data/val.bin")
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--operator-schedule",
        choices=(
            "attention_mlp",
            "mlp_attention",
            "alternating_attention_mlp",
            "alternating_mlp_attention",
        ),
        default=None,
        help="Evaluation-only override for causal operator-order interventions",
    )
    parser.add_argument(
        "--operator-diagnostic-batches",
        type=int,
        default=0,
        help="Compute the extra AM-vs-MA defect on this many batches per depth",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.operator_diagnostic_batches < 0:
        raise ValueError("--operator-diagnostic-batches must be non-negative")
    if args.operator_diagnostic_batches > args.batches:
        raise ValueError("operator diagnostic batches cannot exceed evaluation batches")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LoopedQwen3ForCausalLM.from_pretrained(args.checkpoint).to(device)
    cfg = model.config
    model.eval()
    dataset = BinaryTokenDataset(args.val_file, cfg.max_position_embeddings)
    amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" else nullcontext

    results = []
    evaluation_started = time.time()
    total_work = args.batches * sum(args.loops)
    print(
        f"evaluation plan: depths={args.loops}; batches/depth={args.batches}; "
        f"batch_size={args.batch_size}; weighted_work={total_work:,} loop-batches; device={device}",
        f"operator_schedule={args.operator_schedule or cfg.loop_operator_schedule}; "
        f"operator_diagnostic_batches={args.operator_diagnostic_batches}",
        flush=True,
    )
    progress = tqdm(
        total=total_work,
        desc="evaluation",
        unit="loop-batch",
        dynamic_ncols=True,
    )
    for loops in args.loops:
        depth_started = time.time()
        generator = torch.Generator().manual_seed(1234)
        losses = []
        update_sums = torch.zeros(loops, device=device)
        hidden_norm_sums = torch.zeros(loops, device=device)
        cosine_sums = torch.zeros(loops, device=device)
        update_cosine_sums = torch.zeros(loops, device=device)
        diversity_sums = torch.zeros(loops, device=device)
        attention_update_sums = torch.zeros(loops, device=device)
        mlp_update_sums = torch.zeros(loops, device=device)
        operator_defect_sums = torch.zeros(loops, device=device)
        operator_diagnostic_count = 0
        for batch_idx in range(args.batches):
            x, _ = dataset.get_batch(args.batch_size, device, generator)
            collect_operator_diagnostics = batch_idx < args.operator_diagnostic_batches
            with amp():
                output = model(
                    input_ids=x,
                    labels=x,
                    use_cache=False,
                    num_loops=loops,
                    loop_operator_schedule=args.operator_schedule,
                    return_loop_diagnostics=True,
                    return_operator_diagnostics=collect_operator_diagnostics,
                )
            losses.append(output.loss.float())
            update_sums += torch.stack(model.model.last_relative_updates)
            hidden_norm_sums += torch.stack(model.model.last_hidden_norms)
            cosine_sums += torch.stack(model.model.last_cosine_to_previous)
            update_cosine_sums += torch.stack(
                model.model.last_update_cosine_to_previous_update
            )
            diversity_sums += torch.stack(model.model.last_directional_diversity)
            if collect_operator_diagnostics:
                attention_update_sums += torch.stack(
                    model.model.last_attention_relative_updates
                )
                mlp_update_sums += torch.stack(model.model.last_mlp_relative_updates)
                operator_defect_sums += torch.stack(model.model.last_operator_defects)
                operator_diagnostic_count += 1
            running_loss = torch.stack(losses).mean().item()
            progress.update(loops)
            progress.set_postfix(
                loops=loops,
                batch=f"{batch_idx + 1}/{args.batches}",
                loss=f"{running_loss:.4f}",
            )
        loss = torch.stack(losses).mean().item()
        depth_elapsed = time.time() - depth_started
        result = {
            "loops": loops,
            "loss": loss,
            "perplexity": math.exp(min(loss, 20)),
            "elapsed_seconds": depth_elapsed,
            "batches_per_second": args.batches / max(depth_elapsed, 1e-8),
            "loop_update_alpha_by_loop": [
                value.cpu().item() for value in model.model.last_loop_update_alphas
            ],
            "relative_update_by_loop": (update_sums / args.batches).cpu().tolist(),
            "hidden_norm_by_loop": (hidden_norm_sums / args.batches).cpu().tolist(),
            "cosine_to_previous_by_loop": (cosine_sums / args.batches).cpu().tolist(),
            "update_cosine_to_previous_update_by_loop": (
                update_cosine_sums / args.batches
            ).cpu().tolist(),
            "directional_diversity_by_loop": (
                diversity_sums / args.batches
            ).cpu().tolist(),
            "operator_order_by_loop": list(model.model.last_operator_orders),
            "operator_diagnostic_batches": operator_diagnostic_count,
            "attention_relative_update_by_loop": (
                (attention_update_sums / operator_diagnostic_count).cpu().tolist()
                if operator_diagnostic_count
                else None
            ),
            "mlp_relative_update_by_loop": (
                (mlp_update_sums / operator_diagnostic_count).cpu().tolist()
                if operator_diagnostic_count
                else None
            ),
            "operator_defect_by_loop": (
                (operator_defect_sums / operator_diagnostic_count).cpu().tolist()
                if operator_diagnostic_count
                else None
            ),
        }
        results.append(result)
        tqdm.write(
            f"depth={loops}: loss={loss:.4f}; ppl={result['perplexity']:.2f}; "
            f"time={depth_elapsed:.1f}s; batches/s={result['batches_per_second']:.2f}"
        )

    progress.close()
    evaluation_elapsed = time.time() - evaluation_started

    alpha = model.model.current_loop_update_alpha().detach().float().item()
    payload = {
        "checkpoint": args.checkpoint,
        "loop_update_mode": cfg.loop_update_mode,
        "loop_update_alpha": alpha,
        "loop_update_alpha_config": cfg.loop_update_alpha,
        "loop_update_start_loop": cfg.loop_update_start_loop,
        "loop_update_schedule_slope_config": cfg.loop_update_schedule_slope,
        "loop_update_schedule_slope_learned": (
            model.model.loop_update_log_slope.detach().float().item()
            if hasattr(model.model, "loop_update_log_slope")
            else None
        ),
        "loop_input_dropout": cfg.loop_input_dropout,
        "loop_input_dropout_start": cfg.loop_input_dropout_start,
        "loop_noise_std": cfg.loop_noise_std,
        "loop_noise_mode": cfg.loop_noise_mode,
        "loop_noise_start_loop": cfg.loop_noise_start_loop,
        "loop_noise_warmup_steps": cfg.loop_noise_warmup_steps,
        "loop_noise_after_last_loop": cfg.loop_noise_after_last_loop,
        "loop_operator_schedule_config": cfg.loop_operator_schedule,
        "loop_operator_schedule_evaluated": (
            args.operator_schedule or cfg.loop_operator_schedule
        ),
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote evaluation to {args.output}")
    print(f"evaluation finished in {evaluation_elapsed / 60:.1f} min", flush=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
