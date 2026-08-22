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
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
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
        for batch_idx in range(args.batches):
            x, _ = dataset.get_batch(args.batch_size, device, generator)
            with amp():
                output = model(
                    input_ids=x,
                    labels=x,
                    use_cache=False,
                    num_loops=loops,
                    return_loop_diagnostics=True,
                )
            losses.append(output.loss.float())
            update_sums += torch.stack(model.model.last_relative_updates)
            hidden_norm_sums += torch.stack(model.model.last_hidden_norms)
            cosine_sums += torch.stack(model.model.last_cosine_to_previous)
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
            "relative_update_by_loop": (update_sums / args.batches).cpu().tolist(),
            "hidden_norm_by_loop": (hidden_norm_sums / args.batches).cpu().tolist(),
            "cosine_to_previous_by_loop": (cosine_sums / args.batches).cpu().tolist(),
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
        "loop_input_dropout": cfg.loop_input_dropout,
        "loop_input_dropout_start": cfg.loop_input_dropout_start,
        "loop_noise_std": cfg.loop_noise_std,
        "loop_noise_mode": cfg.loop_noise_mode,
        "loop_noise_start_loop": cfg.loop_noise_start_loop,
        "loop_noise_warmup_steps": cfg.loop_noise_warmup_steps,
        "loop_noise_after_last_loop": cfg.loop_noise_after_last_loop,
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
