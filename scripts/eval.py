#!/usr/bin/env python3
import argparse
import json
import math
from contextlib import nullcontext

import torch

from looped_lm import LoopedQwen3ForCausalLM
from looped_lm.data import BinaryTokenDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint at several loop counts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-file", default="data/val.bin")
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
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
    for loops in args.loops:
        generator = torch.Generator().manual_seed(1234)
        losses = []
        update_sums = torch.zeros(loops, device=device)
        for _ in range(args.batches):
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
        loss = torch.stack(losses).mean().item()
        result = {
            "loops": loops,
            "loss": loss,
            "perplexity": math.exp(min(loss, 20)),
            "relative_update_by_loop": (update_sums / args.batches).cpu().tolist(),
        }
        results.append(result)
        print(json.dumps(result))

    print(json.dumps({"checkpoint": args.checkpoint, "results": results}, indent=2))


if __name__ == "__main__":
    main()
