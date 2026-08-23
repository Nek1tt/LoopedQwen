#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream, tokenize and pack a FineWeb subset")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--val-tokens", type=int, default=1_000_000)
    parser.add_argument("--dataset-config", default="sample-10BT")
    return parser.parse_args()


def write_exact_tokens(dataset, tokenizer, path: Path, target: int, progress_name: str) -> int:
    written = 0
    eos = tokenizer.eos_token_id
    progress = tqdm(
        total=target,
        unit="tok",
        unit_scale=True,
        desc=progress_name,
        dynamic_ncols=True,
    )
    with path.open("wb") as f:
        for row in dataset:
            ids = tokenizer.encode(row["text"], add_special_tokens=False)
            ids.append(eos)
            remaining = target - written
            array = np.asarray(ids[:remaining], dtype=np.uint16)
            array.tofile(f)
            written += len(array)
            progress.update(len(array))
            if written >= target:
                break
    progress.close()
    if written != target:
        raise RuntimeError(f"Dataset ended after {written:,} tokens; expected {target:,}")
    return written


def main() -> None:
    args = parse_args()
    if args.train_tokens <= 0 or args.val_tokens <= 0:
        raise ValueError("Token counts must be positive")

    print(f"[data 1/3] Loading tokenizer from {args.tokenizer}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if len(tokenizer) >= 65536:
        raise ValueError("This data format uses uint16; tokenizer vocab must be below 65,536")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[data 2/3] Connecting to FineWeb ({args.dataset_config}); "
        "the first streamed document may take 1-3 minutes...",
        flush=True,
    )
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.dataset_config,
        split="train",
        streaming=True,
    ).shuffle(seed=42, buffer_size=10_000)

    # Split the stream by documents before packing. Validation is fixed and disjoint.
    val_stream = dataset.take(20_000)
    train_stream = dataset.skip(20_000)
    print(
        f"[data 3/3] Packing {args.val_tokens:,} validation and "
        f"{args.train_tokens:,} training tokens...",
        flush=True,
    )
    val_written = write_exact_tokens(
        val_stream, tokenizer, output_dir / "val.bin", args.val_tokens, "validation"
    )
    train_written = write_exact_tokens(
        train_stream, tokenizer, output_dir / "train.bin", args.train_tokens, "training"
    )
    metadata = {
        "dataset": "HuggingFaceFW/fineweb",
        "dataset_config": args.dataset_config,
        "tokenizer": str(args.tokenizer),
        "vocab_size": len(tokenizer),
        "dtype": "uint16",
        "train_tokens": train_written,
        "val_tokens": val_written,
        "seed": 42,
        "validation_documents_reserved": 20000,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
