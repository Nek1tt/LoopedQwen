#!/usr/bin/env python3
import argparse
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast


EOS_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on FineWeb")
    parser.add_argument("--output-dir", default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--documents", type=int, default=100000)
    parser.add_argument("--dataset-config", default="sample-10BT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[tokenizer 1/3] Connecting to FineWeb ({args.dataset_config}); "
        "the first streamed document may take 1-3 minutes...",
        flush=True,
    )
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=args.dataset_config,
        split="train",
        streaming=True,
    )

    print(
        f"[tokenizer 2/3] Reading {args.documents:,} documents. "
        "The progress bar will gain a measured ETA after the first documents arrive.",
        flush=True,
    )
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    documents = tqdm(
        islice(dataset, args.documents),
        total=args.documents,
        desc="FineWeb tokenizer documents",
        unit="doc",
        dynamic_ncols=True,
    )
    texts = (row["text"] for row in documents)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=args.documents)

    print("[tokenizer 3/3] Finalizing and saving vocabulary...", flush=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=EOS_TOKEN,
        bos_token=EOS_TOKEN,
        pad_token=EOS_TOKEN,
    )
    fast.model_max_length = 1_000_000
    fast.save_pretrained(output_dir)
    print(f"Saved tokenizer with {len(fast):,} tokens to {output_dir}")


if __name__ == "__main__":
    main()
