#!/usr/bin/env python3
import argparse

from transformers import AutoTokenizer

from looped_lm import LoopedQwen3ForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the best HF checkpoint and tokenizer")
    parser.add_argument("--checkpoint", default="outputs/baseline/best_hf")
    parser.add_argument("--repo-id", required=True, help="For example username/looped-qwen3-baseline")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = LoopedQwen3ForCausalLM.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model.push_to_hub(args.repo_id, private=args.private, safe_serialization=True)
    tokenizer.push_to_hub(args.repo_id, private=args.private)
    print(f"Uploaded model and tokenizer to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

