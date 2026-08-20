#!/usr/bin/env python3
import torch

from looped_lm import LoopedQwen3Config, LoopedQwen3ForCausalLM


def main() -> None:
    cfg = LoopedQwen3Config(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=384,
        num_hidden_layers=2,
        num_loops=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        use_cache=False,
    )
    model = LoopedQwen3ForCausalLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 32))
    output = model(
        input_ids=tokens,
        labels=tokens,
        use_cache=False,
        return_loop_diagnostics=True,
    )
    output.loss.backward()
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"loss: {output.loss.item():.4f}")
    print("relative updates:", [round(x.item(), 4) for x in model.model.last_relative_updates])
    print("sanity check passed")


if __name__ == "__main__":
    main()
