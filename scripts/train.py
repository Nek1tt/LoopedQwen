#!/usr/bin/env python3
import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from looped_lm import LoopedQwen3Config, LoopedQwen3ForCausalLM
from looped_lm.data import BinaryTokenDataset
from looped_lm.utils import autocast_dtype, cosine_lr, load_yaml, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the looped Transformer")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--resume", default=None, help="Path to last_state.pt")
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model,
    dataset,
    batches,
    batch_size,
    device,
    amp_context,
    seed=1234,
    description="validation",
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    progress = tqdm(
        range(batches),
        desc=description,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for _ in progress:
        x, _ = dataset.get_batch(batch_size, device, generator)
        with amp_context():
            # Hugging Face's causal-LM loss performs the one-token shift.
            loss = model(input_ids=x, labels=x, use_cache=False).loss
        losses.append(loss.float())
        progress.set_postfix(loss=f"{torch.stack(losses).mean().item():.4f}")
    model.train()
    return torch.stack(losses).mean().item()


def save_resume_state(path, model, optimizer, step, tokens_seen, best_val, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val,
            "run_config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = config["training"]
    model_cfg = LoopedQwen3Config(**config["model"])
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = LoopedQwen3ForCausalLM(model_cfg).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"device={device}; parameters={parameter_count:,}; loops={model_cfg.num_loops}")
    if parameter_count > 10_000_000:
        raise ValueError(f"Model has {parameter_count:,} parameters, above the 10M task limit")

    tokenizer = AutoTokenizer.from_pretrained(train_cfg["tokenizer_dir"])
    if len(tokenizer) != model_cfg.vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer):,} tokens but model vocab_size={model_cfg.vocab_size:,}"
        )
    train_data = BinaryTokenDataset(train_cfg["train_file"], model_cfg.max_position_embeddings)
    val_data = BinaryTokenDataset(train_cfg["val_file"], model_cfg.max_position_embeddings)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=train_cfg["weight_decay"],
        fused=device.type == "cuda",
    )

    dtype = autocast_dtype(train_cfg["dtype"])
    if device.type == "cuda" and dtype != torch.float32:
        amp_context = lambda: torch.autocast(device_type="cuda", dtype=dtype)
    else:
        amp_context = nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)

    tokens_per_micro = train_cfg["micro_batch_size"] * model_cfg.max_position_embeddings
    tokens_per_step = tokens_per_micro * train_cfg["gradient_accumulation_steps"]
    # Floor division is intentional: never exceed the task's processed-token budget.
    total_steps = train_cfg["max_train_tokens"] // tokens_per_step
    if total_steps < 1:
        raise ValueError("max_train_tokens is smaller than one optimizer step")
    start_step, tokens_seen, best_val = 0, 0, float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"] + 1
        tokens_seen = checkpoint["tokens_seen"]
        best_val = checkpoint["best_val_loss"]
        print(f"resumed at step={start_step}; tokens_seen={tokens_seen:,}")

    if train_cfg.get("compile", False):
        model = torch.compile(model)

    log_path = output_dir / "metrics.jsonl"
    model.train()
    started = time.time()
    session_start_tokens = tokens_seen
    print(
        f"training plan: steps={total_steps:,}; start={start_step:,}; "
        f"tokens/step={tokens_per_step:,}; total_tokens={total_steps * tokens_per_step:,}; "
        f"effective_depth={model_cfg.num_hidden_layers * model_cfg.num_loops}; "
        f"eval_every={train_cfg['eval_interval']} steps x {train_cfg['eval_batches']} batches",
        flush=True,
    )
    progress = tqdm(
        range(start_step, total_steps),
        total=total_steps,
        initial=start_step,
        desc=f"train R={model_cfg.num_loops}",
        unit="step",
        dynamic_ncols=True,
    )
    latest_val_loss = None
    for step in progress:
        step_started = time.time()
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        noise_multiplier = raw_model.set_loop_noise_step(step)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(train_cfg["gradient_accumulation_steps"]):
            batch_index = step * train_cfg["gradient_accumulation_steps"] + micro_step
            x = train_data.get_sequential_batch(
                train_cfg["micro_batch_size"], batch_index, device
            )
            with amp_context():
                loss = model(input_ids=x, labels=x, use_cache=False).loss
                loss = loss / train_cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
            accumulated_loss += loss.detach().float().item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        lr = cosine_lr(
            step,
            total_steps,
            train_cfg["warmup_steps"],
            train_cfg["learning_rate"],
            train_cfg["min_learning_rate"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.step(optimizer)
        scaler.update()
        update_start_idx = max(model_cfg.loop_update_start_loop - 1, 0)
        alpha_start = raw_model.model.current_loop_update_alpha(update_start_idx).detach().float().item()
        alpha_last = raw_model.model.current_loop_update_alpha(model_cfg.num_loops - 1).detach().float().item()
        tokens_seen = (step + 1) * tokens_per_step
        elapsed = max(time.time() - started, 1e-6)
        tokens_per_second = (tokens_seen - session_start_tokens) / elapsed
        grad_norm_value = float(grad_norm)
        postfix = {
            "loss": f"{accumulated_loss:.4f}",
            "lr": f"{lr:.2e}",
            "grad": f"{grad_norm_value:.2f}",
            "tok/s": f"{tokens_per_second:,.0f}",
            "α2": f"{alpha_start:.3f}",
            "αR": f"{alpha_last:.3f}",
        }
        if latest_val_loss is not None:
            postfix["val"] = f"{latest_val_loss:.4f}"
        progress.set_postfix(postfix)

        if step % train_cfg["log_interval"] == 0 or step == total_steps - 1:
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": accumulated_loss,
                "lr": lr,
                "grad_norm": grad_norm_value,
                "tokens_per_second": tokens_per_second,
                "step_seconds": time.time() - step_started,
                "session_elapsed_seconds": elapsed,
                "loop_noise_multiplier": noise_multiplier,
                "loop_update_alpha_start": alpha_start,
                "loop_update_alpha_last": alpha_last,
            }
            tqdm.write(json.dumps(record))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        should_eval = step % train_cfg["eval_interval"] == 0 or step == total_steps - 1
        if should_eval:
            val_loss = estimate_loss(
                model,
                val_data,
                train_cfg["eval_batches"],
                train_cfg["micro_batch_size"],
                device,
                amp_context,
                description=f"validation step {step + 1}/{total_steps}",
            )
            latest_val_loss = val_loss
            progress.set_postfix({**postfix, "val": f"{val_loss:.4f}"})
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "val_loss": val_loss,
                "val_perplexity": math.exp(min(val_loss, 20)),
                "session_elapsed_seconds": time.time() - started,
            }
            tqdm.write(json.dumps(record))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            if val_loss < best_val:
                best_val = val_loss
                best_dir = output_dir / "best_hf"
                raw_model.save_pretrained(best_dir, safe_serialization=True)
                tokenizer.save_pretrained(best_dir)
                (best_dir / "training_summary.json").write_text(
                    json.dumps(
                        {
                            "step": step,
                            "tokens_seen": tokens_seen,
                            "best_val_loss": best_val,
                            "parameters": parameter_count,
                            "loop_update_alpha_start": alpha_start,
                            "loop_update_alpha_last": alpha_last,
                            "loop_update_schedule_slope": (
                                raw_model.model.loop_update_log_slope.detach().float().item()
                                if hasattr(raw_model.model, "loop_update_log_slope")
                                else None
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            # Save resumable state at every evaluation, important for Colab sessions.
            save_resume_state(
                output_dir / "last_state.pt",
                raw_model,
                optimizer,
                step,
                tokens_seen,
                best_val,
                config,
            )

        if step > 0 and step % train_cfg["save_interval"] == 0:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_resume_state(
                output_dir / "last_state.pt", raw_model, optimizer, step, tokens_seen, best_val, config
            )

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_resume_state(
        output_dir / "last_state.pt",
        raw_model,
        optimizer,
        total_steps - 1,
        tokens_seen,
        best_val,
        config,
    )
    progress.close()
    print(
        f"finished in {(time.time() - started) / 60:.1f} min; "
        f"best validation loss={best_val:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
