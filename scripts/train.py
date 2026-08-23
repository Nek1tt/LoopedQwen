#!/usr/bin/env python3
import argparse
import json
import math
import random
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
    loop_depths=None,
) -> tuple[float, dict[int, float]]:
    model.eval()
    depths = tuple(loop_depths or (model.config.num_loops,))
    if any(depth < 1 for depth in depths):
        raise ValueError("validation loop depths must be positive")
    losses_by_depth = {}
    progress = tqdm(
        total=batches * sum(depths),
        desc=description,
        unit="loop-batch",
        leave=False,
        dynamic_ncols=True,
    )
    for depth in depths:
        # Reusing the seed makes every depth see exactly the same validation batches.
        generator = torch.Generator().manual_seed(seed)
        losses = []
        for batch_idx in range(batches):
            x, _ = dataset.get_batch(batch_size, device, generator)
            with amp_context():
                # Hugging Face's causal-LM loss performs the one-token shift.
                loss = model(
                    input_ids=x,
                    labels=x,
                    use_cache=False,
                    num_loops=depth,
                ).loss
            losses.append(loss.float())
            running = torch.stack(losses).mean().item()
            progress.update(depth)
            progress.set_postfix(
                R=depth,
                batch=f"{batch_idx + 1}/{batches}",
                loss=f"{running:.4f}",
            )
        losses_by_depth[depth] = torch.stack(losses).mean().item()
        tqdm.write(f"validation R={depth}: loss={losses_by_depth[depth]:.4f}")
    progress.close()
    model.train()
    mean_loss = sum(losses_by_depth.values()) / len(losses_by_depth)
    return mean_loss, losses_by_depth


def depth_plan(training: dict, default_depth: int, seed: int, batch_index: int):
    """Return a deterministic compute depth, supervised depths and loss weights."""
    mode = training.get("depth_sampling", "fixed")
    if mode == "fixed":
        depth = default_depth
    elif mode == "uniform":
        minimum = int(training["min_train_loops"])
        maximum = int(training["max_train_loops"])
        # A local RNG makes the schedule independent of dropout/data RNG and resume-safe.
        depth = random.Random(seed + 1_000_003 * batch_index).randint(minimum, maximum)
    else:
        raise ValueError("training.depth_sampling must be fixed or uniform")

    if not training.get("intermediate_lm_losses", False):
        return depth, (depth,), (1.0,)

    base = int(training.get("intermediate_loss_base_loop", training["min_train_loops"]))
    if base > depth:
        raise ValueError("intermediate_loss_base_loop cannot exceed sampled depth")
    midpoint = (base + depth) // 2
    supervised = tuple(sorted({base, midpoint, depth}))
    if len(supervised) == 1:
        weights = (1.0,)
    else:
        final_weight = float(training.get("final_loss_weight", 0.5))
        auxiliary_weight = (1.0 - final_weight) / (len(supervised) - 1)
        weights = tuple(
            final_weight if loop == depth else auxiliary_weight for loop in supervised
        )
    return depth, supervised, weights


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

    validation_loops = tuple(train_cfg.get("validation_loops", [model_cfg.num_loops]))
    if not validation_loops or any(depth < 1 for depth in validation_loops):
        raise ValueError("training.validation_loops must contain positive depths")
    if train_cfg.get("depth_sampling", "fixed") == "uniform":
        minimum = int(train_cfg["min_train_loops"])
        maximum = int(train_cfg["max_train_loops"])
        if not 1 <= minimum <= maximum:
            raise ValueError("Require 1 <= min_train_loops <= max_train_loops")
        if maximum > model_cfg.num_loops:
            raise ValueError("max_train_loops cannot exceed model.num_loops")
    final_loss_weight = float(train_cfg.get("final_loss_weight", 0.5))
    if not 0.0 < final_loss_weight <= 1.0:
        raise ValueError("training.final_loss_weight must be in (0, 1]")
    token_loss_weighting = train_cfg.get("token_loss_weighting", "uniform")
    if token_loss_weighting not in {"uniform", "previous_loss"}:
        raise ValueError("training.token_loss_weighting must be uniform or previous_loss")
    hard_token_gamma = float(train_cfg.get("hard_token_gamma", 0.5))
    hard_token_min_weight = float(train_cfg.get("hard_token_min_weight", 0.25))
    hard_token_max_weight = float(train_cfg.get("hard_token_max_weight", 4.0))
    hard_token_uniform_mix = float(train_cfg.get("hard_token_uniform_mix", 0.5))
    if hard_token_gamma <= 0.0:
        raise ValueError("training.hard_token_gamma must be positive")
    if not 0.0 < hard_token_min_weight <= hard_token_max_weight:
        raise ValueError("Require 0 < hard_token_min_weight <= hard_token_max_weight")
    if not 0.0 <= hard_token_uniform_mix <= 1.0:
        raise ValueError("training.hard_token_uniform_mix must be in [0, 1]")

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
        f"depth_sampling={train_cfg.get('depth_sampling', 'fixed')}; "
        f"train_depth={train_cfg.get('min_train_loops', model_cfg.num_loops)}–"
        f"{train_cfg.get('max_train_loops', model_cfg.num_loops)}; "
        f"intermediate_losses={train_cfg.get('intermediate_lm_losses', False)}; "
        f"token_weighting={token_loss_weighting}; "
        f"operator_schedule={model_cfg.loop_operator_schedule}; "
        f"validation_depths={list(validation_loops)}; "
        f"eval_every={train_cfg['eval_interval']} steps x {train_cfg['eval_batches']} batches/depth",
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
        sampled_depths = []
        component_loss_sums = {}
        component_loss_counts = {}
        uniform_loss_sums = {}
        corrective_loss_sums = {}
        hard_weight_sums = {}
        last_supervised = ()
        for micro_step in range(train_cfg["gradient_accumulation_steps"]):
            batch_index = step * train_cfg["gradient_accumulation_steps"] + micro_step
            sampled_depth, supervised_loops, supervision_weights = depth_plan(
                train_cfg, model_cfg.num_loops, config["seed"], batch_index
            )
            sampled_depths.append(sampled_depth)
            last_supervised = supervised_loops
            x = train_data.get_sequential_batch(
                train_cfg["micro_batch_size"], batch_index, device
            )
            with amp_context():
                output = model(
                    input_ids=x,
                    labels=x,
                    use_cache=False,
                    num_loops=sampled_depth,
                    supervision_loops=supervised_loops,
                    supervision_weights=supervision_weights,
                    token_loss_weighting=token_loss_weighting,
                    hard_token_gamma=hard_token_gamma,
                    hard_token_min_weight=hard_token_min_weight,
                    hard_token_max_weight=hard_token_max_weight,
                    hard_token_uniform_mix=hard_token_uniform_mix,
                )
                loss = output.loss
                loss = loss / train_cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
            accumulated_loss += loss.detach().float().item()
            for loop, component in zip(supervised_loops, output.loop_losses):
                component_loss_sums[loop] = component_loss_sums.get(loop, 0.0) + component.detach().float().item()
                component_loss_counts[loop] = component_loss_counts.get(loop, 0) + 1
            for loop, uniform, corrective, hard_weight in zip(
                supervised_loops,
                output.loop_uniform_losses,
                output.loop_corrective_losses,
                output.loop_hard_weight_means,
            ):
                uniform_loss_sums[loop] = uniform_loss_sums.get(loop, 0.0) + uniform.detach().float().item()
                corrective_loss_sums[loop] = corrective_loss_sums.get(loop, 0.0) + corrective.detach().float().item()
                hard_weight_sums[loop] = hard_weight_sums.get(loop, 0.0) + hard_weight.detach().float().item()

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
        alpha_last = raw_model.model.current_loop_update_alpha(sampled_depths[-1] - 1).detach().float().item()
        tokens_seen = (step + 1) * tokens_per_step
        elapsed = max(time.time() - started, 1e-6)
        tokens_per_second = (tokens_seen - session_start_tokens) / elapsed
        grad_norm_value = float(grad_norm)
        postfix = {
            "loss": f"{accumulated_loss:.4f}",
            "lr": f"{lr:.2e}",
            "grad": f"{grad_norm_value:.2f}",
            "tok/s": f"{tokens_per_second:,.0f}",
            "Rμ": f"{sum(sampled_depths) / len(sampled_depths):.1f}",
            "heads": "/".join(map(str, last_supervised)),
            "tok-w": "hard" if token_loss_weighting == "previous_loss" else "uniform",
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
                "loop_operator_schedule": model_cfg.loop_operator_schedule,
                "sampled_depths": sampled_depths,
                "mean_sampled_depth": sum(sampled_depths) / len(sampled_depths),
                "component_losses": {
                    str(loop): component_loss_sums[loop] / component_loss_counts[loop]
                    for loop in sorted(component_loss_sums)
                },
                "uniform_component_losses": {
                    str(loop): uniform_loss_sums[loop] / component_loss_counts[loop]
                    for loop in sorted(uniform_loss_sums)
                },
                "corrective_component_losses": {
                    str(loop): corrective_loss_sums[loop] / component_loss_counts[loop]
                    for loop in sorted(corrective_loss_sums)
                },
                "mean_hard_token_weights": {
                    str(loop): hard_weight_sums[loop] / component_loss_counts[loop]
                    for loop in sorted(hard_weight_sums)
                },
                "token_loss_weighting": token_loss_weighting,
                "hard_token_gamma": hard_token_gamma,
                "hard_token_uniform_mix": hard_token_uniform_mix,
            }
            tqdm.write(json.dumps(record))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        should_eval = step % train_cfg["eval_interval"] == 0 or step == total_steps - 1
        if should_eval:
            val_loss, val_losses_by_depth = estimate_loss(
                model,
                val_data,
                train_cfg["eval_batches"],
                train_cfg["micro_batch_size"],
                device,
                amp_context,
                description=f"validation step {step + 1}/{total_steps}",
                loop_depths=validation_loops,
            )
            latest_val_loss = val_loss
            progress.set_postfix({**postfix, "val": f"{val_loss:.4f}"})
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "val_loss": val_loss,
                "val_perplexity": math.exp(min(val_loss, 20)),
                "val_loss_by_depth": {
                    str(depth): loss for depth, loss in val_losses_by_depth.items()
                },
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
                            "best_val_loss_by_depth": {
                                str(depth): loss
                                for depth, loss in val_losses_by_depth.items()
                            },
                            "parameters": parameter_count,
                            "loop_operator_schedule": model_cfg.loop_operator_schedule,
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
