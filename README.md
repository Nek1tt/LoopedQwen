# Looped Transformer baseline on FineWeb

Minimal, inspectable baseline for the T-Lab Looped Models task. It subclasses
the **official Hugging Face Qwen3 implementation** and reuses the **same stack
of Qwen3 decoder layers** at
every loop:

```text
tokens -> embedding -> [shared layer 1 -> shared layer 2] x R -> norm -> LM head
```

Attention, Q/K RMSNorm, RoPE, SwiGLU, causal masking, initialization, the
causal-LM loss and Hugging Face serialization are provided by Transformers.
The only architectural change is the recurrent traversal of `model.layers`.

The baseline deliberately contains no loop embeddings, intermediate losses,
noise, early exit, or other research modifications. It is a clean control on
top of which those ideas can be tested.

## Baseline configuration

| Item | Value |
|---|---:|
| Vocabulary | 16,000 byte-level BPE |
| Hidden size | 256 |
| Shared layers | 2 |
| Loops | 4 |
| Effective depth | 8 |
| Query / KV heads | 4 / 2 |
| SwiGLU width | 768 |
| Context | 512 |
| Parameters | 5,670,400 |
| Train budget | at most 100M processed tokens |

The embedding and LM-head weights are tied. The training script refuses to run
if the model exceeds 10M parameters.

## Repository layout

```text
configs/                 experiment configurations
src/looped_lm/modeling_looped_qwen3.py  thin Qwen3 subclass and loop
src/looped_lm/data.py    memory-mapped token batches
scripts/train_tokenizer.py
scripts/prepare_data.py
scripts/train.py
scripts/eval.py
scripts/upload_to_hub.py
scripts/sanity_check.py
experiments/             isolated experiment configs, runners and results
experiments/003_loop_state_noise/  completed 16-loop noise experiment
experiments/004_normalized_loop_updates/  projected-update experiment
experiments/005_anytime_depth/  random-depth + intermediate-loss experiment
tests/                   small CPU tests
REPORT.md                experiment report template
```

## Experiment 003: noise between recurrent passes

The third experiment trains a 16-loop control and two scale-relative Gaussian
noise variants. Neither relative nor norm-preserving noise at `sigma=0.03`
improved PPL at 16 loops or extrapolation to 32 loops. The hidden-state norm
continued to grow while consecutive states became almost collinear. See
[`experiments/003_loop_state_noise/README.md`](experiments/003_loop_state_noise/README.md)
for the hypothesis, complete results, diagnostics and run commands.

## Experiment 004: normalized projected loop updates

The fourth experiment prevents recurrent residual-norm growth directly. After
the first ordinary loop, every update is RMS-normalized, gated and projected
back to the first loop's per-token RMS. The method keeps the norm fixed and
preserves non-trivial updates through the trained depth, but slightly worsens
PPL@16 and strongly overfits the exact 16-loop horizon: extrapolation to 20–32
loops degrades. Fixed and learned gates behave similarly; the fixed gate is
reconstructed from serialized config so Hugging Face round trips preserve it.
See
[`experiments/004_normalized_loop_updates/README.md`](experiments/004_normalized_loop_updates/README.md)
for the method and experiment matrix.

## Experiment 005: random depth and intermediate losses

The fifth experiment trains the projected recurrent model at uniformly sampled
depths from 8 through 24. Random depth removes the sharp 16-loop specialization
seen in Experiment 004. Adding LM losses at loop 8, a midpoint and the sampled
terminal depth produces the best result: PPL 1111.30 at 16 loops and 1136.32 at
32 loops, versus 1428.02 and 1964.71 for the fixed-depth Experiment 004
reference. Quality is stable across stopping depths but is not yet monotonic:
PPL reaches its minimum near 12 loops and then slowly worsens. See
[`experiments/005_anytime_depth/README.md`](experiments/005_anytime_depth/README.md)
for the checkpoint-selection analysis and complete diagnostics.

## Local setup

```bash
git clone https://github.com/Nek1tt/LoopedQwen.git
cd looped-transformer-baseline
pip install -e .
python scripts/sanity_check.py
pytest -q
```

## 1. Train the tokenizer

Do not use the original Qwen tokenizer: its large vocabulary makes the
embedding table alone exceed this task's parameter limit.

```bash
python scripts/train_tokenizer.py \
  --output-dir tokenizer \
  --vocab-size 16000 \
  --documents 100000
```

## 2. Prepare a fixed FineWeb subset

This command streams FineWeb and writes disjoint packed token streams. It does
not download the full dataset.

```bash
python scripts/prepare_data.py \
  --tokenizer tokenizer \
  --output-dir data \
  --train-tokens 100000000 \
  --val-tokens 1000000
```

`data/metadata.json` records the tokenizer, split and exact token counts.
Training consumes non-overlapping contexts sequentially, so the 99,975,168
processed tokens are unique within the prepared 100M-token stream. The global
batch index determines the offset, making resume deterministic.

## 3. Smoke test before spending GPU time

```bash
python scripts/train.py --config configs/smoke.yaml
```

The smoke configuration uses 1M processed tokens and a smaller CPU-friendly
model. For a quicker data test, prepare at least 1.1M train tokens first.

## 4. Train the baseline

```bash
python scripts/train.py --config configs/baseline.yaml
```

The output directory contains:

- `best_hf/`: best checkpoint in Hugging Face `save_pretrained()` format;
- `last_state.pt`: resumable optimizer and model state;
- `metrics.jsonl`: training and validation metrics;
- `run_config.json`: exact run configuration.

Resume an interrupted Colab session after restoring the output directory:

```bash
python scripts/train.py \
  --config configs/baseline.yaml \
  --resume outputs/baseline/last_state.pt
```

## 5. Evaluate whether later loops remain useful

```bash
python scripts/eval.py \
  --checkpoint outputs/baseline/best_hf \
  --val-file data/val.bin \
  --loops 1 2 4 8 16 \
  --batches 100
```

The evaluator uses identical validation batches for every loop count and
reports loss, perplexity and

```text
||h_(r+1) - h_r|| / ||h_r||
```

for each loop. This is the first diagnostic for saturation or convergence.

## First experiment matrix

Copy `configs/baseline.yaml` and change only `model.num_loops`:

| Run | Train loops | Eval loops |
|---|---:|---|
| control | 1 | 1, 2, 4, 8 |
| loop-2 | 2 | 1, 2, 4, 8 |
| loop-4 | 4 | 1, 2, 4, 8, 16 |
| loop-8 | 8 | 1, 2, 4, 8, 16 |

Keep the tokenizer, data files, seed and processed-token budget fixed. Report
parameter count and loop-dependent FLOPs separately: sharing weights reduces
parameters, not computation.

## Push to GitHub

```bash
git init
git add .
git commit -m "Add looped Transformer FineWeb baseline"
git branch -M main
git remote add origin https://github.com/YOUR_NAME/YOUR_REPOSITORY.git
git push -u origin main
```

Data, tokenizers and checkpoints are intentionally ignored by Git. Upload the
best checkpoint and tokenizer to a public Hugging Face model repository when
the experiment is complete:

```bash
hf auth login
python scripts/upload_to_hub.py \
  --checkpoint outputs/baseline/best_hf \
  --repo-id YOUR_NAME/looped-qwen3-baseline
```

The uploaded checkpoint includes the custom loop wrapper and can be loaded in
a clean environment:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "YOUR_NAME/looped-qwen3-baseline",
    trust_remote_code=True,
)
```

## Notes and current limitations

- This is a single-GPU baseline intended for Colab, not a distributed trainer.
- The repository pins Transformers 5.15.1 because the subclass follows that
  official Qwen3 API. Upgrade only together with the compatibility tests.
- KV caching is intentionally disabled. A normal Qwen3 cache has one slot per
  physical layer and is not correct for repeated effective-depth positions.
- Validation contexts are sampled from one fixed, document-disjoint token file.
- Training contexts traverse the shuffled-document stream sequentially. The
  budget counts tokens actually passed to the model.
- Evaluation beyond the trained loop count is intentional but is not guaranteed
  to improve perplexity.
- The default is `float16` for compatibility with common Colab T4 GPUs. On
  Ampere or newer GPUs, `bfloat16` is also available and usually preferable.
