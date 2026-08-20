# Looped Transformer: experiment report

## 1. Research question

Can a fixed-size shared Transformer block continue to improve next-token
predictions when it is applied for substantially more recurrent loops?

## 2. Baseline

Describe the tokenizer, FineWeb subset, architecture, parameter count, processed
token budget, optimizer, hardware and wall-clock time. State clearly that the
complete Attention + MLP stack is shared between loops.

## 3. Evaluation protocol

- Fixed validation token file and fixed sampled contexts.
- Validation loss and perplexity at different loop counts.
- Relative hidden-state update by loop.
- Parameter count and approximate compute reported separately.

## 4. Baseline results

| Run | Parameters | Train loops | Eval loops | Val loss | Perplexity |
|---|---:|---:|---:|---:|---:|
| | | | | | |

Include plots of validation loss versus evaluation loops and relative update
versus loop index.

## 5. Proposed modification

Write the hypothesis before running the experiment. Explain why the change
could prevent saturation and whether its parameter cost remains negligible at
larger scale.

## 6. Ablations

Change one important factor at a time and keep data, tokenizer and training
budget fixed.

## 7. Negative results

For every failed idea, distinguish optimization failure, instability,
overfitting and genuine loop saturation. Include evidence rather than only the
final perplexity.

## 8. Scaling argument

Discuss expected behavior as model width, training tokens and loop count grow.
Separate improvements caused by extra parameters from improvements caused by
useful recurrent computation.

