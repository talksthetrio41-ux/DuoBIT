# DUOBIT-EST walkthrough

Native 2-bit training without master weights, with compressed training state.

## What is stored

- **Inference:** 2-bit (or ternary/binary) codes `z` plus per-group scales `s`.
- **Training only:** int8 QPEFA residual, int8 block-wise first moment, factored second moment. Discarded at checkpoint time.

## Tiny decoder (94,528 params, 50 steps, CPU)

| Run | Val | Train MB | Infer MB | Acc. |
|---|---|---|---|---|
| FP32 Adam | 2.52 | 1.082 | 0.361 | 70% |
| Latent-weight STE | 2.83 | 1.082 | 0.361 | 40% |
| DUOBIT 2-bit QPEFA-8 | **1.23** | **0.288** | **0.056** | **100%** |
| DUOBIT 2-bit QPEFA-4 | 1.25 | 0.246 | 0.056 | 100% |

Linear maps train at 20.3 bits/weight (16.3 with a 4-bit residual) versus 96 for FP32 Adam.

## Reproduce

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 scripts/run_experiments.py
```
