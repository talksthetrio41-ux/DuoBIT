# DUOBIT-EST

**Error-Compensated Stochastic Transition Training of Language Models at Two Bits and Below Without Master Weights**

Pratyush Bhardwaj · Independent researcher, India · August 2026

Native 2-bit / ternary / binary pretraining **without a full-precision master-weight tensor**. Linear maps are stored as discrete codes `z` and per-group scales `s`. Sub-threshold updates accumulate in a quantized error-feedback residual (QPEFA). Adam moments are 8-bit block-wise with a factored second moment. Training memory and inference memory both drop.

- Paper: [`paper/duobit_est.md`](paper/duobit_est.md)
- Package: `duobit/` (PyTorch)
- Experiments: `scripts/run_experiments.py`
- Tests: `python3 -m pytest tests/ -v` (20 passing)

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest matplotlib
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 scripts/run_experiments.py
```

## Why this is not BitNet

| | FP32 Adam | Latent-weight STE (BitNet pattern) | DUOBIT QPEFA-8 |
|---|---|---|---|
| Persistent weights | FP32 `W` | FP32 master `W`, quantized in forward | 2-bit codes + scales |
| Training residual | none | none (STE) | int8 QPEFA |
| Optimizer | FP32 `m,v` | FP32 `m,v` | int8 `m`, factored `v` |
| Linear train bits/wt | 96 | 96 | **20.3** |
| Linear infer bits/wt | 32 | 32 (checkpoint is still FP32 unless you export) | **3.0** at `G=32` |

## Tiny-run headline (CPU, 94,528 params, 50 steps)

| Run | Val loss | PPL | Train MB | Infer MB | Grammar acc. |
|---|---|---|---|---|---|
| FP32 Adam | 2.52 | 12.43 | 1.082 | 0.361 | 70% |
| Latent-weight STE | 2.83 | 16.96 | 1.082 | 0.361 | 40% |
| **DUOBIT 2-bit QPEFA-8** | **1.23** | **3.41** | **0.288** | **0.056** | **100%** |
| DUOBIT 2-bit QPEFA-4 | 1.25 | 3.49 | 0.246 | 0.056 | 100% |
| Ternary QPEFA-8 | 1.32 | 3.75 | 0.283 | 0.052 | 100% |
| Binary QPEFA-8 | 1.85 | 6.36 | 0.277 | 0.046 | 100% |

These are small-scale verification numbers, not 1B-parameter claims. See the paper.
