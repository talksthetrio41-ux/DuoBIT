# DUOBIT-EST v3

**Error-Compensated Stochastic Transition Training of Language Models at Two Bits and Below Without Master Weights**

Pratyush Bhardwaj · Independent researcher, India · August 2026

Native 2-bit / ternary / binary pretraining **without a full-precision master-weight tensor**. Linear maps are stored as discrete codes `z` and per-group scales `s`. Version 3 adds a persistent error-feedback accumulator (PEFA) so codes actually flip during training.

- Paper (markdown): [`paper/duobit_est.md`](paper/duobit_est.md)
- Package: `duobit/` (PyTorch)
- Verification: `scripts/phase3_verify.py`
- Tests: `python3 -m pytest tests/ -v` (15 passing)

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest matplotlib
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 scripts/phase3_verify.py
```

## What v3 changes

| | v1 | v3 |
|---|---|---|
| Persistent weights | 2-bit codes + scales | same, plus ternary / binary |
| Residual | injected into Adam `m` (ECO) | **PEFA** tensor in optimizer state |
| Transition rate | collapses to 0% | stays ~9–15% on the tiny LM |
| Master FP32 `W` | none | none |

## Tiny-run headline (CPU, 94,528 params, 50 steps)

| Run | Val loss | PPL | Mean flip % | Packed MB | Grammar acc. |
|---|---|---|---|---|---|
| FP32 Adam | 2.52 | 12.43 | — | 0.361 | 70% |
| DUOBIT v3 2-bit PEFA | **1.46** | **4.29** | 15.0 | **0.056** | **100%** |
| DUOBIT v1 ECO only | 3.31 | 27.39 | 3.3 → 0.05 | 0.056 | 40% |
| Ternary PEFA | 2.42 | 11.22 | 6.9 | 0.052 | 80% |
| Binary PEFA | 3.08 | 21.80 | 1.0 | 0.046 | 30% |

These are small-scale verification numbers, not 1B-parameter claims. See the paper.
