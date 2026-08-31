# DUOBIT-EST

**Error-Compensated Stochastic Transition Training of Language Models at Two Bits and Below Without Master Weights**

Pratyush Bhardwaj · Independent researcher, India · August 2026

Native 2-bit / ternary / binary pretraining **without a full-precision master-weight tensor**. Linear maps are stored as discrete codes `z` and per-group scales `s`. Sub-threshold updates accumulate in a quantized error-feedback residual (QPEFA). Adam moments are 8-bit block-wise with a factored second moment. Training memory and inference memory both drop.

- Paper: [`paper/duobit_est.md`](paper/duobit_est.md)
- Package: `duobit/` (PyTorch)
- **GPU results: [`docs/fineweb_v2_results.md`](docs/fineweb_v2_results.md)** ·
  ablation: [`docs/fineweb_v2_ablation.md`](docs/fineweb_v2_ablation.md)
- Experiments: `scripts/run_experiments.py` (CPU),
  `scripts/kaggle_duobit_fineweb.py` (GPU / DDP)
- Tests: `python3 -m pytest tests/ -v` (30 passing)
- Config pre-flight: `python3 scripts/validate_configs.py`

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest matplotlib
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 scripts/run_experiments.py
```

## FineWeb-EDU, 77M parameters, 3000 steps, 2x Tesla T4

Same architecture, same 49.2M streamed tokens, same seed, same step budget.
Full write-up and caveats in [`docs/fineweb_v2_results.md`](docs/fineweb_v2_results.md).

| | **DUOBIT v2** | DUOBIT v1 | FP32 AdamW | FP16 AMP |
|---|---|---|---|---|
| Val loss | **5.0462** | 5.5374 | **4.5968** | 4.5969 |
| Val perplexity | **155.4** | 254.0 | **99.2** | 99.2 |
| Gap to FP32 (nats) | **+0.449** | +0.941 | — | +0.000 |
| Linear train bits/wt | **19.33** | 18.58 | 96.00 | 96.00 |
| Linear infer bits/wt | **2.25** | 2.25 | 32.00 | 16.00 |
| Persistent train state | **413 MiB** | 409 MiB | 883 MiB | 883 MiB |
| Inference footprint | **112 MiB** | 112 MiB | 294 MiB | 196 MiB |
| Throughput | 13,470 tok/s | 13,184 tok/s | 14,083 tok/s | **35,508 tok/s** |

Tuning the method (see the [ablation](docs/fineweb_v2_ablation.md)) moved DuoBIT
**0.49 nats**, cutting perplexity from 254 to 155 and closing **52%** of the gap
to FP32. It is closer, **not equal**: at 2.25 inference bits per linear weight
it remains 0.449 nats behind. Both models are well short of a compute-optimal
token budget, so this gap is not necessarily the gap at convergence.

Read the trade honestly: **DuoBIT buys memory, not speed.** FP16 mixed precision
matches FP32 quality to 0.0001 nats at 2.5x the throughput, because T4 tensor
cores do FP16 matmuls natively; DuoBIT runs at 0.956x FP32. What DuoBIT gives is
**4.97x** less training state and **14.2x** less inference storage *on the
linear maps* (2.14x and 2.63x whole-model, diluted by a 25.7M-parameter FP32
embedding table). A packed bit-serial GEMM, which would remove the ephemeral
dequantized FP32 tile, is what would turn the compression into speed.

## Why this is not BitNet

| | FP32 Adam | Latent-weight STE (BitNet pattern) | DUOBIT v2 |
|---|---|---|---|
| Persistent weights | FP32 `W` | FP32 master `W`, quantized in forward | 2-bit codes + trained scales |
| Training residual | none | none (STE) | int8 QPEFA |
| Optimizer | FP32 `m,v` | FP32 `m,v` | int8 `m`, factored `v` |
| Linear train bits/wt | 96 | 96 | **19.3** at `G=128` |
| Linear infer bits/wt | 32 | 32 (checkpoint is still FP32 unless you export) | **2.25** at `G=128` |

The group scales are trained from their exact gradient, but they are not master
weights: they are part of the stored 2-bit representation, kept at inference and
already charged at `32/G` bits per weight. Training them costs `3*32/G` bits per
weight of optimizer state (0.75 at `G=128`), which the memory report includes.

## Tiny-run headline (CPU, 94,528 params, 50 steps, v1 numbers)

| Run | Val loss | PPL | Train MB | Infer MB | Grammar acc. |
|---|---|---|---|---|---|
| FP32 Adam | 2.52 | 12.43 | 1.082 | 0.361 | 70% |
| Latent-weight STE | 2.83 | 16.96 | 1.082 | 0.361 | 40% |
| **DUOBIT 2-bit QPEFA-8** | **1.23** | **3.41** | **0.288** | **0.056** | **100%** |
| DUOBIT 2-bit QPEFA-4 | 1.25 | 3.49 | 0.246 | 0.056 | 100% |
| Ternary QPEFA-8 | 1.32 | 3.75 | 0.283 | 0.052 | 100% |
| Binary QPEFA-8 | 1.85 | 6.36 | 0.277 | 0.046 | 100% |

These are v1 numbers on a 50-step synthetic-grammar task and are kept for
continuity. They did not predict behaviour at scale -- the same v1 configuration
that wins here is 0.94 nats behind FP32 on FineWeb-EDU at 77M parameters -- so
weight the GPU results above instead. Under v2 this tiny task scores worse (val
2.28 vs 1.23) while the 77M FineWeb-EDU run improves by 0.49 nats; at 94k
parameters and 50 steps the benchmark is too small and too short to separate
configurations.
