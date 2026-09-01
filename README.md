# DUOBIT-EST

**Error-Compensated Stochastic Transition Training of Language Models at Two Bits and Below Without Master Weights**

Pratyush Bhardwaj · Independent researcher, India · August 2026

Native 2-bit / ternary / binary pretraining **without a full-precision master-weight tensor**. Linear maps are stored as discrete codes `z` and per-group scales `s`. Sub-threshold updates accumulate in a quantized error-feedback residual (QPEFA). Adam moments are 8-bit block-wise with a factored second moment. Training memory and inference memory both drop.

- Paper: [`paper/duobit_est.md`](paper/duobit_est.md)
- Package: `duobit/` (PyTorch)
- **GPU results: [`docs/fineweb_v3_results.md`](docs/fineweb_v3_results.md)** ·
  ablation: [`docs/fineweb_v3_ablation.md`](docs/fineweb_v3_ablation.md)
- Previous generation: [`v2 results`](docs/fineweb_v2_results.md) ·
  [`v2 ablation`](docs/fineweb_v2_ablation.md)
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
Full write-up and caveats in [`docs/fineweb_v3_results.md`](docs/fineweb_v3_results.md).

| | **DUOBIT v3** | DUOBIT v2 | DUOBIT v1 | FP32 AdamW | FP16 AMP |
|---|---|---|---|---|---|
| Val loss (pooled) | **5.013** | 5.062 | 5.537 | **4.598** | 4.598 |
| Val perplexity | **150.5** | 160.5 | 254.0 | **99.4** | 99.3 |
| Gap to FP32 (nats) | **+0.415** | +0.465 | +0.941 | — | −0.001 |
| Linear train bits/wt | **15.32** | 19.33 | 18.58 | 96.00 | 96.00 |
| Linear infer bits/wt | **2.25** | 2.25 | 2.25 | 32.00 | 16.00 |
| **FP32 params remaining** | **8,704** | 25.7M | 25.7M | 77.2M | 77.2M |
| Persistent train state | **141 MiB** | 413 MiB | 409 MiB | 883 MiB | 883 MiB |
| Inference footprint | **20.7 MiB** | 112 MiB | 112 MiB | 294 MiB | 196 MiB |
| Throughput | 14,167 tok/s | 14,131 tok/s | 13,184 tok/s | 14,654 tok/s | **34,939 tok/s** |

v3 quantizes the **token embedding table** with the same representation as the
linear maps, which is where the memory actually was: at 77M the FP32 table was
71% of training state and 88% of the inference footprint. After it, **8,704 of
77,161,984 parameters are FP32** — 0.011% — so a 77M model exports at **20.7
MiB**, and the whole-model ratio finally approaches the linear-map ratio:
**6.26x** less training state and **14.2x** less inference storage than FP32
(9.5x than an FP16 export).

Read the result honestly on both axes:

- **Compression: unambiguous.** Exact accounting, no error bars needed.
- **Quality: neutral, not improved.** v2 -> v3 is +0.049 nats at ~1.5σ — *not*
  resolved. A 500-step sweep measured +0.157 nats at 3.1σ, but the
  [ablation](docs/fineweb_v3_ablation.md) named regularisation as the likely
  mechanism and predicted the gain would shrink with more tokens. At 6x the
  tokens it did. v3 buys a 2.3x smaller training footprint and a 5.4x smaller
  export **at v2's quality**.
- **Still 0.415 nats behind FP32**, at 2.25 inference bits per linear weight.
  Closer than v1's 0.941, not equal.
- **DuoBIT buys memory, not speed.** FP16 AMP matches FP32 quality to 0.0005
  nats at 2.4x the throughput, because T4 tensor cores do FP16 matmuls
  natively; DuoBIT runs at 0.96x FP32. A packed bit-serial GEMM, removing the
  ephemeral dequantized FP32 tile, is what would turn compression into speed.

**The discrete path is noisy.** Seven runs of one configuration at the 500-step
scale span 0.131 nats — two of them byte-identical in the same session, 0.097
apart — while FP32 spans 0.004. A code transition is a step function of the
accumulated residual, so a perturbation far below FP32 rounding decides whether
a weight flips. Differences below ~0.13 nats (0.06 at 3000 steps) are not
resolved by a single run, which is why these tables quote pooled means and why
two rows of the v2 ablation are now marked unresolved.

## Why this is not BitNet

| | FP32 Adam | Latent-weight STE (BitNet pattern) | DUOBIT v3 |
|---|---|---|---|
| Persistent weights | FP32 `W` | FP32 master `W`, quantized in forward | 2-bit codes + trained scales |
| Embedding table | FP32 | FP32 | **2-bit codes + trained scales** |
| Training residual | none | none (STE) | int8 QPEFA |
| Optimizer | FP32 `m,v` | FP32 `m,v` | int4 `m`, factored `v` |
| Linear train bits/wt | 96 | 96 | **15.3** at `G=128` |
| Linear infer bits/wt | 32 | 32 (checkpoint is still FP32 unless you export) | **2.25** at `G=128` |
| FP32 params at 77M | 77.2M | 77.2M | **8,704** |

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
