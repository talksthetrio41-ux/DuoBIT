# DUOBIT-EST: Architecture and Developer Guide

**Author:** Pratyush Bhardwaj  
**Version:** 3.0.0 (QPEFA + compressed Adam + trained group scales + 2-bit embedding table)

Native 2-bit / ternary / binary pretraining **without master weights**. Linear maps persist as discrete codes $z$ and group scales $s$. The training residual is an integer QPEFA tensor. Adam $m$ is block-wise int8. Adam $v$ is factored ($O(m+n)$). The group scales are trained from their exact analytic gradient.

## Headline

FineWeb-EDU, 77.2M parameters, 3000 steps (49.2M tokens), 2x Tesla T4, identical
data/seed/budget across runs:

| | DUOBIT v3 | DUOBIT v2 | DUOBIT v1 | FP32 AdamW | FP16 AMP |
|---|---|---|---|---|---|
| val loss / PPL (pooled) | **5.013 / 150.5** | 5.062 / 160.5 | 5.537 / 254.0 | **4.598 / 99.4** | 4.598 / 99.3 |
| linear train bits/wt | **15.32** | 19.33 | 18.58 | 96.00 | 96.00 |
| linear infer bits/wt | 2.25 | 2.25 | 2.25 | 32.00 | 16.00 |
| FP32 params remaining | **8,704** | 25.7M | 25.7M | 77.2M | 77.2M |
| train state / inference | **141 / 20.7 MiB** | 413 / 112 | 409 / 112 | 883 / 294 | 883 / 196 |
| throughput | 14,167 tok/s | 14,131 | 13,184 | 14,654 | 34,939 |

v3 quantizes the token embedding table, which was 71% of training state and 88%
of the inference footprint. **6.26x** less training state and **14.2x** less
inference storage than FP32, whole-model, with 0.011% of parameters left in
FP32. Quality is **neutral vs v2** (+0.049 nats, ~1.5σ, not resolved) and still
**0.415 nats behind FP32**. DuoBIT buys memory, not speed -- FP16 AMP matches
FP32 quality at 2.4x the throughput. Results: `docs/fineweb_v3_results.md`,
ablation: `docs/fineweb_v3_ablation.md`.

## Run-to-run variance (read before trusting any single number)

The discrete path does **not** reproduce across runs, even with a fixed seed.
Seven runs of one configuration at the 500-step sweep scale: mean 6.1164, std
0.0473, **range 0.131 nats** -- two of them byte-identical in the same session,
0.097 apart. At 77M/3000 steps the replicate spread is 0.059. FP32 over the
same sessions spans 0.004.

A code transition is a step function of the accumulated QPEFA residual, so a
perturbation far below FP32 rounding decides whether a weight flips, and GPU
reduction order is not deterministic across runs. Fixing the seed fixes the
stochastic-rounding hash, not the arithmetic feeding it. **Do not report an
effect below ~0.13 nats (500 steps) or ~0.06 (3000 steps) from single runs;
replicate and pool.** Two rows of the v2 ablation failed this test and are
marked unresolved.

## What changed in v2 (read this first)

v1 had exactly one continuous degree of freedom, the group scale $s_g$, and
never trained it. The "EMA scale update" recomputed the MSE-optimal scale from
the already-dequantized weight, and since $s^\star = \sum_i (s c_i)c_i / \sum_i
c_i^2 = s$ it is an algebraic fixed point. Measured over a 3000-step
FineWeb-EDU run, the mean scale moved by $2\times10^{-7}$ relative. Four
corrections follow from that:

1. **Trained scales** (`learn_scales`). $\partial\mathcal{L}/\partial s_g =
   \sum_{i\in g} g_i C[z_i]$ is exact with the codes fixed, so $s$ takes its own
   Adam step (relative, floored at `scale_min_frac` of its init). The scale is
   already part of the persistent 2-bit representation and is kept at
   inference, so this is *not* a master weight; it adds $3\cdot 32/G$ bits per
   weight of optimizer state (0.75 at $G=128$), which `memory.py` accounts for.
2. **Variance-preserving scale init** (`scale_init="var"`). The MSE fit shrinks
   the weight std to $0.956\times$ at every fan-in, compounding to $0.49\times$
   the activation scale over an 8-layer stack. The `var` fit reproduces the
   intended per-group std instead.
3. **`pefa_clip` 4.0 -> 1.0.** The accumulator only reaches $\approx0.333 s_g$
   after quantization, so clipping at $4 s_g$ spent most of the int8 residual
   grid on unreachable values. One integer step goes from 4.7% to 1.2% of a
   codebook gap.
4. **No weight decay on the discrete path** (`duobit_weight_decay=0`), and a
   **scale-relative discrete LR** (`scale_relative_lr`), $\Delta w = \eta s_g
   \cdot \text{Adam dir}$, so one rate is commensurate with the gap everywhere.

`scale_update_freq`/`scale_ema_alpha` now default to 0; they exist only to
reproduce v1 in the ablation.

## Math (short)

$$w_i = s_g C[z_i], \quad C\in\{\{-1,+1\},\{-1,0,+1\},\{-1,-\rho,+\rho,+1\}\}$$

$$s_g^\star = \frac{\sum w_i c_i}{\sum c_i^2+\varepsilon}$$

QPEFA: $e \leftarrow e+(\tilde W-\hat W)$; $(z,\hat W,e)\leftarrow Q(\hat W+e)$; store $e$ as $\mathrm{SR}(e \cdot q_{\max}/(\lambda s_g))$ in 4 or 8 bits.

## Layout

```
duobit/                 # PyTorch package
  config.py             # DuobitConfig (qpefa_bits, moment_bits, ...)
  layers/duobit_linear.py
  layers/ste_linear.py  # latent-weight STE baseline
  optim/duobit_adam.py  # QPEFA + compressed Adam
  quantization/         # codebook, qpefa, blockwise, stochastic, memory
  model/transformer.py  # SwiGLU, RoPE, RMSNorm
  training/
scripts/run_experiments.py
tests/                  # 20 tests
paper/duobit_est.md
```

## Commands

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 scripts/run_experiments.py
```

## GPU experiments (Kaggle, dual T4)

`scripts/kaggle_duobit_fineweb.py` is the self-contained DDP experiment: DuoBIT
vs FP32 vs FP16-AMP on streamed FineWeb-EDU with a GPT-2 tokenizer, custom CUDA
kernels (with a parity-checked PyTorch fallback), and an N-run driver
(`--runs-json`) so an ablation varies one knob per run over a shared token
stream, seed, architecture and step budget.

```bash
# local CPU verification of the whole path
python3 scripts/kaggle_duobit_fineweb.py --smoke --dataset docs.jsonl

# push / poll / fetch a dual-T4 Kaggle run
python3 scripts/kaggle_run.py push   <name> --args '<argv json list>'
python3 scripts/kaggle_run.py status <name>
python3 scripts/kaggle_run.py fetch  <name> --out DIR
python3 scripts/pick_best_config.py DIR       # ablation table + winning config
python3 scripts/collect_results.py  DIR --tag <tag> --figures
```

**Selecting the accelerator.** Kaggle has no separate "T4 x2" value. Set
`enable_gpu: true` *and* `machine_shape: "NvidiaTeslaT4"` in
`kernel-metadata.json` (equivalently `kaggle kernels push --accelerator
NvidiaTeslaT4`): that machine shape *is* the dual-T4 machine, and
`torch.cuda.device_count()` returns 2 inside the session. The alternatives are
`NvidiaTeslaP100` (one P100) and `Tpu1VmV38`. The experiment script spawns one
DDP rank per visible GPU, so both T4s are used. GPU quota is 30 h/week
nominally but the account quota is reported by
`get_accelerator_quota_statistics()` -- check it before a long run.

## Next

1. Packed bit-serial GEMM (remove the ephemeral dequant tile)
2. W2A4
3. Longer-horizon FineWeb-EDU runs (the current budget is compute-limited, not
   method-limited)
