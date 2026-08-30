# DUOBIT-EST: Architecture and Developer Guide

**Author:** Pratyush Bhardwaj  
**Version:** 2.0.0 (QPEFA + compressed Adam + trained group scales)

Native 2-bit / ternary / binary pretraining **without master weights**. Linear maps persist as discrete codes $z$ and group scales $s$. The training residual is an integer QPEFA tensor. Adam $m$ is block-wise int8. Adam $v$ is factored ($O(m+n)$). The group scales are trained from their exact analytic gradient.

## Headline

FineWeb-EDU, 77.2M parameters, 3000 steps (49.2M tokens), 2x Tesla T4, identical
data/seed/budget across runs:

| | DUOBIT v2 | DUOBIT v1 | FP32 AdamW | FP16 AMP |
|---|---|---|---|---|
| val loss / PPL | **5.046 / 155.4** | 5.537 / 254.0 | **4.597 / 99.2** | 4.597 / 99.2 |
| linear train bits/wt | 19.33 | 18.58 | 96.00 | 96.00 |
| linear infer bits/wt | 2.25 | 2.25 | 32.00 | 16.00 |
| train state / inference | 413 / 112 MiB | 409 / 112 | 883 / 294 | 883 / 196 |
| throughput | 13,470 tok/s | 13,184 | 14,083 | 35,508 |

Tuning moved DuoBIT 0.49 nats and closed 52% of the gap to FP32; it is closer,
not equal. DuoBIT buys memory (4.97x training state, 14.2x inference storage on
the linear maps), not speed -- FP16 AMP matches FP32 quality at 2.5x the
throughput. Full results and caveats: `docs/fineweb_v2_results.md`, ablation:
`docs/fineweb_v2_ablation.md`.

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
