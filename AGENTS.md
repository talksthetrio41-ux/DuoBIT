# DUOBIT-EST: Architecture and Developer Guide

**Author:** Pratyush Bhardwaj  
**Version:** 2.0.0 (QPEFA + compressed Adam + trained group scales)

Native 2-bit / ternary / binary pretraining **without master weights**. Linear maps persist as discrete codes $z$ and group scales $s$. The training residual is an integer QPEFA tensor. Adam $m$ is block-wise int8. Adam $v$ is factored ($O(m+n)$). The group scales are trained from their exact analytic gradient.

## Headline

On a 94,528-parameter decoder (50 steps, CPU):

- 2-bit QPEFA-8 val 1.23 / PPL 3.41 / 100% grammar / 0.056 MB infer / 0.288 MB train / 20.3 linear train bits/wt
- FP32 Adam val 2.52 / PPL 12.43 / 70% acc / 0.361 MB infer / 1.082 MB train / 96 bits/wt
- Latent-weight STE val 2.83 / 40% acc / same memory as FP32
- QPEFA-4 val 1.25 / 16.3 train bits/wt

Paper: `paper/duobit_est.md`

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
