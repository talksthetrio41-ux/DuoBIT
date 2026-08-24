# DUOBIT-EST v3 — Architecture & Developer Guide

**Author:** Pratyush Bhardwaj  
**Version:** 0.3.0 (PEFA)

Native 2-bit / ternary / binary pretraining **without master weights**. Linear maps persist as discrete codes $z$ and group scales $s$. Optimizer states (Adam $m,v$ and PEFA residual $e$) are training-only.

## v3 headline

v1 ECO injected quantization error into Adam momentum. Codes froze (transition rate → 0%). v3 stores a persistent error-feedback accumulator $e$ so virtual weights cross codebook gaps. Tiny-LM verification (94,528 params, 50 steps, CPU):

- 2-bit PEFA val 1.46 / PPL 4.29 / 100% grammar acc / 0.056 MB
- FP32 val 2.52 / PPL 12.43 / 70% acc / 0.361 MB
- v1 ECO-only val 3.31 / final flip rate 0.05%

Paper: `paper/duobit_est.md`

## Math (short)

$$w_i = s_g C[z_i], \quad C\in\{\{-1,+1\},\{-1,0,+1\},\{-1,-\rho,+\rho,+1\}\}$$

$$s_g^\star = \frac{\sum w_i c_i}{\sum c_i^2+\varepsilon}$$

PEFA: $e \leftarrow e+(\tilde W-\hat W)$; $(z,\hat W,e)\leftarrow Q(\hat W+e)$.

## Layout

```
duobit/                 # PyTorch package
  config.py             # DuobitConfig (use_pefa, n_levels, ...)
  layers/duobit_linear.py
  optim/duobit_adam.py  # PEFA + ECO
  quantization/         # codebook, stochastic, hadamard, memory
  model/transformer.py  # SwiGLU, RoPE, RMSNorm
  training/
scripts/phase3_verify.py
tests/                  # 15 tests
paper/duobit_est.md
```

## Commands

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 scripts/phase3_verify.py
```

## Next

1. PEFA at 5.25M+ on GPU
2. WikiText-2 / TinyStories
3. Packed bit-serial GEMM
4. W2A4
