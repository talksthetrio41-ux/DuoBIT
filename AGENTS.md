# DUOBIT-EST: Architecture and Developer Guide

**Author:** Pratyush Bhardwaj  
**Version:** 1.0.0 (QPEFA + compressed Adam)

Native 2-bit / ternary / binary pretraining **without master weights**. Linear maps persist as discrete codes $z$ and group scales $s$. The training residual is an integer QPEFA tensor. Adam $m$ is block-wise int8. Adam $v$ is factored ($O(m+n)$).

## Headline

On a 94,528-parameter decoder (50 steps, CPU):

- 2-bit QPEFA-8 val 1.23 / PPL 3.41 / 100% grammar / 0.056 MB infer / 0.288 MB train / 20.3 linear train bits/wt
- FP32 Adam val 2.52 / PPL 12.43 / 70% acc / 0.361 MB infer / 1.082 MB train / 96 bits/wt
- Latent-weight STE val 2.83 / 40% acc / same memory as FP32
- QPEFA-4 val 1.25 / 16.3 train bits/wt

Paper: `paper/duobit_est.md`

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

## Next

1. QPEFA at 5.25M+ on GPU
2. WikiText-2 / TinyStories
3. Packed bit-serial GEMM (remove the ephemeral dequant tile)
4. W2A4
