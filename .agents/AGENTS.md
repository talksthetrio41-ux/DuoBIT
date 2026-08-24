# DUOBIT-EST Architecture & Developer Guide

This file contains context on **DUOBIT-EST** (*Error-compensated Stochastic Transition training*), the codebase structure, mathematical principles, empirical findings, and instructions for AI agents resuming work on this repository.

---

## 1. Executive Summary & Vision

**DUOBIT** is a native 2-bit pretraining framework designed to train Large Language Models (LLMs) from scratch directly in 2-bit quantized representations without master weights.

Key properties:
1. **Zero master-weight memory.** Weights are persistently stored as 2-bit codes $z_i$ (`uint8` values in `{0,1,2,3}`) and per-group float scales $s_g`.
2. **QPEFA.** Sub-threshold updates accumulate in an integer residual (4 or 8 bits) until they cross a codebook gap. Codes actually flip during training.
3. **Compressed Adam.** Block-wise 8-bit first moment and Adafactor-style factored second moment. Linear training state is ~20 bits/weight versus 96 for FP32 Adam or latent-weight STE.
4. **Training and inference both shrink.** Tiny-LM verification: 0.288 MB train / 0.056 MB infer versus 1.082 / 0.361 for FP32, at better grammar loss.

---

## 2. Mathematical Foundation

### Codebook
Group size $G = 128$ (32 on the tiny decoder). For each group $g$:
$$w_i = s_g \cdot C[z_i]$$
$$C = \{-1, -\rho, +\rho, +1\}, \quad \rho \approx 1/3$$
(or ternary $\{-1,0,+1\}$, or binary $\{-1,+1\}$).

### MSE group scale
$$s_g^* = \frac{\sum_{i \in g} w_i c_i}{\sum_{i \in g} c_i^2 + \epsilon}$$

### QPEFA
$$e \leftarrow \mathrm{clip}(e + (\tilde W - \hat W), \pm\lambda s_g)$$
$$z, \hat W, e \leftarrow Q(\hat W + e)$$
Store $e$ as stochastic-rounded $q = \mathrm{SR}(e \cdot q_{\max} / (\lambda s_g))$ in 4 or 8 bits.

### Compressed Adam
- $m$: block-wise int8 + per-block FP32 absmax scale
- $v$: factored row/col second moments, $O(m+n)$

---

## 3. Codebase Structure

```text
duobit/
  config.py
  layers/duobit_linear.py
  layers/ste_linear.py
  model/transformer.py
  model/utils.py
  optim/duobit_adam.py
  quantization/codebook.py, qpefa.py, blockwise.py, stochastic.py, hadamard.py, memory.py
  training/metrics.py, trainer.py
scripts/run_experiments.py
tests/
paper/duobit_est.md
```

---

## 4. Empirical hyperparameters

- `duobit_lr = 8e-3` (discrete layers)
- `lr = 1e-3` (embeddings, norms)
- `qpefa_bits = 8` (4 is close)
- `moment_bits = 8`
- `factored_second_moment = True`
- `pefa_clip = 4.0` (2.0 for 4-bit residual)
- `scale_ema_alpha = 0.01`, `scale_update_freq = 50`
- `enable_error_compensation = False` (injecting residual into $m$ is redundant once QPEFA carries it)
- `min_transitions_per_group = 0`
- `use_swiglu = True`

---

## 5. Commands

```bash
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 scripts/run_experiments.py
```

---

## 6. Next

1. GPU run at 5.25M+
2. WikiText-2 / TinyStories
3. Packed bit-serial GEMM
4. W2A4
