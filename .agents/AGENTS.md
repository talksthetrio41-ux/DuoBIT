# DUOBIT-EST Architecture & Developer Guide

This file contains context on **DUOBIT-EST** (*Error-compensated Stochastic Transition training*), the codebase structure, mathematical principles, empirical findings, and instructions for AI agents resuming work on this repository.

---

## 1. Executive Summary & Vision

**DUOBIT** is a native 2-bit pretraining framework designed to train Large Language Models (LLMs) from scratch directly in 2-bit quantized representations without master weights.

Key properties:
1. **Zero master-weight memory.** Weights are persistently stored as 2-bit codes $z_i$ (`uint8` values in `{0,1,2,3}`) and per-group float scales $s_g`.
2. **QPEFA.** Sub-threshold updates accumulate in an integer residual (4 or 8 bits) until they cross a codebook gap. Codes actually flip during training.
3. **Compressed Adam.** Block-wise 8-bit first moment and Adafactor-style factored second moment. Linear training state is ~19-21 bits/weight versus 96 for FP32 Adam or latent-weight STE.
4. **Trained group scales (v2).** $\partial\mathcal{L}/\partial s_g = \sum_{i\in g} g_i C[z_i]$ is exact with the codes fixed, so the scales take their own Adam step. They are part of the persistent 2-bit representation and are kept at inference, so this is not a master weight; it costs $3\cdot32/G$ bits/weight of optimizer state.
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

## 4. Empirical hyperparameters (v2 defaults)

- `duobit_lr = 8e-3` (discrete layers), `scale_relative_lr = True`
- `lr = 1e-3` (embeddings, norms)
- `learn_scales = True`, `scale_lr = 1e-3`, `scale_min_frac = 0.05`
- `scale_init = "var"` (the MSE fit shrinks the weight std by 4.4% per layer)
- `qpefa_bits = 8` (4 is close)
- `moment_bits = 8`
- `factored_second_moment = True`
- `pefa_clip = 1.0` (4.0 wasted 12x of the int8 residual grid)
- `duobit_weight_decay = 0.0`, `weight_decay = 0.01` on embeddings/norms
- `scale_update_freq = 0`, `scale_ema_alpha = 0.0` -- the v1 MSE-EMA rescale is
  an algebraic fixed point on dequantized weights and never moved a scale
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

1. Packed bit-serial GEMM (remove the ephemeral dequant tile)
2. W2A4
3. Longer-horizon FineWeb-EDU runs

See the root `AGENTS.md` for the Kaggle dual-T4 workflow and the v2 change list.
