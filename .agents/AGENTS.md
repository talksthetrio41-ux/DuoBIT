# DUOBIT-EST Architecture & Developer Guide (AGENTS.md)

This file contains comprehensive context on **DUOBIT-EST** (*Error-compensated Stochastic Transition training*), the codebase structure, mathematical principles, empirical findings, and instructions for AI agents resuming work on this repository.

---

## 1. Executive Summary & Vision

**DUOBIT** is a native 2-bit pretraining framework designed to train Large Language Models (LLMs) **from scratch** directly in 2-bit quantized representations without master weights (0 bytes FP32 weight memory overhead).

Key Breakthroughs:
1. **Zero Master Weight Memory**: Weights are persistently stored as 2-bit codes $z_i \in \{0, 1, 2, 3\}$ (`uint8`) and per-group float scales $s_g$ (`float32`).
2. **Error-Compensated Momentum (ECO)**: Sub-threshold gradient updates are injected back into the first-moment Adam momentum tensor $m_t$ rather than accumulating in FP32 master parameters.
3. **Capacity Scaling Parity**: Scaling model width from $d_{\text{model}} = 256$ (5.25M params) to $d_{\text{model}} = 512$ (19.9M params) achieves **100% identical sequence generation parity** with FP32 baseline while saving **65% weight memory** (6.94 MB vs 20.01 MB).

---

## 2. Mathematical Foundation

### Codebook Representation
Group size $G = 128$. For each group $g$:
$$w_i = s_g \cdot C[z_i], \quad z_i \in \{0, 1, 2, 3\}$$
Symmetric codebook:
$$C = \{-1, -\rho, +\rho, +1\}, \quad \rho \approx 1/3$$

### Closed-Form MSE Group Scale Fitting
Rather than max-abs scales which cause scale amplification loops during pretraining, per-group scale is initialized with closed-form MSE optimal fitting:
$$s_g^* = \frac{\sum_{i \in g} w_i c_i}{\sum_{i \in g} c_i^2 + \epsilon}$$

### Discrete Stochastic Transitions
Candidate continuous update:
$$\tilde{w}_{t+1} = w_t - \eta_{\text{duobit}} \cdot m_{t+1}$$
(or Adam-normalized $\tilde{w}_{t+1} = w_t - \eta_{\text{duobit}} \cdot \frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}$).

Adjacent level transition probability:
$$p_i = \operatorname{clip}\left(\frac{|\tilde{w}_i - w_i|}{\tau \cdot s_g |c_k - c_j|}, 0, 1\right)$$
Balanced Gumbel-TopK rounding selects $K_g = \text{round}\left(\sum_{i \in g} p_i\right)$ codes to transition per group.

### Error Compensation (ECO)
Quantization error $e_{t+1} = \tilde{w}_{t+1} - w_{t+1}$ is rescaled and injected into $m$:
$$\Delta m = -e_{t+1} \cdot \frac{\sqrt{v_{t+1}} + \epsilon}{\eta_{\text{duobit}}}$$
clamped to $[-1.0, 1.0]$ for pretraining stability.

---

## 3. Codebase Structure

```text
workspace/
├── duobit/                     # Core PyTorch Package
│   ├── config.py               # DuobitConfig dataclass
│   ├── layers/
│   │   └── duobit_linear.py    # DuobitLinear module (codes & scales buffers)
│   ├── model/
│   │   ├── transformer.py      # DuobitTransformer decoder (SwiGLU, RoPE, RMSNorm)
│   │   └── utils.py            # RMSNorm, RotaryEmbedding
│   ├── optim/
│   │   └── duobit_adam.py      # DuobitAdam optimizer (ECO, discrete transitions)
│   ├── quantization/
│   │   ├── codebook.py         # SymmetricCodebook & MSE scale fitting
│   │   ├── stochastic.py       # Transition probs & Gumbel-TopK rounding
│   │   └── hadamard.py         # Fast Walsh-Hadamard Transform (FWHT)
│   └── training/
│       ├── metrics.py          # MetricTracker (transition %, perplexity, levels %)
│       └── trainer.py          # Trainer loop (cosine schedule, eval)
├── scripts/                    # Experiments & Benchmarks
│   ├── phase2_ablation.py      # 5-way Phase 2 ablation study
│   ├── phase2_capacity_match.py # Capacity scaling parity benchmark (d=256, 512, 640)
│   ├── phase2_text_benchmark.py # WikiText-2 real text corpus benchmark
│   ├── tune_duobit.py          # Phase 1 hyperparameter tuning script
│   └── run_500step_experiment.py # Initial 500-step baseline run
├── tests/                      # Unit & Integration Tests (12 passing)
│   ├── test_codebook.py        # Codebook levels & quantization roundtrip
│   ├── test_stochastic.py      # Transition probs & Gumbel topK
│   ├── test_optimizer.py       # DuobitAdam single layer & mixed params
│   ├── test_training.py       # End-to-end Transformer pipeline integration
│   └── test_transitions.py    # Phase 2 temperature, scale EMA, SwiGLU tests
├── setup.py                    # Package setup file
└── README.md                   # Project README
```

---

## 4. Empirical Hyperparameters & Findings

1. **Discrete Transition LR (`duobit_lr`)**:
   - Standard FP32 learning rate ($1\text{e-}3$) is too small for discrete level step changes.
   - **`duobit_lr = 8.0e-3`** was found to be optimal across LR sweeps.

2. **Group Scale Stability**:
   - Dynamic recomputation of scales $s_g = \max(|W|)$ during pretraining causes positive feedback scale explosion.
   - **Fix**: Use closed-form MSE scales $s_g^*$ at init and adapt via ultra-slow EMA ($\alpha = 0.01$) every 50 steps.

3. **Transition Dynamics**:
   - Forcing minimum group transitions (`min_transitions_per_group >= 1`) injects random noise that degrades pretraining.
   - **Best Setting**: `min_transitions_per_group = 0`, `transition_temperature = 0.8` for organic step transitions.

4. **Architecture**:
   - **SwiGLU FFN** (`use_swiglu = True`) accelerates pretraining loss reduction by ~0.15 loss points over Squared ReLU.

---

## 5. How to Resume Work & Run Experiments

### Run All Unit Tests
```bash
python3 -m pytest tests/ -v
```

### Run Phase 2 Ablation Study (Synthetic Sequence)
```bash
python3 scripts/phase2_ablation.py
```

### Run Capacity Scaling Benchmark
```bash
python3 scripts/phase2_capacity_match.py
```

### Run Real Text Corpus Benchmark (WikiText-2)
```bash
python3 scripts/phase2_text_benchmark.py
```

---

## 6. Next Engineering Horizons

1. **CUDA Matmul Kernels**: Implement custom CUDA kernels for bit-serial 2-bit matrix multiplication ($a b_0 + b b_1$) for zero-dequantization GEMM speedup on GPUs.
2. **Activation Quantization (W2A8)**: Enable sub-8-bit activation quantization (`activation_bits = 8`) end-to-end.
3. **Scaling to 1B+ Parameters**: Run multi-GPU DDP pretraining experiments at 100M to 1B model scale.
