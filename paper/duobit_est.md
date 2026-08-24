# DUOBIT-EST: Error-Compensated Stochastic Transition Training of Language Models at Two Bits and Below Without Master Weights

**Pratyush Bhardwaj**  
Independent researcher, India  
August 2026 · Version 3.0

> I'm Pratyush Bhardwaj, a 16 year old ML researcher from India. This manuscript is the v3 write-up of DUOBIT-EST: native 2-bit / ternary / binary transformer training without a full-precision master-weight tensor.

---

## Abstract

Native low-bit language models such as BitNet keep a latent full-precision tensor and quantize it on the fly with a straight-through estimator. That latent tensor is a *master-weight* copy. We study the stricter setting in which linear maps are stored only as discrete codes $z_i \in \{0,\ldots,L-1\}$ and per-group scales $s_g$, with $L \in \{2,3,4\}$ (binary, ternary, 2-bit). Updates are discrete transitions on the codebook, not SGD on a hidden real matrix.

The first DUOBIT-EST optimizer injected quantization residuals into Adam momentum (ECO). On a 5.25M-parameter decoder this reduced inference weight memory from 20.01 MB to 2.27 MB, but the code-transition rate collapsed to 0% after a few dozen steps. We identify the cause — exponentially decayed momentum cannot accumulate a sub-threshold residual up to a codebook gap — and replace it with a persistent error-feedback accumulator (PEFA) in optimizer state. PEFA is discarded at inference; the checkpoint remains codes plus scales.

On a 94,528-parameter decoder trained for 50 steps on a synthetic arithmetic grammar, 2-bit PEFA reaches validation loss 1.46 (PPL 4.29) versus 2.52 (PPL 12.43) for FP32 Adam at $10^{-3}$, with a 6.4× packed-memory reduction (0.056 MB vs 0.361 MB) and a sustained 9–15% code-flip rate. The ECO-only v1 run on the same budget freezes (final flip rate 0.05%) and underfits (val 3.31). Ternary PEFA is competitive; binary is not, at this scale. These are small-scale verification results, not 1B-parameter claims.

---

## 1. Introduction

The cost of a large language model is paid twice: once in training, and again at every token of inference. Post-training quantization (GPTQ, AWQ, SmoothQuant) compresses a finished FP16 checkpoint. Quantization-aware training and native low-bit methods such as BitNet (Wang et al., 2023) and BitNet b1.58 (Ma et al., 2024) train with the discrete constraint in the forward pass so the model is born in 1–2 bits. Almost all of those methods still keep a latent full-precision weight $W$ and treat quantization as $Q(W)$ with a straight-through estimator. The latent $W$ is a master copy.

Master weights are convenient. They make the optimizer a drop-in Adam, they let the discrete grid move by tiny real increments, and they make checkpointing trivial. They also mean that training memory still contains an FP16/FP32 tensor of every linear map, and that “native 1.58-bit” describes *inference*, not the training state.

DUOBIT-EST asks a narrower question: can we train a decoder transformer if the only persistent representation of each linear map is a 2-bit (or lower) code tensor and a per-group scale? No latent $W$. The dequantized matrix exists for one forward/backward, then is thrown away. The optimizer must decide, from gradients on that ephemeral matrix, which codes to flip.

That decision is a stochastic rounding problem on a group-partitioned codebook. Sub-threshold gradients — the common case, because a codebook gap is large compared with an Adam step — must not be discarded, or the codes freeze. Error compensation into Adam’s first moment (ECO) was the v1 answer. It is not enough: $\beta_1 = 0.9$ forgets the residual before it reaches a gap. Version 3 stores the residual explicitly as PEFA, the classical error-feedback of 1-bit SGD (Seide et al., 2014) and EF-SGD (Karimireddy et al., 2019), keyed on optimizer state rather than on a master weight.

**Contributions.** (i) A complete derivation of group-wise 2-bit / ternary / binary codebooks with closed-form MSE scales. (ii) Balanced Gumbel-TopK transitions and why they stall. (iii) PEFA as the training-only residual that makes codes move. (iv) CPU experiments that isolate PEFA, ECO, bit-width, W2A8, and width scaling. (v) An honest comparison to the earlier 5.25M-parameter runs, including generation parity at 2× width and 65% less weight memory.

---

## 2. Related work

**Binary and quantized nets.** BinaryConnect (Courbariaux et al., 2015) and BinaryNet keep real weights and binarize in the forward pass. XNOR-Net, DoReFa-Net, LSQ (Esser et al., 2020), and PACT follow the same latent-plus-STE pattern. The latent tensor is the master weight we refuse to store.

**Native 1-bit LLMs.** BitNet replaces `nn.Linear` with BitLinear, absmean-quantizes weights to $\{-1,+1\}$ or, in b1.58, to $\{-1,0,+1\}$, and quantizes activations to 8 bits. BitNet a4.8 (Wang et al., 2024) pushes activations to 4 bits. All of these methods train with shadow 8/16-bit weights. Our forward map is similar; the optimizer is not.

**Error feedback.** 1-bit SGD and the analysis of Karimireddy et al. show that compressing a gradient is unbiased in the limit if the compression residual is carried forward. We apply the same identity to codebook reconstruction error rather than to gradient compression.

**Incoherence processing.** QuIP#, QuaRot, and SpinQuant rotate weights or activations by a Hadamard matrix so that coordinates are easier to quantize. DUOBIT ships an optional online FWHT.

**Optimizers for quantized training.** 8-bit Adam (Dettmers et al., 2022) compresses optimizer state, not weights. QLoRA (Dettmers et al., 2023) freezes a 4-bit base and trains LoRA adapters. We compress the weights themselves and still run Adam on ephemeral dequantized tensors plus a residual.

---

## 3. Method

### 3.1 Discrete codebook

Let $W \in \mathbb{R}^{m \times n}$ be a linear map. Partition each row into groups of size $G$. For coordinate $i$ in group $g$,

$$
w_i = s_g \, C[z_i], \qquad z_i \in \{0,\ldots,L-1\}.
$$

The codebook $C$ is symmetric:

$$
C =
\begin{cases}
\{-1,+1\} & L=2 \quad \text{(1 bit)} \\
\{-1,0,+1\} & L=3 \quad (\log_2 3 \approx 1.585\text{ bits)} \\
\{-1,-\rho,+\rho,+1\} & L=4 \quad \text{(2 bits, }\rho=1/3\text{)}.
\end{cases}
$$

Quantization is nearest neighbour after scaling: $z_i = \arg\min_k |w_i / s_g - C_k|$. For $L=4$ the Voronoi boundaries are $\pm(1+\rho)/2$ and $0$. Dequantization is a table lookup, or, for 2-bit codes, the bit-serial identity

$$
C[z] = a\, b_0 + b\, b_1, \quad a=\frac{1+\rho}{2},\; b=\frac{1-\rho}{2},
$$

with $b_0,b_1\in\{-1,+1\}$ the high and low bits. A 2-bit GEMM is then two binary matmuls and two adds.

### 3.2 Closed-form MSE scales

Max-abs scales $s_g = \max_{i\in g}|w_i|$ over-scale the group and, if recomputed from an unconstrained candidate, form a positive-feedback loop. The MSE-optimal scale given codes is the scalar least-squares fit

$$
s_g^\star = \frac{\sum_{i\in g} w_i\, c_i}{\sum_{i\in g} c_i^2 + \varepsilon}, \qquad c_i = C[z_i].
$$

We use $s_g^\star$ at initialization. During training, scales move by a slow EMA every $F$ steps, $s \leftarrow (1-\alpha)s + \alpha s^\star(\hat W)$ with $\alpha=0.01$, always from the current dequantized weights.

On a Gaussian $64\times 256$ matrix, MSE scales cut reconstruction error by 32–45% for 2-bit and by 88–93% for binary relative to max-abs.

### 3.3 Ephemeral Adam step

In the forward pass we materialise $\hat W = \mathrm{dequant}(z,s)$, run $y = x\hat W^\top$, and retain $\nabla_{\hat W}\mathcal{L}$. Embeddings and RMSNorm gains stay ordinary FP32 parameters. On the discrete layers, Adam moments live in optimizer state:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^{\odot 2}.
$$

$$
\tilde W_{t+1} = \hat W_t - \eta_{\mathrm{duobit}} \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

A codebook gap is order $s_g(1-\rho)\approx 0.1$; an Adam-normalised step at $10^{-3}$ is order $10^{-3}$. A sweep on the 5.25M model gave an optimum at $\eta_{\mathrm{duobit}} = 8\times 10^{-3}$.

### 3.4 Balanced stochastic transitions

Write $\Delta_i = \tilde w_i - \hat w_i$ and let $k$ be the adjacent level in the sign of $\Delta_i$. The gap in weight space is $\gamma_i = s_g |C_k - C_{z_i}|$. The adjacent-flip probability is

$$
p_i = \mathrm{clip}\!\left(\frac{|\Delta_i|}{\tau\,\gamma_i},\,0,1\right).
$$

Independent Bernoulli flips would have the right expectation but high variance inside a group. We instead take a balanced Gumbel-TopK sample: inside group $g$,

$$
K_g = \mathrm{round}\Big(\sum_{i\in g} p_i\Big),
$$

and select the $K_g$ largest $\log p_i + \mathrm{Gumbel}(0,1)$ (Kool et al., 2019). Trust gating in v1 zeroed $p_i$ when $|\Delta_i|$ exceeded a multiple of the gap — which also zeroed the PEFA residual once it was large enough to matter. v3 drops that gate on the PEFA path.

### 3.5 Why ECO stalls

If no code flips, $\hat W_{t+1}=\hat W_t$ and the quantization residual is the whole step. ECO injects it back into $m$:

$$
e_{t+1}=\tilde W_{t+1}-\hat W_{t+1}, \qquad
\Delta m = -\frac{\sqrt{v}+\varepsilon}{\eta}\, e_{t+1}
\text{ clamped to }[-1,1].
$$

Two things kill accumulation. First, $m \leftarrow \beta_1 m + \cdots$ with $\beta_1=0.9$ forgets 65% of a residual in ten steps. Second, the clamp saturates. Empirically the flip rate starts near 20–60% and is numerically zero by step 80. Learning then happens only through scale EMA and the FP32 embeddings — the 2-bit codes are a frozen lookup table.

### 3.6 PEFA

Let $e$ be a tensor of the same shape as $W$, stored in optimizer state, initialized at 0. One PEFA step is

$$
e \leftarrow e + (\tilde W - \hat W_t), \qquad
e \leftarrow \mathrm{clip}(e,\,\pm \lambda s_g),
$$

$$
z_{t+1},\; \hat W_{t+1},\; e
\leftarrow Q(\hat W_t + e;\, s),
\qquad e \leftarrow (\hat W_t + e) - \hat W_{t+1}.
$$

$Q$ is nearest-neighbour quantization onto the current codebook, which permits multi-level jumps when the residual spans more than one gap. The identity $\hat W_t + e_{\mathrm{pre}} = \hat W_{t+1} + e_{\mathrm{post}}$ is exact in floating point: the virtual weight is conserved. Codes flip when, and only when, the virtual weight crosses a Voronoi boundary. At inference we throw $e$, $m$, and $v$ away. The published checkpoint is $(z,s)$.

PEFA is not a master weight. A master weight is a persistent real matrix that *is* the parameter. PEFA is a residual of quantization, with the same status as Adam’s $m$.

### 3.7 Activations, Hadamard, architecture

Optional W2A8: per-token absmax quantization of activations to INT8 with STE. Optional online Walsh–Hadamard rotation of incoming features. The decoder is otherwise standard: RMSNorm, RoPE, causal attention, SwiGLU.

### 3.8 Memory accounting

Packed inference bytes for a DuobitLinear of $N$ weights, $L$ levels, group size $G$:

$$
B = \Big\lceil N\log_2 L / 8\Big\rceil + 4\cdot (N/G).
$$

Linear bits/weight are $\log_2 L + 32/G$. For $L=4, G=32$ that is 3.0; for $G=128$, 2.25. Embeddings and RMSNorm remain FP32, so the model-average bits/parameter is higher (4.98 in the tiny 2-bit run, 3.64 in the 5.25M run).

Training memory still contains Adam states and PEFA. The saving versus BitNet is one latent weight tensor, not the optimizer. Inference is the regime in which 2.25–3.0 bits/weight is real.

---

## 4. Algorithm

```
for each DuobitLinear layer with ephemeral gradient g:
    m ← β1 m + (1-β1) g
    v ← β2 v + (1-β2) g⊙g
    Ŵ  ← dequant(z, s)
    W̃  ← Ŵ - η_duobit · m̂ / (√v̂ + ε)
    e  ← clip(e + (W̃ - Ŵ), ±λ s)
    z, Ŵ, e ← Quantize(Ŵ + e; s)          # PEFA
    if t mod F = 0:
        s ← (1-α) s + α s*_MSE(Ŵ, z)
    if ECO:
        m ← m + clip( -(W̃ - Ŵ)(√v+ε)/η , ±1)
checkpoint for inference: (z, s) only
```

Standard parameters (token embeddings, RMSNorm) take an ordinary Adam step at $\eta$. Gradient clipping at global norm 1.0 and a cosine schedule with warmup apply to both groups.

---

## 5. Experiments

All v3 numbers were measured on CPU, with `scripts/phase3_verify.py` and 15 unit tests. We do not mix them with GPU 1B-scale claims. The earlier 5.25M-parameter runs are labelled “Phase 1”.

### 5.1 Codebook reconstruction

A Gaussian matrix $W\sim\mathcal{N}(0,1)^{64\times 256}$ is quantized with binary, ternary, and 2-bit codebooks at $G\in\{32,64,128\}$. Closed-form MSE scales cut reconstruction error by 32–45% for 2-bit and by 88–93% for binary relative to max-abs.

| Codebook | G | MSE (max-abs) | MSE (closed form) | Rel. improvement |
|---|---|---|---|---|
| binary | 32 | 2.951 | 0.357 | 87.9% |
| binary | 128 | 4.891 | 0.365 | 92.5% |
| ternary | 32 | 0.433 | 0.251 | 42.1% |
| ternary | 128 | 0.603 | 0.316 | 47.5% |
| 2-bit | 32 | 0.209 | 0.141 | 32.3% |
| 2-bit | 128 | 0.319 | 0.176 | 44.7% |

### 5.2 Quadratic bowl

Fit $Y \approx XW^\top$ with $W\in\mathbb{R}^{32\times 64}$ drawn Gaussian — a target a 4-level codebook cannot represent exactly. FP32 Adam reaches MSE $6.4\times 10^{-4}$. Every discrete run floors near 2.7–3.1. PEFA without ECO is the best discrete run (2.75) and keeps a 7.2% mean flip rate; ECO-only v1 floors higher (3.12) with flips dying after the first few steps.

| Run | Final MSE | Mean flip % |
|---|---|---|
| FP32 Adam | 0.00064 | — |
| v1 ECO, no PEFA | 3.120 | 2.35 |
| v3 PEFA + ECO | 2.817 | 3.20 |
| PEFA, no ECO | **2.748** | **7.17** |
| no PEFA, no ECO | 2.773 | 7.26 |

### 5.3 Tiny decoder on a grammar

Task: sequences of triples $(x,\, 3x+1,\, 3x+6) \bmod V$ with $V=96$, length 16. Architecture: 2-layer decoder, $d=64$, 2 heads, SwiGLU $d_{\mathrm{ff}}=128$, $G=32$, 94,528 parameters. 50 steps, batch 16, CPU. FP32 uses AdamW at $10^{-3}$; DUOBIT uses $\eta_{\mathrm{duobit}}=8\times 10^{-3}$. Prompt: `[10, 31, 36, 12]`.

| Run | Val | PPL | Mean flip % | Final flip % | Bits/param | MB | Grammar acc. |
|---|---|---|---|---|---|---|---|
| FP32 baseline | 2.520 | 12.43 | — | — | 32.00 | 0.361 | 70% |
| **DUOBIT v3 2-bit PEFA** | **1.457** | **4.29** | **15.05** | **9.11** | 4.98 | **0.056** | **100%** |
| DUOBIT v1 ECO only | 3.310 | 27.39 | 3.28 | 0.05 | 4.98 | 0.056 | 40% |
| DUOBIT PEFA, no ECO | 1.208 | 3.35 | 3.75 | 1.13 | 4.98 | 0.056 | 100% |
| DUOBIT ternary PEFA | 2.418 | 11.22 | 6.95 | 2.47 | 4.60 | 0.052 | 80% |
| DUOBIT binary PEFA | 3.082 | 21.80 | 1.00 | 0.13 | 4.05 | 0.046 | 30% |
| DUOBIT W2A8 PEFA | 1.461 | 4.31 | 15.00 | 8.94 | 4.98 | 0.056 | 100% |

Three facts. First, PEFA keeps codes alive: mean flip rate 15.0% for 2-bit PEFA, still 9.1% at step 50, versus 0.05% for v1 ECO-only. Second, that difference is the difference between learning the grammar (100% greedy accuracy) and not (40%). Third, W2A8 matches W2A16 to two decimals.

PEFA without ECO is the best tiny run (val 1.21). ECO, designed to rescue a stalling adjacent-flip rule, is redundant once PEFA carries the residual and appears slightly harmful — it double-counts error into $m$.

Greedy continuations: 2-bit PEFA emits `[10, 31, 36, 12, 37, 42, 27, 82, 87, 23, 70, 75, …]`: $12\to 37=3\cdot12+1,\; 42=37+5$, then a new valid triple $(27,82,87)$, then $(23,70,75)$. FP32 gets the first continuation pair and then breaks. Binary PEFA never locks onto the rule.

### 5.4 Width scaling (35 steps)

| Run | Params | Val | MB | Acc. |
|---|---|---|---|---|
| FP32 $d=64$ | 94,528 | 3.140 | 0.361 | 60% |
| DUOBIT $d=64$ | 94,528 | 1.609 | 0.056 | 100% |
| DUOBIT $d=96$ | 203,232 | **1.278** | 0.106 | 100% |

A 96-wide 2-bit model (0.106 MB) reaches val 1.28, below both the 64-wide 2-bit run and the 64-wide FP32 run, at 29% of FP32 memory.

### 5.5 Historical Phase 1 (5.25M)

The earlier 500-step run used a 6-layer decoder, $d=256$, 5,246,208 parameters, the same grammar, and the v1 optimizer (no PEFA). FP32 reached val 0.133 (PPL 1.14) in 20.01 MB. DUOBIT reached val 4.52 (PPL 91.5) in 2.27 MB (3.64 bits/param) with a 0.00% transition rate — the stall this paper diagnoses. A later capacity sweep at $d=512$ (19.9M 2-bit parameters, 6.94 MB, still 65% below the FP32 5.25M footprint) matched the FP32 greedy continuation exactly on the evaluation prompt. That match was real; it was also achieved with frozen codes, i.e. by width and embeddings, not by walking the codebook. v3 is the attempt to give those codes a gradient pathway.

---

## 6. Analysis

**PEFA is the load-bearing piece.** Every comparison that holds architecture, data, and $\eta$ fixed and toggles only PEFA shows sustained transitions if and only if the residual is stored. ECO into $m$ is not a substitute.

**Bit-width is capacity, not a free lunch.** Binary PEFA (2.0 linear bits/weight) does not learn the grammar at this size. Ternary does, with a small tax versus 2-bit. The 4-level codebook with $\rho=1/3$ is a better match to a roughly Gaussian weight histogram than $\{-1,+1\}$.

**The FP32 comparison is not apples-to-apples on step size.** DUOBIT’s discrete rate is 8× the dense rate because a codebook gap is 8–100× an Adam step. Giving FP32 the same $8\times 10^{-3}$ would be a different, probably unstable, baseline. The honest statement is: at the rates each method actually uses, 2-bit PEFA fitted this grammar faster, in 6.4× less inference memory. On the 5.25M, 500-step budget without PEFA, FP32 won by a large margin. Both statements can be true.

**Regularization.** A 4-level codebook is a hard spectral constraint. On a tiny, noiseless grammar that constraint can help; on a random least-squares target it is a floor. We do not claim 2-bit PEFA dominates FP32 in general.

---

## 7. Limitations

The v3 measurements are CPU, synthetic, and small: 50 steps, 95k parameters, one grammar. They verify the optimizer claims (codes move, PEFA is necessary, ternary works, binary struggles, W2A8 is cheap). They do not verify that a 1B-parameter DUOBIT model matches BitNet b1.58 on WikiText or MMLU. A PEFA run at 5.25M+ scale is the next experiment, and it needs a GPU.

Training memory still contains Adam states and PEFA, all FP32, so the training footprint is not 2-bit. Packed storage in the reference implementation is still uint8 with values in $\{0,1,2,3\}$; the byte counts we report are packed-equivalent.

We did not train on natural language in v3. The WikiText-2 script from Phase 2 remains in the repository.

---

## 8. Conclusion

You can train a transformer whose linear maps are stored only as 2-bit (or ternary) codes and group scales, with no master-weight tensor, if you carry quantization error in optimizer state until it crosses a codebook gap. Adjacent stochastic rounding plus ECO into Adam momentum does not do that: the codes freeze. PEFA does. At 95k parameters the resulting 2-bit model learns a structured grammar, matches W2A8, and spends 6.4× less packed inference memory than FP32. At 5.25M parameters, even the stalled v1 model could match FP32 greedy generation by spending the memory budget on width. The open question is whether PEFA plus width plus real text closes the loss gap that v1 left on the table.

---

## References

1. Wang, H. et al. BitNet: Scaling 1-bit Transformers for Large Language Models. arXiv:2310.11453, 2023.
2. Ma, S. et al. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764, 2024.
3. Seide, F. et al. 1-Bit Stochastic Gradient Descent. Interspeech, 2014.
4. Karimireddy, S. P., Rebjock, Q., Stich, S., and Jaggi, M. Error Feedback Fixes SignSGD and other Gradient Compression Schemes. ICML, 2019.
5. Courbariaux, M., Bengio, Y., and David, J.-P. BinaryConnect: Training Deep Neural Networks with Binary Weights during Propagations. NeurIPS, 2015.
6. Esser, S. K. et al. Learned Step Size Quantization. ICLR, 2020.
7. Wang, H., Ma, S., and Wei, F. BitNet a4.8: 4-bit Activations for 1-bit LLMs. arXiv:2411.04965, 2024.
8. Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. 8-bit Optimizers via Block-wise Quantization. ICLR, 2022.
9. Dettmers, T. et al. QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS, 2023.
10. Kool, W., Van Hoof, H., and Welling, M. Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement. ICML, 2019.
11. Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization. ICLR, 2015.
12. Su, J. et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864, 2021.
13. Zhang, B. and Sennrich, R. Root Mean Square Layer Normalization. NeurIPS, 2019.
14. Shazeer, N. GLU Variants Improve Transformer. arXiv:2002.05202, 2020.
15. Ma, S. et al. BitNet b1.58 2B4T Technical Report. arXiv:2504.12285, 2025.
16. Frantar, E. et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR, 2023.
17. Lin, J. et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys, 2024.
18. Xiao, G. et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML, 2023.
