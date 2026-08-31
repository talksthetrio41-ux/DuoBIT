# DUOBIT-EST: Error-Compensated Stochastic Transition Training of Language Models at Two Bits and Below Without Master Weights

**Pratyush Bhardwaj**  
Independent researcher, India  
August 2026

> I'm Pratyush Bhardwaj, a 16 year old ML researcher from India. This manuscript describes DUOBIT-EST: native 2-bit, ternary, and binary transformer training without a full-precision master-weight tensor, with a quantized error-feedback residual and compressed optimizer state so that both training and inference memory drop.

---

## Abstract

Native low-bit language models such as BitNet keep a latent full-precision tensor and quantize it on the fly with a straight-through estimator. That latent tensor is a master-weight copy. Training memory is then the same order as ordinary Adam: the weight, a first moment, and a second moment, typically 12 bytes per parameter. "1.58-bit" describes the inference checkpoint, not the training state.

DUOBIT-EST trains a decoder transformer whose linear maps are stored only as discrete codes $z_i \in \{0,\ldots,L-1\}$ and per-group scales $s_g$, with $L \in \{2,3,4\}$ (binary, ternary, 2-bit). There is no latent $W$. Sub-threshold Adam steps accumulate in a quantized persistent error-feedback accumulator (QPEFA), an integer residual relative to the group scale, until the virtual weight crosses a codebook Voronoi boundary. First-moment tensors are stored block-wise in 8 bits. The second moment is factored into row and column statistics, $O(m+n)$ rather than $O(mn)$. At inference, QPEFA and the moments are discarded. The published checkpoint is $(z, s)$.

On a 94,528-parameter decoder trained for 50 steps on a synthetic arithmetic grammar, 2-bit QPEFA-8 reaches validation loss 1.23 (PPL 3.41) versus 2.52 (PPL 12.43) for FP32 Adam and 2.83 (PPL 16.96) for a latent-weight STE baseline that follows the BitNet training pattern. Packed inference memory is 0.056 MB versus 0.361 MB (6.4×). Persistent training memory is 0.288 MB versus 1.082 MB (3.8×). Linear maps train at 20.3 bits per weight versus 96. A 4-bit residual is close (val 1.25, 16.3 train bits/weight). Ternary matches the grammar. Binary pays a quality tax (val 1.85) but still fits the rule. These are small-scale verification results, not 1B-parameter claims.

---

## 1. Introduction

The cost of a large language model is paid twice: once in training, and again at every token of inference. Post-training quantization (GPTQ, AWQ, SmoothQuant) compresses a finished FP16 checkpoint. Quantization-aware training and native low-bit methods such as BitNet (Wang et al., 2023) and BitNet b1.58 (Ma et al., 2024) train with the discrete constraint in the forward pass so the model is born in 1–2 bits. Almost all of those methods still keep a latent full-precision weight $W$ and treat quantization as $Q(W)$ with a straight-through estimator. The latent $W$ is a master copy.

Master weights are convenient. They make the optimizer a drop-in Adam, they let the discrete grid move by tiny real increments, and they make checkpointing trivial. They also mean that training memory still contains an FP16 or FP32 tensor of every linear map, plus Adam's $m$ and $v$. A method that is 1.58-bit at inference can still be 12-byte-per-weight at training time.

DUOBIT-EST asks a stricter question: can we train a decoder transformer if the only persistent representation of each linear map is a 2-bit (or lower) code tensor, a per-group scale, and a *compressed* training residual? No latent $W$. The dequantized matrix exists for one forward and backward, then is thrown away. The optimizer must decide, from gradients on that ephemeral matrix, which codes to flip, and it must do so without storing three full-precision tensors per weight.

That decision is a stochastic rounding problem on a group-partitioned codebook. Sub-threshold gradients, the common case because a codebook gap is large compared with an Adam step, must not be discarded, or the codes freeze. Storing the residual in FP32 would reintroduce a master-sized tensor and cancel the training-memory win. The move in this paper is to quantize the residual itself (QPEFA) and to compress Adam state: 8-bit block-wise first moments and a factored second moment.

**Contributions.** (i) A complete derivation of group-wise 2-bit, ternary, and binary codebooks with closed-form MSE scales. (ii) QPEFA: an integer error-feedback residual, stochastic-rounded relative to the group scale, that makes codes move without an FP32 residual tensor. (iii) Compressed Adam for discrete layers: block-wise 8-bit $m$ and Adafactor-style factored $v$. (iv) Memory accounting that reports persistent training state and packed inference state against FP32 Adam and latent-weight STE. (v) CPU experiments that isolate residual bit-width, codebook width, W2A8, and model width.

---

## 2. Related work

**Binary and quantized nets.** BinaryConnect (Courbariaux et al., 2015) and BinaryNet keep real weights and binarize in the forward pass. XNOR-Net, DoReFa-Net, LSQ (Esser et al., 2020), and PACT follow the same latent-plus-STE pattern. The latent tensor is the master weight we refuse to store.

**Native 1-bit LLMs.** BitNet replaces `nn.Linear` with BitLinear, absmean-quantizes weights to $\{-1,+1\}$ or, in b1.58, to $\{-1,0,+1\}$, and quantizes activations to 8 bits. BitNet a4.8 (Wang et al., 2024) pushes activations to 4 bits. All of these methods train with shadow 8/16-bit weights. Our forward map is similar. The optimizer and the persistent state are not.

**Error feedback.** 1-bit SGD (Seide et al., 2014) and the analysis of Karimireddy et al. (2019) show that compressing a gradient is unbiased in the limit if the compression residual is carried forward. We apply the same identity to codebook reconstruction error rather than to gradient compression, then store that residual in 4 or 8 bits.

**Incoherence processing.** QuIP#, QuaRot, and SpinQuant rotate weights or activations by a Hadamard matrix so that coordinates are easier to quantize. DUOBIT ships an optional online FWHT.

**Optimizers for quantized training.** 8-bit Adam (Dettmers et al., 2022) compresses optimizer state, not weights. Adafactor (Shazeer and Stern, 2018) factors second moments. QLoRA (Dettmers et al., 2023) freezes a 4-bit base and trains LoRA adapters. We compress the weights themselves *and* the residual *and* the moments.

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

On a Gaussian $64\times 256$ matrix, MSE scales cut reconstruction error by 32–45% for 2-bit and by 88–93% for binary relative to max-abs.

**The EMA scale update is a fixed point (corrected).** Earlier versions of this
work used $s_g^\star$ at initialization and then moved the scales by a slow EMA
every $F$ steps, $s \leftarrow (1-\alpha)s + \alpha s^\star(\hat W)$, always
recomputed from the current *dequantized* weights. That update cannot move
anything. Substituting $\hat w_i = s_g c_i$ into the least-squares fit gives

$$
s^\star(\hat W)_g
= \frac{\sum_i (s_g c_i) c_i}{\sum_i c_i^2 + \varepsilon}
= s_g \cdot \frac{\sum_i c_i^2}{\sum_i c_i^2 + \varepsilon}
\;\approx\; s_g ,
$$

so the EMA target *is* the current scale and the recursion is stationary. In a
3000-step FineWeb-EDU run the measured mean scale moved by $2\times10^{-7}$
relative between step 1 and step 3000. The consequence is severe: the group
scale is the method's only continuous degree of freedom, and it was frozen at
its initialization for the entire run. Section 3.9 replaces the EMA with the
exact scale gradient.

**Variance-preserving initialization.** The MSE fit is also the wrong objective
at initialization. It minimises $\lVert w - s\,C[z] \rVert$ for the particular
random draw, and in doing so it systematically shrinks the weight: measured on
`kaiming_uniform`, $\operatorname{std}(s^\star C[z]) = 0.956\,
\operatorname{std}(w)$ at every fan-in. Composed over the $2L$ linear maps of an
$L$-layer decoder that compounds to $0.956^{2L}$ — $0.49\times$ the activation
scale at $L=8$, $0.24\times$ at $L=16$ — which weakens every residual branch
against its skip path and shrinks the logits before a single step is taken. For
a random init what matters is the forward variance, not the particular draw, so
we rescale each group to reproduce its intended standard deviation:

$$
s_g^{\mathrm{var}} = s_g^\star \cdot
\operatorname{clip}\!\left(
\frac{\operatorname{std}_{i \in g}(w_i)}{\operatorname{std}_{i \in g}(s_g^\star c_i)},
\; 0.5,\; 2 \right).
$$

### 3.3 Ephemeral Adam step

In the forward pass we materialise $\hat W = \mathrm{dequant}(z,s)$, run $y = x\hat W^\top$, and retain $\nabla_{\hat W}\mathcal{L}$. Embeddings and RMSNorm gains stay ordinary FP32 parameters. On the discrete layers, Adam moments live in compressed optimizer state. The candidate update is the usual bias-corrected Adam step on the ephemeral matrix:

$$
\tilde W_{t+1} = \hat W_t - \eta_{\mathrm{duobit}} \frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

A codebook gap is order $s_g(1-\rho)\approx 0.1$; an Adam-normalised step at $10^{-3}$ is order $10^{-3}$. Discrete layers therefore use $\eta_{\mathrm{duobit}} = 8\times 10^{-3}$.

### 3.4 Why a residual is required

If no code flips, $\hat W_{t+1}=\hat W_t$ and the quantization residual is the whole step. Injecting that residual into Adam's first moment does not accumulate: $m \leftarrow \beta_1 m + \cdots$ with $\beta_1=0.9$ forgets 65% of a residual in ten steps, and a clamp saturates. The codes freeze. Learning then happens only through scale EMA and the FP32 embeddings.

The classical fix is error feedback (Seide et al., 2014; Karimireddy et al., 2019): carry $e$ explicitly so the virtual weight $\hat W + e$ is conserved through quantization. An FP32 $e$ of shape $W$ is a master-weight tensor by another name. Training memory does not drop.

### 3.5 QPEFA

The residual is bounded. After nearest-neighbour quantization it lives inside a Voronoi cell, and we clip it to $\pm \lambda s_g$ to allow a few multi-level jumps without letting an outlier dominate. A bounded tensor with a known per-group scale is a good candidate for integer quantization.

The clip $\lambda$ sets the resolution, and the earlier default wasted most of
it. One integer step of the residual is $\lambda s_g / q_{\max}$, i.e. a
fraction $\lambda / (q_{\max}(1-\rho))$ of a codebook gap. At $\lambda = 4$ and
$b = 8$ that is 4.7% of a gap; at $\lambda = 1$ it is 1.2%. Since the whole
purpose of the accumulator is to hold updates *smaller* than a gap until they
sum to one, resolution is the quantity that matters, and $\lambda$ should be
just large enough to hold the accumulated step. We use $\lambda = 1$.

Let $q_{\max} = 2^{b-1}-1$. Store a signed integer $q$ of $b$ bits (default $b=8$, ablation $b=4$) and reconstruct

$$
e = \frac{q}{q_{\max}} \cdot \lambda \, s_g.
$$

One QPEFA step is:

$$
e \leftarrow \mathrm{clip}\big(e + (\tilde W - \hat W_t),\, \pm \lambda s_g\big),
$$

$$
z_{t+1},\; \hat W_{t+1},\; e_{\mathrm{true}}
\leftarrow Q(\hat W_t + e;\, s),
$$

$$
q \leftarrow \mathrm{SR}\Big(e_{\mathrm{true}} \cdot \frac{q_{\max}}{\lambda s_g}\Big),
$$

where $\mathrm{SR}$ is stochastic rounding, $q = \lfloor x + U(0,1)\rfloor$, which is unbiased: $\mathbb{E}[q] = x$ inside the representable range. $Q$ is nearest-neighbour quantization onto the current codebook and permits multi-level jumps when the residual spans more than one gap.

QPEFA is not a master weight. A master weight is a persistent real matrix that *is* the parameter. QPEFA is a quantization residual with the same status as Adam's $m$, stored in 4 or 8 bits rather than 32.

### 3.6 Compressed Adam

**First moment.** Following 8-bit Adam (Dettmers et al., 2022), $m$ is split into blocks of size $B$ (default 128). Each block stores an FP32 absmax scale and signed integer values. The tensor is dequantized only for the optimizer step, updated, and immediately re-quantized. Persistent state is $b$ bits per weight plus 32 bits per block.

**Second moment.** A per-weight $v$ is the most expensive remaining tensor. We factor it in the Adafactor style (Shazeer and Stern, 2018). For $G \in \mathbb{R}^{m \times n}$,

$$
r \leftarrow \beta_2 r + (1-\beta_2)\,\mathrm{mean}_{\mathrm{col}}(G^{\odot 2}),
\quad
c \leftarrow \beta_2 c + (1-\beta_2)\,\mathrm{mean}_{\mathrm{row}}(G^{\odot 2}),
$$

$$
\hat v_{ij} = \frac{r_i \, c_j}{\mathrm{mean}(r)+\varepsilon}.
$$

Storage is $4(m+n)$ bytes, not $4mn$. For discrete code flips we only need a reasonable adaptive rate, not a high-fidelity per-coordinate second moment.

Embeddings and RMSNorm stay on ordinary FP32 Adam. They are a small fraction of a large decoder. They are not a small fraction of a 95k-parameter toy, which is why we report both linear-only and full-model training bytes.

### 3.7 Activations, Hadamard, architecture

Optional W2A8: per-token absmax quantization of activations to INT8 with STE. Optional online Walsh–Hadamard rotation of incoming features. The decoder is otherwise standard: RMSNorm, RoPE, causal attention, SwiGLU.

### 3.8 Trainable group scales

The codes carry the *shape* of a group and the scale carries its *magnitude*.
Once the EMA is recognised as a fixed point (Section 3.2), the magnitude has no
learning signal at all, and a 2-bit code with a frozen scale cannot represent a
weight whose magnitude needs to change. The fix does not require a master
weight, because the scale is already part of the persistent representation: it
is stored, kept at inference, and already charged at $32/G$ bits per weight.

Holding the codes fixed, $w_i = s_g C[z_i]$ is linear in $s_g$, so the gradient
is exact and costs one reduction over each group:

$$
\frac{\partial \mathcal{L}}{\partial s_g}
= \sum_{i \in g} \frac{\partial \mathcal{L}}{\partial w_i} \, C[z_i].
$$

We take an ordinary Adam step on $s$ with its own rate, in relative form so a
group's magnitude moves by a bounded fraction per step regardless of fan-in,
and floor it against its initialization so a group cannot collapse to zero and
become unrecoverable:

$$
s_g \leftarrow \max\!\left(
s_g\left(1 - \eta_s \frac{\hat m^{(s)}_g}{\sqrt{\hat v^{(s)}_g}+\varepsilon}\right),
\; \kappa\, s_g^{(0)} \right), \qquad \kappa = 0.05 .
$$

The cost is three FP32 values per group — two Adam moments and the
initialization reference — i.e. $3 \cdot 32/G$ bits per weight, or 0.75 at
$G=128$, against the $\approx 19$ bits per weight the discrete path already
uses. All three are optimizer state and are discarded at inference; the
inference footprint is unchanged.

**Scale-relative discrete rate.** With the scales now moving, a single absolute
$\eta_{\mathrm{duobit}}$ is no longer commensurate with the codebook gap across
layers, since the gap is $s_g(1-\rho)$ and $s_g$ varies with fan-in and with
training. We therefore measure the discrete step in units of the group scale,

$$
\tilde W_{t+1} = \hat W_t - \eta_{\mathrm{duobit}}\, s_g\,
\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon},
$$

so $\eta_{\mathrm{duobit}}$ has a single meaning everywhere: the fraction of a
group scale traversed by a unit Adam step.

**Weight decay.** Decoupled weight decay on the discrete path is meaningless:
the magnitude of $w_i$ lives in $s_g$, not in $z_i$, so decaying $\hat W$ only
biases transitions toward the inner levels. It is set to zero; embeddings and
norm gains keep ordinary AdamW decay.

### 3.9 The embedding table

Sections 3.1-3.8 all act on the linear maps, and on a real decoder that is not
where the memory is. Measured on the 77.2M-parameter FineWeb-EDU model of
Section 5, with every linear map at 2-bit codes and compressed optimizer state:

| | training state | share | inference | share |
|---|---|---|---|---|
| linear maps (2-bit) | 118.5 MiB | 29% | 13.8 MiB | 12% |
| embedding table + norms (FP32) | 294.6 MiB | 71% | 98.2 MiB | 88% |

The remaining levers on the linear side are correspondingly small: halving the
QPEFA and first-moment bit-widths to 4 removes 12% of training state and
nothing at inference; doubling the group size removes 0.7% of each. Holding the
embedding table in the same representation removes **57% of training state and
82% of the inference footprint**.

A lookup table is a matrix, so it needs no new machinery: store $z$ and $s_g$
per row-group exactly as in Section 3.1, materialise
$\hat E = \mathrm{dequant}(z, s)$ for the lookup, and let the optimizer of
Sections 3.5-3.8 act on $\nabla_{\hat E}\mathcal{L}$. The persistent state is
$\log_2 L + 32/G$ bits per entry, and no FP32 table exists at any point.

The one structural difference is sparsity. Only rows for tokens present in the
batch receive a gradient -- about 93-95% of rows are zero at typical batch
sizes -- so most rows see a long run of exactly zero updates punctuated by
occasional ones. This is the regime QPEFA was built for: the residual holds a
row's accumulated sub-threshold movement indefinitely and flips its codes only
when the accumulation crosses a Voronoi boundary, so a rare token's updates are
neither discarded nor amplified. Note that the factored second moment is a
weaker fit here than for a dense matrix, since its row statistic decays toward
zero for tokens that have not appeared recently; the first moment decays faster
($\beta_1^k$ against $\beta_2^{k/2}$), so stale rows take smaller steps rather
than exploding ones.

### 3.10 Memory accounting

Packed inference bytes for a DuobitLinear of $N$ weights, $L$ levels, group size $G$:

$$
B_{\mathrm{inf}} = \Big\lceil N\log_2 L / 8\Big\rceil + 4\cdot (N/G).
$$

Linear bits/weight at inference are $\log_2 L + 32/G$. For $L=4, G=32$ that is 3.0. For $G=128$, 2.25.

Persistent training bytes for the same layer, with QPEFA bit-width $b_e$, moment bit-width $b_m$, block size $B$:

$$
B_{\mathrm{train}} = B_{\mathrm{inf}} + \Big\lceil N b_e / 8\Big\rceil + \Big\lceil N b_m / 8\Big\rceil + 4\lceil N/B\rceil + 4(m+n) + 12\,(N/G),
$$

where the last term is the trained-scale state of Section 3.8 (two Adam moments
and the initialization reference, $3\cdot32/G$ bits per weight). The same
formula applies to a quantized embedding table (Section 3.9), with $m$ the
vocabulary size and $n$ the model dimension.

For $b_e=b_m=8$, $G=32$, $B=64$ this is 20.3 bits per linear weight in the tiny run, versus 96 for FP32 Adam ($W+m+v$) and versus 96 for latent-weight STE (master $W$ in FP32 plus $m,v$). The 4-bit residual variant is 16.3 bits/weight.

Peak step memory still materialises an ephemeral dequantized tile for GEMM, as BitNet does. The number that differs, and the number we report, is persistent state: what remains allocated between steps and what a checkpoint must hold.

---

## 4. Algorithm

```
for each DuobitLinear layer with ephemeral gradient g:
    m ← dequant8(m_q)                         # block-wise 8-bit first moment
    m ← β1 m + (1-β1) g
    r ← β2 r + (1-β2) mean_col(g⊙g)           # factored second moment
    c ← β2 c + (1-β2) mean_row(g⊙g)
    v̂ ← outer(r, c) / mean(r)
    Ŵ  ← dequant(z, s)
    W̃  ← Ŵ - η_duobit · m̂ / (√v̂ + ε)
    e  ← dequant_qpefa(q, s, λ)
    e  ← clip(e + (W̃ - Ŵ), ±λ s)
    z, Ŵ, e ← Quantize(Ŵ + e; s)
    q  ← SR(e · qmax / (λ s))                 # integer residual
    m_q ← quant8(m)
    if t mod F = 0:
        s ← (1-α) s + α s*_MSE(Ŵ, z)
checkpoint for inference: (z, s) only
```

Standard parameters (token embeddings, RMSNorm) take an ordinary Adam step at $\eta$. Gradient clipping at global norm 1.0 applies to both registered parameters and ephemeral dequantized weights. A cosine schedule with warmup applies to both groups.

---

## 5. Experiments

All numbers were measured on CPU with `scripts/run_experiments.py` and 20 unit tests. We do not mix them with GPU 1B-scale claims.

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

Fit $Y \approx XW^\top$ with $W\in\mathbb{R}^{32\times 64}$ drawn Gaussian, a target a 4-level codebook cannot represent exactly. FP32 Adam reaches MSE $6.4\times 10^{-4}$. Every discrete run floors near 2.46–2.75. QPEFA-8 matches an FP32 residual (2.744 vs 2.748) at 4.8× less linear training state (5,312 vs 25,344 bytes). Latent-weight STE floors a little lower (2.46) because the master copy can sit off-grid even though the forward map is quantized. Codes under QPEFA-8 keep a 4.0% mean flip rate.

| Run | Final MSE | Mean flip % | Train bytes |
|---|---|---|---|
| FP32 Adam | 0.00064 | n/a | 24,576 |
| Latent-weight STE | 2.458 | n/a | 24,576 |
| DUOBIT QPEFA-8 | 2.744 | 3.98 | **5,312** |
| DUOBIT QPEFA-4 | 2.747 | 3.28 | **4,288** |
| DUOBIT FP32 residual | 2.748 | 7.17 | 25,344 |

### 5.3 Tiny decoder on a grammar

Task: sequences of triples $(x,\, 3x+1,\, 3x+6) \bmod V$ with $V=96$, length 16. Architecture: 2-layer decoder, $d=64$, 2 heads, SwiGLU $d_{\mathrm{ff}}=128$, $G=32$, 94,528 parameters. 50 steps, batch 16, CPU. FP32 and STE use AdamW at $10^{-3}$. DUOBIT uses $\eta_{\mathrm{duobit}}=8\times 10^{-3}$. Prompt: `[10, 31, 36, 12]`.

| Run | Val | PPL | Mean flip % | Train bits/wt | Train MB | Infer MB | Grammar acc. |
|---|---|---|---|---|---|---|---|
| FP32 Adam | 2.520 | 12.43 | n/a | 96.0 | 1.082 | 0.361 | 70% |
| Latent-weight STE | 2.831 | 16.96 | n/a | 96.0 | 1.082 | 0.361 | 40% |
| **DUOBIT 2-bit QPEFA-8** | **1.227** | **3.41** | 2.03 | **20.3** | **0.288** | **0.056** | **100%** |
| DUOBIT 2-bit QPEFA-4 | 1.250 | 3.49 | 1.79 | **16.3** | **0.246** | **0.056** | **100%** |
| DUOBIT ternary QPEFA-8 | 1.323 | 3.75 | 1.56 | 19.9 | 0.283 | 0.052 | 100% |
| DUOBIT binary QPEFA-8 | 1.851 | 6.36 | 0.83 | 19.3 | 0.277 | 0.046 | 100% |
| DUOBIT W2A8 QPEFA-8 | 1.227 | 3.41 | 1.98 | 20.3 | 0.288 | 0.056 | 100% |

Three facts. First, QPEFA keeps codes alive: mean flip rate 2.0% for 2-bit QPEFA-8, still 0.84% at step 50. Second, that discrete model learns the grammar (100% greedy accuracy, val 1.23) while FP32 Adam at $10^{-3}$ reaches 70% and val 2.52, and while latent-weight STE reaches 40% and val 2.83. Third, W2A8 matches W2A16 to three decimals, and a 4-bit residual is within 0.03 val of the 8-bit residual at 16.3 train bits/weight.

The 4-level codebook is a hard spectral constraint. On a tiny, noiseless grammar that constraint can help. On the random least-squares target in §5.2 it is a floor. We do not claim 2-bit QPEFA dominates FP32 in general. We claim it can match, and here exceed, FP32 quality on this task while storing 4.7× less linear training state and 6.4× less inference memory.

Greedy continuations: 2-bit QPEFA-8 emits `[10, 31, 36, 12, 37, 42, 21, 64, 69, …]`. $12\to 37=3\cdot12+1$, $42=37+5$, then a new valid triple $(21,64,69)$. FP32 gets the first continuation pair and then a mix of valid and broken triples.

### 5.4 Width scaling (35 steps)

| Run | Params | Val | Train MB | Infer MB | Acc. |
|---|---|---|---|---|---|
| FP32 $d=64$ | 94,528 | 3.140 | 1.082 | 0.361 | 60% |
| DUOBIT $d=64$ | 94,528 | 1.265 | 0.288 | 0.056 | 100% |
| DUOBIT $d=96$ | 203,232 | **1.160** | 0.574 | 0.106 | 100% |

A 96-wide 2-bit model (0.106 MB inference, 0.574 MB training) reaches val 1.16, below both the 64-wide 2-bit run and the 64-wide FP32 run, at 29% of FP32 inference memory and 53% of FP32 training memory.

---

## 6. Analysis

**QPEFA is load-bearing and compressible.** An explicit residual is what lets codes cross codebook gaps. Storing that residual in 8 bits, stochastic-rounded against the group scale, matches an FP32 residual on the quadratic bowl (2.744 vs 2.748) and wins on the grammar. Four bits is close. The residual does not need to be a master weight.

**Compressed Adam is the rest of the training-memory win.** Block-wise 8-bit $m$ plus factored $v$ takes linear training state from 96 bits/weight (FP32 $W+m+v$, or QPEFA in FP32 plus FP32 $m,v$) down to 20.3. The 4-bit residual variant reaches 16.3. Full-model compression on the toy is 3.8× because embeddings still use FP32 Adam. At decoder scale the linear maps dominate and the ratio approaches the linear figure.

**Bit-width is capacity, not a free lunch.** Binary QPEFA-8 (2.0 linear inference bits/weight) fits the grammar on greedy decode but sits at val 1.85, a clear tax versus 2-bit (1.23) and ternary (1.32). The 4-level codebook with $\rho=1/3$ is a better match to a roughly Gaussian weight histogram than $\{-1,+1\}$.

**The FP32 comparison is not apples-to-apples on step size.** DUOBIT's discrete rate is 8× the dense rate because a codebook gap is 8–100× an Adam step. Giving FP32 the same $8\times 10^{-3}$ would be a different, probably unstable, baseline. The honest statement is: at the rates each method actually uses, 2-bit QPEFA-8 fitted this grammar faster, in 3.8× less training memory and 6.4× less inference memory. Latent-weight STE, which *does* use the same codebook in the forward pass, lost on quality and saved nothing in training.

**Regularization.** A 4-level codebook is a hard spectral constraint. On a tiny, noiseless grammar that constraint can help. On a random least-squares target it is a floor.

---

## 7. Limitations

The measurements are CPU, synthetic, and small: 50 steps, 95k parameters, one grammar. They verify the optimizer claims (codes move, QPEFA matches an FP32 residual, 8-bit moments suffice, ternary works, binary pays a tax, W2A8 is cheap, training memory drops). They do not verify that a 1B-parameter DUOBIT model matches BitNet b1.58 on WikiText or MMLU.

Peak step memory still contains an ephemeral dequantized weight for GEMM. Packed bit-serial matmul would remove that tile. It is not in the reference implementation. Packed storage of codes in the reference is still uint8 with values in $\{0,1,2,3\}$; the byte counts we report are packed-equivalent. QPEFA and the first moment *are* stored as int8 tensors.

Embeddings and RMSNorm remain FP32 with FP32 Adam. We did not train on natural language in this run.

---

## 8. Conclusion

You can train a transformer whose linear maps are stored only as 2-bit (or ternary, or binary) codes and group scales, with no master-weight tensor, if you carry quantization error in a *quantized* residual until it crosses a codebook gap and if you refuse to store Adam state in FP32. QPEFA plus 8-bit block-wise first moments plus a factored second moment takes linear training state from 96 bits per weight to 20.3 (16.3 with a 4-bit residual). The inference checkpoint is 3.0 bits per linear weight at $G=32$. At 95k parameters the resulting 2-bit model learns a structured grammar, matches W2A8, and spends 3.8× less persistent training memory and 6.4× less packed inference memory than FP32 Adam. The open question is whether the same compressed state closes the loss gap on real text at 100M–1B scale.

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
15. Shazeer, N. and Stern, M. Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. ICML, 2018.
16. Ma, S. et al. BitNet b1.58 2B4T Technical Report. arXiv:2504.12285, 2025.
17. Frantar, E. et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR, 2023.
18. Lin, J. et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys, 2024.
19. Xiao, G. et al. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML, 2023.
