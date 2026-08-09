# Phase 2: DUOBIT-EST Enhancements & Tuning

Phase 1 proved DUOBIT-EST can match FP32 token generation at 65% less weight memory. Phase 2 closes the **val-loss gap** (2.59 vs 0.13) and hardens the system for real-world pretraining.

## Phase 1 Recap

| Metric | FP32 ($d\!=\!256$) | DUOBIT Std ($d\!=\!256$) | DUOBIT 2x ($d\!=\!512$) |
|---|---|---|---|
| Weight Memory | 20.01 MB | 2.27 MB | 6.94 MB |
| Val Loss (500 steps) | **0.133** | 2.588 | 2.836 |
| Token Gen Match | Reference | Approx | **100% exact** |
| Transition Rate | N/A | **0.00%** | **0.00%** |

> [!WARNING]
> **Critical observation**: Transition rate is stuck at 0.00% across all runs. The 2-bit codes are not actually transitioning during training — all learning is happening through the continuous scales and momentum pathway alone. Fixing this is the single highest-impact improvement.

---

## User Review Required

> [!IMPORTANT]
> **Scope decision**: This plan contains 5 enhancement tracks. They are ordered by expected impact. Please confirm if you want all 5, or want to prioritize a subset.

> [!IMPORTANT]
> **Real-data readiness**: Track 5 introduces a real text corpus (TinyStories or WikiText-2). Confirm if you'd like to keep the synthetic arithmetic task, switch to real text, or benchmark both.

---

## Open Questions

1. **GPU target**: Are we targeting CPU-only experiments, or should we optimize for CUDA kernels (e.g., fused 2-bit matmul)?
2. **Model scale ceiling**: Should we push beyond 20M params in Phase 2, or keep experiments small and fast?
3. **Activation quantization**: The W2A8 path exists in code but was never activated. Should we include W2A8 experiments in Phase 2?

---

## Proposed Changes

### Track 1: Fix Transition Dynamics (CRITICAL)

The 0.00% transition rate means codes never flip. Root cause analysis:

**Problem diagnosis**: In [duobit_adam.py](file:///kaggle/working/workspace/duobit/optim/duobit_adam.py#L158), the candidate step `w_cand = w_curr - d_lr * step_direction` produces candidates whose distance from `w_curr` is much smaller than the level gap `s_g * |c_k - c_j|`. The trust gating and balanced rounding then round all transition probabilities to K_g = 0.

**Root cause**: `d_lr = 8e-3` with Adam-normalized step directions of magnitude ~0.01–0.1 produces `|w_cand - w_curr|` ≈ 1e-4, while level gaps are ~0.01–0.1. The ratio gives `p_i` ≈ 0.001–0.01 per weight, and with group_size=128, `K_g = round(sum(p_i))` ≈ round(0.1–1.3) ≈ 0 or 1.

**Proposed fixes**:

#### [MODIFY] [duobit_adam.py](file:///kaggle/working/workspace/duobit/optim/duobit_adam.py)
- **Decouple transition LR from Adam normalization**: Instead of `w_cand = w_curr - d_lr * (m_hat / sqrt(v_hat))`, use raw gradient momentum: `w_cand = w_curr - d_lr * m_hat`. The Adam denominator `sqrt(v_hat)` crushes the step magnitude to ~1.0 uniformly, erasing the per-weight signal the transition probability formula needs.
- **Add transition temperature parameter** `tau` (default 1.0): Scale `p_i = clip(|diff| / (tau * gap), 0, 1)` so `tau < 1` amplifies transition probabilities during early training.
- **Add minimum transition budget** `min_transitions_per_group` (default 0): Ensure at least this many codes flip per group per step, preventing total stagnation.

#### [MODIFY] [stochastic.py](file:///kaggle/working/workspace/duobit/quantization/stochastic.py)
- Accept `temperature` parameter in `compute_transition_probs`.
- Accept `min_k` parameter in `balanced_stochastic_round` to enforce `K_g >= min_k` when `sum(p_i) > 0`.

#### [MODIFY] [config.py](file:///kaggle/working/workspace/duobit/config.py)
- Add `transition_temperature: float = 1.0`
- Add `min_transitions_per_group: int = 0`
- Add `use_raw_momentum: bool = True` (bypass Adam denominator for code transitions)

---

### Track 2: Slow Group Scale EMA Update

Currently group scales $s_g$ are frozen at initialization. This is stable but prevents the model from adapting scale distribution as training shifts the weight landscape.

#### [MODIFY] [duobit_adam.py](file:///kaggle/working/workspace/duobit/optim/duobit_adam.py#L180-L182)
- After code update, periodically recompute MSE-optimal scales from the *current* dequantized weights (not from `w_cand`).
- Apply EMA blending: `s_new = alpha * s_mse_current + (1 - alpha) * s_old` with `alpha = 0.01`.
- Only update every `scale_update_freq` steps (default 100).
- This avoids the scale-explosion feedback loop (recomputing from unconstrained `w_cand`) while allowing gradual adaptation.

#### [MODIFY] [config.py](file:///kaggle/working/workspace/duobit/config.py)
- Add `scale_ema_alpha: float = 0.01`

---

### Track 3: SwiGLU FFN + Gradient Checkpointing

The current FFN uses Squared ReLU. SwiGLU is the standard for modern LLMs (LLaMA, Mistral) and typically provides 0.5–1.0 lower perplexity at equivalent compute.

#### [MODIFY] [transformer.py](file:///kaggle/working/workspace/duobit/model/transformer.py#L48-L64)
- Replace `DuobitFFN` with SwiGLU variant: `gate_proj`, `up_proj`, `down_proj`.
- `h = F.silu(gate_proj(x)) * up_proj(x); return down_proj(h)`
- Adjust `d_ff` default from `4 * d_model` to `(8/3) * d_model` rounded to nearest multiple of group_size (standard SwiGLU sizing).

#### [MODIFY] [config.py](file:///kaggle/working/workspace/duobit/config.py)
- Add `use_swiglu: bool = True`

#### [MODIFY] [transformer.py](file:///kaggle/working/workspace/duobit/model/transformer.py#L81-L117)
- Add optional `torch.utils.checkpoint` wrapping for each transformer block to reduce activation memory at 2x scaled models.

---

### Track 4: Comprehensive Ablation & Longer Training

Run a proper ablation study isolating each enhancement's contribution.

#### [NEW] [scripts/phase2_ablation.py](file:///kaggle/working/workspace/scripts/phase2_ablation.py)

Ablation matrix (1000 steps each, $d=256$, same synthetic dataset):

| Run | Transition Fix | Scale EMA | SwiGLU | Description |
|---|---|---|---|---|
| A | ❌ | ❌ | ❌ | Phase 1 baseline (reproduce) |
| B | ✅ | ❌ | ❌ | Transition fix only |
| C | ✅ | ✅ | ❌ | + Scale EMA |
| D | ✅ | ✅ | ✅ | Full Phase 2 |
| E (FP32) | N/A | N/A | ✅ | FP32 reference with SwiGLU |

Track metrics: val loss, transition rate %, level distribution entropy, generation accuracy, training speed (steps/sec).

#### [NEW] [scripts/phase2_capacity_match.py](file:///kaggle/working/workspace/scripts/phase2_capacity_match.py)

After identifying best config from ablation, run capacity-matched comparison:
- FP32 ($d=256$, ~5.2M params, 20 MB)
- DUOBIT Phase 2 ($d=256$, ~5.2M params, 2.3 MB) — same param count
- DUOBIT Phase 2 scaled ($d=512$, ~20M params, 6.9 MB) — same memory budget
- Run for 2000 steps to allow convergence

---

### Track 5: Real Text Corpus Benchmark (Optional)

#### [NEW] [scripts/phase2_text_benchmark.py](file:///kaggle/working/workspace/scripts/phase2_text_benchmark.py)
- Download TinyStories or WikiText-2 via HuggingFace `datasets`.
- Tokenize with GPT-2 tokenizer (vocab_size=50257, matching current config).
- Train FP32 vs DUOBIT Phase 2 on identical data for 5000 steps.
- Report: val perplexity, text generation samples (temperature=0.7, top-p=0.9), bits-per-character.

---

## Verification Plan

### Automated Tests

```bash
# Existing tests still pass
python -m pytest tests/ -v

# New transition dynamics test
python -m pytest tests/test_transitions.py -v

# Phase 2 ablation (1000 steps × 5 configs)
python scripts/phase2_ablation.py

# Phase 2 capacity match (2000 steps × 3 configs)
python scripts/phase2_capacity_match.py
```

### Key Success Criteria

| Metric | Phase 1 | Phase 2 Target |
|---|---|---|
| Transition Rate | 0.00% | **> 1%** (codes actually flipping) |
| DUOBIT $d=256$ Val Loss (1000 steps) | ~2.59 | **< 1.5** |
| DUOBIT 2x $d=512$ Val Loss (2000 steps) | ~2.84 | **< 0.5** (approaching FP32) |
| Level Distribution Entropy | Unknown (likely peaked) | **> 1.5 bits** (balanced usage of all 4 levels) |

### Manual Verification
- Inspect transition rate curves: should show healthy 1–5% flip rate
- Verify level distribution histograms approach balanced usage across {−1, −ρ, +ρ, +1}
- Compare generated text samples qualitatively (if Track 5 is included)
