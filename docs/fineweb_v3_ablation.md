# DuoBIT v3 on FineWeb-EDU: compressing the embedding table

v2 left the method 0.449 nats behind FP32 while holding 2.14x less training
state, and profiling said where the remaining memory was: at 77M parameters the
FP32 embedding table is **71% of persistent training state and 88% of the
inference footprint**, against 29% and 12% for all the 2-bit linear maps
combined. Every knob on the linear side was rearranging the small share.

So v3 quantizes the token embedding table with the same representation as the
linear maps — uint8 codes plus trained per-group scales, no master weight —
and measures what that costs in quality. It turns out to cost nothing, and the
first thing this document has to establish is how much "nothing" is worth
given how noisy the discrete path is.

**Setup.** Identical to the v2 ablation: 49.2M-parameter decoder (6 layers,
d_model 384, 6 heads, SwiGLU d_ff 1024, RMSNorm, RoPE), 500 steps, batch
16/GPU x 512 tokens on 2x Tesla T4 (DDP), streamed `HuggingFaceFW/fineweb-edu`
`sample-100BT` with the GPT-2 tokenizer, 32,768 held-out validation tokens per
rank from files disjoint from training, seed 1337, 8.2M tokens per run. Every
run branches from the v2 winner (`A8`).

Raw artifacts: [`results/fineweb-v3-ablation/`](../results/fineweb-v3-ablation/).

## First: the noise floor

Sweep C's anchor run was the v2 winner's exact configuration and was expected
to reproduce its 6.0731. It did not. So the configuration was run seven times
in total, across three sessions and twice within one session:

| measurement | val loss |
|---|---|
| `A8` (sweep A) | 6.0731 |
| `C0-base` (sweep C) | 6.1578 |
| `D0-base` (sweep D) | 6.0977 |
| `D0b-base-rep` (sweep D, same session as `D0`, identical config and seed) | 6.1942 |
| `D2-fp32emb-lr4e-3` | 6.0632 |
| `D3-fp32emb-lr4e-3-nowd` | 6.0971 |
| `D4-fp32emb-nowd` | 6.1315 |

**mean 6.1164, std 0.0473, range 0.1310 nats.**

The FP32 baseline over the same sessions is 5.8948 / 5.8990 / 5.8972 — a range
of 0.004 nats. The variance is therefore not the harness, the data stream or
the seed; it is the discrete path. A code transition is a step function of the
accumulated QPEFA residual, so a perturbation far below FP32 rounding — GPU
reduction order is not deterministic across runs — decides whether a weight
flips, and a flip moves that weight by a full codebook gap. Fixing the seed
fixes the stochastic-rounding hash, not the arithmetic feeding it.

Two consequences, and they apply to the v2 ablation retroactively:

1. **A single run resolves a difference of ~0.13 nats, not less.** Effects
   below that need replicates, and the v2 ablation's `pefa_clip` (0.041) and
   weight-decay (0.024) rows do not survive this standard. They are annotated
   in [`fineweb_v2_ablation.md`](fineweb_v2_ablation.md).
2. **Quote pooled means, not best runs.** `A8`'s 6.0731 is the low end of its
   own sampling distribution.

## The result

| run | val loss | vs FP32 | vs pooled base | train state | inference | linear train b/wt |
|---|---|---|---|---|---|---|
| `fp32` | **5.8972** | — | −0.219 | 563.3 MiB | 187.8 MiB | 96.00 |
| `C1-emb` | 5.9126 | +0.015 | −0.204 | **113.6 MiB** | **13.2 MiB** | 19.34 |
| `D5-emb-m4` | 5.9625 | +0.065 | −0.154 | **90.1 MiB** | **13.2 MiB** | 15.34 |
| `D8-emb-dlr2e-3` | 5.9659 | +0.069 | −0.151 | 113.5 MiB | 13.2 MiB | 19.34 |
| `D1-emb` | 6.0059 | +0.109 | −0.111 | 113.5 MiB | 13.2 MiB | 19.34 |
| `D7-emb-dlr8e-3` | 6.0235 | +0.126 | −0.093 | 113.5 MiB | 13.2 MiB | 19.34 |
| `D6-emb-q4m4` | 6.0545 | +0.157 | −0.062 | **66.6 MiB** | **13.2 MiB** | 11.34 |
| *pooled dense-embedding base* | *6.1164* | *+0.219* | — | *289.9 MiB* | *81.7 MiB* | *19.35* |
| `C10-q4m4` (4-bit, dense embeddings) | 6.2590 | +0.362 | +0.143 | 187.0 MiB | 81.7 MiB | 11.35 |
| `C8-slr1e-1` (`scale_lr` 0.1) | 6.4889 | +0.592 | +0.373 | 289.9 MiB | 81.7 MiB | 19.35 |

Pooling the two measurements of the plain quantized-embedding configuration
(`C1-emb` 5.9126, `D1-emb` 6.0059; mean 5.9592) against the seven-run dense
mean of 6.1164:

> **quantizing the embedding table is worth +0.157 nats, sem 0.050, ~3.1σ** —
> while cutting persistent training state 2.55x and the inference footprint
> 6.2x.

Every configuration that quantizes the table beats every configuration that
does not. With seven dense and six quantized runs, that separation arises by
chance once in `C(13,6)` = 1,716 labellings; counting `C10` and `C8` as the
dense runs they also are, once in 5,005. Two honest qualifications: the runs
within each group are not identically configured, so this is a rank
observation rather than a designed permutation test, and **the separation
margin is only 0.0087 nats** (`D6` at 6.0545 against `D2` at 6.0632) — one
further dense run landing low would break it. The pooled 3.1σ comparison is
the more robust of the two statements.

## Why it is not a learning-rate artifact

With `--quant-embeddings 0` the table trains through AdamW at `--lr` (1e-3).
With `1` it becomes a DuoBIT module and trains through the discrete path at
`--duobit-lr` (4e-3) with no weight decay. That is a 4x rate change and a decay
change riding along with the representation change, so the comparison as first
run was confounded. `--embed-lr` and `--embed-wd` were added to break it, and
sweep D spends three runs on the controls:

| control | val loss | vs `D0-base` |
|---|---|---|
| `D0-base` (dense, `lr` 1e-3, `wd` 1e-2) | 6.0977 | — |
| `D2-fp32emb-lr4e-3` (dense at the discrete rate) | 6.0632 | −0.0345 |
| `D3-fp32emb-lr4e-3-nowd` (both changes) | 6.0971 | −0.0006 |
| `D4-fp32emb-nowd` (decay only) | 6.1315 | +0.0338 |

All three sit inside the noise band, and none approaches the 0.157-nat effect.
Weight decay was the weaker suspect anyway and can be dismissed analytically:
it is applied to all 6.4M entries every step whether or not the token appeared,
but at `lr*wd` = 1e-5 that is a 0.5% shrink over 500 steps.

**So the gain belongs to the representation, not to the schedule.** The
plausible mechanism is regularisation: 8.2M tokens over a 50,257-row table
leaves most rows seen a handful of times, and a 2-bit row with a trained group
scale cannot memorise a rare token the way an unconstrained FP32 row can. That
is a hypothesis this sweep does not test — it predicts the gain should shrink
with a larger token budget, which is worth checking before leaning on it.

One mechanism was tested and ruled out in advance. Embedding gradients are
~95% row-sparse, and the factored second moment (`v̂ = outer(v_row, v_col) /
mean(v_row)`) was the obvious suspect for mishandling them, since an untouched
row's `v_row` decays as `β₂ᵏ` while `v_col` averages over all rows. Simulated
against a full per-element `v` with Zipf-distributed visits, the update
direction holds at **cosine 0.990** (against 0.9988 for dense gradients), with
a mild systematic penalty across token-frequency deciles (0.06 direction error
on the most frequent tenth rising to 0.19 on the rarest). Real, but far too
small to be the story, so a non-factored second moment for the table is not
the fix it looked like it might be.

## The compression ladder

With the table quantized, the linear-map knobs finally act on a majority of
the memory. Bits per weight on the linear maps, and the whole-model totals:

| configuration | linear train b/wt | train state | vs FP32 | inference | vs FP32 |
|---|---|---|---|---|---|
| v2 (dense embeddings) | 19.35 | 289.9 MiB | 1.94x | 81.7 MiB | 2.30x |
| + 2-bit embeddings | 19.34 | 113.6 MiB | **4.96x** | 13.2 MiB | **14.2x** |
| + 4-bit moments | 15.34 | 90.1 MiB | **6.25x** | 13.2 MiB | 14.2x |
| + 4-bit QPEFA and moments | 11.34 | 66.6 MiB | **8.46x** | 13.2 MiB | 14.2x |

Inference is unchanged along the last two rows because QPEFA and the Adam
moments are training state only; what they buy is the training footprint.

Isolating the two 4-bit knobs gives a clear asymmetry, and they should not be
adopted together:

- **4-bit Adam moments are free.** `D5` (5.9625) against the 8-bit
  quantized-embedding mean (5.9592) is **+0.003 nats** — far inside the noise
  band, for 19.34 -> 15.34 bits per weight and 113.6 -> 90.1 MiB.
- **4-bit QPEFA costs about 0.09 nats.** `D6` against `D5`, which differ only
  in the residual width, is **+0.092 nats** — comparable to the noise band on
  a single pair, so treat it as an upper-bound-ish point estimate rather than
  a resolved number, but it is the only compression step here that visibly
  costs anything.

So `moment_bits=4` is taken into the recommended configuration and
`qpefa_bits=4` is left as an option for when 66.6 MiB matters more than a
tenth of a nat. That the residual is the sensitive one is consistent with its
role: it is the only place sub-threshold updates survive between transitions,
and halving its width coarsens the accumulator that decides when a code flips.

Two knobs did not survive. **4-bit without quantized embeddings** (`C10-q4m4`,
6.2590) is the worst non-degenerate run in either sweep: it pays the residual's
quality cost to compress the 29% of memory that was never the problem.
**`scale_lr` 1e-1** (`C8`, 6.4889) drives the mean group scale to 1.688x its
initialisation and loses 0.37 nats; 1e-2 stays the right rate.

Re-scanning the discrete learning rate with embeddings quantized (`D7` 8e-3,
`D1` 4e-3, `D8` 2e-3, spanning 6.0235 / 6.0059 / 5.9659) puts every point
inside one noise band of the others. 4e-3 is retained because v2 resolved it
against 8e-3, not because this scan re-confirmed it.
