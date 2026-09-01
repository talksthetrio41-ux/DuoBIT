# DuoBIT v2 vs FP32 and FP16 on FineWeb-EDU

Head-to-head at the same scale, data, seed and step budget as the v1 reference
run, so the tuning gain is measurable directly.

**Setup.** 77,161,984-parameter decoder (8 layers, d_model 512, 8 heads, SwiGLU
d_ff 1408, RMSNorm, RoPE, causal SDPA), 3000 steps, batch 8/GPU x 1024 tokens on
2x Tesla T4 (DDP, NCCL), 49,152,000 tokens of streamed
`HuggingFaceFW/fineweb-edu` `sample-100BT` with the GPT-2 tokenizer, 131,072
held-out validation tokens per rank drawn from files disjoint from training,
seed 1337. All runs consume the identical token sequence.

DuoBIT configuration is the sweep winner (`A8`): 2-bit codebook
`{-1,-1/3,+1/3,+1}`, group size 128, variance-preserving scale init, trained
group scales at `scale_lr` 1e-2, `duobit_lr` 4e-3 (absolute), QPEFA 8-bit with
clip 1.0, no weight decay on the discrete path. See
[`fineweb_v2_ablation.md`](fineweb_v2_ablation.md) for how it was chosen.

Raw artifacts: [`results/fineweb-v2-final/`](../results/fineweb-v2-final/).

## Results

| | **DuoBIT v2** | DuoBIT v1 | FP32 AdamW | FP16 AMP AdamW |
|---|---|---|---|---|
| final val loss | **5.0462** | 5.5374 | **4.5968** | 4.5969 |
| final val perplexity | **155.43** | 254.02 | **99.16** | 99.17 |
| gap to FP32 (nats) | **+0.4494** | +0.9406 | — | +0.0001 |
| linear train bits/weight | **19.33** | 18.58 | 96.00 | 96.00 |
| linear inference bits/weight | **2.25** | 2.25 | 32.00 | 16.00 |
| persistent train state | **413.1 MiB** | 408.5 MiB | 883.0 MiB | 883.0 MiB |
| inference footprint | **112.0 MiB** | 112.0 MiB | 294.3 MiB | 196.3 MiB |
| throughput | 13,470 tok/s | 13,184 tok/s | 14,083 tok/s | **35,508 tok/s** |
| peak GPU memory | 10,182 MiB | 10,180 MiB | 10,219 MiB | **8,579 MiB** |
| wall time | 62.3 min | 63.7 min | 59.4 min | **24.3 min** |
| mean code transition rate | 1.489% | 0.930% | — | — |
| final scale / init scale | 0.824 | 1.000 | — | — |
| non-finite steps | 0 | 0 | 0 | 0 |

Persistent training state is what must be held for the whole run: for DuoBIT the
2-bit codes, FP32 group scales, int8 QPEFA residual, block-wise int8 first
moment and factored second moment, plus the trained-scale state; for FP32 and
FP16 the FP32 weight and both FP32 Adam moments. Mixed precision keeps FP32
master weights, so its *training* state is identical to FP32 and only its
inference export halves. Embeddings and RMSNorm gains are FP32 in every regime
and are included in every total.

## Validation curve

| step | DuoBIT v2 | DuoBIT v1 | FP32 | FP16 |
|---|---|---|---|---|
| 250 | 6.6961 | 7.1101 | 6.1595 | 6.1613 |
| 500 | 6.3617 | 6.7520 | 5.6833 | 5.6878 |
| 1000 | 5.9175 | 6.3662 | 5.2196 | 5.2156 |
| 1500 | 5.7036 | 6.2824 | 4.9924 | 4.9922 |
| 2000 | 5.3760 | 5.9251 | 4.7694 | 4.7683 |
| 2500 | 5.2049 | 5.7286 | 4.6671 | 4.6671 |
| 3000 | **5.0462** | 5.5374 | **4.5968** | 4.5969 |

v2 leads v1 at every checkpoint and the margin widens with the horizon (0.41
nats at step 250, 0.49 at step 3000). v1 also had a non-monotone excursion at
step 1250 (6.6425, above its own step-1000 value); v2's curve is monotone
throughout.

## What this says

**The tuning worked, and it is not parity.** v1 -> v2 is **0.491 nats**, a
perplexity drop from 254.0 to 155.4 (39% lower), and it closes **52%** of the
gap to the FP32 baseline. DuoBIT at 2.25 inference bits per linear weight is
still **0.449 nats behind** an FP32 model of the same shape trained on the same
tokens. That is the honest headline: substantially closer, not equal.

**The compression is real and is the point.** Against FP32, DuoBIT holds
**2.14x** less persistent training state and a **2.63x** smaller inference
footprint; against FP16 mixed precision, **2.14x** and **1.75x**. On the linear
maps alone -- where the method actually acts -- it is **4.97x** on training
state (19.33 vs 96 bits/weight) and **14.2x** at inference (2.25 vs 32), or
7.1x against an FP16 export. The whole-model ratios are lower because the
25.7M-parameter embedding table (33% of the model) stays FP32 in every regime;
at larger model sizes, where embeddings are a smaller fraction, the whole-model
ratio approaches the linear-map ratio.

**Mixed precision is the throughput answer, DuoBIT is the memory answer.**
FP16 AMP matches FP32 quality to 0.0001 nats while running **2.5x faster** and
using less peak memory, because T4 tensor cores do FP16 matmuls natively.
DuoBIT runs at 0.956x FP32 throughput (up from 0.915x in v1) — it buys memory,
not speed. Nothing here suggests 2-bit training is currently the faster option
on this hardware; the ephemeral dequantized tile still goes through an FP32
GEMM, and a packed bit-serial kernel is what would change that.

**The trained scales moved, and more at longer horizons.** The mean group scale
ended at 0.824 of its initialization, against 0.951-0.979 over the 500-step
sweep and exactly 1.000 in v1, where the update was an algebraic fixed point.
The code transition rate also rose from 0.93% to 1.49%: the discrete weights are
genuinely more active, not merely better initialized.

**Cross-rank determinism held.** 12/12 consistency checks passed over the run,
confirming that codes and trained scales stay bitwise identical on both GPUs
under `broadcast_buffers=False`, with nothing synchronising them but identical
arithmetic and stochastic rounding hashed on (element index, step).

**The FP32 baseline reproduced across sessions** to 1.0e-3 nats (4.5957 in the
v1 session, 4.5968 here), which is what licenses comparing the v1 DuoBIT number
to the v2 one at all.

## Caveats

- **Both models are heavily undertrained.** 49.2M tokens for 77.2M parameters is
  far under a compute-optimal budget; FP32 itself only reaches perplexity 99.
  The gap measured at this budget need not be the gap at convergence, and could
  move in either direction.
- **The discrete learning rate scan was still monotone at its lower edge**
  (2e-2 worse than 8e-3 worse than 4e-3). 4e-3 is the best *measured* value, not
  a located optimum. The weekly GPU quota (6 h, of which this run used 2.5)
  did not allow a fourth 3000-step run to chase it, so some headroom is
  probably unclaimed.
- **The scale-relative discrete LR did not help** at either scale tested, and
  the final configuration keeps the absolute rate despite the relative
  parametrisation being the more principled one.
- **Single seed per configuration, and the variance was later measured and is
  large.** Seven runs of one DuoBIT configuration at the 500-step sweep scale
  gave mean 6.1164, std 0.0473, range 0.1310 nats — including two runs in the
  *same session* with byte-identical config and seed that differed by 0.097.
  The FP32 baseline reproduces across sessions to 0.004 nats, so this is
  specific to the discrete path: a code transition is a step function of the
  accumulated residual, so a perturbation far below FP32 rounding decides
  whether a weight flips. The 0.49-nat v1 -> v2 headline is far outside that
  band; the finer ablation rows are not, and are annotated accordingly in
  [`fineweb_v2_ablation.md`](fineweb_v2_ablation.md). The variance at this
  3000-step scale is measured in the v3 run rather than assumed equal.
- **A memory-accounting bug in this harness was found and fixed after the runs.**
  The dense baselines charged their linear weights twice (once as linear maps,
  once as parameters), inflating only the FP32/FP16 memory columns. The DuoBIT
  rows were never affected (its codes and scales are buffers, not parameters),
  nor was any quality or throughput number. The baseline memory figures above
  were recomputed with the corrected rule and agree exactly with the FP32
  reference that each DuoBIT run computed independently.
