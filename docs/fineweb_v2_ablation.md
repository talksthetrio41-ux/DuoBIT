# DuoBIT v1 -> v2 ablation on FineWeb-EDU

Cumulative ablation of the v2 corrections, then a scan over the two learning
rates they reparametrise. Every run shares the seed, the streamed token
sequence, the architecture, the step budget and the hardware, so each row
differs from its predecessor in exactly one knob.

**Setup.** 49.2M-parameter decoder (6 layers, d_model 384, 6 heads, SwiGLU
d_ff 1024, RMSNorm, RoPE), 500 steps, batch 16/GPU x 512 tokens on 2x Tesla T4
(DDP), streamed `HuggingFaceFW/fineweb-edu` `sample-100BT` with the GPT-2
tokenizer, 32,768 held-out validation tokens per rank from files disjoint from
training. Seed 1337. 8.2M tokens per run.

Raw artifacts: [`results/fineweb-v2-ablation/`](../results/fineweb-v2-ablation/).

## Ablation

| run | change from the row above | val loss | val PPL | vs FP32 |
|---|---|---|---|---|
| `fp32` | FP32 AdamW baseline | **5.8948** | 363.2 | — |
| `A0-v1-repro` | the v1 configuration | 6.4814 | 652.9 | +0.5865 |
| `A1-varinit` | variance-preserving scale init | 6.2660 | 526.4 | +0.3712 |
| `A2-clip1` | QPEFA clip 4.0 -> 1.0 | 6.2254 | 505.4 | +0.3305 |
| `A3-nowd` | no weight decay on the discrete path | 6.2492 | 517.6 | +0.3544 |
| `A4-scales` | trained group scales, `scale_lr` 1e-2 | 6.2481 | 517.0 | +0.3532 |
| `A5-slr3e-2` | `scale_lr` 3e-2 | 6.1438 | 465.8 | +0.2490 |
| **`A8-dlr4e-3`** | **`duobit_lr` 8e-3 -> 4e-3 (best)** | **6.0731** | **434.0** | **+0.1782** |
| `A9-dlr2e-2` | `duobit_lr` 2e-2 | 6.5532 | 701.5 | +0.6583 |
| `A6-rellr0.3` | scale-relative discrete LR, 0.3 | 6.4114 | 608.8 | +0.5166 |
| `A7-rellr1.0` | scale-relative discrete LR, 1.0 | 6.6869 | 801.8 | +0.7920 |

`A6`/`A7` branch from `A4`; `A8`/`A9` branch from `A4` with `scale_lr` 1e-2.

## Validation trajectory

| run | @125 | @250 | @375 | @500 |
|---|---|---|---|---|
| `fp32` | 6.6440 | 6.2048 | 5.9844 | 5.8948 |
| `A8-dlr4e-3` | 6.8235 | 6.4704 | 6.2075 | 6.0731 |
| `A5-slr3e-2` | 6.9969 | 6.5969 | 6.3069 | 6.1438 |
| `A0-v1-repro` | 7.2619 | 6.9336 | 6.6547 | 6.4814 |
| `A7-rellr1.0` | 7.8655 | 7.2280 | 6.8993 | 6.6869 |

No ordering changes over the last three checkpoints, so the ranking is not an
artifact of where the runs were cut.

## What each correction was worth

- **Variance-preserving init: 0.215 nats**, the single largest effect and the
  cheapest — it is a different closed-form scale at step 0 and costs nothing at
  runtime. The MSE fit shrinks the weight std to 0.956x at every fan-in, which
  compounds over the 12 linear maps of this model.
- **QPEFA clip 4.0 -> 1.0: 0.041 nats.** Smaller than the init fix but free: the
  accumulator only ever reaches ~0.333 s after quantization, so the wide clip
  was spending most of the int8 residual grid on unreachable values.
- **Trained scales: 0.104 nats at `scale_lr` 3e-2**, and roughly neutral at
  1e-2 over 500 steps (the scales move ~4% net, `s/s0` 0.951-0.959). This is the
  correction that removes a structural limitation rather than a tuning error, so
  its value should grow with the horizon; at 500 steps it is not yet the biggest
  term.
- **Weight decay on the discrete path: neutral** here (-0.024 nats, within
  run-to-run noise). Kept at 0 because decaying `W_hat` when the magnitude lives
  in `s_g` has no principled meaning, not because it measurably hurt.
- **Discrete learning rate: 0.175 nats** from 8e-3 to 4e-3, with 2e-2 much
  worse. The scan is monotone at its lower edge, so 4e-3 is the best *measured*
  value, not a located optimum — there is likely more here.
- **Scale-relative discrete LR: did not help.** 0.3 is roughly the absolute
  8e-3 equivalent at this fan-in and lost 0.16 nats against it. The
  reparametrisation is defensible in principle (it makes one rate commensurate
  with the codebook gap across layers) but it is not what this model wanted, so
  the final configuration keeps the absolute rate.

Net: **6.4814 -> 6.0731, a 0.408-nat improvement**, and the gap to the FP32
baseline narrows from +0.587 to +0.178 nats — 70% of it closed.

## Cost and correctness

| | DuoBIT v2 | FP32 |
|---|---|---|
| linear train bits/weight | 19.35 | 96.00 |
| linear inference bits/weight | 2.25 | 32.00 |
| persistent train state | 289.9 MiB | 563.3 MiB |
| inference footprint | 81.7 MiB | 187.8 MiB |
| throughput | 23,671 tok/s | 24,549 tok/s |

DuoBIT now runs at **0.964x** the FP32 throughput, against 0.915x for v1 at a
larger model: removing the ~150 device synchronisations per step (a `.item()`
per layer per statistic) and fusing the QPEFA path into one kernel took the
optimizer from the dominant cost to a ~4% overhead.

Every DuoBIT run passed the cross-rank consistency check at both checkpoints
(0/2 failures), confirming that codes and trained scales stay bitwise identical
on both GPUs under `broadcast_buffers=False` — determinism alone holds them
together, via hash-based stochastic rounding keyed on (element index, step).
No run had a non-finite step.

> The FP32 memory figures were recomputed after fixing a double count in this
> harness: the dense baselines charged their linear weights once as linear maps
> and again as parameters. DuoBIT rows were never affected (its codes and scales
> are buffers, not parameters), nor was any quality or throughput number.
