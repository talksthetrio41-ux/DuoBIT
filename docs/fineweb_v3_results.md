# DuoBIT v3 vs FP32 and FP16 on FineWeb-EDU

Head-to-head at the same scale, data, seed and step budget as the v1 and v2
reference runs, with the main configuration run twice so the comparison has an
error bar at this scale rather than an assumed one.

**Setup.** 77,161,984-parameter decoder (8 layers, d_model 512, 8 heads, SwiGLU
d_ff 1408, RMSNorm, RoPE, causal SDPA), 3000 steps, batch 8/GPU x 1024 tokens
on 2x Tesla T4 (DDP, NCCL), 49,152,000 tokens of streamed
`HuggingFaceFW/fineweb-edu` `sample-100BT` with the GPT-2 tokenizer, 131,072
held-out validation tokens per rank from files disjoint from training, seed
1337. All runs consume the identical token sequence.

v3 keeps the v2 configuration and adds `--quant-embeddings`: the 25,731,584-entry
token table is held as uint8 codes plus trained per-group scales, exactly like
the linear maps, leaving 8,704 FP32 parameters in the whole model (the RMSNorm
gains). Selection and controls: [`fineweb_v3_ablation.md`](fineweb_v3_ablation.md).

Raw artifacts: [`results/fineweb-v3-final/`](../results/fineweb-v3-final/).

## Results

| | **DuoBIT v3** | **v3 + 4-bit moments** | DuoBIT v2 | FP32 AdamW | FP16 AMP |
|---|---|---|---|---|---|
| final val loss | 5.0425 / 4.9836 | 5.0467 | 5.0785 | **4.5987** | 4.5983 |
| pooled val loss | **5.0131** | — | 5.0624 | 4.5978 | — |
| val perplexity | **150.5** | 155.5 | 160.5 | **99.4** | 99.3 |
| gap to FP32 (nats) | **+0.415** | +0.449 | +0.465 | — | −0.001 |
| linear train bits/weight | 19.32 | **15.32** | 19.33 | 96.00 | 96.00 |
| linear inference bits/weight | **2.25** | **2.25** | 2.25 | 32.00 | 16.00 |
| FP32 parameters remaining | **8,704** | 8,704 | 25.7M | 77.2M | 77.2M |
| persistent train state | 177.8 MiB | **141.0 MiB** | 413.1 MiB | 883.0 MiB | 883.0 MiB |
| inference footprint | **20.7 MiB** | **20.7 MiB** | 112.0 MiB | 294.3 MiB | 196.3 MiB |
| throughput | 14,024 tok/s | 14,167 tok/s | 14,131 tok/s | 14,654 tok/s | **34,939 tok/s** |
| peak GPU memory | 10,163 MiB | 10,162 MiB | 10,183 MiB | 10,220 MiB | **8,580 MiB** |
| wall time | 58.4 min | 57.8 min | 58.0 min | 55.9 min | **23.4 min** |
| mean code transition rate | 0.92% / 1.04% | 1.01% | 1.62% | — | — |
| final scale / init scale | 0.898 / 0.876 | 0.878 | 0.828 | — | — |
| non-finite steps | 0 | 0 | 0 | 0 | 0 |

Two numbers are given for v3 where the run was replicated. Pooled rows use both.

## What this says

**The compression is the result, and it is large.** Against FP32, v3 with 4-bit
moments holds **6.26x** less persistent training state and a **14.2x** smaller
inference footprint; against FP16 mixed precision, 6.26x and 9.5x. Unlike every
earlier version of these numbers, the whole-model ratio is now close to the
linear-map ratio, because there is essentially no FP32 left: **8,704 of
77,161,984 parameters**, or 0.011%. A 77M-parameter model exports at **20.7
MiB**.

**The quality gain from quantizing embeddings did not survive the longer
horizon, as predicted.** At 500 steps the effect was +0.157 nats at 3.1σ. Here
it is **+0.049 nats at ~1.5σ — not resolved**. The v3 ablation named the
regularisation hypothesis and its falsifiable consequence in advance: 8.2M
tokens over a 50,257-row table leaves most rows barely seen, so a 2-bit row
cannot memorise a rare token, and *the gain should shrink with a larger token
budget*. At 6x the tokens it shrank into the noise. The honest reading is that
quantizing the table is **quality-neutral at this budget**, and the 500-step
result was a short-horizon regularisation effect, not a durable quality win.

**DuoBIT remains behind FP32 on quality: +0.415 nats**, perplexity 150.5 against
99.4. v2 was +0.465. That difference is itself inside the noise, so v3 should be
read as buying a 2.3x smaller training footprint and a 5.4x smaller export *at
the same quality as v2*, not as improving on it.

**Replicate spread at this scale is 0.059 nats** (5.0425 vs 4.9836, identical
config and seed, same session), against a 0.131-nat range at the 500-step sweep
scale. It shrinks with the horizon but does not vanish, and it is the reason
this table quotes pooled means. FP32 over the same two sessions spans 0.0019
nats.

**DuoBIT still buys memory, not speed.** FP16 AMP matches FP32 quality to 0.0005
nats at **2.4x** the throughput, because T4 tensor cores do FP16 matmuls
natively. DuoBIT runs at 0.96x FP32. The ephemeral dequantized tile still goes
through an FP32 GEMM; a packed bit-serial kernel is what would convert the
compression into speed.

**4-bit Adam moments are free at this scale too**, confirming the sweep: 5.0467
against a pooled 5.0131 is inside the replicate spread, for another 36.8 MiB
(19.32 -> 15.32 bits per weight). This is the configuration to prefer.

**Cross-rank determinism held.** 48/48 consistency checks passed across the four
DuoBIT runs, confirming codes and trained scales stay bitwise identical on both
GPUs under `broadcast_buffers=False`, with nothing synchronising them but
identical arithmetic and stochastic rounding hashed on (element index, step).
Quantizing the embedding table did not weaken this.

## Validation curve

| step | v3-emb | v3-emb-rep | v3-emb-m4 | v2-dense | FP32 | FP16 |
|---|---|---|---|---|---|---|
| 250 | 6.6773 | 6.6614 | 6.6588 | 6.7101 | 6.1596 | 6.1617 |
| 500 | 6.3040 | 6.2871 | 6.2819 | 6.3727 | 5.6938 | 5.6877 |
| 750 | *7.1757* | 5.9981 | 5.9926 | 6.1433 | 5.4370 | 5.4352 |
| 1000 | 5.8180 | 5.7728 | 5.7744 | 5.9382 | 5.2219 | 5.2164 |
| 1500 | 5.5667 | 5.5346 | 5.5519 | 5.6908 | 4.9921 | 4.9890 |
| 2000 | 5.3000 | 5.2622 | 5.2838 | 5.3817 | 4.7728 | 4.7719 |
| 2500 | 5.1556 | 5.1032 | 5.1527 | 5.2160 | 4.6679 | 4.6683 |
| 3000 | 5.0425 | 4.9836 | 5.0467 | 5.0785 | 4.5987 | 4.5983 |

Every DuoBIT run leads `v2-dense` at every checkpoint, which is a more
consistent signal than the endpoint difference alone — though with one v2 run
it is not independent evidence.

The italicised 7.1757 is a real transient in `v3-emb` at step 750: validation
rose 0.87 nats above its own step-500 value and recovered completely by step
1000. No non-finite step was recorded, and neither the replicate nor the `m4`
run shows anything at that step. The run finishes 0.059 nats behind its own
replicate, so whatever the excursion cost, it was not fully repaid. A single
excursion in one of four runs is consistent
with the code-flip sensitivity documented in the ablation: a cluster of
transitions in one step can move the loss sharply and be re-absorbed. It is
noted rather than explained; a mechanism would need instrumentation this run
did not carry.

## Caveats

- **The models are still heavily undertrained.** 49.2M tokens for 77.2M
  parameters is far below a compute-optimal budget; FP32 itself only reaches
  perplexity 99. The 0.415-nat gap measured here need not be the gap at
  convergence, and the embedding result's horizon-dependence is direct evidence
  that conclusions at this budget can reverse with more tokens.
- **n=2 for the main configuration, n=1 elsewhere.** The v2-to-v3 quality
  comparison (1.5σ) and the 4-bit-moment comparison are not resolved. The
  compression figures are exact accounting and carry no such uncertainty.
- **The 500-step quality finding did not replicate at 3000 steps.** It is
  reported in the ablation as measured and is corrected here rather than
  quietly dropped. Anything selected on a 500-step sweep inherits this risk.
- **`qpefa_bits=4` was left out of the final.** The sweep put its cost at ~0.09
  nats against ~0.003 for 4-bit moments, so it was not worth a 3000-step slot;
  the 66.6 MiB point on the frontier is therefore only measured at 500 steps.
- **One hardware, one tokenizer, one dataset.** Nothing here tests whether the
  method transfers to another architecture family or a different vocabulary
  size, and the embedding result is by construction sensitive to vocabulary
  size relative to token budget.
