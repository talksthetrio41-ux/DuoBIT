#!/usr/bin/env python3
"""Does the factored second moment mis-fit row-sparse embedding gradients?

The concern: for a linear map every row gets a gradient every step, so
v_row is an EMA over a dense signal. An embedding table is different --
in one step only the rows for tokens actually in the batch get a
gradient, so v_row[o] for an untouched row decays as beta2^k while
v_col[i] (a mean over ALL rows) stays roughly constant. The product
v_row*v_col/mean(v_row) can then be far from the true per-element v,
and the error is *systematic in row frequency*, not noise.

This measures the direction error the approximation actually causes,
against a full per-element v (what Adam would do) as ground truth.
"""
import math
import torch

torch.manual_seed(0)

V, D = 8192, 512
STEPS = 400
TOKENS = 8192          # tokens per step, as in batch 16 x seq 512
B1, B2, EPS = 0.9, 0.999, 1e-8

# Zipf token frequency: a few very common tokens, a long thin tail. This
# is what makes the row-visit pattern bursty rather than uniform.
rank = torch.arange(1, V + 1, dtype=torch.float64)
p = (1.0 / rank ** 1.05)
p = (p / p.sum()).float()


def run(sparse: bool):
    """Return (rel err of v_hat, cosine of update dir, per-decile err)."""
    m = torch.zeros(V, D)
    v_full = torch.zeros(V, D)          # ground truth per-element
    v_row = torch.zeros(V)
    v_col = torch.zeros(D)

    cos_hist, rel_hist = [], []
    # error broken out by token frequency decile, to see whether the
    # mis-fit tracks how often a row is visited
    dec = torch.clamp((torch.arange(V).float() / V * 10).long(), max=9)
    dec_err = torch.zeros(10)
    dec_n = torch.zeros(10)

    for t in range(1, STEPS + 1):
        g = torch.zeros(V, D)
        if sparse:
            ids = torch.multinomial(p, TOKENS, replacement=True)
            hit, cnt = torch.unique(ids, return_counts=True)
            # gradient of a row is the sum over its occurrences in the batch
            g[hit] = torch.randn(len(hit), D) * cnt.float().sqrt().unsqueeze(1)
        else:
            g = torch.randn(V, D)

        g2 = g * g
        m.mul_(B1).add_(g, alpha=1 - B1)
        v_full.mul_(B2).add_(g2, alpha=1 - B2)
        v_row.mul_(B2).add_(g2.mean(dim=1), alpha=1 - B2)
        v_col.mul_(B2).add_(g2.mean(dim=0), alpha=1 - B2)

        bc1, bc2 = 1 - B1 ** t, 1 - B2 ** t
        denom = max(float(v_row.mean()), EPS)
        v_hat_f = (torch.outer(v_row, v_col) / denom) / bc2   # factored
        v_hat_t = v_full / bc2                                # true
        mh = m / bc1

        d_f = mh / (v_hat_f.sqrt() + EPS)
        d_t = mh / (v_hat_t.sqrt() + EPS)

        if t > 50 and t % 10 == 0:
            cos = torch.nn.functional.cosine_similarity(
                d_f.flatten(), d_t.flatten(), dim=0)
            rel = ((v_hat_f - v_hat_t).pow(2).sum()
                   / v_hat_t.pow(2).sum().clamp(min=1e-30)).sqrt()
            cos_hist.append(float(cos))
            rel_hist.append(float(rel))
            # per-row direction error, bucketed by frequency decile
            rerr = (d_f - d_t).norm(dim=1) / d_t.norm(dim=1).clamp(min=1e-30)
            dec_err.index_add_(0, dec, rerr)
            dec_n.index_add_(0, dec, torch.ones(V))

    return (sum(rel_hist) / len(rel_hist),
            sum(cos_hist) / len(cos_hist),
            dec_err / dec_n.clamp(min=1))


for name, sparse in (("dense (linear map)", False), ("row-sparse (embedding)", True)):
    rel, cos, dec = run(sparse)
    print(f"{name:<26} v_hat rel err {rel:7.4f}   dir cosine {cos:.4f}")
    if sparse:
        print("   direction err by token-frequency decile "
              "(0 = most frequent rows):")
        print("   " + "  ".join(f"{i}:{float(d):.2f}" for i, d in enumerate(dec)))

# Measured result (400 steps, V=8192, D=512, 8192 tokens/step, Zipf s=1.05):
#
#   dense (linear map)       v_hat rel err 0.1021   dir cosine 0.9988
#   row-sparse (embedding)   v_hat rel err 0.0925   dir cosine 0.9900
#   direction err by token-frequency decile (0 = most frequent):
#     0:0.06  1:0.08  2:0.10  3:0.12  4:0.13  5:0.14  6:0.15  7:0.16  8:0.18  9:0.19
#
# So the approximation does degrade on sparse gradients, and the degradation is
# systematic in row frequency rather than noise -- but at cosine 0.990 it is far
# too small to explain a tenth-nat effect. This is why DuobitEmbedding keeps the
# factored second moment instead of carrying a full O(mn) one for the table.
