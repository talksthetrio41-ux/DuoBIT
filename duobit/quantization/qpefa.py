"""Quantized persistent error-feedback accumulator (QPEFA).

The residual e lives in a bounded range relative to the group scale, so it
can be stored as a signed integer without a master-weight tensor:

    e = (q / qmax) * clip * s_g,    q in {-qmax, ..., qmax}

Stochastic rounding keeps the integer residual unbiased, which is what lets
sub-threshold Adam steps accumulate until a codebook Voronoi boundary is
crossed. At inference the residual is discarded.
"""
from __future__ import annotations

from typing import Tuple

import torch


def qpefa_qmax(bits: int) -> int:
    if bits < 2:
        raise ValueError(f"QPEFA bits must be >= 2, got {bits}")
    return (1 << (bits - 1)) - 1


def _broadcast_group_scale(scale: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 3 and like.dim() == 2:
        out_f, in_f = like.shape
        n_groups = scale.shape[1]
        gsz = in_f // n_groups
        return scale.expand(out_f, n_groups, gsz).reshape(out_f, in_f)
    return scale.expand_as(like)


def residual_range(scale: torch.Tensor, like: torch.Tensor, pefa_clip: float) -> torch.Tensor:
    """Per-coordinate clip range |e| <= pefa_clip * s_g."""
    return pefa_clip * _broadcast_group_scale(scale, like).clamp(min=1e-8)


def quantize_residual(
    error: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    pefa_clip: float,
    stochastic: bool = True,
) -> torch.Tensor:
    """Map a floating residual onto a signed integer grid.

    Returns an int8 tensor even for 4-bit grids (values fit in [-7, 7]).
    Memory accounting uses `bits`, not the storage dtype.
    """
    if bits >= 16:
        return error
    qmax = float(qpefa_qmax(bits))
    max_res = residual_range(scale, error, pefa_clip)
    norm = (error / max_res).clamp(-1.0, 1.0) * qmax
    if stochastic:
        # Unbiased stochastic rounding: q = floor(x + U(0,1)).
        q = torch.floor(norm + torch.rand_like(norm))
    else:
        q = torch.round(norm)
    return q.clamp(-qmax, qmax).to(torch.int8)


def dequantize_residual(
    q_error: torch.Tensor,
    scale: torch.Tensor,
    bits: int,
    pefa_clip: float,
) -> torch.Tensor:
    if bits >= 16:
        return q_error
    qmax = float(qpefa_qmax(bits))
    max_res = residual_range(scale, q_error, pefa_clip)
    return q_error.to(dtype=max_res.dtype) * (max_res / qmax)


def residual_nbytes(n_weights: int, bits: int) -> int:
    if bits >= 16:
        return n_weights * 4
    return int((n_weights * bits + 7) // 8)
