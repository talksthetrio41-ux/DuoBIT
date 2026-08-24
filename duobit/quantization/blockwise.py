"""Block-wise integer quantization for optimizer moments.

Follows the 8-bit Adam construction (Dettmers et al., ICLR 2022): split a
tensor into fixed-size blocks, store a per-block absmax scale in FP32, and
store the values as signed integers. Dequantized only for the optimizer
step, then immediately re-quantized. Persistent training state stays integer.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def moment_qmax(bits: int) -> int:
    if bits < 2:
        raise ValueError(f"moment bits must be >= 2, got {bits}")
    return (1 << (bits - 1)) - 1


def quantize_blockwise(
    x: torch.Tensor,
    bits: int = 8,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Size, int]:
    """Returns (q, block_scale, orig_shape, pad)."""
    if bits >= 16:
        return x, torch.ones((), device=x.device, dtype=x.dtype), x.shape, 0

    qmax = float(moment_qmax(bits))
    orig_shape = x.shape
    flat = x.reshape(-1)
    n = flat.numel()
    pad = (block_size - n % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    scale = blocks.abs().amax(dim=-1).clamp(min=1e-8) / qmax
    q = torch.round(blocks / scale.unsqueeze(-1)).clamp(-qmax, qmax).to(torch.int8)
    return q, scale, orig_shape, pad


def dequantize_blockwise(
    q: torch.Tensor,
    scale: torch.Tensor,
    orig_shape: torch.Size,
    pad: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if q.dtype in (torch.float16, torch.float32, torch.bfloat16) and scale.ndim == 0:
        return q.to(dtype)
    blocks = q.to(dtype) * scale.to(dtype).unsqueeze(-1)
    flat = blocks.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.reshape(orig_shape)


def blockwise_nbytes(n_weights: int, bits: int, block_size: int) -> int:
    """Packed integer payload plus FP32 per-block scales."""
    if bits >= 16:
        return n_weights * 4
    n_blocks = (n_weights + block_size - 1) // block_size
    return int((n_weights * bits + 7) // 8) + n_blocks * 4
