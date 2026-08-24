"""Latent-weight straight-through linear map (BitNet training pattern).

Keeps an FP32 master-weight Parameter and quantizes it on the fly with STE.
This is the baseline DUOBIT is measured against: same codebook in the
forward pass, but training memory still contains a full-precision W plus
Adam moments.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from duobit.quantization.codebook import (
    SymmetricCodebook,
    compute_group_scales,
    compute_mse_group_scales,
)


class SteQuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        rho: float = 1.0 / 3.0,
        n_levels: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_levels = n_levels
        self.codebook = SymmetricCodebook(rho=rho, n_levels=n_levels)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _quantized_weight(self) -> torch.Tensor:
        w = self.weight
        scales = compute_group_scales(w, group_size=self.group_size)
        codes = self.codebook.quantize(w, scales)
        scales = compute_mse_group_scales(
            w, codes, self.codebook, group_size=self.group_size
        )
        w_q = self.codebook.dequantize(codes, scales)
        return w + (w_q - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._quantized_weight(), self.bias)
