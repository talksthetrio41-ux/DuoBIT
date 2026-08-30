import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from duobit.quantization.codebook import (
    SymmetricCodebook,
    compute_group_scales,
    compute_mse_group_scales,
    compute_var_group_scales,
)
from duobit.quantization.hadamard import OnlineHadamard


class DuobitLinear(nn.Module):
    """Linear layer whose persistent weights are discrete codes + group scales.

    No FP32 master-weight Parameter is registered. The dequantized matrix is
    ephemeral: materialised for the forward/backward, then discarded. The
    optimizer reads `ephemeral_w.grad` and writes new `codes` / `scales`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        rho: float = 1.0 / 3.0,
        learnable_rho: bool = False,
        activation_bits: int = 16,
        use_hadamard: bool = False,
        n_levels: int = 4,
        scale_init: str = "var",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.learnable_rho = learnable_rho
        self.activation_bits = activation_bits
        self.use_hadamard = use_hadamard
        self.n_levels = n_levels
        self.scale_init = scale_init

        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by group_size ({group_size})"
            )

        if learnable_rho:
            self.rho = nn.Parameter(torch.tensor([rho], dtype=torch.float32))
        else:
            self.rho = rho

        self.codebook = SymmetricCodebook(rho=rho, n_levels=n_levels)

        n_groups_per_row = in_features // group_size
        self.register_buffer("codes", torch.zeros((out_features, in_features), dtype=torch.uint8))
        self.register_buffer(
            "scales", torch.ones((out_features, n_groups_per_row, 1), dtype=torch.float32)
        )
        # Init scale, kept so a trained scale can be floored relative to it: a
        # group whose scale collapses to zero can never recover.
        self.register_buffer(
            "scales_init", torch.ones((out_features, n_groups_per_row, 1), dtype=torch.float32)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.hadamard = OnlineHadamard(in_features) if use_hadamard else None
        self.ephemeral_w: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self):
        init_w = torch.empty((self.out_features, self.in_features))
        nn.init.kaiming_uniform_(init_w, a=math.sqrt(5))
        scales = compute_group_scales(init_w, group_size=self.group_size)
        codes = self.codebook.quantize(init_w, scales)
        rho_val = self.rho if isinstance(self.rho, nn.Parameter) else None
        fit = compute_var_group_scales if self.scale_init == "var" \
            else compute_mse_group_scales
        fitted = fit(init_w, codes, self.codebook, group_size=self.group_size,
                     rho=rho_val)
        self.codes.copy_(codes)
        self.scales.copy_(fitted)
        self.scales_init.copy_(fitted)

    def get_dequantized_weight(self) -> torch.Tensor:
        rho_val = self.rho if isinstance(self.rho, nn.Parameter) else None
        return self.codebook.dequantize(self.codes, self.scales, rho=rho_val)

    def _quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_bits >= 16:
            return x
        qmax = float(2 ** (self.activation_bits - 1) - 1)
        scale = torch.amax(torch.abs(x), dim=-1, keepdim=True) / qmax
        scale = torch.clamp(scale, min=1e-5)
        x_q = torch.round(x / scale).clamp(-qmax - 1, qmax)
        return x + (x_q * scale - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_hadamard and self.hadamard is not None:
            x = self.hadamard(x)
        if self.activation_bits < 16:
            x = self._quantize_activations(x)

        w_dequant = self.get_dequantized_weight()
        if self.training:
            w_dequant.requires_grad_(True)
            w_dequant.retain_grad()
            self.ephemeral_w = w_dequant
        return F.linear(x, w_dequant, self.bias)

    def inference_nbytes(self) -> int:
        """Packed inference footprint: codes at log2(L) bits + FP32 scales."""
        import math

        code_bits = math.log2(self.n_levels) * self.codes.numel()
        scale_bytes = self.scales.numel() * 4
        bias_bytes = 0 if self.bias is None else self.bias.numel() * 4
        return int(math.ceil(code_bits / 8.0)) + scale_bytes + bias_bytes
