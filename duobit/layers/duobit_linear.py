import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from duobit.quantization.codebook import SymmetricCodebook, compute_group_scales, compute_mse_group_scales
from duobit.quantization.hadamard import OnlineHadamard

class DuobitLinear(nn.Module):
    """
    DuobitLinear layer replacing standard nn.Linear.
    Weights are persistently stored as 2-bit codes (uint8) and per-group scales (float32/bfloat16).
    Dequantization is tile-local/ephemeral during forward and backward.
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
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.learnable_rho = learnable_rho
        self.activation_bits = activation_bits
        self.use_hadamard = use_hadamard

        if in_features % group_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")

        if learnable_rho:
            self.rho = nn.Parameter(torch.tensor([rho], dtype=torch.float32))
        else:
            self.rho = rho

        self.codebook = SymmetricCodebook(rho=rho)

        # Register persistent 2-bit weight codes and per-group scales as buffers
        n_groups_per_row = in_features // group_size
        
        self.register_buffer("codes", torch.zeros((out_features, in_features), dtype=torch.uint8))
        self.register_buffer("scales", torch.ones((out_features, n_groups_per_row, 1), dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        if self.use_hadamard:
            self.hadamard = OnlineHadamard(in_features)
        else:
            self.hadamard = None

        # Ephemeral dequantized weight handle for optimizer to inspect gradient
        self.ephemeral_w: Optional[torch.Tensor] = None

        self.reset_parameters()

    def reset_parameters(self):
        init_w = torch.empty((self.out_features, self.in_features))
        nn.init.kaiming_uniform_(init_w, a=math.sqrt(5))
        
        # Max-abs initial scale estimate
        scales = compute_group_scales(init_w, group_size=self.group_size)
        codes = self.codebook.quantize(init_w, scales)

        # Refine scale using closed-form MSE optimal fitting
        rho_val = self.rho if isinstance(self.rho, nn.Parameter) else None
        mse_scales = compute_mse_group_scales(init_w, codes, self.codebook, group_size=self.group_size, rho=rho_val)

        self.codes.copy_(codes)
        self.scales.copy_(mse_scales)

    def get_dequantized_weight(self) -> torch.Tensor:
        rho_val = self.rho if isinstance(self.rho, nn.Parameter) else None
        return self.codebook.dequantize(self.codes, self.scales, rho=rho_val)

    def _quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_bits >= 16:
            return x
        # Per-token absmax 8-bit quantization with STE
        scale = torch.amax(torch.abs(x), dim=-1, keepdim=True) / 127.0
        scale = torch.clamp(scale, min=1e-5)
        x_q = torch.round(x / scale)
        x_q = torch.clamp(x_q, -128, 127)
        # STE
        return x + (x_q * scale - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_hadamard and self.hadamard is not None:
            x = self.hadamard(x)

        if self.activation_bits < 16:
            x = self._quantize_activations(x)

        # Dequantize 2-bit weight on-the-fly
        w_dequant = self.get_dequantized_weight()
        
        # Retain gradient hook for optimizer
        if self.training:
            w_dequant.requires_grad_(True)
            w_dequant.retain_grad()
            self.ephemeral_w = w_dequant

        return F.linear(x, w_dequant, self.bias)
