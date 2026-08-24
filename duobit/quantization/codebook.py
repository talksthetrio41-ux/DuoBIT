import math
from typing import Optional, Tuple, Union

import torch


class SymmetricCodebook:
    """Symmetric codebook over n_levels in {2, 3, 4}.

    Binary (1-bit):     C = {-1, +1}
    Ternary (~1.58-bit): C = {-1, 0, +1}
    2-bit:              C = {-1, -ρ, +ρ, +1} with ρ ≈ 1/3
    """

    def __init__(self, rho: Union[float, torch.Tensor] = 1.0 / 3.0, n_levels: int = 4):
        if n_levels not in (2, 3, 4):
            raise ValueError(f"n_levels must be 2, 3, or 4, got {n_levels}")
        self.rho = rho
        self.n_levels = n_levels
        self._update_levels()

    def set_rho(self, rho: Union[float, torch.Tensor]):
        self.rho = rho
        self._update_levels()

    def _update_levels(self):
        if isinstance(self.rho, torch.Tensor):
            r = float(self.rho.item())
        else:
            r = float(self.rho)
        if self.n_levels == 2:
            self.levels = [-1.0, 1.0]
        elif self.n_levels == 3:
            self.levels = [-1.0, 0.0, 1.0]
        else:
            self.levels = [-1.0, -r, r, 1.0]

    def get_levels_tensor(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.n_levels == 2:
            return torch.tensor([-1.0, 1.0], device=device, dtype=dtype)
        if self.n_levels == 3:
            return torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
        if rho is not None:
            r = rho.to(device=device, dtype=dtype).reshape(())
            one = torch.ones((), device=device, dtype=dtype)
            return torch.stack([-one, -r, r, one])
        return torch.tensor(self.levels, device=device, dtype=dtype)

    def _apply_scale_div(self, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if scale.dim() == 3 and w.dim() == 2:
            out_f, in_f = w.shape
            n_groups = scale.shape[1]
            group_size = in_f // n_groups
            w_grouped = w.view(out_f, n_groups, group_size)
            return (w_grouped / (scale + 1e-8)).view(w.shape)
        return w / (scale + 1e-8)

    def _apply_scale_mul(self, c_val: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if scale.dim() == 3 and c_val.dim() == 2:
            out_f, in_f = c_val.shape
            n_groups = scale.shape[1]
            group_size = in_f // n_groups
            c_grouped = c_val.view(out_f, n_groups, group_size)
            return (scale * c_grouped).view(c_val.shape)
        return scale * c_val

    def quantize(
        self,
        w: torch.Tensor,
        scale: torch.Tensor,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Nearest-neighbour quantize onto the codebook after per-group scaling."""
        w_scaled = self._apply_scale_div(w, scale)
        levels_t = self.get_levels_tensor(w.device, dtype=w.dtype, rho=rho)
        dist = (w_scaled.unsqueeze(-1) - levels_t).abs()
        return dist.argmin(dim=-1).to(torch.uint8)

    def dequantize(
        self,
        codes: torch.Tensor,
        scale: torch.Tensor,
        rho: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        levels_t = self.get_levels_tensor(codes.device, dtype=scale.dtype, rho=rho)
        n = levels_t.numel()
        c_val = levels_t[codes.long().clamp(0, n - 1)]
        return self._apply_scale_mul(c_val, scale)

    def bit_serial_decompose(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """Decompose 2-bit codes as a*b0 + b*b1 with b0,b1 in {-1,+1}."""
        if self.n_levels != 4:
            raise ValueError("bit-serial decomposition is defined for 4-level (2-bit) codes")
        a = (1.0 + float(self.rho)) / 2.0
        b = (1.0 - float(self.rho)) / 2.0
        b0 = torch.where(codes >= 2, torch.ones_like(codes, dtype=torch.float32), -torch.ones_like(codes, dtype=torch.float32))
        b1 = torch.where((codes % 2) == 1, torch.ones_like(codes, dtype=torch.float32), -torch.ones_like(codes, dtype=torch.float32))
        return b0, b1, a, b


def compute_group_scales(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    if w.dim() == 2:
        out_f, in_f = w.shape
        n_groups = in_f // group_size
        flat_w = w.view(out_f, n_groups, group_size)
        max_val = torch.amax(torch.abs(flat_w), dim=-1, keepdim=True)
        return torch.clamp(max_val, min=1e-5)
    flat_w = w.reshape(-1, group_size)
    max_val = torch.amax(torch.abs(flat_w), dim=-1, keepdim=True)
    return torch.clamp(max_val, min=1e-5)


def compute_mse_group_scales(
    w: torch.Tensor,
    codes: torch.Tensor,
    codebook: SymmetricCodebook,
    group_size: int = 128,
    rho: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Closed-form MSE-optimal group scale s* = Σ w_i c_i / (Σ c_i² + ε)."""
    levels_t = codebook.get_levels_tensor(w.device, dtype=w.dtype, rho=rho)
    c_val = levels_t[codes.long().clamp(0, levels_t.numel() - 1)]

    if w.dim() == 2:
        out_f, in_f = w.shape
        n_groups = in_f // group_size
        w_g = w.view(out_f, n_groups, group_size)
        c_g = c_val.view(out_f, n_groups, group_size)
        num = torch.sum(w_g * c_g, dim=-1, keepdim=True)
        den = torch.sum(c_g ** 2, dim=-1, keepdim=True) + 1e-8
        return torch.clamp(num / den, min=1e-5)

    flat_w = w.reshape(-1, group_size)
    flat_c = c_val.reshape(-1, group_size)
    num = torch.sum(flat_w * flat_c, dim=-1, keepdim=True)
    den = torch.sum(flat_c ** 2, dim=-1, keepdim=True) + 1e-8
    return torch.clamp(num / den, min=1e-5)


def bits_per_weight(n_levels: int, group_size: int, scale_bits: int = 32) -> float:
    """Inference bits/weight including per-group scale overhead."""
    code_bits = math.log2(n_levels)
    return code_bits + scale_bits / float(group_size)
