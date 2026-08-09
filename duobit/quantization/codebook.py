import torch
import torch.nn as nn
from typing import Tuple, Union, Optional

class SymmetricCodebook:
    """
    Symmetric 4-level balanced codebook C = {-1, -rho, +rho, +1}
    mapped from 2-bit codes z in {0, 1, 2, 3}.
    """
    def __init__(self, rho: Union[float, torch.Tensor] = 1.0 / 3.0):
        self.rho = rho
        self._update_levels()

    def set_rho(self, rho: Union[float, torch.Tensor]):
        self.rho = rho
        self._update_levels()

    def _update_levels(self):
        # Code levels: 0 -> -1.0, 1 -> -rho, 2 -> +rho, 3 -> +1.0
        if isinstance(self.rho, torch.Tensor):
            r = self.rho.item()
        else:
            r = float(self.rho)
        self.levels = [-1.0, -r, r, 1.0]

    def get_levels_tensor(self, device: torch.device, dtype: torch.dtype = torch.float32, rho: Optional[torch.Tensor] = None) -> torch.Tensor:
        if rho is not None:
            r = rho.to(device=device, dtype=dtype)
            return torch.stack([-torch.tensor(1.0, device=device, dtype=dtype), -r, r, torch.tensor(1.0, device=device, dtype=dtype)])
        return torch.tensor(self.levels, device=device, dtype=dtype)

    def quantize(self, w: torch.Tensor, scale: torch.Tensor, rho: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Quantize high-precision continuous tensor w using per-group scale.
        w: (out_features, in_features) or (N_groups * group_size,)
        scale: (out_features, n_groups, 1) or broadcastable shape
        Returns:
            codes: torch.Tensor of dtype uint8 with values in {0, 1, 2, 3}
        """
        orig_shape = w.shape
        r_val = float(rho.item()) if rho is not None else (self.rho if isinstance(self.rho, (float, int)) else float(self.rho.item()))

        if scale.dim() == 3 and w.dim() == 2:
            out_f, in_f = w.shape
            n_groups = scale.shape[1]
            group_size = in_f // n_groups
            w_grouped = w.view(out_f, n_groups, group_size)
            w_scaled = w_grouped / (scale + 1e-8)
            w_scaled = w_scaled.view(orig_shape)
        else:
            w_scaled = w / (scale + 1e-8)
        
        m01 = -(1.0 + r_val) / 2.0
        m12 = 0.0
        m23 = (1.0 + r_val) / 2.0

        codes = torch.zeros_like(w, dtype=torch.uint8)
        codes = torch.where(w_scaled >= m01, torch.tensor(1, dtype=torch.uint8, device=w.device), codes)
        codes = torch.where(w_scaled >= m12, torch.tensor(2, dtype=torch.uint8, device=w.device), codes)
        codes = torch.where(w_scaled >= m23, torch.tensor(3, dtype=torch.uint8, device=w.device), codes)
        return codes

    def dequantize(self, codes: torch.Tensor, scale: torch.Tensor, rho: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Dequantize 2-bit codes z in {0, 1, 2, 3} back to float tensor.
        codes: (out_features, in_features)
        scale: (out_features, n_groups, 1) or broadcastable scale tensor
        Returns:
            w: continuous float tensor s_g * C[z_i]
        """
        orig_shape = codes.shape
        levels_t = self.get_levels_tensor(codes.device, dtype=scale.dtype, rho=rho)
        
        # Gather level value using codes as index: c_val has shape of codes
        c_val = levels_t[codes.long()]

        if scale.dim() == 3 and codes.dim() == 2:
            out_f, in_f = codes.shape
            n_groups = scale.shape[1]
            group_size = in_f // n_groups
            c_val_grouped = c_val.view(out_f, n_groups, group_size)
            w_grouped = scale * c_val_grouped
            return w_grouped.view(orig_shape)
        else:
            return scale * c_val

    def bit_serial_decompose(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        a = (1.0 + float(self.rho)) / 2.0
        b = (1.0 - float(self.rho)) / 2.0

        b0 = torch.where(codes >= 2, torch.tensor(1.0, device=codes.device), torch.tensor(-1.0, device=codes.device))
        b1 = torch.where((codes % 2) == 1, torch.tensor(1.0, device=codes.device), torch.tensor(-1.0, device=codes.device))
        
        return b0, b1, a, b


def compute_group_scales(w: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    orig_shape = w.shape
    if w.dim() == 2:
        out_f, in_f = orig_shape
        n_groups = in_f // group_size
        flat_w = w.view(out_f, n_groups, group_size)
        max_val = torch.amax(torch.abs(flat_w), dim=-1, keepdim=True)
        scales = torch.clamp(max_val, min=1e-5)
        return scales
    else:
        flat_w = w.reshape(-1, group_size)
        max_val = torch.amax(torch.abs(flat_w), dim=-1, keepdim=True)
        scales = torch.clamp(max_val, min=1e-5)
        return scales


def compute_mse_group_scales(w: torch.Tensor, codes: torch.Tensor, codebook: SymmetricCodebook, group_size: int = 128, rho: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes MSE-optimal group scale s_g = sum(w_i * c_i) / sum(c_i^2 + eps).
    """
    device = w.device
    dtype = w.dtype
    levels_t = codebook.get_levels_tensor(device, dtype=dtype, rho=rho)
    c_val = levels_t[codes.long()] # shape same as w

    if w.dim() == 2:
        out_f, in_f = w.shape
        n_groups = in_f // group_size
        w_g = w.view(out_f, n_groups, group_size)
        c_g = c_val.view(out_f, n_groups, group_size)

        num = torch.sum(w_g * c_g, dim=-1, keepdim=True)
        den = torch.sum(c_g ** 2, dim=-1, keepdim=True) + 1e-8

        s_mse = num / den
        return torch.clamp(s_mse, min=1e-5)
    else:
        flat_w = w.reshape(-1, group_size)
        flat_c = c_val.reshape(-1, group_size)

        num = torch.sum(flat_w * flat_c, dim=-1, keepdim=True)
        den = torch.sum(flat_c ** 2, dim=-1, keepdim=True) + 1e-8

        s_mse = num / den
        return torch.clamp(s_mse, min=1e-5)
