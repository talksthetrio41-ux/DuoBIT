import math
from typing import Dict

import torch
import torch.nn as nn

from duobit.layers.duobit_linear import DuobitLinear


def inference_memory_report(model: nn.Module) -> Dict[str, float]:
    """Packed inference memory: discrete codes + scales + unquantized tensors.

    Embeddings, RMSNorm gains, and biases stay FP32. Linear maps that are
    DuobitLinear contribute log2(L) bits per weight plus 32-bit group scales.
    """
    code_bits = 0.0
    scale_bytes = 0
    fp32_bytes = 0
    n_linear_w = 0
    n_other = 0

    counted = set()
    for module in model.modules():
        if isinstance(module, DuobitLinear):
            n = module.codes.numel()
            n_linear_w += n
            code_bits += math.log2(module.n_levels) * n
            scale_bytes += module.scales.numel() * 4
            if module.bias is not None:
                fp32_bytes += module.bias.numel() * 4
                n_other += module.bias.numel()
            counted.add(id(module.codes))
            counted.add(id(module.scales))
            if module.bias is not None:
                counted.add(id(module.bias))

    for p in model.parameters():
        if id(p) in counted:
            continue
        fp32_bytes += p.numel() * 4
        n_other += p.numel()

    code_bytes = math.ceil(code_bits / 8.0)
    total = code_bytes + scale_bytes + fp32_bytes
    n_params = n_linear_w + n_other
    return {
        "n_params": float(n_params),
        "n_linear_weights": float(n_linear_w),
        "code_bytes": float(code_bytes),
        "scale_bytes": float(scale_bytes),
        "fp32_bytes": float(fp32_bytes),
        "total_bytes": float(total),
        "total_mb": total / (1024.0 * 1024.0),
        "avg_bits_per_param": (8.0 * total / n_params) if n_params else 0.0,
        "linear_bits_per_weight": (
            (8.0 * (code_bytes + scale_bytes) / n_linear_w) if n_linear_w else 0.0
        ),
    }


def fp32_memory_mb(n_params: int) -> float:
    return n_params * 4 / (1024.0 * 1024.0)
