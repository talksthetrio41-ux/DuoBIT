import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from duobit.layers import DUOBIT_MODULES, DuobitLinear
from duobit.layers.ste_linear import SteQuantLinear
from duobit.quantization.blockwise import blockwise_nbytes
from duobit.quantization.qpefa import residual_nbytes


def packed_code_bytes(n_weights: int, n_levels: int) -> int:
    return int(math.ceil(math.log2(n_levels) * n_weights / 8.0))


def inference_memory_report(model: nn.Module) -> Dict[str, float]:
    """Packed inference memory: discrete codes + scales + unquantized tensors.

    Embeddings, RMSNorm gains, and biases stay FP32. Linear maps that are
    DuobitLinear contribute log2(L) bits per weight plus 32-bit group scales.
    STE latent-weight layers are counted as FP32 (they are a training pattern).
    """
    code_bits = 0.0
    scale_bytes = 0
    fp32_bytes = 0
    n_linear_w = 0
    n_other = 0

    counted = set()
    for module in model.modules():
        if isinstance(module, DUOBIT_MODULES):
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
        elif isinstance(module, SteQuantLinear):
            n = module.weight.numel()
            n_linear_w += n
            fp32_bytes += n * 4
            counted.add(id(module.weight))
            if module.bias is not None:
                fp32_bytes += module.bias.numel() * 4
                n_other += module.bias.numel()
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


def linear_training_bytes(
    n_weights: int,
    n_levels: int,
    group_size: int,
    qpefa_bits: int,
    moment_bits: int,
    factored_second_moment: bool,
    block_size: int,
    out_features: int,
    in_features: int,
    learn_scales: bool = False,
) -> Dict[str, int]:
    """Persistent training bytes for one discrete linear map."""
    codes = packed_code_bytes(n_weights, n_levels)
    scales = 4 * (n_weights // group_size)
    residual = residual_nbytes(n_weights, qpefa_bits)
    first_moment = blockwise_nbytes(n_weights, moment_bits, block_size)
    if factored_second_moment:
        second_moment = 4 * (out_features + in_features)
    else:
        second_moment = blockwise_nbytes(n_weights, moment_bits, block_size)
    # Training the scales costs, per group: two FP32 Adam moments plus the FP32
    # init-scale reference that floors them. 3*32/G bits per weight, 0.75 at
    # G=128. All three are optimizer state and are dropped at inference.
    scale_moments = 12 * (n_weights // group_size) if learn_scales else 0
    total = (codes + scales + residual + first_moment + second_moment
             + scale_moments)
    return {
        "codes": codes,
        "scales": scales,
        "residual": residual,
        "first_moment": first_moment,
        "second_moment": second_moment,
        "scale_moments": scale_moments,
        "total": total,
    }


def fp32_adam_training_bytes(n_weights: int) -> int:
    """W + m + v, all FP32."""
    return n_weights * 12


def bitnet_training_bytes(n_weights: int, weight_bytes: int = 4) -> int:
    """Latent W (FP32 or FP16) + FP32 Adam m,v."""
    return n_weights * (weight_bytes + 8)


def training_memory_report(
    model: nn.Module,
    optimizer: Optional[Any] = None,
) -> Dict[str, float]:
    """Persistent training memory: packed codes, QPEFA, compressed moments, FP32 rest.

    Does not include activations or the ephemeral dequantized matmul tile.
    That peak is shared with BitNet-style STE (both materialise a dequantized
    weight for GEMM). The number that differs is the *persistent* state.
    """
    inf = inference_memory_report(model)
    qpefa_bits = 8
    moment_bits = 8
    factored = True
    block_size = 128
    learn_scales = False
    if optimizer is not None and getattr(optimizer, "param_groups", None):
        g = optimizer.param_groups[0]
        qpefa_bits = int(g.get("qpefa_bits", qpefa_bits))
        moment_bits = int(g.get("moment_bits", moment_bits))
        factored = bool(g.get("factored_second_moment", factored))
        block_size = int(g.get("block_size", block_size))
        learn_scales = bool(g.get("learn_scales", False))

    linear_train = 0
    linear_fp32_adam = 0
    linear_bitnet = 0
    n_linear = 0
    for module in model.modules():
        if isinstance(module, DUOBIT_MODULES):
            n = module.codes.numel()
            n_linear += n
            br = linear_training_bytes(
                n_weights=n,
                n_levels=module.n_levels,
                group_size=module.group_size,
                qpefa_bits=qpefa_bits,
                moment_bits=moment_bits,
                factored_second_moment=factored,
                block_size=block_size,
                out_features=module.out_features,
                in_features=module.in_features,
                learn_scales=learn_scales,
            )
            linear_train += br["total"]
            linear_fp32_adam += fp32_adam_training_bytes(n)
            linear_bitnet += bitnet_training_bytes(n, weight_bytes=4)
        elif isinstance(module, SteQuantLinear):
            n = module.weight.numel()
            n_linear += n
            linear_train += bitnet_training_bytes(n, weight_bytes=4)
            linear_fp32_adam += fp32_adam_training_bytes(n)
            linear_bitnet += bitnet_training_bytes(n, weight_bytes=4)
        elif isinstance(module, nn.Linear) and not isinstance(module, DUOBIT_MODULES):
            n = module.weight.numel()
            n_linear += n
            linear_train += fp32_adam_training_bytes(n)
            linear_fp32_adam += fp32_adam_training_bytes(n)
            linear_bitnet += fp32_adam_training_bytes(n)

    other = int(inf["fp32_bytes"])
    # FP32 tensors (embeddings, norms) still carry Adam m,v.
    other_train = other * 3
    total = linear_train + other_train
    fp32_total = linear_fp32_adam + other_train
    bitnet_total = linear_bitnet + other_train
    return {
        "n_params": inf["n_params"],
        "n_linear_weights": float(n_linear),
        "linear_train_bytes": float(linear_train),
        "other_train_bytes": float(other_train),
        "total_bytes": float(total),
        "total_mb": total / (1024.0 * 1024.0),
        "fp32_adam_linear_bytes": float(linear_fp32_adam),
        "fp32_adam_total_bytes": float(fp32_total),
        "fp32_adam_total_mb": fp32_total / (1024.0 * 1024.0),
        "bitnet_linear_bytes": float(linear_bitnet),
        "bitnet_total_bytes": float(bitnet_total),
        "bitnet_total_mb": bitnet_total / (1024.0 * 1024.0),
        "linear_compression_vs_fp32": (
            (linear_fp32_adam / linear_train) if linear_train else 0.0
        ),
        "linear_bits_per_weight_train": (
            (8.0 * linear_train / n_linear) if n_linear else 0.0
        ),
        "inference_mb": inf["total_mb"],
        "qpefa_bits": float(qpefa_bits),
        "moment_bits": float(moment_bits),
        "factored_second_moment": 1.0 if factored else 0.0,
    }
