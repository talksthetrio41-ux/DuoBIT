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


class DuobitEmbedding(nn.Module):
    """Token embedding table stored as discrete codes + per-group scales.

    A lookup table is just another matrix, so this reuses `DuobitLinear`'s
    persistent representation exactly -- uint8 codes, FP32 per-group scales,
    no master weight -- and exposes the same attributes, which lets
    `DuobitAdam`, the memory report and the DDP consistency check treat both
    through one path. The dequantized table is ephemeral: materialised for the
    lookup, then discarded.

    This is the dominant compression lever on a real decoder. In the 77M-param
    FineWeb-EDU reference model the FP32 embedding table is 71% of persistent
    training state and 88% of the inference footprint, against 29% and 12% for
    all the 2-bit linear maps put together.

    Note that embedding gradients are extremely sparse -- only the rows for
    tokens present in the batch receive one, ~93-95% zero per step at typical
    batch sizes -- so the QPEFA residual, which accumulates sub-threshold
    updates per row until they cross a codebook gap, is doing most of the work
    here.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        group_size: int = 128,
        rho: float = 1.0 / 3.0,
        n_levels: int = 4,
        init_std: float = 0.0,
        scale_init: str = "var",
    ):
        super().__init__()
        if embedding_dim % group_size != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # aliases so the optimizer's factored second moment can size itself the
        # same way it does for a linear map
        self.out_features = num_embeddings
        self.in_features = embedding_dim
        self.group_size = group_size
        self.n_levels = n_levels
        self.rho = rho
        # nn.Embedding initialises N(0, 1); keep that unless told otherwise
        self.init_std = float(init_std) if init_std > 0 else 1.0
        self.scale_init = scale_init
        self.codebook = SymmetricCodebook(rho=rho, n_levels=n_levels)

        n_groups = embedding_dim // group_size
        self.register_buffer(
            "codes", torch.zeros((num_embeddings, embedding_dim), dtype=torch.uint8)
        )
        self.register_buffer(
            "scales", torch.ones((num_embeddings, n_groups, 1), dtype=torch.float32)
        )
        self.register_buffer(
            "scales_init", torch.ones((num_embeddings, n_groups, 1), dtype=torch.float32)
        )
        self.bias = None
        self.ephemeral_w: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self):
        init_w = torch.empty((self.num_embeddings, self.embedding_dim))
        init_w.normal_(0.0, self.init_std)
        scales = compute_group_scales(init_w, group_size=self.group_size)
        codes = self.codebook.quantize(init_w, scales)
        fit = compute_var_group_scales if self.scale_init == "var" \
            else compute_mse_group_scales
        fitted = fit(init_w, codes, self.codebook, group_size=self.group_size)
        self.codes.copy_(codes)
        self.scales.copy_(fitted)
        self.scales_init.copy_(fitted)

    def get_dequantized_weight(self) -> torch.Tensor:
        return self.codebook.dequantize(self.codes, self.scales)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w = self.get_dequantized_weight()
        if self.training:
            w.requires_grad_(True)
            w.retain_grad()
            self.ephemeral_w = w
        return F.embedding(input_ids, w)

    def inference_nbytes(self) -> int:
        """Packed inference footprint: codes at log2(L) bits + FP32 scales."""
        code_bits = math.log2(self.n_levels) * self.codes.numel()
        return int(math.ceil(code_bits / 8.0)) + self.scales.numel() * 4
