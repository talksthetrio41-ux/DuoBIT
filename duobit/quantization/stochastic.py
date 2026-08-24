from typing import Tuple

import torch

from duobit.quantization.codebook import SymmetricCodebook


def compute_transition_probs(
    w_current: torch.Tensor,
    w_candidate: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor,
    codebook: SymmetricCodebook,
    trust_threshold: float = 2.0,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adjacent-level transition probabilities.

    p_i = clip(|w̃_i - w_i| / (τ · s_g · |c_k - c_j|), 0, 1)

    Trust gating zeros probabilities whose residual exceeds
    trust_threshold · gap. Callers that accumulate residuals (PEFA)
    should pass a large trust_threshold or use `quantize_with_residual`.
    """
    device = codes.device
    dtype = w_current.dtype
    levels_t = codebook.get_levels_tensor(device, dtype=dtype)
    n_levels = levels_t.numel()

    diff = w_candidate - w_current
    direction = torch.sign(diff).to(torch.int16)

    target_codes = torch.clamp(codes.to(torch.int16) + direction, 0, n_levels - 1).to(torch.uint8)

    c_j = levels_t[codes.long().clamp(0, n_levels - 1)]
    c_k = levels_t[target_codes.long()]
    c_diff = c_k - c_j

    if scale.dim() == 3 and c_diff.dim() == 2:
        out_f, in_f = c_diff.shape
        n_groups = scale.shape[1]
        group_size = in_f // n_groups
        scaled_c_diff = (scale * c_diff.view(out_f, n_groups, group_size)).view(out_f, in_f)
    else:
        scaled_c_diff = scale * c_diff

    temp = max(1e-4, float(temperature))
    level_gap = torch.abs(scaled_c_diff) * temp
    safe_gap = torch.where(level_gap > 1e-8, level_gap, torch.ones((), device=device, dtype=dtype))

    probs = torch.clip(torch.abs(diff) / safe_gap, 0.0, 1.0)
    probs = torch.where(target_codes == codes, torch.zeros((), device=device, dtype=dtype), probs)

    trust_mask = torch.abs(diff) < (trust_threshold * safe_gap)
    gated_probs = torch.where(trust_mask, probs, torch.zeros((), device=device, dtype=dtype))
    return gated_probs, target_codes, trust_mask


def balanced_stochastic_round(
    probs: torch.Tensor,
    target_codes: torch.Tensor,
    current_codes: torch.Tensor,
    group_size: int = 128,
    stochastic: bool = True,
    min_transitions_per_group: int = 0,
) -> torch.Tensor:
    """Group-balanced Gumbel-TopK rounding.

    K_g = round(Σ_{i∈g} p_i) codes transition inside each group, so the
    expected number of flips matches Σ p_i (unbiased, low variance).
    """
    orig_shape = probs.shape
    device = probs.device

    flat_p = probs.reshape(-1, group_size)
    flat_target = target_codes.reshape(-1, group_size)
    flat_current = current_codes.reshape(-1, group_size)

    expected_k = torch.sum(flat_p, dim=-1)
    K_g = torch.round(expected_k).long()
    if min_transitions_per_group > 0:
        valid_counts = torch.sum((flat_target != flat_current).long(), dim=-1)
        min_k_tensor = torch.minimum(
            torch.tensor(min_transitions_per_group, device=device), valid_counts
        )
        K_g = torch.maximum(K_g, min_k_tensor)

    if stochastic:
        eps = 1e-10
        p_clamped = torch.clamp(flat_p, min=eps, max=1.0 - eps)
        u = torch.rand_like(flat_p)
        gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
        score = torch.log(p_clamped) + gumbel_noise
    else:
        score = flat_p

    valid_mask = flat_target != flat_current
    score = torch.where(valid_mask, score, torch.full((), -1e9, device=device, dtype=score.dtype))

    sorted_indices = torch.argsort(score, dim=-1, descending=True)
    ranks = torch.argsort(sorted_indices, dim=-1)
    selected_mask = (ranks < K_g.unsqueeze(-1)) & valid_mask

    flat_new = torch.where(selected_mask, flat_target, flat_current)
    return flat_new.reshape(orig_shape)


def quantize_with_residual(
    w_virtual: torch.Tensor,
    scale: torch.Tensor,
    codebook: SymmetricCodebook,
    rho=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest codebook reconstruction of a virtual (continuous) tensor.

    Returns (codes, w_quant, residual) with residual = w_virtual - w_quant.
    This is the PEFA / error-feedback quantizer.
    """
    codes = codebook.quantize(w_virtual, scale, rho=rho)
    w_quant = codebook.dequantize(codes, scale, rho=rho)
    residual = w_virtual - w_quant
    return codes, w_quant, residual
