import torch
from typing import Tuple, Optional
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
    """
    Computes adjacent-level transition probabilities p_i, transition target codes next_codes,
    and trust gating masks.

    w_current: s_g * C[z_t]
    w_candidate: W_tilde_{t+1}
    codes: z_t in {0, 1, 2, 3}
    scale: per-group scale s_g of shape (out_f, n_groups, 1) or broadcastable
    temperature: scaling factor for level gap (lower = higher probability)

    Returns:
        probs: transition probabilities p_i in [0, 1]
        target_codes: z_target (adjacent level code z_t + 1 or z_t - 1)
        trust_mask: boolean mask indicating if transition is trusted
    """
    device = codes.device
    dtype = w_current.dtype
    levels_t = codebook.get_levels_tensor(device, dtype=dtype)

    diff = w_candidate - w_current
    # Direction: +1 if stepping up, -1 if stepping down, 0 if no step
    direction = torch.sign(diff).to(torch.int8)

    # Adjacent code target
    target_codes = torch.clamp(codes.to(torch.int16) + direction, 0, 3).to(torch.uint8)

    c_j = levels_t[codes.long()]
    c_k = levels_t[target_codes.long()]

    c_diff = c_k - c_j
    if scale.dim() == 3 and c_diff.dim() == 2:
        out_f, in_f = c_diff.shape
        n_groups = scale.shape[1]
        group_size = in_f // n_groups
        c_diff_g = c_diff.view(out_f, n_groups, group_size)
        scaled_c_diff = scale * c_diff_g
        scaled_c_diff = scaled_c_diff.view(out_f, in_f)
    else:
        scaled_c_diff = scale * c_diff

    # Level gap in continuous space: s_g * |c_k - c_j| * temperature
    temp = max(1e-4, temperature)
    level_gap = torch.abs(scaled_c_diff) * temp
    
    # Avoid div by zero when target_codes == codes (i.e. at boundaries or direction == 0)
    safe_gap = torch.where(level_gap > 1e-8, level_gap, torch.tensor(1.0, device=device, dtype=dtype))

    # Raw transition probability p_i = clip(|w_tilde - w| / gap, 0, 1)
    probs = torch.clip(torch.abs(diff) / safe_gap, 0.0, 1.0)
    probs = torch.where(target_codes == codes, torch.tensor(0.0, device=device, dtype=dtype), probs)

    # Trust gating: if residual |w_tilde - w| exceeds threshold * level_gap, gate out transition
    trust_mask = torch.abs(diff) < (trust_threshold * safe_gap)
    gated_probs = torch.where(trust_mask, probs, torch.tensor(0.0, device=device, dtype=dtype))

    return gated_probs, target_codes, trust_mask


def balanced_stochastic_round(
    probs: torch.Tensor,
    target_codes: torch.Tensor,
    current_codes: torch.Tensor,
    group_size: int = 128,
    stochastic: bool = True,
    min_transitions_per_group: int = 0,
) -> torch.Tensor:
    """
    Applies balanced group stochastic rounding:
    Within each group g, K_g = max(min_k, round(sum_{i in g} p_i)) weights transition to target_codes.
    
    probs shape: (..., N)
    target_codes shape: (..., N)
    current_codes shape: (..., N)
    
    Returns:
        new_codes: (..., N) updated 2-bit codes in {0, 1, 2, 3}
    """
    orig_shape = probs.shape
    device = probs.device
    
    flat_p = probs.reshape(-1, group_size)
    flat_target = target_codes.reshape(-1, group_size)
    flat_current = current_codes.reshape(-1, group_size)

    # Target transition count per group
    expected_k = torch.sum(flat_p, dim=-1)
    K_g = torch.round(expected_k).long() # (N_groups,)
    if min_transitions_per_group > 0:
        # Enforce minimum transition count when valid transitions exist
        valid_counts = torch.sum((flat_target != flat_current).long(), dim=-1)
        min_k_tensor = torch.minimum(torch.tensor(min_transitions_per_group, device=device), valid_counts)
        K_g = torch.maximum(K_g, min_k_tensor)

    if stochastic:
        # Gumbel-topK trick: add Gumbel noise to log(p_i) to sample top K_g without replacement proportional to p_i
        eps = 1e-10
        p_clamped = torch.clamp(flat_p, min=eps, max=1.0 - eps)
        # Uniform noise
        u = torch.rand_like(flat_p)
        gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
        score = torch.log(p_clamped) + gumbel_noise
    else:
        score = flat_p

    # Mask non-transitions to avoid selecting elements where target_codes == current_codes
    valid_mask = flat_target != flat_current
    score = torch.where(valid_mask, score, torch.tensor(-1e9, device=device, dtype=score.dtype))

    # Sort elements per group by score descending
    sorted_indices = torch.argsort(score, dim=-1, descending=True)
    
    # Create rank mask: element at position r gets selected if r < K_g
    ranks = torch.argsort(sorted_indices, dim=-1) # (N_groups, group_size)
    K_g_expanded = K_g.unsqueeze(-1) # (N_groups, 1)
    
    selected_mask = (ranks < K_g_expanded) & valid_mask # (N_groups, group_size)

    # Apply transition to selected elements
    flat_new = torch.where(selected_mask, flat_target, flat_current)
    return flat_new.reshape(orig_shape)

