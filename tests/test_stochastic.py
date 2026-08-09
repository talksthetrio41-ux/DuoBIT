import torch
import pytest
from duobit.quantization.codebook import SymmetricCodebook
from duobit.quantization.stochastic import compute_transition_probs, balanced_stochastic_round

def test_transition_probs():
    cb = SymmetricCodebook(rho=1.0/3.0)
    
    # Current codes: [1, 2] (levels -1/3, +1/3)
    codes = torch.tensor([[1, 2]], dtype=torch.uint8)
    scales = torch.tensor([[1.0]], dtype=torch.float32)
    w_curr = cb.dequantize(codes, scales) # [-1/3, 1/3]
    
    # Candidate weights slightly higher: [-1/3 + 0.1, 1/3 + 0.2]
    # For code 1 (-1/3), adjacent level up is code 2 (+1/3), gap = 2/3
    # Step = 0.1, so prob = 0.1 / (2/3) = 0.15
    w_cand = w_curr + torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    
    probs, target_codes, trust_mask = compute_transition_probs(
        w_curr, w_cand, codes, scales, cb, trust_threshold=2.0
    )
    
    assert torch.equal(target_codes, torch.tensor([[2, 3]], dtype=torch.uint8))
    assert torch.allclose(probs[0, 0], torch.tensor(0.1 / (2.0/3.0)), atol=1e-4)

def test_balanced_stochastic_round():
    # Group of size 4 with transition probs sum to 1.0 => exactly 1 element should transition
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32)
    current_codes = torch.tensor([[1, 1, 1, 1]], dtype=torch.uint8)
    target_codes = torch.tensor([[2, 2, 2, 2]], dtype=torch.uint8)
    
    new_codes = balanced_stochastic_round(
        probs, target_codes, current_codes, group_size=4, stochastic=False
    )
    
    # Deterministic top-1 should transition index 0 (prob 0.4)
    expected = torch.tensor([[2, 1, 1, 1]], dtype=torch.uint8)
    assert torch.equal(new_codes, expected)
