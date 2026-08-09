import torch
import torch.nn as nn
import pytest
from duobit.config import DuobitConfig
from duobit.layers.duobit_linear import DuobitLinear
from duobit.optim.duobit_adam import DuobitAdam
from duobit.model.transformer import DuobitTransformer
from duobit.quantization.stochastic import compute_transition_probs, balanced_stochastic_round

def test_transition_temperature_and_min_k():
    torch.manual_seed(42)
    w_curr = torch.randn(64, 128)
    w_cand = w_curr + 0.1 * torch.randn_like(w_curr)
    codes = torch.randint(0, 4, (64, 128), dtype=torch.uint8)
    scales = torch.ones(64, 1, 1) * 0.5
    from duobit.quantization.codebook import SymmetricCodebook
    codebook = SymmetricCodebook()

    # Temperature = 0.5 should increase transition probabilities vs temp = 2.0
    probs_hot, target_codes_hot, _ = compute_transition_probs(
        w_curr, w_cand, codes, scales, codebook, temperature=0.5
    )
    probs_cold, target_codes_cold, _ = compute_transition_probs(
        w_curr, w_cand, codes, scales, codebook, temperature=2.0
    )

    assert probs_hot.mean() > probs_cold.mean()

    # Test min_transitions_per_group
    new_codes_min = balanced_stochastic_round(
        probs_cold, target_codes_cold, codes, group_size=128, min_transitions_per_group=5
    )
    changes_per_group = (new_codes_min != codes).sum(dim=-1)
    assert (changes_per_group >= 5).all()

def test_scale_ema_update():
    torch.manual_seed(42)
    layer = DuobitLinear(in_features=128, out_features=64, group_size=128)
    model = nn.Sequential(layer)
    optimizer = DuobitAdam(model, lr=0.01, scale_update_freq=1, scale_ema_alpha=0.1)

    initial_scales = layer.scales.clone()
    x = torch.randn(8, 128)
    for _ in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = out.pow(2).sum()
        loss.backward()
        optimizer.step()

    # Over 10 steps, codes transition and scales adapt via EMA
    # Verify scale updates work without NaN/Inf
    assert not torch.isnan(layer.scales).any()
    assert not torch.isinf(layer.scales).any()
    assert (layer.scales > 0).all()


def test_swiglu_transformer():
    config = DuobitConfig(
        vocab_size=100,
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_ff=256,
        use_swiglu=True,
        group_size=64,
    )
    model = DuobitTransformer(config, use_duobit=True)
    input_ids = torch.randint(0, 100, (2, 16))
    logits, loss = model(input_ids, targets=input_ids)
    assert logits.shape == (2, 16, 100)
    assert loss is not None
