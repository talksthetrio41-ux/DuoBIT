import torch
import torch.nn as nn

from duobit.layers.duobit_linear import DuobitLinear
from duobit.optim.duobit_adam import DuobitAdam
from duobit.quantization.codebook import SymmetricCodebook, compute_mse_group_scales, compute_group_scales
from duobit.quantization.stochastic import quantize_with_residual


def test_pefa_codes_keep_moving():
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = DuobitLinear(64, 32, group_size=32, n_levels=4)

        def forward(self, x):
            return self.lin(x)

    model = M()
    opt = DuobitAdam(model, lr=0.05, duobit_lr=0.05, use_pefa=True, scale_update_freq=0)
    x = torch.randn(16, 64)
    y = torch.randn(16, 32)
    initial = model.lin.codes.clone()
    changed_steps = 0
    for _ in range(40):
        opt.zero_grad()
        pred = model(x)
        (pred - y).pow(2).mean().backward()
        opt.step()
        if opt.last_transition_frac > 0:
            changed_steps += 1
    assert (model.lin.codes != initial).any(), "PEFA should flip some codes"
    assert changed_steps >= 5, f"expected sustained transitions, got {changed_steps} active steps"


def test_pefa_residual_is_exact():
    torch.manual_seed(1)
    cb = SymmetricCodebook(n_levels=4)
    w = torch.randn(8, 64)
    scales = compute_group_scales(w, group_size=32)
    codes = cb.quantize(w, scales)
    scales = compute_mse_group_scales(w, codes, cb, group_size=32)
    codes, w_q, resid = quantize_with_residual(w, scales, cb)
    recon = w_q + resid
    assert torch.allclose(recon, w, atol=1e-5)
    assert codes.max() <= 3


def test_ternary_and_binary_codebooks():
    w = torch.tensor([[-0.9, -0.1, 0.2, 0.8]])
    scales = torch.tensor([[[1.0]]])
    bin_cb = SymmetricCodebook(n_levels=2)
    ter_cb = SymmetricCodebook(n_levels=3)
    b_codes = bin_cb.quantize(w, scales)
    t_codes = ter_cb.quantize(w, scales)
    assert b_codes.tolist() == [[0, 0, 1, 1]]
    assert t_codes.tolist() == [[0, 1, 1, 2]]
    assert bin_cb.levels == [-1.0, 1.0]
    assert ter_cb.levels == [-1.0, 0.0, 1.0]
