"""Tests for the trainable group scale (v2) and the initialization fit.

The v1 method had exactly one continuous degree of freedom -- the per-group
scale s_g -- and never trained it: the "EMA scale update" recomputed the
MSE-optimal scale from the already-dequantized weight, which is an algebraic
fixed point. These tests pin down the fixed point, the exact scale gradient
that replaces it, and the variance-preserving initialization.
"""
import math

import torch
import torch.nn as nn

from duobit.layers.duobit_linear import DuobitLinear
from duobit.optim.duobit_adam import DuobitAdam
from duobit.quantization.codebook import (
    SymmetricCodebook,
    compute_group_scales,
    compute_mse_group_scales,
    compute_scale_grad,
    compute_var_group_scales,
)
from duobit.quantization.memory import training_memory_report


def test_mse_rescale_on_dequantized_weight_is_a_fixed_point():
    """The v1 scale update could not move the scales, by construction."""
    torch.manual_seed(0)
    cb = SymmetricCodebook(rho=1.0 / 3.0, n_levels=4)
    w = torch.randn(8, 256) * 0.03
    s = compute_group_scales(w, group_size=128)
    codes = cb.quantize(w, s)
    s = compute_mse_group_scales(w, codes, cb, group_size=128)

    for _ in range(5):
        w_post = cb.dequantize(codes, s)          # w_post = s * C[z] exactly
        s_target = compute_mse_group_scales(w_post, codes, cb, group_size=128)
        assert torch.allclose(s_target, s, rtol=1e-5, atol=1e-9)
        s = 0.99 * s + 0.01 * s_target


def test_scale_gradient_matches_autograd():
    """dL/ds_g = sum_{i in g} g_i C[z_i] is exact, not an approximation."""
    torch.manual_seed(0)
    cb = SymmetricCodebook(rho=1.0 / 3.0, n_levels=4)
    out_f, in_f, gsz = 16, 256, 128
    n_groups = in_f // gsz
    codes = torch.randint(0, 4, (out_f, in_f), dtype=torch.uint8)
    s = (torch.rand(out_f, n_groups, 1) * 0.05 + 0.01).requires_grad_(True)

    c = cb.get_levels_tensor(codes.device)[codes.long()].view(out_f, n_groups, gsz)
    w = (c * s).view(out_f, in_f)
    upstream = torch.randn(out_f, in_f)
    (w * upstream).sum().backward()

    analytic = compute_scale_grad(upstream, codes, cb, group_size=gsz)
    assert torch.allclose(analytic, s.grad, atol=1e-5)


def test_var_group_scales_preserve_weight_std():
    """The MSE fit shrinks the weight std; the variance fit does not."""
    torch.manual_seed(0)
    cb = SymmetricCodebook(rho=1.0 / 3.0, n_levels=4)
    for fan_in in (256, 512, 1024):
        w = torch.empty(512, fan_in)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        s0 = compute_group_scales(w, group_size=128)
        codes = cb.quantize(w, s0)

        s_mse = compute_mse_group_scales(w, codes, cb, group_size=128)
        s_var = compute_var_group_scales(w, codes, cb, group_size=128)
        r_mse = float(cb.dequantize(codes, s_mse).std() / w.std())
        r_var = float(cb.dequantize(codes, s_var).std() / w.std())

        assert r_mse < 0.97, f"expected MSE shrinkage, got {r_mse}"
        assert abs(r_var - 1.0) < 0.02, f"variance fit off: {r_var}"


def test_layer_records_init_scale_for_the_floor():
    layer = DuobitLinear(256, 64, group_size=128, n_levels=4)
    assert torch.equal(layer.scales, layer.scales_init)
    assert layer.scales_init.shape == layer.scales.shape


def _fit_one_layer(learn_scales: bool, steps: int = 120, seed: int = 0):
    """Fit a fixed linear target with a single DuoBIT layer; return final loss."""
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = DuobitLinear(128, 64, group_size=32, n_levels=4)

        def forward(self, x):
            return self.lin(x)

    model = M()
    opt = DuobitAdam(
        model,
        lr=0.0,
        duobit_lr=0.02,
        weight_decay=0.0,
        duobit_weight_decay=0.0,
        scale_update_freq=0,
        learn_scales=learn_scales,
        scale_lr=0.02,
        scale_relative_lr=True,
    )
    x = torch.randn(64, 128)
    # a target the layer cannot reach at its init magnitude, so the scale has
    # to grow for the loss to come down
    target = x @ (torch.randn(128, 64) * 0.35)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad()
        loss = (model(x) - target).pow(2).mean()
        loss.backward()
        opt.step()
    return float(loss.detach()), model.lin


def test_learned_scales_move_and_improve_the_fit():
    loss_off, layer_off = _fit_one_layer(learn_scales=False)
    loss_on, layer_on = _fit_one_layer(learn_scales=True)

    ratio_off = float((layer_off.scales / layer_off.scales_init).mean())
    ratio_on = float((layer_on.scales / layer_on.scales_init).mean())

    assert abs(ratio_off - 1.0) < 1e-6, "scales must stay frozen when not learned"
    assert ratio_on > 1.05, f"scales should grow toward the target, got {ratio_on}"
    assert loss_on < loss_off, f"learned scales should fit better: {loss_on} vs {loss_off}"


def test_scale_floor_is_respected():
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = DuobitLinear(64, 32, group_size=32, n_levels=4)

        def forward(self, x):
            return self.lin(x)

    model = M()
    opt = DuobitAdam(model, lr=0.0, duobit_lr=0.0, weight_decay=0.0,
                     learn_scales=True, scale_lr=0.5, scale_min_frac=0.25,
                     scale_update_freq=0)
    x = torch.randn(32, 64)
    for _ in range(60):
        opt.zero_grad()
        # drive every scale toward zero
        model(x).pow(2).mean().backward()
        opt.step()
    floor = model.lin.scales_init * 0.25
    assert torch.all(model.lin.scales >= floor - 1e-9)


def test_training_state_accounts_for_the_trained_scales():
    """Trained scales cost 3*32/G bits/weight and must show up in the report."""
    torch.manual_seed(0)
    model = nn.Sequential(DuobitLinear(256, 256, group_size=64, n_levels=4))
    base = training_memory_report(model, DuobitAdam(model, learn_scales=False))
    with_scales = training_memory_report(model, DuobitAdam(model, learn_scales=True))

    n = 256 * 256
    delta_bits = (with_scales["linear_train_bytes"] - base["linear_train_bytes"]) * 8 / n
    assert abs(delta_bits - 3 * 32 / 64) < 1e-6
    # still far below FP32 Adam and a latent-weight (BitNet-style) optimizer
    assert with_scales["linear_train_bytes"] < 0.35 * with_scales["fp32_adam_linear_bytes"]
    assert with_scales["linear_bits_per_weight_train"] < 26.0
