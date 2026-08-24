import torch
import torch.nn as nn

from duobit.layers.duobit_linear import DuobitLinear
from duobit.optim.duobit_adam import DuobitAdam
from duobit.quantization.blockwise import dequantize_blockwise, quantize_blockwise
from duobit.quantization.codebook import compute_group_scales
from duobit.quantization.memory import training_memory_report
from duobit.quantization.qpefa import dequantize_residual, quantize_residual


def test_qpefa_roundtrip_near_identity():
    torch.manual_seed(0)
    w = torch.randn(16, 64)
    scales = compute_group_scales(w, group_size=32)
    err = 0.05 * torch.randn_like(w)
    q = quantize_residual(err, scales, bits=8, pefa_clip=4.0, stochastic=False)
    recon = dequantize_residual(q, scales, bits=8, pefa_clip=4.0)
    mse = float(((err - recon) ** 2).mean())
    assert mse < 5e-3
    assert q.dtype == torch.int8


def test_qpefa_stochastic_is_unbiased():
    torch.manual_seed(1)
    scale = torch.ones(1, 1, 1)
    err = torch.full((8, 32), 0.013)
    acc = torch.zeros_like(err)
    n = 200
    for _ in range(n):
        q = quantize_residual(err, scale, bits=8, pefa_clip=4.0, stochastic=True)
        acc = acc + dequantize_residual(q, scale, bits=8, pefa_clip=4.0)
    mean = acc / n
    assert abs(float(mean.mean()) - 0.013) < 1e-3


def test_blockwise_moment_roundtrip():
    torch.manual_seed(2)
    x = torch.randn(32, 64)
    q, scale, shape, pad = quantize_blockwise(x, bits=8, block_size=64)
    recon = dequantize_blockwise(q, scale, shape, pad)
    rel = float(((x - recon) ** 2).mean() / (x.pow(2).mean() + 1e-8))
    assert rel < 0.02
    assert q.dtype == torch.int8


def test_compressed_optimizer_flips_codes():
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = DuobitLinear(64, 32, group_size=32, n_levels=4)

        def forward(self, x):
            return self.lin(x)

    model = M()
    opt = DuobitAdam(
        model,
        lr=0.05,
        duobit_lr=0.05,
        use_pefa=True,
        qpefa_bits=8,
        moment_bits=8,
        factored_second_moment=True,
        scale_update_freq=0,
        weight_decay=0.0,
    )
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
    assert (model.lin.codes != initial).any(), "QPEFA should flip some codes"
    assert changed_steps >= 5, f"expected sustained transitions, got {changed_steps}"
    st = opt.state[model.lin]
    assert "error_q" in st
    assert st["error_q"].dtype == torch.int8
    assert "exp_avg_q" in st
    assert st["exp_avg_q"].dtype == torch.int8
    assert "v_row" in st and "v_col" in st


def test_training_memory_beats_fp32_and_bitnet():
    torch.manual_seed(0)
    layer = DuobitLinear(256, 256, group_size=64, n_levels=4)
    model = nn.Sequential(layer)
    opt = DuobitAdam(
        model,
        qpefa_bits=8,
        moment_bits=8,
        factored_second_moment=True,
        block_size=128,
    )
    report = training_memory_report(model, opt)
    assert report["linear_train_bytes"] < 0.35 * report["fp32_adam_linear_bytes"]
    assert report["linear_train_bytes"] < 0.40 * report["bitnet_linear_bytes"]
    assert report["linear_bits_per_weight_train"] < 24.0
