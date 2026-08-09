import torch
import pytest
from duobit.quantization.codebook import SymmetricCodebook, compute_group_scales

def test_symmetric_codebook_levels():
    cb = SymmetricCodebook(rho=1.0/3.0)
    assert cb.levels == [-1.0, -1.0/3.0, 1.0/3.0, 1.0]

def test_quantize_dequantize_roundtrip():
    cb = SymmetricCodebook(rho=1.0/3.0)
    # Generate test weights
    w = torch.tensor([[-0.9, -0.3, 0.2, 0.95]], dtype=torch.float32)
    scales = torch.tensor([[1.0]], dtype=torch.float32) # scale 1.0
    
    codes = cb.quantize(w, scales)
    # Level boundaries:
    # m01 = -2/3 = -0.6667
    # m12 = 0.0
    # m23 = 2/3 = 0.6667
    # -0.9 < -0.6667 => code 0
    # -0.3 is between -0.6667 and 0.0 => code 1
    # 0.2 is between 0.0 and 0.6667 => code 2
    # 0.95 > 0.6667 => code 3
    assert torch.equal(codes, torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8))
    
    w_dequant = cb.dequantize(codes, scales)
    expected = torch.tensor([[-1.0, -1.0/3.0, 1.0/3.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(w_dequant, expected, atol=1e-5)

def test_bit_serial_decomposition():
    cb = SymmetricCodebook(rho=1.0/3.0)
    codes = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
    b0, b1, a, b = cb.bit_serial_decompose(codes)
    
    assert a == pytest.approx(2.0/3.0)
    assert b == pytest.approx(1.0/3.0)
    
    # Check linear combo a*b0 + b*b1
    recon = a * b0 + b * b1
    expected = torch.tensor([-1.0, -1.0/3.0, 1.0/3.0, 1.0], dtype=torch.float32)
    assert torch.allclose(recon, expected, atol=1e-5)

def test_compute_group_scales():
    w = torch.randn(2, 256)
    scales = compute_group_scales(w, group_size=128)
    assert scales.shape == (2, 2, 1)
    
    # Scale for group should equal max abs value in that group
    group0_max = torch.max(torch.abs(w[0, :128]))
    assert torch.allclose(scales[0, 0, 0], group0_max, atol=1e-5)
