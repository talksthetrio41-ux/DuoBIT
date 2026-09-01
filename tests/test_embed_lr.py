"""A dense embedding table can take its own learning rate and weight decay.

Comparing `--quant-embeddings 0` against `1` is only a test of quantization if
both tables train at the same rate. Without a separate knob the dense table
trains at `--lr` while the quantized one trains at `--duobit-lr`, so any quality
difference confounds the representation with a learning-rate change. These
tests pin the knob that removes the confound.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HARNESS = Path(__file__).resolve().parent.parent / "scripts" / "kaggle_duobit_fineweb.py"


@pytest.fixture(scope="module")
def kdf():
    spec = importlib.util.spec_from_file_location("kdf_test", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kdf_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(kdf, **over):
    a = kdf.build_argparser().parse_args([
        "--d-model", "128", "--n-layers", "1", "--n-heads", "2",
        "--d-ff", "256", "--group-size", "128"])
    for k, v in over.items():
        setattr(a, k, v)
    return a


def _embedding_step(kdf, **over):
    """One optimizer step; returns the embedding table's displacement."""
    torch.manual_seed(0)
    args = _args(kdf, **over)
    net = kdf.Transformer(args, use_duobit=True)
    opt = kdf.DuobitAdam(net, args, torch.device("cpu"), seed=0)
    before = net.tok_embeddings.weight.detach().clone()
    ids = torch.randint(0, net.vocab_size, (2, 33))
    _, loss = net(ids, targets=ids)
    loss.backward()
    opt.step()
    return (net.tok_embeddings.weight.detach() - before)


def test_embed_lr_zero_falls_back_to_lr(kdf):
    """The default (0) must reproduce the shared rate exactly."""
    d_default = _embedding_step(kdf, lr=1e-3, embed_lr=0.0)
    d_explicit = _embedding_step(kdf, lr=1e-3, embed_lr=1e-3)
    assert torch.allclose(d_default, d_explicit, atol=0, rtol=0)


def test_embed_lr_scales_the_embedding_step(kdf):
    """A 4x embedding rate moves the table exactly 4x further on the first step.

    Both terms of the step are proportional to lr -- the decoupled decay
    (-lr*wd*p) and the Adam term (-lr*m_hat/(sqrt(v_hat)+eps)) -- so the
    displacement ratio is the learning-rate ratio algebraically.

    Run with wd=0 so the only moving entries are the rows the batch actually
    touched. With decay on, every one of the table's 6.4M entries moves by
    -p*lr*wd instead, and although that term is proportional to lr too, the
    ratio only reads 3.993 +- 0.007: `fl(p*(1 - 1e-5))` keeps about two digits
    of a 1e-5 change to an O(1) float32, so the measured displacement is
    dominated by rounding. That is arithmetic, not the optimizer -- with wd=0
    the ratio here is exactly 4.0.
    """
    d1 = _embedding_step(kdf, lr=1e-3, embed_lr=1e-3, wd=0.0)
    d4 = _embedding_step(kdf, lr=1e-3, embed_lr=4e-3, wd=0.0)
    moved = d1.abs() > 1e-12
    assert moved.any(), "no embedding row moved; the test is not exercising the path"
    ratio = d4[moved] / d1[moved]
    assert torch.allclose(ratio, torch.full_like(ratio, 4.0), rtol=1e-3)


def test_embed_lr_does_not_touch_other_params(kdf):
    """Only the embedding table sees the separate rate."""
    torch.manual_seed(0)

    def norm_step(embed_lr):
        torch.manual_seed(0)
        args = _args(kdf, lr=1e-3, embed_lr=embed_lr)
        net = kdf.Transformer(args, use_duobit=True)
        opt = kdf.DuobitAdam(net, args, torch.device("cpu"), seed=0)
        before = net.norm.weight.detach().clone()
        ids = torch.randint(0, net.vocab_size, (2, 33))
        _, loss = net(ids, targets=ids)
        loss.backward()
        opt.step()
        return net.norm.weight.detach() - before

    assert torch.allclose(norm_step(1e-3), norm_step(4e-2), atol=0, rtol=0)


def test_embed_wd_negative_falls_back_to_wd(kdf):
    d_default = _embedding_step(kdf, wd=1e-2, embed_wd=-1.0)
    d_explicit = _embedding_step(kdf, wd=1e-2, embed_wd=1e-2)
    assert torch.allclose(d_default, d_explicit, atol=0, rtol=0)


def test_embed_wd_zero_removes_the_decay(kdf):
    """Decay is applied multiplicatively before the Adam step, so switching it
    off changes the displacement by exactly lr*wd times the table."""
    torch.manual_seed(0)
    args = _args(kdf, lr=1e-3, wd=0.5, embed_wd=0.0)
    net = kdf.Transformer(args, use_duobit=True)
    opt = kdf.DuobitAdam(net, args, torch.device("cpu"), seed=0)
    w0 = net.tok_embeddings.weight.detach().clone()
    ids = torch.randint(0, net.vocab_size, (2, 33))
    _, loss = net(ids, targets=ids)
    loss.backward()
    opt.step()
    # rows with no gradient and no decay must be exactly unchanged
    touched = torch.zeros(net.vocab_size, dtype=torch.bool)
    touched[ids.flatten()] = True
    untouched = ~touched
    assert untouched.any()
    assert torch.equal(net.tok_embeddings.weight.detach()[untouched], w0[untouched])


def test_quantized_embedding_ignores_embed_lr(kdf):
    """With --quant-embeddings the table is a DuoBIT module, not a parameter,
    so it is stepped by duobit_lr and the dense knob must not apply to it."""
    args = _args(kdf, quant_embeddings=1, embed_lr=4e-3)
    net = kdf.Transformer(args, use_duobit=True)
    opt = kdf.DuobitAdam(net, args, torch.device("cpu"), seed=0)
    assert isinstance(net.tok_embeddings, kdf.DuobitEmbedding)
    assert opt.embed_param_ids == set()
