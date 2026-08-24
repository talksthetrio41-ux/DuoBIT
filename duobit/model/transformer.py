from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from duobit.config import DuobitConfig
from duobit.layers.duobit_linear import DuobitLinear
from duobit.layers.ste_linear import SteQuantLinear
from duobit.model.utils import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


def _linear_factory(config: DuobitConfig, use_duobit: Union[bool, str], use_hadamard: Optional[bool] = None):
    had = config.use_hadamard if use_hadamard is None else use_hadamard
    mode = config.quant_mode
    if use_duobit == "ste" or mode == "ste":
        return lambda in_f, out_f: SteQuantLinear(
            in_f,
            out_f,
            group_size=config.group_size,
            rho=config.rho,
            n_levels=config.resolved_n_levels(),
            bias=False,
        )
    if use_duobit:
        return lambda in_f, out_f: DuobitLinear(
            in_f,
            out_f,
            bias=False,
            group_size=config.group_size,
            rho=config.rho,
            use_hadamard=had,
            activation_bits=config.activation_bits,
            n_levels=config.resolved_n_levels(),
        )
    return lambda in_f, out_f: nn.Linear(in_f, out_f, bias=False)


class DuobitAttention(nn.Module):
    def __init__(self, config: DuobitConfig, use_duobit: Union[bool, str] = True):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        linear_cls = _linear_factory(config, use_duobit)
        self.q_proj = linear_cls(config.d_model, config.d_model)
        self.k_proj = linear_cls(config.d_model, config.d_model)
        self.v_proj = linear_cls(config.d_model, config.d_model)
        self.out_proj = linear_cls(config.d_model, config.d_model)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=config.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(v, seq_len=s)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)


class DuobitFFN(nn.Module):
    def __init__(self, config: DuobitConfig, use_duobit: Union[bool, str] = True):
        super().__init__()
        self.use_swiglu = config.use_swiglu
        linear_cls = _linear_factory(config, use_duobit)
        if self.use_swiglu:
            self.gate_proj = linear_cls(config.d_model, config.d_ff)
            self.up_proj = linear_cls(config.d_model, config.d_ff)
            self.down_proj = linear_cls(config.d_ff, config.d_model)
        else:
            self.gate_up_proj = linear_cls(config.d_model, config.d_ff)
            self.down_proj = linear_cls(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            h = F.silu(self.gate_proj(x)) * self.up_proj(x)
            return self.down_proj(h)
        h = F.relu(self.gate_up_proj(x)) ** 2
        return self.down_proj(h)


class DuobitTransformerBlock(nn.Module):
    def __init__(self, config: DuobitConfig, use_duobit: Union[bool, str] = True):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = DuobitAttention(config, use_duobit=use_duobit)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = DuobitFFN(config, use_duobit=use_duobit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DuobitTransformer(nn.Module):
    """GPT-style decoder with DuobitLinear, STE latent weights, or dense Linear."""

    def __init__(self, config: DuobitConfig, use_duobit: Union[bool, str] = True):
        super().__init__()
        self.config = config
        if config.quant_mode == "ste":
            use_duobit = "ste"
        elif config.quant_mode == "fp32":
            use_duobit = False
        self.use_duobit = use_duobit
        self.gradient_checkpointing = config.gradient_checkpointing
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [DuobitTransformerBlock(config, use_duobit=use_duobit) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.d_model)
        linear_cls = _linear_factory(config, use_duobit, use_hadamard=False)
        self.lm_head = linear_cls(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)
            )
        return logits, loss

    def count_parameters(self) -> int:
        n = 0
        for m in self.modules():
            if isinstance(m, DuobitLinear):
                n += m.codes.numel()
                if m.bias is not None:
                    n += m.bias.numel()
            elif isinstance(m, SteQuantLinear):
                n += m.weight.numel()
                if m.bias is not None:
                    n += m.bias.numel()
            elif isinstance(m, (nn.Linear, nn.Embedding, RMSNorm)):
                for p in m.parameters(recurse=False):
                    n += p.numel()
        return n
