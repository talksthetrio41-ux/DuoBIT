# DUOBIT-EST: Error-Compensated Stochastic Transition Training

DUOBIT-EST is a training framework designed to train Large Language Models natively in 2-bit precision without full-precision master weights.

## Key Features
- **Native 2-bit Weight Codes**: Weights represented as discrete 2-bit codes $z_i \in \{0, 1, 2, 3\}$ mapped through a symmetric 4-level codebook $C = \{-1, -\rho, +\rho, +1\}$ with group scales $s_g$.
- **Latent-Free Optimizer**: Eliminates continuous master weight storage by injecting ephemeral quantization residuals directly into optimizer momentum (ECO-style error compensation).
- **Balanced Stochastic Transitions**: Uses group-level dependent stochastic rounding to guarantee unbiased updates with low variance.
- **Stabilizers**: Hadamard normalization to smooth activation/gradient distributions, and transition-trust gating to prevent noise-driven level flips.

## Package Architecture
- `duobit.quantization`: Symmetric codebooks, group scale computation, bit-serial decomposition, balanced stochastic rounding, and Hadamard transforms.
- `duobit.layers`: `DuobitLinear` layer replacing standard `nn.Linear`.
- `duobit.optim`: `DuobitAdam` optimizer implementing error-compensated momentum updates and stochastic weight code transitions.
- `duobit.model`: GPT-style decoder Transformer built with `DuobitLinear` modules.
- `duobit.training`: Trainer and evaluation utilities.
