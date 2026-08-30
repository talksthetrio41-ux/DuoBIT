from dataclasses import dataclass


@dataclass
class DuobitConfig:
    """DUOBIT-EST configuration.

    Persistent weight storage is discrete codes plus per-group scales.
    The training residual (QPEFA) and Adam moments are integer / factored
    optimizer state, discarded at inference. There is no FP32 master-weight
    tensor at any point.
    """

    # Codebook
    group_size: int = 128
    rho: float = 1.0 / 3.0
    learnable_rho: bool = False
    n_levels: int = 4  # 2 = binary, 3 = ternary (~1.58-bit), 4 = 2-bit

    # Quantization / precision
    activation_bits: int = 16  # 8 for W2A8, 16 for W2A16
    weight_bits: int = 2  # informational; derived from n_levels if needed
    quant_mode: str = "duobit"  # duobit | fp32 | ste

    # Optimizer
    lr: float = 1e-3
    duobit_lr: float = 8e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Error feedback
    enable_error_compensation: bool = False
    use_pefa: bool = True
    # The reachable post-quantization residual is 0.5 * gap = 0.333 * s_g, so a
    # clip of 4.0 spent most of the integer residual grid on values the
    # accumulator never reaches. 1.0 keeps the headroom and quadruples the
    # effective resolution at the same bit width.
    pefa_clip: float = 1.0
    qpefa_bits: int = 8
    qpefa_stochastic: bool = True
    trust_threshold: float = 4.0
    balanced_rounding: bool = True
    use_mse_scales: bool = True

    # Compressed optimizer state
    moment_bits: int = 8
    factored_second_moment: bool = True
    block_size: int = 128

    # Trained group scales. The scale is part of the persistent 2-bit
    # representation (kept at inference, already charged at 32/G bits per
    # weight), so training it does not reintroduce master weights; it adds
    # 2 * 32 / G bits per weight of optimizer state.
    learn_scales: bool = True
    scale_lr: float = 1e-3
    scale_lr_absolute: bool = False
    scale_min_frac: float = 0.05
    scale_relative_lr: bool = True
    duobit_weight_decay: float = 0.0
    scale_init: str = "var"  # "var" preserves the group std, "mse" is the v1 fit

    # Transition dynamics
    transition_temperature: float = 0.8
    min_transitions_per_group: int = 0
    use_raw_momentum: bool = False

    # Hadamard & legacy scale EMA. Recomputing the MSE scale from the already
    # dequantized weight is an algebraic fixed point (s* = sum(s c)c / sum(c^2)
    # = s), so the EMA never moved the scales; it is disabled by default and
    # kept only to reproduce v1.
    use_hadamard: bool = False
    scale_update_freq: int = 0
    scale_ema_alpha: float = 0.0

    # Architecture
    use_swiglu: bool = True
    gradient_checkpointing: bool = False

    vocab_size: int = 50257
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.0

    def resolved_n_levels(self) -> int:
        if self.n_levels in (2, 3, 4):
            return self.n_levels
        if self.weight_bits <= 1:
            return 2
        return 4

    def bits_per_code(self) -> float:
        n = self.resolved_n_levels()
        if n == 2:
            return 1.0
        if n == 3:
            return 1.58496250072  # log2(3)
        return 2.0
