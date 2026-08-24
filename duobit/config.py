from dataclasses import dataclass


@dataclass
class DuobitConfig:
    """DUOBIT-EST v3 configuration.

    Persistent weight storage is discrete codes + per-group scales.
    Optimizer states (Adam moments, PEFA residual) are training-only and
    discarded at inference — there is no FP32 master-weight tensor.
    """

    # Codebook
    group_size: int = 128
    rho: float = 1.0 / 3.0
    learnable_rho: bool = False
    n_levels: int = 4  # 2 = binary, 3 = ternary (~1.58-bit), 4 = 2-bit

    # Quantization / precision
    activation_bits: int = 16  # 8 for W2A8, 16 for W2A16
    weight_bits: int = 2  # informational; derived from n_levels if needed

    # Optimizer
    lr: float = 1e-3
    duobit_lr: float = 8e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01

    # Error compensation, PEFA, trust gating
    enable_error_compensation: bool = True
    use_pefa: bool = True  # v3: persistent error-feedback accumulator
    pefa_clip: float = 8.0  # clip residual relative to group scale
    trust_threshold: float = 4.0
    balanced_rounding: bool = True
    use_mse_scales: bool = True

    # Transition dynamics
    transition_temperature: float = 0.8
    min_transitions_per_group: int = 0
    use_raw_momentum: bool = False

    # Hadamard & scale EMA
    use_hadamard: bool = False
    scale_update_freq: int = 50
    scale_ema_alpha: float = 0.01

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
