from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DuobitConfig:
    # Codebook parameters
    group_size: int = 128
    rho: float = 1.0 / 3.0
    learnable_rho: bool = False
    
    # Quantization / Precision
    activation_bits: int = 8  # 8 for W2A8, 16 for W2A16, 4 for W2A4
    weight_bits: int = 2
    
    # Optimizer / Stochastic transitions
    lr: float = 1e-3
    duobit_lr: float = 5e-3  # Specialized higher LR for 2-bit weight code transitions
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    
    # Error compensation & Trust gating
    enable_error_compensation: bool = True
    trust_threshold: float = 2.0  # Gating threshold relative to grid step
    balanced_rounding: bool = True
    use_mse_scales: bool = True
    
    # Transition dynamics & scaling
    transition_temperature: float = 1.0
    min_transitions_per_group: int = 0
    use_raw_momentum: bool = False  # Use Adam-rescaled steps by default
    
    # Hadamard & Stabilizers
    use_hadamard: bool = False
    scale_update_freq: int = 100  # Step frequency to update group scales
    scale_ema_alpha: float = 0.01  # EMA rate for periodic scale updates
    
    # Architecture features
    use_swiglu: bool = True
    gradient_checkpointing: bool = False
    
    # Model architecture defaults
    vocab_size: int = 50257
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.0

