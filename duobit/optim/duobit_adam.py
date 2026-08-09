import math
import torch
from torch.optim import Optimizer
from typing import List, Dict, Any, Union, Iterable, Optional

from duobit.layers.duobit_linear import DuobitLinear
from duobit.quantization.stochastic import compute_transition_probs, balanced_stochastic_round
from duobit.quantization.codebook import compute_group_scales, compute_mse_group_scales

class DuobitAdam(Optimizer):
    """
    DUOBIT-EST Optimizer with support for:
    - Independent duobit_lr for discrete weight transitions
    - Raw momentum vs Adam-rescaled candidate steps
    - Transition temperature & minimum group transitions
    - Slow EMA group scale adaptation
    - Closed-form MSE group scale fitting
    - ECO residual injection into momentum
    - Learnable codebook rho parameter clamping
    """
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        duobit_lr: Optional[float] = None,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        trust_threshold: float = 2.0,
        enable_error_compensation: bool = True,
        balanced_rounding: bool = True,
        use_mse_scales: bool = True,
        scale_update_freq: int = 100,
        scale_ema_alpha: float = 0.01,
        transition_temperature: float = 1.0,
        min_transitions_per_group: int = 0,
        use_raw_momentum: bool = False,
    ):
        self.duobit_lr = duobit_lr if duobit_lr is not None else lr

        defaults = dict(
            lr=lr,
            duobit_lr=self.duobit_lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            trust_threshold=trust_threshold,
            enable_error_compensation=enable_error_compensation,
            balanced_rounding=balanced_rounding,
            use_mse_scales=use_mse_scales,
            scale_update_freq=scale_update_freq,
            scale_ema_alpha=scale_ema_alpha,
            transition_temperature=transition_temperature,
            min_transitions_per_group=min_transitions_per_group,
            use_raw_momentum=use_raw_momentum,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            self.dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            params = [self.dummy_param]
            
        super().__init__(params, defaults)

        self.duobit_layers: List[DuobitLinear] = []
        for module in model.modules():
            if isinstance(module, DuobitLinear):
                self.duobit_layers.append(module)

        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        # 1. Update standard parameters & learnable rho parameters
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step_t
                bias_correction2 = 1.0 - beta2 ** step_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Clamp learnable rho parameters to valid range [0.1, 0.9]
                if p.numel() == 1 and hasattr(p, "_is_rho"):
                    p.data.clamp_(0.1, 0.9)

        # 2. Update DuobitLinear 2-bit weight codes using DUOBIT-EST
        for layer in self.duobit_layers:
            if layer.ephemeral_w is None or layer.ephemeral_w.grad is None:
                continue

            grad = layer.ephemeral_w.grad.data
            state = self.state[layer]

            group_defaults = self.param_groups[0]
            d_lr = group_defaults.get("duobit_lr", group_defaults["lr"])
            beta1 = group_defaults["beta1"]
            beta2 = group_defaults["beta2"]
            eps = group_defaults["eps"]
            wd = group_defaults["weight_decay"]
            trust_thresh = group_defaults["trust_threshold"]
            enable_ec = group_defaults["enable_error_compensation"]
            balanced_round = group_defaults["balanced_rounding"]
            use_mse = group_defaults["use_mse_scales"]
            scale_freq = group_defaults["scale_update_freq"]
            scale_alpha = group_defaults.get("scale_ema_alpha", 0.01)
            temp = group_defaults.get("transition_temperature", 1.0)
            min_k = group_defaults.get("min_transitions_per_group", 0)
            use_raw_mom = group_defaults.get("use_raw_momentum", True)

            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(layer.codes, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(layer.codes, dtype=torch.float32)

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            step_t = state["step"]

            w_curr = layer.get_dequantized_weight()

            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1 ** step_t
            bias_correction2 = 1.0 - beta2 ** step_t

            m_hat = exp_avg / bias_correction1
            v_hat = exp_avg_sq / bias_correction2

            if use_raw_mom:
                step_direction = m_hat
            else:
                step_direction = m_hat / (torch.sqrt(v_hat) + eps)

            if wd != 0:
                step_direction = step_direction + wd * w_curr

            # Use d_lr (duobit_lr) for 2-bit weight transition candidate step W_tilde
            w_cand = w_curr - d_lr * step_direction

            rho_val = layer.rho if isinstance(layer.rho, torch.nn.Parameter) else None
            probs, target_codes, trust_mask = compute_transition_probs(
                w_current=w_curr,
                w_candidate=w_cand,
                codes=layer.codes,
                scale=layer.scales,
                codebook=layer.codebook,
                trust_threshold=trust_thresh,
                temperature=temp,
            )

            new_codes = balanced_stochastic_round(
                probs=probs,
                target_codes=target_codes,
                current_codes=layer.codes,
                group_size=layer.group_size,
                stochastic=balanced_round,
                min_transitions_per_group=min_k,
            )

            layer.codes.copy_(new_codes)

            # Slow EMA group scale adaptation from current dequantized weights
            if scale_freq > 0 and step_t % scale_freq == 0 and scale_alpha > 0:
                w_post_code = layer.get_dequantized_weight()
                if use_mse:
                    s_target = compute_mse_group_scales(
                        w_post_code, layer.codes, layer.codebook, group_size=layer.group_size, rho=rho_val
                    )
                else:
                    s_target = compute_group_scales(w_post_code, group_size=layer.group_size)
                
                # Blend with slow EMA to avoid scale explosion
                layer.scales.copy_((1.0 - scale_alpha) * layer.scales + scale_alpha * s_target)

            w_new = layer.get_dequantized_weight()

            if enable_ec:
                quant_residual = w_cand - w_new
                rescaled_err = -quant_residual * (torch.sqrt(v_hat) + eps) / (d_lr + 1e-8)
                rescaled_err = torch.clamp(rescaled_err, min=-1.0, max=1.0)
                exp_avg.add_(rescaled_err)

            layer.ephemeral_w = None

        return loss

