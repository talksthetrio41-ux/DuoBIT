import math
from typing import List, Optional

import torch
from torch.optim import Optimizer

from duobit.layers.duobit_linear import DuobitLinear
from duobit.quantization.blockwise import dequantize_blockwise, quantize_blockwise
from duobit.quantization.codebook import (
    compute_group_scales,
    compute_mse_group_scales,
    compute_scale_grad,
)
from duobit.quantization.qpefa import dequantize_residual, quantize_residual
from duobit.quantization.stochastic import (
    balanced_stochastic_round,
    compute_transition_probs,
    quantize_with_residual,
)


class DuobitAdam(Optimizer):
    """DUOBIT-EST optimizer.

    Discrete codes are updated from an Adam candidate step. Sub-threshold
    updates accumulate in a quantized error-feedback residual (QPEFA) until
    they cross a codebook gap. First-moment tensors are stored block-wise in
    integer format. The second moment is factored into row and column
    statistics by default (Adafactor-style), so it is O(m+n) rather than O(mn).

        e ← e + (W̃ − W_t)
        z, W_{t+1}, e ← Quantize(W_t + e)
        store e as an integer residual relative to the group scale

    Inference checkpoints store only (codes, scales). The residual and the
    Adam moments are discarded.

    The per-group scales are trained from their exact analytic gradient
    ``dL/ds_g = sum_{i in g} g_i C[z_i]``. They are part of the persistent 2-bit
    representation, not master weights: they are kept at inference and already
    counted at 32/G bits per weight. Training them adds only their own two Adam
    moments (2 * 32 / G bits per weight, 0.5 at G=128), and it is what gives the
    method a continuous degree of freedom -- without it the scales are frozen at
    initialization for the whole run.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        duobit_lr: Optional[float] = None,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        trust_threshold: float = 4.0,
        enable_error_compensation: bool = False,
        balanced_rounding: bool = True,
        use_mse_scales: bool = True,
        scale_update_freq: int = 50,
        scale_ema_alpha: float = 0.01,
        transition_temperature: float = 0.8,
        min_transitions_per_group: int = 0,
        use_raw_momentum: bool = False,
        use_pefa: bool = True,
        pefa_clip: float = 4.0,
        qpefa_bits: int = 8,
        qpefa_stochastic: bool = True,
        moment_bits: int = 8,
        factored_second_moment: bool = True,
        block_size: int = 128,
        learn_scales: bool = True,
        scale_lr: float = 1e-3,
        scale_lr_absolute: bool = False,
        scale_min_frac: float = 0.05,
        scale_relative_lr: bool = True,
        duobit_weight_decay: float = 0.0,
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
            use_pefa=use_pefa,
            pefa_clip=pefa_clip,
            qpefa_bits=qpefa_bits,
            qpefa_stochastic=qpefa_stochastic,
            moment_bits=moment_bits,
            factored_second_moment=factored_second_moment,
            block_size=block_size,
            learn_scales=learn_scales,
            scale_lr=scale_lr,
            scale_lr_absolute=scale_lr_absolute,
            scale_min_frac=scale_min_frac,
            scale_relative_lr=scale_relative_lr,
            duobit_weight_decay=duobit_weight_decay,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            self.dummy_param = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
            params = [self.dummy_param]

        super().__init__(params, defaults)

        self.duobit_layers: List[DuobitLinear] = [
            m for m in model.modules() if isinstance(m, DuobitLinear)
        ]
        self.step_count = 0
        self.last_transition_frac = 0.0

    def clip_ephemeral_grads(self, max_norm: float = 1.0) -> None:
        tensors = []
        for layer in self.duobit_layers:
            if layer.ephemeral_w is not None and layer.ephemeral_w.grad is not None:
                tensors.append(layer.ephemeral_w.grad)
        if tensors:
            torch.nn.utils.clip_grad_norm_(tensors, max_norm)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1
        self._step_standard_params()
        self._step_duobit_layers()
        return loss

    def _step_standard_params(self):
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

                if p.numel() == 1 and hasattr(p, "_is_rho"):
                    p.data.clamp_(0.1, 0.9)

    def _load_first_moment(self, state, like: torch.Tensor, bits: int, block_size: int):
        if bits >= 16:
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(like, dtype=torch.float32)
            return state["exp_avg"]
        if "exp_avg_q" not in state:
            q, scale, shape, pad = quantize_blockwise(
                torch.zeros_like(like, dtype=torch.float32), bits=bits, block_size=block_size
            )
            state["exp_avg_q"] = q
            state["exp_avg_scale"] = scale
            state["exp_avg_shape"] = shape
            state["exp_avg_pad"] = pad
        return dequantize_blockwise(
            state["exp_avg_q"],
            state["exp_avg_scale"],
            state["exp_avg_shape"],
            state["exp_avg_pad"],
            dtype=torch.float32,
        )

    def _store_first_moment(self, state, m: torch.Tensor, bits: int, block_size: int):
        if bits >= 16:
            state["exp_avg"].copy_(m)
            return
        q, scale, shape, pad = quantize_blockwise(m, bits=bits, block_size=block_size)
        state["exp_avg_q"] = q
        state["exp_avg_scale"] = scale
        state["exp_avg_shape"] = shape
        state["exp_avg_pad"] = pad

    def _load_second_moment(
        self,
        state,
        like: torch.Tensor,
        bits: int,
        block_size: int,
        factored: bool,
    ):
        if factored:
            if "v_row" not in state:
                state["v_row"] = torch.zeros(like.shape[0], dtype=torch.float32, device=like.device)
                state["v_col"] = torch.zeros(like.shape[1], dtype=torch.float32, device=like.device)
            return None
        if bits >= 16:
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(like, dtype=torch.float32)
            return state["exp_avg_sq"]
        if "exp_avg_sq_q" not in state:
            q, scale, shape, pad = quantize_blockwise(
                torch.zeros_like(like, dtype=torch.float32), bits=bits, block_size=block_size
            )
            state["exp_avg_sq_q"] = q
            state["exp_avg_sq_scale"] = scale
            state["exp_avg_sq_shape"] = shape
            state["exp_avg_sq_pad"] = pad
        return dequantize_blockwise(
            state["exp_avg_sq_q"],
            state["exp_avg_sq_scale"],
            state["exp_avg_sq_shape"],
            state["exp_avg_sq_pad"],
            dtype=torch.float32,
        )

    def _store_second_moment(self, state, v: torch.Tensor, bits: int, block_size: int, factored: bool):
        if factored:
            return
        if bits >= 16:
            state["exp_avg_sq"].copy_(v)
            return
        q, scale, shape, pad = quantize_blockwise(v, bits=bits, block_size=block_size)
        state["exp_avg_sq_q"] = q
        state["exp_avg_sq_scale"] = scale
        state["exp_avg_sq_shape"] = shape
        state["exp_avg_sq_pad"] = pad

    def _load_error(self, state, layer: DuobitLinear, bits: int, pefa_clip: float):
        if bits >= 16:
            if "error" not in state:
                state["error"] = torch.zeros(
                    layer.codes.shape, dtype=torch.float32, device=layer.codes.device
                )
            return state["error"]
        if "error_q" not in state:
            state["error_q"] = torch.zeros(
                layer.codes.shape, dtype=torch.int8, device=layer.codes.device
            )
        return dequantize_residual(state["error_q"], layer.scales, bits, pefa_clip)

    def _store_error(
        self,
        state,
        err: torch.Tensor,
        layer: DuobitLinear,
        bits: int,
        pefa_clip: float,
        stochastic: bool,
    ):
        if bits >= 16:
            state["error"].copy_(err)
            return
        state["error_q"] = quantize_residual(
            err, layer.scales, bits, pefa_clip, stochastic=stochastic
        )

    def _step_group_scales(self, layer, state, grad, step_t, beta1, beta2, eps,
                           scale_lr, scale_lr_abs, scale_min_frac):
        """Adam on the per-group scales from the exact analytic gradient."""
        rho_val = layer.rho if isinstance(layer.rho, torch.nn.Parameter) else None
        gs = compute_scale_grad(
            grad, layer.codes, layer.codebook, group_size=layer.group_size,
            rho=rho_val,
        )
        if "scale_m" not in state:
            state["scale_m"] = torch.zeros_like(layer.scales)
            state["scale_v"] = torch.zeros_like(layer.scales)
        state["scale_m"].mul_(beta1).add_(gs, alpha=1.0 - beta1)
        state["scale_v"].mul_(beta2).addcmul_(gs, gs, value=1.0 - beta2)
        m_hat = state["scale_m"] / (1.0 - beta1 ** step_t)
        v_hat = state["scale_v"] / (1.0 - beta2 ** step_t)
        direction = m_hat / (v_hat.sqrt() + eps)
        if scale_lr_abs:
            layer.scales.add_(direction, alpha=-scale_lr)
        else:
            layer.scales.addcmul_(layer.scales, direction, value=-scale_lr)
        floor = layer.scales_init * scale_min_frac
        torch.maximum(layer.scales, floor, out=layer.scales)

    def _step_duobit_layers(self):
        changed = 0
        total = 0
        group_defaults = self.param_groups[0]

        for layer in self.duobit_layers:
            if layer.ephemeral_w is None or layer.ephemeral_w.grad is None:
                continue

            grad = layer.ephemeral_w.grad.data
            state = self.state[layer]
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
            use_raw_mom = group_defaults.get("use_raw_momentum", False)
            use_pefa = group_defaults.get("use_pefa", True)
            pefa_clip = group_defaults.get("pefa_clip", 4.0)
            qpefa_bits = int(group_defaults.get("qpefa_bits", 8))
            qpefa_stoch = bool(group_defaults.get("qpefa_stochastic", True))
            moment_bits = int(group_defaults.get("moment_bits", 8))
            factored = bool(group_defaults.get("factored_second_moment", True))
            block_size = int(group_defaults.get("block_size", 128))

            learn_scales = bool(group_defaults.get("learn_scales", False))
            scale_lr = float(group_defaults.get("scale_lr", 1e-3))
            scale_lr_abs = bool(group_defaults.get("scale_lr_absolute", False))
            scale_min_frac = float(group_defaults.get("scale_min_frac", 0.05))
            scale_rel_lr = bool(group_defaults.get("scale_relative_lr", False))
            duobit_wd = float(group_defaults.get("duobit_weight_decay", 0.0))

            if "step" not in state:
                state["step"] = 0

            state["step"] += 1
            step_t = state["step"]

            if learn_scales:
                self._step_group_scales(
                    layer, state, grad, step_t, beta1, beta2, eps,
                    scale_lr, scale_lr_abs, scale_min_frac,
                )

            w_curr = layer.get_dequantized_weight()
            prev_codes = layer.codes.clone()

            m = self._load_first_moment(state, w_curr, moment_bits, block_size)
            v = self._load_second_moment(state, w_curr, moment_bits, block_size, factored)

            m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

            if factored:
                g2 = grad * grad
                state["v_row"].mul_(beta2).add_(g2.mean(dim=1), alpha=1.0 - beta2)
                state["v_col"].mul_(beta2).add_(g2.mean(dim=0), alpha=1.0 - beta2)
                row = state["v_row"].clamp(min=0)
                col = state["v_col"].clamp(min=0)
                denom_mean = row.mean().clamp(min=eps)
                v_hat = torch.outer(row, col) / denom_mean
            else:
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                v_hat = v

            bias_correction1 = 1.0 - beta1 ** step_t
            bias_correction2 = 1.0 - beta2 ** step_t
            m_hat = m / bias_correction1
            v_hat = v_hat / bias_correction2

            if use_raw_mom:
                step_direction = m_hat
            else:
                step_direction = m_hat / (torch.sqrt(v_hat) + eps)

            # Decoupled weight decay is meaningless for codes whose magnitude
            # is carried by s_g: it only biases transitions toward the inner
            # levels. It defaults to 0 for the discrete path.
            if duobit_wd != 0:
                step_direction = step_direction + duobit_wd * w_curr

            if scale_rel_lr:
                # Measure the step in units of the group scale, so one learning
                # rate is commensurate with the codebook gap in every layer even
                # as the scales themselves train.
                out_f, in_f = w_curr.shape
                n_groups = layer.scales.shape[1]
                gsz = in_f // n_groups
                lr_eff = (d_lr * layer.scales).expand(out_f, n_groups, gsz).reshape(
                    out_f, in_f)
                w_cand = w_curr - lr_eff * step_direction
            else:
                w_cand = w_curr - d_lr * step_direction
            rho_val = layer.rho if isinstance(layer.rho, torch.nn.Parameter) else None

            if use_pefa:
                err = self._load_error(state, layer, qpefa_bits, pefa_clip)
                err = err + (w_cand - w_curr)
                if layer.scales.dim() == 3:
                    out_f, in_f = err.shape
                    n_groups = layer.scales.shape[1]
                    gsz = in_f // n_groups
                    max_res = (pefa_clip * layer.scales).expand(out_f, n_groups, gsz).reshape(out_f, in_f)
                    err = err.clamp(-max_res, max_res)
                w_virtual = w_curr + err
                new_codes, w_new, new_err = quantize_with_residual(
                    w_virtual, layer.scales, layer.codebook, rho=rho_val
                )
                layer.codes.copy_(new_codes)
                self._store_error(state, new_err, layer, qpefa_bits, pefa_clip, qpefa_stoch)
            else:
                probs, target_codes, _ = compute_transition_probs(
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
                w_new = layer.get_dequantized_weight()

            if scale_freq > 0 and step_t % scale_freq == 0 and scale_alpha > 0:
                w_post = layer.get_dequantized_weight()
                if use_mse:
                    s_target = compute_mse_group_scales(
                        w_post, layer.codes, layer.codebook, group_size=layer.group_size, rho=rho_val
                    )
                else:
                    s_target = compute_group_scales(w_post, group_size=layer.group_size)
                layer.scales.copy_((1.0 - scale_alpha) * layer.scales + scale_alpha * s_target)
                w_new = layer.get_dequantized_weight()

            if enable_ec:
                quant_residual = w_cand - w_new
                rescaled_err = -quant_residual * (torch.sqrt(v_hat) + eps) / (d_lr + 1e-8)
                rescaled_err = torch.clamp(rescaled_err, min=-1.0, max=1.0)
                m = m + rescaled_err

            self._store_first_moment(state, m, moment_bits, block_size)
            if not factored:
                self._store_second_moment(state, v, moment_bits, block_size, factored)

            changed += int((layer.codes != prev_codes).sum().item())
            total += layer.codes.numel()
            layer.ephemeral_w = None

        self.last_transition_frac = (changed / total) if total else 0.0
