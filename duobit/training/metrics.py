import math
from typing import Any, Dict

import torch

from duobit.layers.duobit_linear import DuobitLinear


class MetricTracker:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.prev_codes: Dict[str, torch.Tensor] = {}
        self._cache_initial_codes()

    def _cache_initial_codes(self):
        for name, module in self.model.named_modules():
            if isinstance(module, DuobitLinear):
                self.prev_codes[name] = module.codes.clone()

    @torch.no_grad()
    def compute_metrics(self, loss_val: float) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "loss": loss_val,
            "perplexity": float(torch.exp(torch.tensor(loss_val)).item()) if loss_val < 20 else float("inf"),
        }

        total_codes = 0
        total_changed = 0
        level_counts = torch.zeros(4, dtype=torch.long)
        n_levels_seen = 2

        for name, module in self.model.named_modules():
            if not isinstance(module, DuobitLinear):
                continue
            curr_codes = module.codes
            prev_c = self.prev_codes.get(name, curr_codes)
            changed = int((curr_codes != prev_c).sum().item())
            num_el = int(curr_codes.numel())
            total_changed += changed
            total_codes += num_el
            n_levels_seen = max(n_levels_seen, module.n_levels)
            for lvl in range(module.n_levels):
                level_counts[lvl] += int((curr_codes == lvl).sum().item())
            self.prev_codes[name] = curr_codes.clone()

        if total_codes > 0:
            metrics["transition_rate"] = total_changed / total_codes
            probs = []
            for lvl in range(n_levels_seen):
                p = level_counts[lvl].item() / total_codes
                metrics[f"level_{lvl}_pct"] = p * 100.0
                if p > 0:
                    probs.append(p)
            entropy = -sum(p * math.log(p, 2) for p in probs)
            metrics["level_entropy_bits"] = entropy
            metrics["level_entropy_norm"] = entropy / math.log2(n_levels_seen)
        return metrics
