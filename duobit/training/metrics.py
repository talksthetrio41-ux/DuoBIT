import torch
from typing import Dict, Any, List
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
        metrics = {
            "loss": loss_val,
            "perplexity": torch.exp(torch.tensor(loss_val)).item(),
        }

        total_codes = 0
        total_changed = 0
        level_counts = torch.zeros(4, dtype=torch.long)

        for name, module in self.model.named_modules():
            if isinstance(module, DuobitLinear):
                curr_codes = module.codes
                prev_c = self.prev_codes.get(name, curr_codes)

                changed = (curr_codes != prev_c).sum().item()
                num_el = curr_codes.numel()

                total_changed += changed
                total_codes += num_el

                for lvl in range(4):
                    level_counts[lvl] += (curr_codes == lvl).sum().item()

                # Update cache
                self.prev_codes[name] = curr_codes.clone()

        if total_codes > 0:
            metrics["transition_rate"] = total_changed / total_codes
            for lvl in range(4):
                metrics[f"level_{lvl}_pct"] = (level_counts[lvl].item() / total_codes) * 100.0

        return metrics
