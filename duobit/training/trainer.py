import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple, Union

from duobit.config import DuobitConfig
from duobit.optim.duobit_adam import DuobitAdam
from duobit.training.metrics import MetricTracker

class Trainer:
    def __init__(
        self,
        config: DuobitConfig,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Union[str, torch.device] = "cpu",
        use_wandb: bool = False,
    ):
        self.config = config
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.use_wandb = use_wandb

        if optimizer is None:
            self.optimizer = DuobitAdam(
                self.model,
                lr=config.lr,
                duobit_lr=config.duobit_lr,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay,
                trust_threshold=config.trust_threshold,
                enable_error_compensation=config.enable_error_compensation,
                balanced_rounding=config.balanced_rounding,
                use_mse_scales=config.use_mse_scales,
                scale_update_freq=config.scale_update_freq,
                scale_ema_alpha=config.scale_ema_alpha,
                transition_temperature=config.transition_temperature,
                min_transitions_per_group=config.min_transitions_per_group,
                use_raw_momentum=config.use_raw_momentum,
            )
        else:
            self.optimizer = optimizer


        self.metric_tracker = MetricTracker(self.model)

    def _get_lr(self, step: int, max_steps: int, warmup_steps: int = 100) -> float:
        if step < warmup_steps:
            return self.config.lr * (step + 1) / warmup_steps
        decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        decay_ratio = min(1.0, max(0.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.lr * 0.1 + coeff * (self.config.lr * 0.9)

    def train(self, max_steps: int, eval_freq: int = 50, warmup_steps: int = 50) -> Dict[str, Any]:
        self.model.train()
        step = 0
        data_iter = iter(self.train_dataloader)
        
        start_time = time.time()
        train_logs = []

        while step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                targets = batch.get("labels", input_ids).to(self.device)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                targets = batch[1].to(self.device) if len(batch) > 1 else input_ids
            else:
                input_ids = batch.to(self.device)
                targets = input_ids

            # LR schedule update
            lr = self._get_lr(step, max_steps, warmup_steps=warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.optimizer.zero_grad()
            logits, loss = self.model(input_ids, targets=targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            step += 1

            if step % 10 == 0 or step == max_steps:
                metrics = self.metric_tracker.compute_metrics(loss.item())
                metrics["step"] = step
                metrics["lr"] = lr
                metrics["sec_per_step"] = (time.time() - start_time) / step
                train_logs.append(metrics)

                if step % 50 == 0 or step == max_steps:
                    trans_rate = metrics.get("transition_rate", 0.0)
                    print(
                        f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | "
                        f"PPL: {metrics['perplexity']:.2f} | TransRate: {trans_rate*100:.2f}% | LR: {lr:.2e}"
                    )

            if eval_freq > 0 and (step % eval_freq == 0 or step == max_steps) and self.val_dataloader is not None:
                val_loss, val_ppl = self.evaluate()
                print(f"--> Step {step} Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")

        return {"train_logs": train_logs}

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in self.val_dataloader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                targets = batch.get("labels", input_ids).to(self.device)
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                targets = batch[1].to(self.device) if len(batch) > 1 else input_ids
            else:
                input_ids = batch.to(self.device)
                targets = input_ids

            _, loss = self.model(input_ids, targets=targets)
            total_loss += loss.item()
            total_batches += 1

        self.model.train()
        avg_loss = total_loss / max(1, total_batches)
        ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        return avg_loss, ppl
