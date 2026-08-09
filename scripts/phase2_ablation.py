import os
import time
import json
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from duobit.config import DuobitConfig
from duobit.model.transformer import DuobitTransformer
from duobit.training.trainer import Trainer

# Synthetic Dataset Generator matching Phase 1 experiment
class SyntheticSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size: int = 500, seq_len: int = 32, num_samples: int = 2000, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.samples = []
        for _ in range(num_samples):
            prompt = torch.randint(10, vocab_size // 4, (seq_len // 3,))
            seq = []
            for t in prompt:
                val1 = t.item()
                val2 = (val1 * 3 + 1) % vocab_size
                val3 = (val2 + 5) % vocab_size
                seq.extend([val1, val2, val3])
            seq = seq[:seq_len]
            self.samples.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def evaluate_generation(model, prompt_tokens: List[int], gen_len: int = 20, device: str = "cpu") -> List[int]:
    model.eval()
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(gen_len):
            logits, _ = model(input_ids)
            next_token = torch.argmax(logits[0, -1, :]).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    model.train()
    return input_ids[0].tolist()


def run_ablation():
    print("=" * 60)
    print("      DUOBIT Phase 2 Enhancement Ablation Study")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset & Dataloaders
    train_dataset = SyntheticSequenceDataset(vocab_size=500, seq_len=32, num_samples=3000, seed=42)
    val_dataset = SyntheticSequenceDataset(vocab_size=500, seq_len=32, num_samples=500, seed=999)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    max_steps = 1000

    configs = {
        "Config A: Phase 1 Baseline": {
            "use_duobit": True,
            "config": DuobitConfig(
                vocab_size=500, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=32, group_size=128,
                duobit_lr=8e-3, use_swiglu=False, scale_ema_alpha=0.0, transition_temperature=1.0, min_transitions_per_group=0
            )
        },
        "Config B: + Temperature Scaling (0.8)": {
            "use_duobit": True,
            "config": DuobitConfig(
                vocab_size=500, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=32, group_size=128,
                duobit_lr=8e-3, use_swiglu=False, scale_ema_alpha=0.0, transition_temperature=0.8, min_transitions_per_group=0
            )
        },
        "Config C: + Scale EMA Adaptation": {
            "use_duobit": True,
            "config": DuobitConfig(
                vocab_size=500, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=32, group_size=128,
                duobit_lr=8e-3, use_swiglu=False, scale_ema_alpha=0.01, scale_update_freq=50, transition_temperature=1.0, min_transitions_per_group=0
            )
        },
        "Config D: Full Phase 2 (+ SwiGLU)": {
            "use_duobit": True,
            "config": DuobitConfig(
                vocab_size=500, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=32, group_size=128,
                duobit_lr=8e-3, use_swiglu=True, scale_ema_alpha=0.01, scale_update_freq=50, transition_temperature=0.8, min_transitions_per_group=0
            )
        },
        "Config E: FP32 Baseline (SwiGLU)": {
            "use_duobit": False,
            "config": DuobitConfig(
                vocab_size=500, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=32, group_size=128,
                lr=1e-3, use_swiglu=True
            )
        }
    }


    results = {}
    prompt_tokens = [10, 31, 36, 12]

    for name, item in configs.items():
        print(f"\n>>> Running {name}...")
        torch.manual_seed(42)
        cfg = item["config"]
        use_duobit = item["use_duobit"]

        model = DuobitTransformer(cfg, use_duobit=use_duobit)
        trainer = Trainer(cfg, model, train_loader, val_loader, device=device)

        start_t = time.time()
        res = trainer.train(max_steps=max_steps, eval_freq=100, warmup_steps=50)
        elapsed = time.time() - start_t

        val_loss, val_ppl = trainer.evaluate()
        gen_tokens = evaluate_generation(model, prompt_tokens, gen_len=20, device=device)

        logs = res["train_logs"]
        steps = [l["step"] for l in logs]
        losses = [l["loss"] for l in logs]
        trans_rates = [l.get("transition_rate", 0.0) for l in logs]

        avg_trans_rate = sum(trans_rates) / max(1, len(trans_rates))

        results[name] = {
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "avg_trans_rate": avg_trans_rate,
            "elapsed_sec": elapsed,
            "steps": steps,
            "losses": losses,
            "trans_rates": trans_rates,
            "gen_tokens": gen_tokens,
        }

        print(f"   -> {name} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Avg TransRate: {avg_trans_rate*100:.2f}% | Time: {elapsed:.1f}s")
        print(f"      Gen Output: {gen_tokens[:12]}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, r in results.items():
        ax1.plot(r["steps"], r["losses"], label=f"{name} (Val Loss: {r['val_loss']:.2f})")
        if "Baseline" not in name or "Config A" in name or "Config B" in name or "Config C" in name or "Config D" in name:
            ax2.plot(r["steps"], [tr * 100 for tr in r["trans_rates"]], label=name)

    ax1.set_title("Training Loss Comparison")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title("DUOBIT Transition Rate (%) per Step")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Transition Rate (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    chart_path = "/root/.gemini/antigravity-ide/brain/51c7ddfc-7552-429f-8f8c-c893e80a2451/duobit_phase2_ablation.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nSaved ablation plot to {chart_path}")

    # Save summary JSON
    summary_path = "/kaggle/working/workspace/duobit_phase2_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved ablation summary to {summary_path}")


if __name__ == "__main__":
    run_ablation()
