import argparse
import time
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from duobit.config import DuobitConfig
from duobit.model.transformer import DuobitTransformer
from duobit.training.trainer import Trainer

ARTIFACTS_DIR = "/root/.gemini/antigravity-ide/brain/51c7ddfc-7552-429f-8f8c-c893e80a2451"

def count_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0
    for module in model.modules():
        if hasattr(module, "codes"):
            total_params += module.codes.numel()
            linear_params += module.codes.numel()
    return total_params, linear_params

def generate_text(model: torch.nn.Module, prompt_ids: torch.Tensor, max_new_tokens: int = 24):
    model.eval()
    device = next(model.parameters()).device
    curr_ids = prompt_ids.clone().to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_cond = curr_ids[:, -model.config.max_seq_len:] if curr_ids.size(1) > model.config.max_seq_len else curr_ids
            logits, _ = model(input_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
    model.train()
    return curr_ids[0].tolist()

def build_structured_dataset(vocab_size=1024, num_samples=1024, seq_len=64):
    torch.manual_seed(42)
    synthetic_data = torch.randint(0, vocab_size, (num_samples, seq_len))
    for i in range(num_samples):
        pattern_start = i % 50
        for j in range(0, seq_len - 1, 3):
            synthetic_data[i, j] = (pattern_start + j * 7) % vocab_size
            synthetic_data[i, j+1] = (synthetic_data[i, j] * 3 + 1) % vocab_size
            synthetic_data[i, j+2] = (synthetic_data[i, j+1] + 5) % vocab_size
    return synthetic_data[:896], synthetic_data[896:]

def main():
    print("==================================================")
    print("   DUOBIT-EST Hyperparameter & Capacity Tuning    ")
    print("==================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    train_data, val_data = build_structured_dataset()
    train_loader = DataLoader(TensorDataset(train_data), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=16, shuffle=False)

    # Prompt for inference generation: [10, 31, 36, 12]
    # Expected pattern: 12 -> 12*3+1=37 -> 37+5=42
    prompt = torch.tensor([[10, 31, 36, 12]], dtype=torch.long)

    # --------------------------------------------------
    # Step 1: Learning Rate Tuning Sweep for DUOBIT 2-Bit Codes
    # --------------------------------------------------
    print(">>> Step 1: Learning Rate Sweep for DUOBIT 2-Bit Code Transitions...")
    lr_candidates = [1e-3, 3e-3, 5e-3, 8e-3, 1.2e-2]
    lr_results = {}

    for d_lr in lr_candidates:
        print(f"Testing duobit_lr = {d_lr:.1e}...")
        torch.manual_seed(42)
        config = DuobitConfig(
            vocab_size=1024,
            d_model=256,
            n_layers=6,
            n_heads=4,
            d_ff=1024,
            max_seq_len=64,
            group_size=128,
            lr=1e-3,
            duobit_lr=d_lr,
            use_mse_scales=True,
            scale_update_freq=100,
        )

        model = DuobitTransformer(config, use_duobit=True)
        trainer = Trainer(config, model, train_loader, val_dataloader=val_loader, device=device)
        res = trainer.train(max_steps=300, eval_freq=100, warmup_steps=30)
        
        final_log = res["train_logs"][-1]
        lr_results[d_lr] = {
            "loss": final_log["loss"],
            "ppl": final_log["perplexity"],
            "transition_rate": final_log.get("transition_rate", 0.0),
        }
        print(f"   -> duobit_lr={d_lr:.1e} | Loss: {final_log['loss']:.4f} | PPL: {final_log['perplexity']:.2f} | TransRate: {final_log.get('transition_rate', 0.0)*100:.2f}%\n")

    # Select best duobit_lr
    best_duobit_lr = min(lr_results.keys(), key=lambda k: lr_results[k]["loss"])
    print(f"Best duobit_lr found: {best_duobit_lr:.1e} (Loss: {lr_results[best_duobit_lr]['loss']:.4f})")

    # --------------------------------------------------
    # Step 2: Capacity Scaling & Parity Comparison (500 steps)
    # --------------------------------------------------
    print("\n==================================================")
    print(">>> Step 2: Capacity Scaling Benchmark (500 steps)")
    print("==================================================")

    experiments = [
        {"name": "FP32 Baseline (d=256)", "d_model": 256, "d_ff": 1024, "duobit": False, "d_lr": 1e-3},
        {"name": "DUOBIT-EST Standard (d=256)", "d_model": 256, "d_ff": 1024, "duobit": True, "d_lr": best_duobit_lr},
        {"name": "DUOBIT-EST Scaled 1.5x (d=384)", "d_model": 384, "d_ff": 1536, "duobit": True, "d_lr": best_duobit_lr},
        {"name": "DUOBIT-EST Scaled 2.0x (d=512)", "d_model": 512, "d_ff": 2048, "duobit": True, "d_lr": best_duobit_lr},
    ]

    exp_results = {}

    for exp in experiments:
        print(f"\nRunning {exp['name']}...")
        torch.manual_seed(42)
        
        config = DuobitConfig(
            vocab_size=1024,
            d_model=exp["d_model"],
            n_layers=6,
            n_heads=4,
            d_ff=exp["d_ff"],
            max_seq_len=64,
            group_size=128,
            lr=1e-3,
            duobit_lr=exp["d_lr"],
            use_mse_scales=True,
            scale_update_freq=100,
        )

        model = DuobitTransformer(config, use_duobit=exp["duobit"])
        tot_params, lin_params = count_parameters(model)
        
        if exp["duobit"]:
            bits = (lin_params * 2.125 + (tot_params - lin_params) * 32.0) / tot_params
            mem_mb = (tot_params * bits / 8.0) / (1024 * 1024)
            optimizer = None
        else:
            bits = 32.0
            mem_mb = (tot_params * 4.0) / (1024 * 1024)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

        print(f"  Params: {tot_params:,} | Avg Bits: {bits:.2f} | Memory: {mem_mb:.2f} MB")

        trainer = Trainer(config, model, train_loader, val_dataloader=val_loader, optimizer=optimizer, device=device)
        
        t0 = time.time()
        res = trainer.train(max_steps=500, eval_freq=100, warmup_steps=50)
        elapsed = time.time() - t0

        gen_tokens = generate_text(model, prompt, max_new_tokens=20)
        
        steps = [log["step"] for log in res["train_logs"]]
        losses = [log["loss"] for log in res["train_logs"]]
        ppls = [log["perplexity"] for log in res["train_logs"]]
        trans = [log.get("transition_rate", 0.0) * 100.0 for log in res["train_logs"]]

        exp_results[exp["name"]] = {
            "params": tot_params,
            "bits": bits,
            "memory_mb": mem_mb,
            "final_loss": losses[-1],
            "final_ppl": ppls[-1],
            "time_sec": elapsed,
            "steps": steps,
            "losses": losses,
            "ppls": ppls,
            "trans_rates": trans,
            "gen_tokens": gen_tokens,
        }

        print(f"  -> Final Loss: {losses[-1]:.4f} | PPL: {ppls[-1]:.2f} | Gen Tokens: {gen_tokens[:12]}")

    # Plot Tuning Results
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = {
        "FP32 Baseline (d=256)": "#ff7f0e",
        "DUOBIT-EST Standard (d=256)": "#1f77b4",
        "DUOBIT-EST Scaled 1.5x (d=384)": "#2ca02c",
        "DUOBIT-EST Scaled 2.0x (d=512)": "#d62728",
    }

    styles = {
        "FP32 Baseline (d=256)": "--",
        "DUOBIT-EST Standard (d=256)": "-",
        "DUOBIT-EST Scaled 1.5x (d=384)": "-",
        "DUOBIT-EST Scaled 2.0x (d=512)": "-",
    }

    for name, stats in exp_results.items():
        axes[0].plot(stats["steps"], stats["losses"], label=name, color=colors[name], linestyle=styles[name], linewidth=2)
        axes[1].plot(stats["steps"], stats["ppls"], label=name, color=colors[name], linestyle=styles[name], linewidth=2)

    axes[0].set_title("Capacity & Hyperparameter Tuning Loss Curves (500 Steps)")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Perplexity Progression (Capacity Scaling)")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    chart_path = os.path.join(ARTIFACTS_DIR, "duobit_tuning_capacity_scaling.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nSaved tuning chart to: {chart_path}")

    # Save summary json
    summary_path = "/kaggle/working/workspace/duobit_tuning_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "best_duobit_lr": best_duobit_lr,
            "lr_sweep": {str(k): v for k, v in lr_results.items()},
            "capacity_sweep": {
                k: {
                    "params": v["params"],
                    "bits": v["bits"],
                    "memory_mb": v["memory_mb"],
                    "final_loss": v["final_loss"],
                    "final_ppl": v["final_ppl"],
                    "time_sec": v["time_sec"],
                    "gen_tokens": v["gen_tokens"],
                }
                for k, v in exp_results.items()
            },
            "chart_path": chart_path,
        }, f, indent=2)

    print("\nDUOBIT-EST Tuning Benchmark Complete!")

if __name__ == "__main__":
    main()
