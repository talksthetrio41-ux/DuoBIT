import argparse
import time
import json
import math
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
    # Total params and DUOBIT Linear params
    total_params = 0
    linear_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
    
    # Add buffer elements from DuobitLinear (codes & scales)
    for module in model.modules():
        if hasattr(module, "codes"):
            total_params += module.codes.numel()
            linear_params += module.codes.numel()
            
    return total_params, linear_params

def generate_text(model: torch.nn.Module, prompt_ids: torch.Tensor, max_new_tokens: int = 30, temperature: float = 0.8):
    model.eval()
    device = next(model.parameters()).device
    curr_ids = prompt_ids.clone().to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if curr_ids.size(1) > model.config.max_seq_len:
                input_cond = curr_ids[:, -model.config.max_seq_len:]
            else:
                input_cond = curr_ids
                
            logits, _ = model(input_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
    model.train()
    return curr_ids[0].tolist()

def main():
    print("==================================================")
    print("   DUOBIT-EST 500-Step Deep Experiment Benchmark  ")
    print("==================================================")

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Config for ~5M parameter Transformer model
    # d_model=256, n_layers=6, n_heads=4, d_ff=1024, vocab_size=1024
    config = DuobitConfig(
        vocab_size=1024,
        d_model=256,
        n_layers=6,
        n_heads=4,
        d_ff=1024,
        max_seq_len=64,
        group_size=128,
        rho=1.0 / 3.0,
        activation_bits=16,
        use_hadamard=False,
        lr=1e-3,
        weight_decay=0.01,
        trust_threshold=2.0,
        enable_error_compensation=True,
        balanced_rounding=True,
        scale_update_freq=1000,
    )

    # 1. Create synthetic dataset with repeating structural grammar patterns for text generation test
    num_samples = 1024
    seq_len = config.max_seq_len
    
    # Generate structured synthetic tokens: A -> B -> C token transition patterns
    synthetic_data = torch.randint(0, config.vocab_size, (num_samples, seq_len))
    for i in range(num_samples):
        # Create deterministic grammar rules inside synthetic text
        pattern_start = i % 50
        for j in range(0, seq_len - 1, 3):
            synthetic_data[i, j] = (pattern_start + j * 7) % config.vocab_size
            synthetic_data[i, j+1] = (synthetic_data[i, j] * 3 + 1) % config.vocab_size
            synthetic_data[i, j+2] = (synthetic_data[i, j+1] + 5) % config.vocab_size

    train_data = synthetic_data[:896]
    val_data = synthetic_data[896:]

    train_loader = DataLoader(TensorDataset(train_data), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=16, shuffle=False)

    # Instantiate models
    duobit_model = DuobitTransformer(config, use_duobit=True)
    baseline_model = DuobitTransformer(config, use_duobit=False)

    duobit_total_params, duobit_linear_params = count_parameters(duobit_model)
    base_total_params, _ = count_parameters(baseline_model)

    print(f"\nModel Parameter Architecture:")
    print(f"- DUOBIT-EST Total Parameters: {duobit_total_params:,} (Linear 2-bit weights: {duobit_linear_params:,})")
    print(f"- Baseline FP32 Total Parameters: {base_total_params:,}")

    # Calculate average bits per parameter
    # DUOBIT-EST Linear weights: 2 bits + (16 bits / 128 group_size) = 2.125 bits per weight!
    # Non-linear weights (Embeddings, RMSNorm): 32 bits
    duobit_total_bits = (duobit_linear_params * 2.125) + ((duobit_total_params - duobit_linear_params) * 32.0)
    duobit_avg_bits = duobit_total_bits / duobit_total_params

    # Memory in MB for parameters
    duobit_param_memory_mb = (duobit_total_bits / 8.0) / (1024 * 1024)
    base_param_memory_mb = (base_total_params * 4.0) / (1024 * 1024) # 32 bits = 4 bytes

    print(f"\nMemory Footprint Analysis (Model Weight Parameters):")
    print(f"  DUOBIT-EST: {duobit_avg_bits:.3f} avg bits/param | {duobit_param_memory_mb:.2f} MB weight memory")
    print(f"  Baseline  : 32.000 avg bits/param | {base_param_memory_mb:.2f} MB weight memory")
    print(f"  --> Memory Savings: {base_param_memory_mb / duobit_param_memory_mb:.2f}x reduction in model weight storage!")

    # Train DUOBIT-EST Model
    print("\n==================================================")
    print(">>> Phase 1: Training DUOBIT-EST Model (500 steps)...")
    print("==================================================")
    duobit_trainer = Trainer(config, duobit_model, train_loader, val_dataloader=val_loader, device=device)
    
    t0 = time.time()
    duobit_results = duobit_trainer.train(max_steps=500, eval_freq=50, warmup_steps=50)
    duobit_time = time.time() - t0

    # Train Baseline FP32 Model
    print("\n==================================================")
    print(">>> Phase 2: Training FP32 Baseline Model (500 steps)...")
    print("==================================================")
    base_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    baseline_trainer = Trainer(config, baseline_model, train_loader, val_dataloader=val_loader, optimizer=base_optimizer, device=device)
    
    t0 = time.time()
    baseline_results = baseline_trainer.train(max_steps=500, eval_freq=50, warmup_steps=50)
    baseline_time = time.time() - t0

    # Inference Test
    prompt = torch.tensor([[10, 31, 36, 12]], dtype=torch.long)
    duobit_gen = generate_text(duobit_model, prompt, max_new_tokens=20)
    baseline_gen = generate_text(baseline_model, prompt, max_new_tokens=20)

    # Plot results
    steps_d = [log["step"] for log in duobit_results["train_logs"]]
    loss_d = [log["loss"] for log in duobit_results["train_logs"]]
    ppl_d = [log["perplexity"] for log in duobit_results["train_logs"]]
    trans_d = [log.get("transition_rate", 0.0) * 100.0 for log in duobit_results["train_logs"]]

    steps_b = [log["step"] for log in baseline_results["train_logs"]]
    loss_b = [log["loss"] for log in baseline_results["train_logs"]]
    ppl_b = [log["perplexity"] for log in baseline_results["train_logs"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Training Loss
    axes[0].plot(steps_d, loss_d, label="DUOBIT-EST (Native 2-bit)", color="#1f77b4", linewidth=2)
    axes[0].plot(steps_b, loss_b, label="Baseline (FP32)", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[0].set_title("Training Loss Comparison (500 Steps)")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Cross Entropy Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Subplot 2: Perplexity
    axes[1].plot(steps_d, ppl_d, label="DUOBIT-EST (Native 2-bit)", color="#1f77b4", linewidth=2)
    axes[1].plot(steps_b, ppl_b, label="Baseline (FP32)", color="#ff7f0e", linewidth=2, linestyle="--")
    axes[1].set_title("Perplexity Progression")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Subplot 3: Code Transition Rate (%)
    axes[2].plot(steps_d, trans_d, label="Weight Code Transition Rate (%)", color="#2ca02c", linewidth=2)
    axes[2].set_title("DUOBIT 2-Bit Weight Transition Rate per Step")
    axes[2].set_xlabel("Steps")
    axes[2].set_ylabel("% Weights Transitioned")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    chart_path = os.path.join(ARTIFACTS_DIR, "duobit_vs_fp32_500steps.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"\nSaved visualization chart to: {chart_path}")

    # Prepare summary data dictionary
    summary_data = {
        "duobit_avg_bits": duobit_avg_bits,
        "duobit_param_memory_mb": duobit_param_memory_mb,
        "base_param_memory_mb": base_param_memory_mb,
        "duobit_final_loss": loss_d[-1],
        "base_final_loss": loss_b[-1],
        "duobit_final_ppl": ppl_d[-1],
        "base_final_ppl": ppl_b[-1],
        "duobit_time": duobit_time,
        "base_time": baseline_time,
        "duobit_ms_per_step": (duobit_time / 500.0) * 1000.0,
        "base_ms_per_step": (baseline_time / 500.0) * 1000.0,
        "duobit_gen": duobit_gen,
        "base_gen": baseline_gen,
        "chart_path": chart_path,
    }

    with open("/kaggle/working/workspace/experiment_500steps_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nFinished 500-step experiment successfully!")

if __name__ == "__main__":
    main()
