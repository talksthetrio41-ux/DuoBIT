import os
import time
import json
import math
import torch
from typing import Dict, List, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from duobit.config import DuobitConfig
from duobit.model.transformer import DuobitTransformer
from duobit.training.trainer import Trainer

# Load real text corpus or fallback
def get_text_dataloaders(seq_len: int = 64, batch_size: int = 16) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        print("Loading WikiText-2 via HuggingFace...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        text = "\n".join([t for t in dataset["text"] if len(t.strip()) > 0])
        tokens = tokenizer.encode(text)
        
        vocab_size = tokenizer.vocab_size
        print(f"Loaded WikiText-2 text corpus with {len(tokens):,} tokens. Vocab size: {vocab_size}")

        n_samples = len(tokens) // seq_len
        input_chunks = []
        for i in range(n_samples):
            chunk = tokens[i * seq_len : (i + 1) * seq_len]
            input_chunks.append(torch.tensor(chunk, dtype=torch.long))

        train_split = int(0.85 * len(input_chunks))
        train_chunks = input_chunks[:train_split]
        val_chunks = input_chunks[train_split:]

        train_loader = torch.utils.data.DataLoader(train_chunks, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_chunks, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, vocab_size

    except Exception as e:
        print(f"Offline / network fallback ({e}). Generating synthetic language corpus...")
        vocab_size = 50257
        torch.manual_seed(42)

        # Synthetic multi-rule token language dataset
        def generate_text_chunks(num_chunks: int):
            chunks = []
            for _ in range(num_chunks):
                chunk = []
                for _ in range(seq_len // 4):
                    base = torch.randint(100, 5000, (1,)).item()
                    t1 = base
                    t2 = (base * 5 + 13) % vocab_size
                    t3 = (t2 + 7) % vocab_size
                    t4 = (t3 * 2) % vocab_size
                    chunk.extend([t1, t2, t3, t4])
                chunks.append(torch.tensor(chunk[:seq_len], dtype=torch.long))
            return chunks

        train_chunks = generate_text_chunks(2000)
        val_chunks = generate_text_chunks(300)

        train_loader = torch.utils.data.DataLoader(train_chunks, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_chunks, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, vocab_size


def get_model_size_info(model: torch.nn.Module) -> Tuple[int, float, float]:
    total_params = 0
    total_bits = 0.0

    for name, p in model.named_parameters():
        total_params += p.numel()
        total_bits += p.numel() * 32.0

    for name, b in model.named_buffers():
        if "codes" in name:
            total_params += b.numel()
            total_bits += b.numel() * 2.0
        elif "scales" in name:
            n_groups = b.numel()
            total_bits += n_groups * 32.0

    avg_bits = total_bits / max(1, total_params)
    memory_mb = total_bits / (8.0 * 1024 * 1024)
    return total_params, avg_bits, memory_mb


def run_text_benchmark():
    print("=" * 60)
    print("      DUOBIT Phase 2 Real Text Pretraining Benchmark")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, vocab_size = get_text_dataloaders(seq_len=64, batch_size=16)
    max_steps = 500

    experiments = {
        "FP32 Baseline (d=256)": {
            "use_duobit": False,
            "config": DuobitConfig(
                vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=64, group_size=128,
                lr=1e-3, use_swiglu=True
            )
        },
        "DUOBIT Phase 2 (d=256)": {
            "use_duobit": True,
            "config": DuobitConfig(
                vocab_size=vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=64, group_size=128,
                duobit_lr=8e-3, use_swiglu=True, scale_ema_alpha=0.01, scale_update_freq=50, transition_temperature=0.8, min_transitions_per_group=0
            )
        },
        "DUOBIT Phase 2 Scaled 2.0x (d=512)": {
            "use_duobit": True,
            "config": DuobitConfig(
                vocab_size=vocab_size, d_model=512, n_layers=4, n_heads=8, d_ff=1408, max_seq_len=64, group_size=128,
                duobit_lr=8e-3, use_swiglu=True, scale_ema_alpha=0.01, scale_update_freq=50, transition_temperature=0.8, min_transitions_per_group=0
            )
        }
    }

    results = {}

    for name, item in experiments.items():
        print(f"\n>>> Running {name}...")
        torch.manual_seed(42)
        cfg = item["config"]
        use_duobit = item["use_duobit"]

        model = DuobitTransformer(cfg, use_duobit=use_duobit)
        params, avg_bits, mem_mb = get_model_size_info(model)
        print(f"  Params: {params:,} | Avg Bits: {avg_bits:.2f} | Weight Memory: {mem_mb:.2f} MB")

        trainer = Trainer(cfg, model, train_loader, val_loader, device=device)

        start_t = time.time()
        res = trainer.train(max_steps=max_steps, eval_freq=100, warmup_steps=50)
        elapsed = time.time() - start_t

        val_loss, val_ppl = trainer.evaluate()

        logs = res["train_logs"]
        steps = [l["step"] for l in logs]
        losses = [l["loss"] for l in logs]
        trans_rates = [l.get("transition_rate", 0.0) for l in logs]
        avg_trans_rate = sum(trans_rates) / max(1, len(trans_rates))

        results[name] = {
            "params": params,
            "bits": avg_bits,
            "memory_mb": mem_mb,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "avg_trans_rate": avg_trans_rate,
            "elapsed_sec": elapsed,
            "steps": steps,
            "losses": losses,
            "trans_rates": trans_rates,
        }

        print(f"  -> {name} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | TransRate: {avg_trans_rate*100:.2f}% | Time: {elapsed:.1f}s")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, r in results.items():
        ax1.plot(r["steps"], r["losses"], label=f"{name} (Val Loss: {r['val_loss']:.2f})")
        if "FP32" not in name:
            ax2.plot(r["steps"], [tr * 100 for tr in r["trans_rates"]], label=name)

    ax1.set_title("Pretraining Loss on Text Corpus (1000 steps)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title("DUOBIT 2-Bit Code Transition Rate (%)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Transition Rate (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    chart_path = "/root/.gemini/antigravity-ide/brain/51c7ddfc-7552-429f-8f8c-c893e80a2451/duobit_phase2_text_benchmark.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    summary_path = "/kaggle/working/workspace/duobit_phase2_text_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved text benchmark plot to {chart_path}")
    print(f"Saved text summary to {summary_path}")


if __name__ == "__main__":
    run_text_benchmark()
