import argparse
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

from duobit.config import DuobitConfig
from duobit.model.transformer import DuobitTransformer
from duobit.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation study for DUOBIT-EST components.")
    parser.add_argument("--steps", type=int, default=100, help="Training steps per ablation setting")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=== DUOBIT-EST Component Ablation Study ===")
    print(f"Device: {args.device} | Steps: {args.steps}\n")

    torch.manual_seed(42)

    config = DuobitConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=1024,
        max_seq_len=64,
        group_size=128,
        lr=1e-3,
    )

    train_data = torch.randint(0, config.vocab_size, (256, config.max_seq_len))
    train_loader = DataLoader(TensorDataset(train_data), batch_size=8, shuffle=True)

    ablations = [
        {"name": "Full DUOBIT-EST", "ec": True, "trust": 2.0, "balanced": True},
        {"name": "No Error Compensation", "ec": False, "trust": 2.0, "balanced": True},
        {"name": "No Trust Gating", "ec": True, "trust": 1e6, "balanced": True},
        {"name": "Independent Stochastic Rounding", "ec": True, "trust": 2.0, "balanced": False},
    ]

    results = {}

    for abl in ablations:
        print(f"\nRunning Ablation: {abl['name']}...")
        torch.manual_seed(42)
        model = DuobitTransformer(config, use_duobit=True)
        
        cfg = DuobitConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            group_size=config.group_size,
            lr=config.lr,
            enable_error_compensation=abl["ec"],
            trust_threshold=abl["trust"],
            balanced_rounding=abl["balanced"],
        )

        trainer = Trainer(cfg, model, train_loader, device=args.device)
        res = trainer.train(max_steps=args.steps, eval_freq=0)
        
        final_loss = res["train_logs"][-1]["loss"]
        final_ppl = res["train_logs"][-1]["perplexity"]
        final_tr = res["train_logs"][-1].get("transition_rate", 0.0)
        
        results[abl["name"]] = {"loss": final_loss, "ppl": final_ppl, "tr": final_tr}

    print("\n" + "=" * 50)
    print("           ABLATION RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Setting':<35} | {'Loss':<8} | {'Transition Rate':<15}")
    print("-" * 65)
    for name, stats in results.items():
        print(f"{name:<35} | {stats['loss']:<8.4f} | {stats['tr']*100:<14.2f}%")

if __name__ == "__main__":
    main()
