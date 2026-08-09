import argparse
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

from duobit.config import DuobitConfig
from duobit.model.transformer import DuobitTransformer
from duobit.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny DUOBIT-EST model and compare with baseline.")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"=== DUOBIT-EST Pretraining Benchmark ===")
    print(f"Device: {args.device} | Steps: {args.steps} | Batch Size: {args.batch_size}")
    print(f"Config: d_model={args.d_model}, n_layers={args.n_layers}, seq_len={args.seq_len}\n")

    torch.manual_seed(42)

    config = DuobitConfig(
        vocab_size=1000,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=4,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_len,
        group_size=128,
        lr=args.lr,
    )

    # Synthetic Dataset
    train_data = torch.randint(0, config.vocab_size, (512, config.max_seq_len))
    val_data = torch.randint(0, config.vocab_size, (128, config.max_seq_len))

    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=args.batch_size, shuffle=False)

    # 1. Train DUOBIT-EST Model (Native 2-bit parameters, no master weights)
    print(">>> 1. Training DUOBIT-EST (Native 2-bit Weights)...")
    duobit_model = DuobitTransformer(config, use_duobit=True)
    duobit_trainer = Trainer(config, duobit_model, train_loader, val_dataloader=val_loader, device=args.device)
    
    t0 = time.time()
    duobit_res = duobit_trainer.train(max_steps=args.steps, eval_freq=args.steps // 2)
    duobit_time = time.time() - t0

    # 2. Train Standard Baseline Model (Continuous Parameters)
    print("\n>>> 2. Training Standard Baseline Model (Continuous Weights)...")
    baseline_model = DuobitTransformer(config, use_duobit=False)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    baseline_trainer = Trainer(config, baseline_model, train_loader, val_dataloader=val_loader, optimizer=baseline_optimizer, device=args.device)
    
    t0 = time.time()
    baseline_res = baseline_trainer.train(max_steps=args.steps, eval_freq=args.steps // 2)
    baseline_time = time.time() - t0

    print("\n=== Benchmark Summary ===")
    print(f"DUOBIT-EST Final Loss: {duobit_res['train_logs'][-1]['loss']:.4f} | PPL: {duobit_res['train_logs'][-1]['perplexity']:.2f} | Time: {duobit_time:.2f}s")
    print(f"Baseline   Final Loss: {baseline_res['train_logs'][-1]['loss']:.4f} | PPL: {baseline_res['train_logs'][-1]['perplexity']:.2f} | Time: {baseline_time:.2f}s")
    
    trans_rate = duobit_res['train_logs'][-1].get('transition_rate', 0.0)
    print(f"DUOBIT-EST Final Step Transition Rate: {trans_rate*100:.2f}%")

if __name__ == "__main__":
    main()
