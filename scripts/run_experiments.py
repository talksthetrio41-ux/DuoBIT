#!/usr/bin/env python3
"""Small-scale verification of DUOBIT-EST.

Runs on CPU:
  1. Codebook reconstruction MSE
  2. Quadratic bowl: discrete compressed training vs FP32 vs latent-weight STE
  3. Tiny decoder on a synthetic arithmetic grammar
  4. Width scaling
  5. Training and inference memory accounting

Writes JSON + PNG figures for the paper and the companion site.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from duobit.config import DuobitConfig
from duobit.layers.duobit_linear import DuobitLinear
from duobit.layers.ste_linear import SteQuantLinear
from duobit.model.transformer import DuobitTransformer
from duobit.optim.duobit_adam import DuobitAdam
from duobit.quantization.codebook import (
    SymmetricCodebook,
    compute_group_scales,
    compute_mse_group_scales,
)
from duobit.quantization.memory import inference_memory_report, training_memory_report
from duobit.training.trainer import Trainer

OUT_DIR = os.environ.get("DUOBIT_OUT", os.path.join(ROOT, "results"))
FIG_DIR = os.path.join(OUT_DIR, "figures")
DATA_DIR = os.path.join(OUT_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

INK = "#1A1814"
SLATE = "#3D4F5F"
RULE = "#C4B8A4"
CREAM = "#F4EFE4"
MUTED = "#6B6459"
ACCENT = "#4E5D4A"


def style_ax(ax):
    ax.set_facecolor(CREAM)
    ax.figure.patch.set_facecolor(CREAM)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=MUTED)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.grid(True, color=RULE, alpha=0.45, linewidth=0.6)


def experiment_codebook_mse(seed: int = 0) -> Dict[str, Any]:
    torch.manual_seed(seed)
    w = torch.randn(64, 256)
    rows = []
    for n_levels, name in [(2, "binary"), (3, "ternary"), (4, "2-bit")]:
        cb = SymmetricCodebook(n_levels=n_levels)
        for g in (32, 64, 128):
            s_max = compute_group_scales(w, group_size=g)
            z_max = cb.quantize(w, s_max)
            rec_max = cb.dequantize(z_max, s_max)
            mse_max = float(((w - rec_max) ** 2).mean())
            s_mse = compute_mse_group_scales(w, z_max, cb, group_size=g)
            z_mse = cb.quantize(w, s_mse)
            rec_mse = cb.dequantize(z_mse, s_mse)
            mse_opt = float(((w - rec_mse) ** 2).mean())
            rows.append(
                {
                    "codebook": name,
                    "n_levels": n_levels,
                    "group_size": g,
                    "mse_maxabs": mse_max,
                    "mse_closed_form": mse_opt,
                    "rel_improvement": (mse_max - mse_opt) / mse_max,
                }
            )
    return {"rows": rows}


def experiment_quadratic(steps: int = 80, seed: int = 0) -> Dict[str, Any]:
    torch.manual_seed(seed)
    in_f, out_f, n = 64, 32, 128
    x = torch.randn(n, in_f)
    true_w = torch.randn(out_f, in_f) * 0.3
    y = x @ true_w.t()

    def run_fp32() -> Dict[str, Any]:
        torch.manual_seed(seed)
        layer = nn.Linear(in_f, out_f, bias=False)
        model = nn.Sequential(layer)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        losses, trans = [], []
        for _ in range(steps):
            opt.zero_grad()
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            trans.append(0.0)
        n_w = out_f * in_f
        return {
            "name": "FP32 Adam",
            "final_loss": losses[-1],
            "losses": losses,
            "trans_rates": trans,
            "mean_trans": 0.0,
            "train_bytes": n_w * 12,
            "infer_bytes": n_w * 4,
        }

    def run_ste() -> Dict[str, Any]:
        torch.manual_seed(seed)
        layer = SteQuantLinear(in_f, out_f, group_size=32, n_levels=4)
        model = nn.Sequential(layer)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        losses, trans = [], []
        for _ in range(steps):
            opt.zero_grad()
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            trans.append(0.0)
        n_w = out_f * in_f
        return {
            "name": "Latent-weight STE",
            "final_loss": losses[-1],
            "losses": losses,
            "trans_rates": trans,
            "mean_trans": 0.0,
            "train_bytes": n_w * 12,
            "infer_bytes": n_w * 4,
        }

    def run_duobit(name: str, **opt_kw) -> Dict[str, Any]:
        torch.manual_seed(seed)
        layer = DuobitLinear(in_f, out_f, group_size=32, n_levels=4)
        model = nn.Sequential(layer)
        opt = DuobitAdam(
            model,
            lr=0.05,
            duobit_lr=0.08,
            scale_update_freq=20,
            scale_ema_alpha=0.05,
            weight_decay=0.0,
            **opt_kw,
        )
        losses, trans = [], []
        for _ in range(steps):
            opt.zero_grad()
            loss = (model(x) - y).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            trans.append(float(getattr(opt, "last_transition_frac", 0.0)))
        mem = training_memory_report(model, opt)
        inf = inference_memory_report(model)
        return {
            "name": name,
            "final_loss": losses[-1],
            "losses": losses,
            "trans_rates": trans,
            "mean_trans": sum(trans) / max(1, len(trans)),
            "train_bytes": int(mem["linear_train_bytes"]),
            "infer_bytes": int(inf["code_bytes"] + inf["scale_bytes"]),
        }

    runs = [
        run_fp32(),
        run_ste(),
        run_duobit(
            "DUOBIT QPEFA-8",
            use_pefa=True,
            qpefa_bits=8,
            moment_bits=8,
            factored_second_moment=True,
            enable_error_compensation=False,
        ),
        run_duobit(
            "DUOBIT QPEFA-4",
            use_pefa=True,
            qpefa_bits=4,
            moment_bits=8,
            factored_second_moment=True,
            pefa_clip=2.0,
            enable_error_compensation=False,
        ),
        run_duobit(
            "DUOBIT FP32 residual",
            use_pefa=True,
            qpefa_bits=32,
            moment_bits=32,
            factored_second_moment=False,
            enable_error_compensation=False,
        ),
    ]
    return {"steps": list(range(1, steps + 1)), "runs": runs}


class GrammarDataset(Dataset):
    """Triples (x, 3x+1, 3x+6) mod vocab, packed into fixed-length sequences."""

    def __init__(self, vocab: int, seq_len: int, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.samples = []
        lo = 8
        hi = max(lo + 1, vocab // 3)
        for _ in range(n):
            prompt = torch.randint(lo, hi, (seq_len // 3 + 2,), generator=g)
            seq: List[int] = []
            for t in prompt.tolist():
                v1 = int(t)
                v2 = (v1 * 3 + 1) % vocab
                v3 = (v2 + 5) % vocab
                seq.extend([v1, v2, v3])
            self.samples.append(torch.tensor(seq[:seq_len], dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


@torch.no_grad()
def greedy_gen(model, prompt: List[int], gen_len: int, device: str) -> List[int]:
    model.eval()
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    for _ in range(gen_len):
        logits, _ = model(ids)
        nxt = int(torch.argmax(logits[0, -1, :]).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    model.train()
    return ids[0].tolist()


def grammar_token_accuracy(tokens: List[int], vocab: int) -> float:
    if len(tokens) < 3:
        return 0.0
    ok = 0
    n = 0
    i = 0
    while i + 2 < len(tokens):
        v1, v2, v3 = tokens[i], tokens[i + 1], tokens[i + 2]
        n += 2
        if v2 == (v1 * 3 + 1) % vocab:
            ok += 1
        if v3 == (v2 + 5) % vocab:
            ok += 1
        i += 3
    return ok / max(1, n)


def make_tiny_config(**kwargs) -> DuobitConfig:
    cfg = DuobitConfig(
        vocab_size=96,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=16,
        group_size=32,
        lr=1e-3,
        duobit_lr=8e-3,
        use_swiglu=True,
        dropout=0.0,
        weight_decay=0.0,
        scale_update_freq=25,
        scale_ema_alpha=0.02,
        transition_temperature=0.8,
        use_pefa=True,
        enable_error_compensation=False,
        n_levels=4,
        activation_bits=16,
        use_hadamard=False,
        qpefa_bits=8,
        moment_bits=8,
        factored_second_moment=True,
        pefa_clip=4.0,
        block_size=64,
        quant_mode="duobit",
    )
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def train_one(
    name: str,
    cfg: DuobitConfig,
    use_duobit,
    train_loader,
    val_loader,
    steps: int,
    device: str,
    prompt: List[int],
) -> Dict[str, Any]:
    torch.manual_seed(42)
    model = DuobitTransformer(cfg, use_duobit=use_duobit)
    if use_duobit is True:
        opt = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    trainer = Trainer(cfg, model, train_loader, val_loader, optimizer=opt, device=device)
    t0 = time.time()
    res = trainer.train(max_steps=steps, eval_freq=max(steps // 2, 20), warmup_steps=10)
    elapsed = time.time() - t0
    val_loss, val_ppl = trainer.evaluate()
    gen = greedy_gen(model, prompt, gen_len=12, device=device)
    inf = inference_memory_report(model)
    train_mem = training_memory_report(model, trainer.optimizer if use_duobit is True else None)
    if use_duobit is not True:
        # STE / FP32: W+m+v for every parameter.
        n = sum(p.numel() for p in model.parameters())
        train_mem = {
            **train_mem,
            "total_bytes": float(n * 12),
            "total_mb": n * 12 / (1024.0 * 1024.0),
            "linear_train_bytes": float(inf["n_linear_weights"] * 12),
            "linear_bits_per_weight_train": 96.0,
            "fp32_adam_total_mb": n * 12 / (1024.0 * 1024.0),
            "bitnet_total_mb": n * 12 / (1024.0 * 1024.0),
        }
    logs = res["train_logs"]
    return {
        "name": name,
        "use_duobit": use_duobit is True,
        "n_levels": cfg.resolved_n_levels() if use_duobit is True else 0,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "elapsed_sec": elapsed,
        "ms_per_step": 1000.0 * elapsed / steps,
        "gen_tokens": gen,
        "grammar_acc": grammar_token_accuracy(gen, cfg.vocab_size),
        "memory": inf,
        "train_memory": {k: (float(v) if not isinstance(v, dict) else v) for k, v in train_mem.items()},
        "n_params": int(inf["n_params"]),
        "memory_mb": inf["total_mb"],
        "train_mb": train_mem["total_mb"],
        "linear_train_bits": train_mem.get("linear_bits_per_weight_train", 96.0),
        "avg_bits": inf["avg_bits_per_param"],
        "linear_bits": inf["linear_bits_per_weight"],
        "steps": [l["step"] for l in logs],
        "losses": [l["loss"] for l in logs],
        "trans_rates": [l.get("transition_rate", 0.0) for l in logs],
        "entropy": [l.get("level_entropy_bits", 0.0) for l in logs],
        "final_trans": logs[-1].get("transition_rate", 0.0) if logs else 0.0,
        "mean_trans": (
            sum(l.get("transition_rate", 0.0) for l in logs) / max(1, len(logs))
        ),
        "final_entropy": logs[-1].get("level_entropy_bits", 0.0) if logs else 0.0,
    }


def experiment_lm(steps: int = 50, device: str = "cpu") -> Dict[str, Any]:
    vocab, seq = 96, 16
    train_ds = GrammarDataset(vocab, seq, n=800, seed=42)
    val_ds = GrammarDataset(vocab, seq, n=160, seed=999)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    prompt = [10, 31, 36, 12]

    specs = [
        ("FP32 Adam", make_tiny_config(quant_mode="fp32"), False),
        ("Latent-weight STE", make_tiny_config(quant_mode="ste"), "ste"),
        (
            "DUOBIT 2-bit QPEFA-8",
            make_tiny_config(qpefa_bits=8, moment_bits=8, factored_second_moment=True),
            True,
        ),
        (
            "DUOBIT 2-bit QPEFA-4",
            make_tiny_config(qpefa_bits=4, moment_bits=8, factored_second_moment=True, pefa_clip=2.0),
            True,
        ),
        (
            "DUOBIT ternary QPEFA-8",
            make_tiny_config(n_levels=3, qpefa_bits=8, moment_bits=8),
            True,
        ),
        (
            "DUOBIT binary QPEFA-8",
            make_tiny_config(n_levels=2, qpefa_bits=8, moment_bits=8),
            True,
        ),
        (
            "DUOBIT W2A8 QPEFA-8",
            make_tiny_config(activation_bits=8, qpefa_bits=8, moment_bits=8),
            True,
        ),
    ]
    runs = []
    for name, cfg, use_d in specs:
        print(f"\n===== {name} =====")
        runs.append(train_one(name, cfg, use_d, train_loader, val_loader, steps, device, prompt))
        r = runs[-1]
        print(
            f"  val={r['val_loss']:.4f} ppl={r['val_ppl']:.2f} "
            f"trans={r['mean_trans']*100:.3f}% infer={r['memory_mb']:.3f}MB "
            f"train={r['train_mb']:.3f}MB acc={r['grammar_acc']:.2f}"
        )
    return {"steps": steps, "prompt": prompt, "runs": runs}


def experiment_width(steps: int = 35, device: str = "cpu") -> Dict[str, Any]:
    vocab, seq = 96, 16
    train_ds = GrammarDataset(vocab, seq, n=600, seed=42)
    val_ds = GrammarDataset(vocab, seq, n=120, seed=7)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    prompt = [10, 31, 36, 12]
    runs = []
    for d, name, use_d in [
        (64, "FP32 d=64", False),
        (64, "DUOBIT d=64", True),
        (96, "DUOBIT d=96", True),
    ]:
        cfg = make_tiny_config(
            d_model=d,
            d_ff=d * 2,
            n_heads=2 if d == 64 else 3,
            quant_mode="fp32" if not use_d else "duobit",
        )
        print(f"\n===== width {name} =====")
        runs.append(train_one(name, cfg, use_d, train_loader, val_loader, steps, device, prompt))
    return {"runs": runs}


def plot_all(mse, quad, lm, width):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style_ax(ax)
    labels = [f"{r['codebook']}\nG={r['group_size']}" for r in mse["rows"]]
    xs = list(range(len(labels)))
    ax.bar([x - 0.18 for x in xs], [r["mse_maxabs"] for r in mse["rows"]], 0.36, color=MUTED, label="max-abs scale")
    ax.bar([x + 0.18 for x in xs], [r["mse_closed_form"] for r in mse["rows"]], 0.36, color=SLATE, label="MSE-optimal scale")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Group-scale fitting")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_codebook_mse.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style_ax(ax)
    palette = [INK, MUTED, SLATE, ACCENT, RULE]
    for i, run in enumerate(quad["runs"]):
        ax.plot(quad["steps"], run["losses"], color=palette[i % len(palette)], lw=1.8, label=run["name"])
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE")
    ax.set_title("Quadratic bowl: discrete vs dense")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_quadratic_loss.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style_ax(ax)
    for i, run in enumerate(quad["runs"]):
        if run["mean_trans"] == 0 and "DUOBIT" not in run["name"]:
            continue
        ax.plot(
            quad["steps"],
            [100 * t for t in run["trans_rates"]],
            color=palette[i % len(palette)],
            lw=1.6,
            label=run["name"],
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Code transition rate (%)")
    ax.set_title("Do 2-bit codes actually move?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_quadratic_transitions.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    style_ax(ax)
    colors = [INK, MUTED, SLATE, ACCENT, "#7A8B99", "#8C5E4A", RULE]
    for i, run in enumerate(lm["runs"]):
        ax.plot(run["steps"], run["losses"], color=colors[i % len(colors)], lw=1.7, label=run["name"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("Tiny decoder on synthetic grammar")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_lm_loss.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    style_ax(ax)
    for i, run in enumerate(lm["runs"]):
        if not run["use_duobit"]:
            continue
        ax.plot(
            run["steps"],
            [100 * t for t in run["trans_rates"]],
            color=colors[i % len(colors)],
            lw=1.6,
            label=run["name"],
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Transition rate (%)")
    ax.set_title("LM code-flip rate under quantized error feedback")
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_lm_transitions.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    style_ax(ax)
    for i, run in enumerate(lm["runs"]):
        ax.scatter(run["train_mb"], run["val_loss"], color=colors[i % len(colors)], s=48, zorder=3)
        ax.annotate(
            run["name"],
            (run["train_mb"], run["val_loss"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color=INK,
        )
    ax.set_xlabel("Persistent training memory (MB)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Quality vs training memory (tiny LM)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_memory_frontier.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    style_ax(ax)
    names = [r["name"] for r in lm["runs"]]
    train_mbs = [r["train_mb"] for r in lm["runs"]]
    infer_mbs = [r["memory_mb"] for r in lm["runs"]]
    xs = list(range(len(names)))
    ax.bar([x - 0.18 for x in xs], train_mbs, 0.36, color=SLATE, label="Training state")
    ax.bar([x + 0.18 for x in xs], infer_mbs, 0.36, color=ACCENT, label="Inference checkpoint")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7, rotation=18, ha="right")
    ax.set_ylabel("MB")
    ax.set_title("Training vs inference footprint")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_train_infer_memory.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    style_ax(ax)
    for i, run in enumerate(width["runs"]):
        ax.plot(
            run["steps"],
            run["losses"],
            color=colors[i % len(colors)],
            lw=1.8,
            label=f"{run['name']} (train {run['train_mb']:.3f} MB)",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("Width scaling at 2-bit storage")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_width_scaling.png"), dpi=160)
    plt.close(fig)


def main():
    device = "cpu"
    print("Device:", device)
    print("=== Experiment 1: codebook MSE ===")
    mse = experiment_codebook_mse()
    print(json.dumps(mse, indent=2))

    print("\n=== Experiment 2: quadratic bowl ===")
    quad = experiment_quadratic(steps=80)
    for r in quad["runs"]:
        print(
            f"  {r['name']:28s} loss={r['final_loss']:.5f} "
            f"mean_trans={r['mean_trans']*100:.3f}% train_B={r['train_bytes']}"
        )

    print("\n=== Experiment 3: tiny LM ===")
    lm = experiment_lm(steps=50, device=device)

    print("\n=== Experiment 4: width scaling ===")
    width = experiment_width(steps=35, device=device)

    plot_all(mse, quad, lm, width)

    payload = {
        "version": "1.0.0",
        "device": device,
        "codebook_mse": mse,
        "quadratic": {
            "steps": quad["steps"],
            "runs": quad["runs"],
        },
        "lm": lm,
        "width": width,
    }
    out_json = os.path.join(DATA_DIR, "results.json")
    with open(out_json, "w") as f:
        json.dump(payload, f)
    print("Wrote", out_json)
    print("Figures in", FIG_DIR)


if __name__ == "__main__":
    main()
