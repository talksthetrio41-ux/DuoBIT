#!/usr/bin/env python3
"""Pre-flight check for every DuoBIT configuration before it costs GPU hours.

A misconfigured run is only visible hours into a Kaggle session, and a sweep
shares one session with every run after it. This builds the real model and the
real optimizer from the experiment harness at a small but structurally
identical shape, takes a few steps on random ids, and asserts the invariants
that a broken config would break first:

  * the loss stays finite,
  * every trained scale stays finite,
  * every code stays inside the codebook.

It also prints the memory ladder, which is the quickest way to confirm that a
compression knob is actually charged in the accounting and not silently
ignored.

    PYTHONPATH=. python3 scripts/validate_configs.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "scripts" / "kaggle_duobit_fineweb.py"

# The v2 winner (`A8`), which every v3 variant branches from.
BASE = dict(scale_init="var", learn_scales=1, scale_lr=0.01,
            scale_relative_lr=0, duobit_lr=0.004, pefa_clip=1.0,
            duobit_wd=0.0, scale_update_freq=0, scale_ema=0.0)

# name -> overrides on top of BASE. The ordering is the compression ladder:
# each row adds one knob to the row above it.
CONFIGS = {
    "quality (v2 A8)": dict(),
    "+2-bit embeddings": dict(quant_embeddings=1),
    "+4-bit qpefa": dict(quant_embeddings=1, qpefa_bits=4),
    "+4-bit moments": dict(quant_embeddings=1, moment_bits=4),
    "max compression": dict(quant_embeddings=1, qpefa_bits=4, moment_bits=4),
    "ternary linears": dict(n_levels=3),
    "ternary + max cmp": dict(n_levels=3, quant_embeddings=1, qpefa_bits=4,
                              moment_bits=4),
}


def load_harness():
    spec = importlib.util.spec_from_file_location("kdf", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kdf"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--configs", default="",
                    help="JSON dict of name -> overrides, replacing the default set")
    a = ap.parse_args()

    kdf = load_harness()
    configs = json.loads(a.configs) if a.configs else CONFIGS
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = kdf.build_argparser()
    failures = []

    print(f"{'config':<20}{'loss@0':>9}{f'loss@{a.steps - 1}':>9}"
          f"{'train MiB':>11}{'infer MiB':>11}{'lin b/wt':>10}{'trans%':>8}")
    print("-" * 78)

    for name, over in configs.items():
        args = parser.parse_args([
            "--steps", str(a.steps), "--d-model", "256", "--n-layers", "2",
            "--n-heads", "4", "--d-ff", "512", "--batch-size", "2",
            "--seq-len", "128", "--group-size", "128"])
        for k, v in {**BASE, **over}.items():
            setattr(args, k, v)

        torch.manual_seed(1337)
        net = kdf.Transformer(args, use_duobit=True).to(dev)
        opt = kdf.DuobitAdam(net, args, dev, seed=1337)

        losses = []
        try:
            for step in range(a.steps):
                g = torch.Generator().manual_seed(step)
                ids = torch.randint(0, net.vocab_size, (2, 129), generator=g).to(dev)
                _, loss = net(ids, targets=ids)
                loss.backward()
                losses.append(float(loss.detach()))
                if not torch.isfinite(loss):
                    raise AssertionError(f"non-finite loss at step {step}")
                opt.step()
                opt.zero_grad()

            for p in net.parameters():
                if not torch.isfinite(p).all():
                    raise AssertionError("non-finite parameter")
            for m in net.modules():
                if isinstance(m, kdf.DUOBIT_MODULES):
                    if not torch.isfinite(m.scales).all():
                        raise AssertionError("non-finite group scale")
                    if int(m.codes.max()) >= m.n_levels:
                        raise AssertionError(
                            f"code {int(m.codes.max())} outside a "
                            f"{m.n_levels}-level codebook")
        except AssertionError as exc:
            failures.append((name, str(exc)))
            print(f"{name:<20}{'FAILED: ' + str(exc)}")
            continue

        rep = kdf.persistent_state_report(net, args, "duobit", opt)
        print(f"{name:<20}{losses[0]:>9.4f}{losses[-1]:>9.4f}"
              f"{rep['persistent_train_mb']:>11.2f}{rep['inference_mb']:>11.2f}"
              f"{rep['linear_train_bits_per_weight']:>10.2f}"
              f"{opt.last_transition_frac * 100:>8.3f}")

    if failures:
        print(f"\n{len(failures)} configuration(s) failed:")
        for name, why in failures:
            print(f"  {name}: {why}")
        return 1
    print(f"\nall {len(configs)} configurations ran {a.steps} steps: "
          "finite loss, finite scales, codes in range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
