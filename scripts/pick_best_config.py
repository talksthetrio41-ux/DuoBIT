#!/usr/bin/env python3
"""Read a finished sweep's summaries and emit the winning DuoBIT configuration.

Prints the ablation table (each run's own overrides next to its validation
loss) and writes the best duobit run's overrides as JSON, ready to paste into
the final head-to-head run.

    python3 scripts/pick_best_config.py <sweep-output-dir> [--out best.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# the knobs the ablation varies; everything else is shared across runs
KNOBS = ["scale_init", "learn_scales", "scale_lr", "scale_relative_lr",
         "duobit_lr", "pefa_clip", "duobit_wd", "scale_update_freq", "scale_ema"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default="/tmp/duobit_best_config.json")
    a = ap.parse_args()

    runs = {}
    for f in sorted(Path(a.src).glob("summary_*.json")):
        s = json.loads(f.read_text())
        runs[s.get("name", f.stem[8:])] = s
    if not runs:
        print(f"no summary_*.json under {a.src}")
        return 1

    base = next((s for s in runs.values() if s.get("mode") == "fp32"), None)
    order = sorted(runs, key=lambda k: runs[k].get("final_val_loss") or 1e9)

    print(f"{'run':<14}{'mode':<7}{'val loss':>10}{'val PPL':>10}"
          f"{'vs fp32':>9}{'trans%':>8}{'s/s0':>7}  config")
    print("-" * 118)
    for k in order:
        s = runs[k]
        vl = s.get("final_val_loss")
        vp = s.get("final_val_ppl")
        delta = (vl - base["final_val_loss"]) if (base and vl is not None) else None
        cfg = s.get("config", {})
        knobs = {j: cfg.get(j) for j in KNOBS if j in cfg}
        tr = s.get("mean_transition_rate")
        sr = s.get("final_scale_ratio")
        print(f"{k:<14}{s.get('mode',''):<7}"
              f"{(f'{vl:.4f}' if vl is not None else 'n/a'):>10}"
              f"{(f'{vp:,.1f}' if vp is not None else 'n/a'):>10}"
              f"{(f'{delta:+.4f}' if delta is not None else '-'):>9}"
              f"{(f'{tr*100:.3f}' if tr is not None else '-'):>8}"
              f"{(f'{sr:.3f}' if sr is not None else '-'):>7}  "
              + (" ".join(f"{j}={v}" for j, v in knobs.items())
                 if s.get("mode") == "duobit" else ""))

    # The best run at the sweep's horizon is not automatically the best at the
    # final horizon: too high a rate wins early and plateaus, too low loses
    # early and catches up. Print the whole curve so the trajectory is visible.
    curves = {}
    for k in order:
        f = Path(a.src) / f"val_curve_{k}.json"
        if f.exists():
            curves[k] = json.loads(f.read_text())
    if curves:
        steps = sorted({c["step"] for v in curves.values() for c in v})
        print(f"\nvalidation curve\n{'run':<14}"
              + "".join(f"{'@'+str(st):>10}" for st in steps))
        print("-" * (14 + 10 * len(steps)))
        for k in order:
            row = {c["step"]: c["val_loss"] for c in curves.get(k, [])}
            print(f"{k:<14}" + "".join(
                f"{row[st]:>10.4f}" if st in row else f"{'-':>10}" for st in steps))

    duobit = [k for k in order if runs[k].get("mode") == "duobit"]
    if not duobit:
        print("\nno duobit runs in this sweep")
        return 1
    best = duobit[0]
    cfg = runs[best].get("config", {})
    overrides = {j: cfg[j] for j in KNOBS if j in cfg}
    Path(a.out).write_text(json.dumps(overrides, indent=1))
    print(f"\nbest duobit run: {best} "
          f"(val {runs[best]['final_val_loss']:.4f})")
    if base:
        d = runs[best]["final_val_loss"] - base["final_val_loss"]
        worst = runs[duobit[-1]]["final_val_loss"]
        print(f"  gap to FP32 baseline: {d:+.4f} nats "
              f"| spread across duobit variants: {worst - runs[best]['final_val_loss']:.4f} nats")
    print(f"  overrides written to {a.out}:\n  {json.dumps(overrides)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
