#!/usr/bin/env python3
"""Fold a finished Kaggle run's output into the repository's results tree.

The experiment script writes its own report inside the Kaggle session; this
copies the durable artifacts (summaries, curves, comparison table, figures)
into `results/<tag>/` and prints a compact table so the numbers can be pasted
into the paper and the README.

    python3 scripts/collect_results.py <kaggle-output-dir> --tag fineweb-v2
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory holding the downloaded kernel output")
    ap.add_argument("--tag", required=True, help="name for results/<tag>/")
    ap.add_argument("--figures", action="store_true", help="also copy figures/")
    a = ap.parse_args()

    src = Path(a.src)
    dst = REPO / "results" / a.tag
    dst.mkdir(parents=True, exist_ok=True)

    for pat in ("summary_*.json", "val_curve_*.json", "comparison.csv",
                "summary_report.md", "config.json", "step_logs_*.csv"):
        for f in sorted(src.glob(pat)):
            shutil.copy(f, dst / f.name)
    if a.figures and (src / "figures").is_dir():
        fig_dst = dst / "figures"
        fig_dst.mkdir(exist_ok=True)
        for f in sorted((src / "figures").glob("*.png")):
            shutil.copy(f, fig_dst / f.name)

    summaries = {}
    for f in sorted(dst.glob("summary_*.json")):
        s = json.loads(f.read_text())
        summaries[s.get("name", f.stem.replace("summary_", ""))] = s
    if not summaries:
        print(f"no summary_*.json under {src}")
        return 1

    order = sorted(summaries, key=lambda k: summaries[k].get("final_val_loss") or 1e9)
    rows = [
        ("final val loss", lambda s: s.get("final_val_loss"), 4),
        ("final val PPL", lambda s: s.get("final_val_ppl"), 2),
        ("train bits/weight", lambda s: s["persistent"]["linear_train_bits_per_weight"], 2),
        ("infer bits/weight", lambda s: s["persistent"]["linear_infer_bits_per_weight"], 2),
        ("train state MB", lambda s: s["persistent"]["persistent_train_mb"], 1),
        ("inference MB", lambda s: s["persistent"]["inference_mb"], 1),
        ("tok/s", lambda s: s.get("avg_step_tok_per_s"), 0),
        ("transition rate", lambda s: s.get("mean_transition_rate"), 5),
        ("scale / init", lambda s: s.get("final_scale_ratio"), 3),
    ]
    w = max(len(k) for k in order) + 2
    print(f"\n{'metric':<24}" + "".join(f"{k:>{w}}" for k in order))
    print("-" * (24 + w * len(order)))
    for label, fn, nd in rows:
        cells = []
        for k in order:
            try:
                cells.append(fmt(fn(summaries[k]), nd))
            except Exception:
                cells.append("n/a")
        print(f"{label:<24}" + "".join(f"{c:>{w}}" for c in cells))
    print(f"\ncopied into {dst.relative_to(REPO)}")
    (dst / "index.json").write_text(json.dumps(
        {k: {"final_val_loss": v.get("final_val_loss"),
             "final_val_ppl": v.get("final_val_ppl"),
             "mode": v.get("mode"),
             "config": v.get("config", {})} for k, v in summaries.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
