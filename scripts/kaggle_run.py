#!/usr/bin/env python3
"""Push, poll and collect a DuoBIT experiment on Kaggle notebooks.

Kaggle exposes GPU sessions through "kernels": a script plus a
`kernel-metadata.json`. Two things about the accelerator are worth writing down,
because the naming is misleading:

  * The accelerator is selected with `enable_gpu: true` *and*
    `machine_shape: "NvidiaTeslaT4"` (equivalently `kaggle kernels push
    --accelerator NvidiaTeslaT4`). `enable_gpu` alone leaves the choice to the
    server default.
  * There is no separate "T4 x2" enum value. Kaggle's T4 machine shape *is* the
    dual-T4 machine: `NvidiaTeslaT4` provisions two Tesla T4s, and
    `torch.cuda.device_count()` returns 2 inside the session. The alternatives
    are `NvidiaTeslaP100` (one P100) and `Tpu1VmV38`. So dual T4 is what you get
    by asking for `NvidiaTeslaT4` and then actually using both devices --
    `kaggle_duobit_fineweb.py` spawns one DDP rank per visible GPU.

Usage:
    python3 scripts/kaggle_run.py push  <name> [--args '<argv json list>']
    python3 scripts/kaggle_run.py status <name>
    python3 scripts/kaggle_run.py fetch  <name> [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "kaggle_duobit_fineweb.py"
USER = "androi890"


def build_dir(name: str, argv: list, work: Path) -> Path:
    """Materialise a kernel folder: the experiment script + its metadata."""
    d = work / f"k_{name}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    code = SCRIPT.read_text()
    argv_lit = json.dumps(argv, indent=4)
    code = code.replace(
        'if __name__ == "__main__":\n    main()',
        f"KAGGLE_ARGV = {argv_lit}\n\n"
        'if __name__ == "__main__":\n'
        "    # prepended, so the same file can be dry-run locally with overrides\n"
        "    sys.argv = [sys.argv[0]] + KAGGLE_ARGV + sys.argv[1:]\n"
        "    main()")
    (d / f"{name}.py").write_text(code)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{name}",
        "title": name.replace("-", " "),
        "code_file": f"{name}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        # enable_gpu + machine_shape NvidiaTeslaT4 == the dual-T4 machine
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }, indent=2))
    return d


def run(cmd: list, **kw) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return (out.stdout or "") + (out.stderr or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "status", "fetch"])
    ap.add_argument("name")
    ap.add_argument("--args", default="[]", help="JSON list of argv for the script")
    ap.add_argument("--out", default=None)
    ap.add_argument("--work", default="/tmp/duobit_kernels")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)

    if a.action == "push":
        d = build_dir(a.name, json.loads(a.args), work)
        print(run(["kaggle", "kernels", "push", "-p", str(d),
                   "--accelerator", "NvidiaTeslaT4"]))
    elif a.action == "status":
        print(run(["kaggle", "kernels", "status", f"{USER}/{a.name}"]))
    else:
        out = Path(a.out or (work / f"out_{a.name}"))
        out.mkdir(parents=True, exist_ok=True)
        print(run(["kaggle", "kernels", "output", f"{USER}/{a.name}",
                   "-p", str(out)])[-2000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
