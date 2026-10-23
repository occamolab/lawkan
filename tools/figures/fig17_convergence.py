#!/usr/bin/env python3
"""fig17_convergence.pdf -- the convergence check (X46 A1).

Held-out error vs training epochs (checkpoint grid 100--8000, 5 seeds),
two-slot correction vs the matched MLP, at N=20/60/80.
Data: the canonical X46 dump, resolved through repro/canonical.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import apply, PALETTE

ROOT = Path(__file__).resolve().parents[2]
_PAPER = ROOT.parent / "quatkan_latex" / "figures"
OUT = _PAPER if _PAPER.is_dir() else (ROOT / "figures")

SIZES = [20, 60, 80]
EPS = list(range(10, 4001, 10))
CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))
DUMP = ROOT / "repro" / "json" / CANON["x46"]["file"]

d = json.load(open(DUMP))
def curve(N, model):
    return [np.mean(d[f"X46:A1:N{N}:{model}:ep{e}"]["per_seed"]) for e in EPS]

apply()
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
for ax, N in zip(axes, SIZES):
    ax.plot(EPS, curve(N, "two_slot"), 
            color=PALETTE["ours"], label="two-slot")
    ax.plot(EPS, curve(N, "mlp"), 
            color=PALETTE["base"], label="matched MLP")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"$N={N}$")
    ax.set_xlabel("epochs")
axes[0].set_ylabel("held-out error (mm)")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig17_convergence.pdf", bbox_inches="tight")
print("saved", OUT / "fig17_convergence.pdf")
