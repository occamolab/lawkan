#!/usr/bin/env python3
"""fig16_kinova_seeds.pdf -- seed stability of the relaxation on Kinova (X32).

Per-seed socket RMS for the reduced edge, the global head and the local
kernel; the global head collapses on the two outlying seeds (8, 9) of the
ten-seed probe, above the reduced edge, while the local kernel stays below
it on every seed.
Data: repro/json/X32_kernel dump.
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
apply()
plt.rcParams.update({
    "font.size": 6.2,
    "axes.labelsize": 6.0,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
})

CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))


def canonical(block):
    return json.loads((ROOT / "repro" / "json" / CANON[block]["file"])
                      .read_text(encoding="utf-8"))


def main():
    x32 = canonical("x32")
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))
    seeds_lbl = ["0", "1", "2", "8", "9"]
    xi = np.arange(len(seeds_lbl))
    for key, color, ls, mk, name in (
            ("X32:edge_b:rms_mm", PALETTE["grey"], "-", "o", "reduced edge"),
            ("X32:base:rms_mm", PALETTE["base2"], "--", "s", "global head"),
            ("X32:kr_1:rms_mm", PALETTE["ours"], "-", "D", "local kernel")):
        ax.plot(xi, x32[key]["per_seed"], color=color, ls=ls, marker=mk, lw=1.2,
                markersize=4, label=name)
    ax.set_xticks(xi)
    ax.set_xticklabels(seeds_lbl)
    ax.set_xlabel("seed")
    ax.set_ylabel("socket RMS (mm)")
    ax.set_title("seed stability (Kinova)")
    ax.legend(fontsize=5.6, frameon=False, loc="upper left")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.15)
    fig.savefig(OUT / "fig16_kinova_seeds.pdf")
    print("Saved:", OUT / "fig16_kinova_seeds.pdf")


if __name__ == "__main__":
    main()
