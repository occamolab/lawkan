#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig12_noise_extrap.pdf -- merged stress panels: noise (E3.2) and pole extrapolation (E3.4).

(a) geodesic error vs input/target noise sigma; (b) the four train/test splits
(in-distribution, pole on axis 1/2, both poles). Canonical run_main dump, resolved through repro/canonical.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import apply, PALETTE

ROOT = Path(__file__).resolve().parents[2]
CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))
DATA = ROOT / "repro" / "json" / CANON["run_main"]["file"]
_PAPER = ROOT.parent / "quatkan_latex" / "figures"
OUT = _PAPER if _PAPER.is_dir() else (ROOT / "figures")
apply()


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    data = {k.replace("QuatKAN", "LawKAN"): v for k, v in data.items()}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.55),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    sigma = [0.01, 0.05, 0.1, 0.2]
    for key, color, marker, label in [
            ("LawKAN", PALETTE["ours"], "o", "LawKAN-SU(2) (54 p)"),
            ("MLP-Quat h=32", PALETTE["base"], "s", "MLP-Quat h=32 (1476 p)")]:
        means = [data[f"E3.2:{s}:{key}"]["mean"] for s in sigma]
        stds = [data[f"E3.2:{s}:{key}"]["std"] for s in sigma]
        ax1.errorbar(sigma, means, yerr=stds, color=color, marker=marker, ms=5,
                     lw=1.3, capsize=2.5, label=label)
    ax1.set_xlim(0, 0.21)
    ax1.set_ylim(0, 22)
    ax1.set_xticks([0.05, 0.1, 0.15, 0.2])
    ax1.set_xlabel(r"noise $\sigma$ (rad)")
    ax1.set_ylabel("geodesic error (deg)")
    ax1.legend(loc="upper left", fontsize=6.8)
    ax1.set_title("(a) noise (E3.2)", fontsize=8)

    splits = ["inside", "axis1_pole", "axis2_pole", "both_poles"]
    groups = ["in-dist.", "axis 1 pole", "axis 2 pole", "both poles"]
    width = 0.26
    x = np.arange(len(splits))
    for i, (key, color, label) in enumerate([
            ("LawKAN", PALETTE["ours"], "LawKAN-SU(2)"),
            ("MLP-Quat h=32", PALETTE["base"], "MLP-Quat h=32"),
            ("MLP h=4", PALETTE["grey"], "MLP h=4")]):
        means = [data[f"E3.4:{key}:{s}"]["mean"] for s in splits]
        ax2.bar(x + (i - 1) * width, means, width=width, color=color, label=label)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=7)
    ax2.set_ylim(0, 32)
    ax2.set_ylabel("geodesic error (deg)")
    ax2.legend(loc="upper left", fontsize=6.8)
    ax2.set_title("(b) pole extrapolation (E3.4)", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig12_noise_extrap.pdf")
    print("Saved:", OUT / "fig12_noise_extrap.pdf")


if __name__ == "__main__":
    main()
