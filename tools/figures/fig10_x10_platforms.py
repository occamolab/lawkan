#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig10_x10_platforms.pdf -- X10 three-platform FK calibration (P48).

Held-out position error vs training-set size for FR3, UR10 and UR5:
constant base/tool fit, dense screw model (ours) and the rotation-only sparse
variant; mean +/- std over the canonical ten seeds (0--9). Data resolved
through repro/canonical.json.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import apply, PALETTE

ROOT = Path(__file__).resolve().parents[2]
QUATKAN = ROOT
CANON = json.loads((QUATKAN / "repro" / "canonical.json").read_text(encoding="utf-8"))
_PAPER = ROOT.parent / "quatkan_latex" / "figures"
OUT = _PAPER if _PAPER.is_dir() else (ROOT / "figures")
apply()

PLATFORMS = [
    ("FR3 (55 p)", "philnad", ""),
    ("UR10 (48 p)", "ipa_A", ""),
    ("UR5 (48 p, pos.)", "pauljac_UR5", "_random"),
]
NS = [20, 40, 80]
SERIES = [
    ("const. base/tool fit", "nominal_gap", PALETTE["grey"], "--", "o"),
    ("dense screw (ours)", "screw_dense", PALETTE["ours"], "-", "s"),
    ("sparse (rot. only)", "screw_sparse", PALETTE["base"], "-.", "^"),
]


def main():
    d = json.loads((QUATKAN / "repro" / "json" /
                    CANON["x10"]["file"]).read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.05))
    for ax, (title, key, suffix) in zip(axes, PLATFORMS):
        ps = d[key]["per_seed"]
        seeds = sorted(ps.keys())
        for label, field, color, ls, mk in SERIES:
            means, stds = [], []
            for n in NS:
                vals = [ps[s][f"N{n}"][f"{field}{suffix}"]["pos_mean"] for s in seeds]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            ax.errorbar(NS, means, yerr=stds, color=color, ls=ls, marker=mk,
                        ms=3.5, lw=1.1, capsize=2, label=label)
        dense80 = np.mean([ps[s]["N80"][f"screw_dense{suffix}"]["pos_mean"] for s in seeds])
        const80 = np.mean([ps[s]["N80"][f"nominal_gap{suffix}"]["pos_mean"] for s in seeds])
        ax.annotate(f"{const80/dense80:.1f}$\\times$", xy=(80, const80), xytext=(-3, 3),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7.5, color=PALETTE["ours"])
        ax.set_title(title, fontsize=8)
        ax.set_xticks(NS)
        ax.set_xticklabels(["20", "40", "80"])
        ax.set_xlabel("train poses")
    axes[0].set_ylabel("held-out position error (mm)")
    axes[0].legend(loc="center left", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_x10_platforms.pdf")
    print("Saved:", OUT / "fig10_x10_platforms.pdf")


if __name__ == "__main__":
    main()
