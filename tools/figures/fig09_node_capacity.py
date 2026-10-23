#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig09_node_capacity.pdf -- capacity curves under the fixed exp+Hamilton node.

Error (geodesic, deg) vs parameter count for two edge families sharing the SAME
node (E31/E31b/E31c) plus no-node controls (E1/E11/E32) and the reported-metric
Bayes floor (E33). Data resolved through repro/canonical.json.
"""
import json
from pathlib import Path

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


def block(name):
    entry = CANON[name]
    return json.loads((QUATKAN / "repro" / "json" / entry["file"]).read_text(encoding="utf-8"))


def pt(d, key):
    v = d[key]
    return v["params"], v["mean"], v.get("std", 0.0)


def main():
    e31, e31b, e31c = block("e31"), block("e31b"), block("e31c")
    rm = block("run_main")
    e32 = block("e32")
    e33 = block("e33")
    e40 = block("e40")

    fig, ax = plt.subplots(figsize=(5.85, 3.79))

    # --- node + Fourier edge (ours) ---
    four = [pt(e31c, "E31c:node+sparse (18p)"),
            pt(e31, "E31:node+KAN (K=1)"),
            pt(rm, "E13:K2")]
    xs, ys, es = zip(*four)
    ax.errorbar(xs, ys, yerr=es, color=PALETTE["ours"], marker="s", ms=5.5, lw=1.3,
                capsize=2.5, zorder=4, label="node + Fourier edge (ours)")

    # --- node + MLP edge ---
    m = [pt(e31b, "E31b:node+MLP h=4 (1L, 70p)"),
         pt(e31b, "E31b:node+MLP h=4 (2L, 110p)"),
         pt(e31, "E31:node+MLP h=8"),
         pt(e31, "E31:node+MLP h=32")]
    m = [pt(e31, "E31:node+MLP h=2 (1L)")] + m
    xs, ys, es = zip(*m)
    ax.errorbar(xs, ys, yerr=es, color=PALETTE["base"], marker="o", ms=5, lw=1.3,
                capsize=2.5, zorder=3, label="node + MLP edge")

    # --- no-node controls: band only (the full matrix is Figure 7) ---
    ax.axhspan(3.26, 3.36, color=PALETTE["grey"], alpha=0.18, zorder=1)
    from matplotlib.patches import Patch
    band = Patch(facecolor=PALETTE["grey"], alpha=0.35,
                 label="no-node controls (3.26--3.36)")

    bx = e32["E32:h=64,lr=0.003"]
    ax.errorbar([4996], [bx["mean"]], yerr=[bx["std"]], color=PALETTE["grey"],
                marker="^", ms=5, mfc="white", lw=0, elinewidth=1.0, capsize=2.5,
                zorder=2, label="no node: tuned (E32)")

    lie = pt(rm, "E11:MLP-Lie h=32")
    ax.errorbar([lie[0]], [lie[1]], yerr=[lie[2]], color="0.15", marker="D", ms=4.5,
                lw=0, elinewidth=1.0, capsize=2.5, zorder=2,
                label="no node: single-exp")

    floor = e40["E40:floor"]["mean"]
    ax.axhline(floor, ls="--", lw=0.9, color="0.3", zorder=1)
    ax.text(16, floor * 0.985, f"Bayes floor (E40): {floor:.2f}\u00b0",
            fontsize=7, va="top", color="0.25")

    ax.annotate("54-p Fourier edge matches\n278-p MLP edge",
                xy=(54, e31["E31:node+KAN (K=1)"]["mean"]), xytext=(19.5, 4.4),
                fontsize=7.5, color=PALETTE["ours"],
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["ours"], lw=0.9,
                                shrinkA=2, shrinkB=5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(14, 7200)
    ax.set_ylim(1.55, 32)
    ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["20", "50", "100", "200", "500", "1000", "2000", "5000"])
    ax.set_yticks([2.5, 3, 5, 10, 20])
    ax.set_yticklabels(["2.5", "3", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("Parameters (log)")
    ax.set_ylabel("Geodesic error, deg (log)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(band); labels.append(band.get_label())
    ax.legend(handles, labels, loc="upper right", fontsize=7.5, ncol=1)

    fig.savefig(OUT / "fig09_node_capacity.pdf")
    print("Saved:", OUT / "fig09_node_capacity.pdf")


if __name__ == "__main__":
    main()
