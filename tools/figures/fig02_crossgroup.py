#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig02_crossgroup.pdf -- Figure A1: the recipe transfers to the GL+ and SE(3) instances.

Source: canonical e16v2 and e22 dumps via repro/canonical.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figstyle import apply, PALETTE

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "repro" / "json"
_PAPER = ROOT.parent / "quatkan_latex" / "figures"
OUT = _PAPER if _PAPER.is_dir() else (ROOT / "figures")
apply()


def main():
    can = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))

    def canonical(k):
        return json.loads((ROOT / "repro" / "json" / can[k]["file"]).read_text(encoding="utf-8"))

    e16 = canonical("e16v2")
    e22 = canonical("e22")

    cats = [
        ("analytic\nedge (ours)", "E16v2:GLnode-structured (analytic supp)",
         "E22:SE3-structured (analytic + exp)", PALETTE["base2"]),
        ("dense/linear\nedge (ours)", "E16v2:GLKAN-exp K=1 (all inputs)",
         "E22:SE3-dense (linear edge + exp)", PALETTE["grey"]),
        ("no exp\n(ours)", "E16v2:GLnode-noexp K=1 (all inputs)",
         "E22:SE3-nodeonly (no exp)", PALETTE["nodeonly"]),
        ("best MLP\n(opponent)", "E16v2:MLP h=64 (unmatched, v1 baseline)", "E22:MLP h=64", PALETTE["base"]),
    ]
    groups = ["$GL^{+}(2,\\mathbb{R})$", "SE(3)"]
    x = np.arange(len(groups))
    width = 0.2

    fig, ax = plt.subplots(figsize=(5.33, 2.91))
    for i, (lab, k16, k22, color) in enumerate(cats):
        vals = [e16[k16]["mean"], e22[k22]["mean"]]
        pos = x + (i - 1.5) * width
        edge = dict(edgecolor="black", linewidth=0.6) if i == 0 else {}
        ax.bar(pos, vals, width=width, color=color, label=lab, log=True, **edge)
        for xi, v in zip(pos, vals):
            ax.annotate(f"{v:.4f}".rstrip("0").rstrip("."), (xi, v),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Error (relative Frobenius / probe units)")
    ax.set_ylim(1e-4, 2.0)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles[:1] + handles[1:], labels[:1] + labels[1:],
                    loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=7.5, ncol=4,
                    frameon=False, title="ours (3 edge variants) vs. opponent", title_fontsize=8,
                    columnspacing=1.2, handletextpad=0.5)
    leg.get_texts()[0].set_fontweight("bold")
    ax.set_title("same node, different charts: the limit is the chart, not the node",
                 fontsize=8.5)
    fig.subplots_adjust(bottom=0.22)

    fig.savefig(OUT / "fig02_crossgroup.pdf")


if __name__ == "__main__":
    main()
