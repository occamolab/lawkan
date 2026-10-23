#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig04_matrix_pareto.pdf -- Figure 2: the opponent matrix as a Pareto front (E1).

Source: canonical dumps, resolved through repro/canonical.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import apply, PALETTE

ROOT = Path(__file__).resolve().parents[2]
CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))


def dump(block):
    return json.loads((ROOT / "repro" / "json" / CANON[block]["file"])
                      .read_text(encoding="utf-8"))


_PAPER = ROOT.parent / "quatkan_latex" / "figures"
OUT = _PAPER if _PAPER.is_dir() else (ROOT / "figures")
apply()

FAMILIES = [
    ("RealKAN", "#CC79A7", ["RealKAN K=3", "RealKAN 2L K=3", "RealKAN-6D h=16"]),
    ("MLP-6D", "#009E73", ["MLP-6D h=4", "MLP-6D h=8", "MLP-6D h=16", "MLP-6D h=32"]),
    ("MLP-9D+SVD", "#E69F00", ["MLP-9D+SVD h=4", "MLP-9D+SVD h=8", "MLP-9D+SVD h=16"]),
    ("MLP-Quat", "#56B4E9", ["MLP-Quat h=4", "MLP-Quat h=8", "MLP-Quat h=16", "MLP-Quat h=32"]),
    ("MLP h=4 (atan2)", PALETTE["grey"], ["MLP h=4 (atan2)"]),
]


def main():
    data = dump("run_main")
    # alias map: canonical dump keys keep the historical QuatKAN spelling
    data = {k.replace("QuatKAN", "LawKAN"): v for k, v in data.items()}
    fig, ax = plt.subplots(figsize=(5.85, 3.79))

    for name, color, keys in FAMILIES:
        params = [data[f"E1:{k}"]["params"] for k in keys]
        means = [data[f"E1:{k}"]["mean"] for k in keys]
        ax.scatter(params, means, s=81, color=color, zorder=3, label=name)

    e38 = dump("e38")
    tuned = [(e38["E38:MLPQuat64 lr3e-3"], "#56B4E9", "MLP-Quat h=64 (tuned)"),
             (e38["E38:MLP6D16 lr1e-2"], "#009E73", "MLP-6D h=16 (tuned)"),
             (e38["E38:MLP4 lr1e-2"], PALETTE["grey"], "MLP h=4 (tuned)")]
    for v, c, lab in tuned:
        ax.scatter([v["params"]], [v["mean"]], s=81, facecolors="none",
                   edgecolors=c, linewidths=1.2, zorder=3, label=lab)

    qk = data["E1:LawKAN K=1"]
    ax.scatter([qk["params"]], [qk["mean"]], s=81, color=PALETTE["ours"], zorder=4,
               label="LawKAN-SU(2) (ours)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(45.7, 6000)
    ax.set_ylim(2.45, 36.8)
    ax.set_xticks([50, 100, 200, 500, 1000, 2000, 5000])
    ax.set_xticklabels(["50", "100", "200", "500", "1000", "2000", "5000"])
    ax.set_yticks([3, 5, 10, 20, 30])
    ax.set_yticklabels(["3", "5", "10", "20", "30"])
    ax.minorticks_off()
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    ax.set_xlabel("Parameters (log)")
    ax.set_ylabel("Geodesic error, deg (log)")

    ax.annotate("", xy=(54, qk["mean"]), xytext=(100, 1.92),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0,
                                shrinkA=2, shrinkB=7),
                annotation_clip=False)
    ax.text(79, 1.78, "LawKAN-SU(2) (ours)\n2.78\u00b0 @ 54 params", color=PALETTE["ours"],
            fontsize=8.5, fontweight="bold", va="top")

    fig.savefig(OUT / "fig04_matrix_pareto.pdf")


if __name__ == "__main__":
    main()
