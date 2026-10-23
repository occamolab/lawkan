#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig13_learned_law.pdf -- the learned law on the real FR3 (X13/X14/S1/S3).

(a) held-out position error vs training size (N=20/40/60/80): exact dense,
    GN-LS, isolated two-slot-psi, tuned residual MLP (baseline comparison);
(b) capacity curve: global (Fourier) / local kernel / two-slot vs N;
(c) synthetic controls S1: exact law (solid) vs deformed law (dashed);
(d) seed stability on Kinova: reduced edge, global head (collapses on 2/5
    seeds), local kernel.
Data: repro/json/X7_bench_main, X14_fourier_law, X22_learned_baseline,
X34_fr3_kernel, X35_recount_fr3, S1_synthetic_law, X32_kernel dumps.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from figstyle import apply, PALETTE

ROOT = Path(__file__).resolve().parents[2]
_PAPER = ROOT.parent / "quatkan_latex" / "figures"
OUT = _PAPER if _PAPER.is_dir() else (ROOT / "figures")
apply()
# This figure is shown at ~column width, so nominal sizes look larger than the
# other figures (displayed at DISPLAY_SCALE); shrink labels to match.
plt.rcParams.update({
    "font.size": 6.2,
    "axes.labelsize": 6.0,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 5.6,
    "ytick.labelsize": 5.6,
})

NS = [20, 40, 60, 80]


CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))


def canonical(block):
    """Resolve a dump through repro/canonical.json (never by 'latest filename')."""
    return json.loads((ROOT / "repro" / "json" / CANON[block]["file"])
                      .read_text(encoding="utf-8"))


def curve2(d, key):
    a = np.array(d[key]["per_seed"], dtype=float)[:, 0]
    return a.mean(), a.std()


def curve1(d, key):
    a = np.array(d[key]["per_seed"], dtype=float)
    return a.mean(), a.std()


def curve3(d, key):
    """mean/std fields (G2bc isolated dump)"""
    return float(d[key]["pos_mm_mean"]), float(d[key]["pos_mm_std"])


def main():
    bench = canonical("x7bench")
    x13 = canonical("x13")
    x14 = canonical("x14")
    g2 = canonical("g2bc_fig")
    s3 = canonical("s3")
    x22 = canonical("x22")
    x35 = canonical("x35")
    x34 = canonical("x34")
    s1 = canonical("s1")
    x32 = canonical("x32")
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4))

    # ---------------- (a) FR3 curves ----------------
    ax = axes[0, 0]
    series = [
        ("exact dense (54 p)", lambda n: curve2(bench, f"X7bench:n{n}:ours"),
         PALETTE["grey"], "-", "o"),
        ("GN-LS (54 p, classical)", lambda n: curve2(bench, f"X7bench:n{n}:gn"),
         PALETTE["base2"], "--", "s"),
        ("isolated two-slot-$\\psi$ (306 p)", lambda n: curve3(x35, f"X35:n{n}:isolated"),
         PALETTE["ours"], "-", "D"),
        ("tuned residual MLP ($18$k--$69$k p)", lambda n: curve1(x22, f"X22:n{n}:selected_mlp"),
         "#56B4E9", "-.", "P"),
    ]
    for label, fn, color, ls, mk in series:
        mu = np.array([fn(n)[0] for n in NS])
        sd = np.array([fn(n)[1] for n in NS])
        ax.errorbar(NS, mu, yerr=sd, color=color, ls=ls, marker=mk, capsize=2.5,
                    label=label, lw=1.2, markersize=4)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("training poses $N$")
    ax.set_ylabel("held-out position error (mm)")
    ax.set_title("(a) real FR3 calibration")
    ax.set_ylim(1.0, 2.7)
    ax.legend(fontsize=5.2, loc="upper right", frameon=False, handlelength=1.5,
              handletextpad=0.4, labelspacing=0.35)

    # ---------------- (b) capacity: global / kernel / two-slot ----------------
    ax = axes[0, 1]
    cap = [
        ("global (Fourier)", lambda n: curve1(x14, f"X14:n{n}:relaxed_fourier"),
         PALETTE["base2"], "--", "s"),
        ("kernel (local)", lambda n: curve3(x34, f"X34:n{n}:kernel"),
         PALETTE["base"], ":", "v"),
        ("two-slot (final)", lambda n: curve3(x35, f"X35:n{n}:isolated"),
         PALETTE["ours"], "-", "D"),
    ]
    NSC = [20, 40, 60, 80]
    for label, fn, color, ls, mk in cap:
        mu = np.array([fn(n)[0] for n in NSC])
        sd = np.array([fn(n)[1] for n in NSC])
        ax.errorbar(NSC, mu, yerr=sd, color=color, ls=ls, marker=mk, capsize=2,
                    label=label, lw=1.2, markersize=4)
    ax.set_xticks(NSC)
    ax.set_xticklabels([str(n) for n in NSC])
    ax.set_xlabel("training poses $N$")
    ax.set_ylabel("held-out position error (mm)")
    ax.set_title("(b) capacity: global / kernel / two-slot")
    ax.legend(fontsize=5.6, frameon=False, loc="upper right")

    # ---------------- (c) synthetic controls (S1): exact vs deformed law ----------------
    ax = axes[1, 0]
    NS_C = [20, 80, 160]
    for law, ls, model, color, mk, name in (
            ("exact", "-", "exact_dense", PALETTE["grey"], "o", "exact"),
            ("deformed", "--", "exact_dense", "#1A1A1A", "o", "exact"),
            ("exact", "-", "relaxed_fourier", PALETTE["ours"], "D", "relaxed"),
            ("deformed", "--", "relaxed_fourier", "#CC79A7", "D", "relaxed")):
        mu = np.array([s1[f"S1:{law}:n{n}:{model}"]["pos_mm_mean"] for n in NS_C])
        sd = np.array([s1[f"S1:{law}:n{n}:{model}"]["pos_mm_std"] for n in NS_C])
        ax.errorbar(NS_C, mu, yerr=sd, color=color, ls=ls, marker=mk, capsize=2,
                    lw=1.2, markersize=4, label=f"{name}, {law} law")
    ax.set_xscale("log")
    ax.set_xticks(NS_C)
    ax.set_xticklabels([str(n) for n in NS_C])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("training poses $N$")
    ax.set_ylabel("held-out position error (mm)")
    ax.set_title("(c) synthetic controls (S1)")
    ax.set_ylim(0.5, 15.0)
    ax.legend(fontsize=5.0, frameon=False, loc="center right")

    # ---------------- (d) seed stability: global vs local head (Kinova) ----------------
    ax = axes[1, 1]
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
    ax.set_title("(d) seed stability (Kinova)")
    ax.legend(fontsize=5.6, frameon=False, loc="upper left")

    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.09,
                        wspace=0.30, hspace=0.42)
    fig.savefig(OUT / "fig13_learned_law.pdf")
    print("Saved:", OUT / "fig13_learned_law.pdf")


if __name__ == "__main__":
    main()
