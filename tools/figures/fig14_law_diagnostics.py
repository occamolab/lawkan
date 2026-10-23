#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig14_law_diagnostics.pdf -- diagnostics and cross-domain checks.

(a) X16 diagnostics on the learned law: input-group ablation (error with the
    feature group zeroed) and the MLP-psi partial dependence along theta_B;
(b) S1: synthetic law controls (exact vs deformed law, error vs N);
(c) S2: relaxed law on the independent UR5 dataset (error vs N).
Data: X16_psi_diagnostics, S1_synthetic_law, S2_ur5gtpn_relaxed,
X11_ur5_gtpn_sweep dumps.
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


CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))


def canonical(block):
    """Resolve a dump through repro/canonical.json (never by 'latest filename')."""
    return json.loads((ROOT / "repro" / "json" / CANON[block]["file"])
                      .read_text(encoding="utf-8"))


def main():
    x16 = canonical("x16iso")
    s1 = canonical("s1")
    s2 = canonical("s2")
    x11 = canonical("x11")
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.15))

    # ---------------- (a) diagnostics ----------------
    ax = axes[0]
    ab = x16.get("X16:ablation:seed0", {})
    order = [("none", "full"), ("A_angles", "A angles"), ("A_trans", "A trans"),
             ("B_angles", "B angles"), ("B_trans", "B trans")]
    vals = [ab.get(k, np.nan) for k, _ in order]
    names = [lab for _, lab in order]
    ax.bar(range(len(vals)), vals,
           color=[PALETTE["grey"]] + [PALETTE["ours"]] * 4)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=6.3, rotation=20)
    ax.set_ylabel("held-out error (mm)")
    ax.set_title("(a) which inputs carry $\\psi$", pad=10)
    if "X16:pd_B_theta" in x16:
        axin = ax.inset_axes([0.50, 0.28, 0.47, 0.30])
        xs = np.array(x16["X16:pd_B_x"])
        axin.plot(xs, x16["X16:pd_B_theta"], color=PALETTE["base"], lw=1.0)
        axin.set_title("$\\|\\psi\\|$ vs $\\theta_B$", fontsize=6.2)
        axin.tick_params(labelsize=5.8)
        axin.set_xticks([0, np.pi / 2, np.pi])
        axin.set_xticklabels(["0", "$\\pi/2$", "$\\pi$"], fontsize=5.8)

    # ---------------- (b) synthetic controls ----------------
    ax = axes[1]
    ns1 = [20, 80, 160]
    for tag, (tlabel, ls, color) in {"exact": ("exact law", "-", PALETTE["base"]),
                                     "deformed": ("deformed law", "--", PALETTE["ours"])}.items():
        mu, sd = [], []
        for n in ns1:
            a = np.array(s1[f"S1:{tag}:n{n}:exact_dense"]["per_seed"])
            mu.append(a.mean()); sd.append(a.std())
        ax.errorbar(ns1, mu, yerr=sd, color=PALETTE["grey"], ls=ls, marker="o",
                    capsize=2, label=f"exact, {tlabel}")
        mu, sd = [], []
        for n in ns1:
            a = np.array(s1[f"S1:{tag}:n{n}:relaxed_fourier"]["per_seed"])
            mu.append(a.mean()); sd.append(a.std())
        ax.errorbar(ns1, mu, yerr=sd, color=color, ls=ls, marker="s", capsize=2,
                    label=f"relaxed, {tlabel}")
    ax.set_xscale("log")
    ax.set_xticks(ns1)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticklabels([str(n) for n in ns1])
    ax.set_xlabel("training poses $N$")
    ax.set_ylabel("held-out position error (mm)")
    ax.set_title("(b) synthetic controls")
    ax.legend(fontsize=6.0, frameon=False)

    # ---------------- (c) UR5-GTPN ----------------
    ax = axes[2]
    NS = [20, 40, 60, 80]
    refs = [("exact (X11)", ["X11:n%d:ours" % n for n in NS], PALETTE["grey"], "-", "s"),
            ("GN-LS (X11)", ["X11:n%d:gn" % n for n in NS], PALETTE["base2"], "--", "o")]
    for label, keys, color, ls, mk in refs:
        mu = [np.mean(x11[k]["per_seed"]) for k in keys]
        sd = [np.std(x11[k]["per_seed"]) for k in keys]
        ax.errorbar(NS, mu, yerr=sd, color=color, ls=ls, marker=mk, capsize=2,
                    label=label, lw=1.1)
    mu = [s2[f"S2:n{n}:relaxed"]["pos_mm_mean"] for n in NS]
    sd = [s2[f"S2:n{n}:relaxed"]["pos_mm_std"] for n in NS]
    ax.errorbar(NS, mu, yerr=sd, color=PALETTE["ours"], ls="-", marker="D",
                capsize=2, label="relaxed law (390 p)", lw=1.1)
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("training poses $N$")
    ax.set_ylabel("held-out position error (mm)")
    ax.set_title("(c) independent UR5 (GTPN)")
    ax.legend(fontsize=6.0, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig14_law_diagnostics.pdf")
    print("Saved:", OUT / "fig14_law_diagnostics.pdf")


if __name__ == "__main__":
    main()
