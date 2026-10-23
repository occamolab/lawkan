#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig15_new_platforms.pdf -- the two new cross-vendor platforms (KUKA, ABB).

(a) MUKCa socket consistency (KUKA iiwa14): held-out RMS scatter, mean +/- std
    over seeds, for the nominal model, the reduced edge (tool + joint biases),
    the dense screw model, the isolated psi (protocol) and the joint control;
(b) RobotCali draw-wire (ABB IRB 120): held-out mean |dL| against training
    size for the constant base/tool fit, the exact screw model, the same-class
    Gauss-Newton fit, the isolated psi and the joint control.

Data: repro/json/MK1_mukca.*.json and RC1_robotcali.*.json (canonical dumps;
missing series are skipped, so the figure can be previewed on partial runs).
Run:  python fig15_new_platforms.py
"""
import glob
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


CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))


def canonical(block):
    """Resolve a dump through repro/canonical.json (never by 'latest filename')."""
    return json.loads((ROOT / "repro" / "json" / CANON[block]["file"])
                      .read_text(encoding="utf-8"))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))

    # ---------------- (a) KUKA socket consistency ----------------
    ax = axes[0]
    try:
        mk = canonical("mk1")
        plat = "kuka14"
        series = [("nominal", PALETTE["grey"]), ("edge_b", PALETTE["base"]),
                  ("dense", PALETTE["base2"]), ("iso", PALETTE["ours"]),
                  ("joint", "#CC79A7")]
        labels, means, stds, colors = [], [], [], []
        for key, col in series:
            k = f"MK1:{plat}:{key}:rms_mm"
            if k not in mk:
                continue
            labels.append({"nominal": "nominal", "edge_b": "edge (bias)",
                           "dense": "dense", "iso": "isolated $\\psi$",
                           "joint": "joint"}[key])
            means.append(mk[k]["mean"]); stds.append(mk[k]["std"])
            colors.append(col)
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=colors, capsize=2, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("socket RMS (mm)")
        ax.set_title("(a) MUKCa, KUKA iiwa14")
        ax.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
        nseed = len(mk.get(f"MK1:{plat}:iso:rms_mm", {}).get("per_seed", []))
        ax.text(0.98, 0.95, f"{nseed} seeds", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5)
    except FileNotFoundError:
        ax.axis("off")

    # ---------------- (b) ABB draw-wire ----------------
    ax = axes[1]
    try:
        rc = canonical("rc1")
        sizes = sorted({int(k.split(":")[1][1:]) for k in rc
                        if k.startswith("RC1:N") and k.endswith(":baseline")})
        series = [("baseline", PALETTE["grey"], "--", "s"),
                  ("exact", PALETTE["base2"], "-", "o"),
                  ("gn", PALETTE["base"], ":", "v"),
                  ("isolated", PALETTE["ours"], "-", "D"),
                  ("joint", "#CC79A7", "--", "^")]
        for key, col, ls, mk_ in series:
            xs, mu, sd = [], [], []
            for n in sizes:
                k = f"RC1:N{n}:{key}"
                if k not in rc:
                    continue
                xs.append(n); mu.append(rc[k]["dL_mm_mean"]); sd.append(rc[k]["dL_mm_std"])
            if xs:
                ax.errorbar(xs, mu, yerr=sd, color=col, ls=ls, marker=mk_,
                            capsize=2, lw=1.1,
                            label={"baseline": "const base/tool", "exact": "exact",
                                   "gn": "GN-LS", "isolated": "isolated $\\psi$",
                                   "joint": "joint"}[key])
        ax.set_xscale("log")
        ax.set_xticks(sizes); ax.set_xticklabels([str(n) for n in sizes])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("training poses $N$")
        ax.set_ylabel("held-out $|\\Delta L|$ (mm)")
        ax.set_title("(b) RobotCali, ABB IRB 120")
        ax.legend(fontsize=6.2, frameon=False)
    except FileNotFoundError:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUT / "fig15_new_platforms.pdf")
    print("Saved:", OUT / "fig15_new_platforms.pdf")


if __name__ == "__main__":
    main()
