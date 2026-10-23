#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig08_imu.pdf -- Figure A5: BROAD real-IMU, error vs horizon (X5).

Source: canonical x5 dump, resolved through repro/canonical.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figstyle import apply, PALETTE

ROOT = Path(__file__).resolve().parents[2]
CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))
DATA = ROOT / "repro" / "json" / CANON["x5"]["file"]
_PAPER = ROOT.parent / "quatkan_latex" / "figures"
OUT = _PAPER if _PAPER.is_dir() else (ROOT / "figures")
apply()


def main():
    d = json.loads(DATA.read_text(encoding="utf-8"))
    keys = ["a-short", "b-1s", "c-5s"]
    horizons = [0.07, 1.0, 5.0]

    fig, ax = plt.subplots(figsize=(4.03, 2.63))

    ax.plot(horizons, [d[f"X5:{k}:BiasIntegrator"]["mean"] for k in keys],
            "--", color=PALETTE["integrator"], marker="s",
            label="strapdown (free)")

    ax.plot([1.0, 5.0], [d[f"X5:{k}:QuatIMU-sparse"]["mean"] for k in ["b-1s", "c-5s"]],
            "-", color=PALETTE["ours"], marker="o",
            label="sparse (18 p, ours)")

    ym = [d[f"X5:{k}:QuatIMU"]["mean"] for k in keys]
    ax.plot(horizons, ym, "-", color=PALETTE["base"], marker="^",
            label="dense (27/36 p, ours)")
    for k, h in zip(keys, horizons):
        seeds = d[f"X5:{k}:QuatIMU"]["per_seed"]
        ax.plot([h] * len(seeds), seeds, "o", color=PALETTE["base"], ms=2.5,
                alpha=0.75, zorder=2)
    ax.annotate("1 of 3 seeds", xy=(5.0, 73.07), xytext=(3.2, 40),
                fontsize=6, color=PALETTE["base"], ha="center",
                arrowprops=dict(arrowstyle="-", color=PALETTE["base"], lw=0.6,
                                shrinkA=2, shrinkB=4))

    ax.plot(horizons, [d[f"X5:{k}:MLP h=32"]["mean"] for k in keys],
            "-", color=PALETTE["grey"], marker="v", label="MLP h=32")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels(["0.07", "1.0", "5.0"])
    ax.set_yticks([0.3, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["0.3", "0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlim(0.03, 7.5)
    ax.set_ylim(0.3, 200)
    ax.set_xlabel("Horizon (s)")
    ax.set_ylabel("Geodesic error (deg)")
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.5, labelspacing=0.35,
              handletextpad=0.5)

    fig.savefig(OUT / "fig08_imu.pdf")


if __name__ == "__main__":
    main()
