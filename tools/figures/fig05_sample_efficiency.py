#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig05_sample_efficiency.pdf -- Figure A3: sample-efficiency curves (E4).

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


def main():
    data = dump("run_main")
    # alias map: canonical dump keys keep the historical QuatKAN spelling
    data = {k.replace("QuatKAN", "LawKAN"): v for k, v in data.items()}
    sizes = [50, 100, 200, 400, 800, 1600]
    series = [
        ("LawKAN", PALETTE["ours"], "o", "LawKAN-SU(2) (54 params, ours)"),
        ("MLP-6D h=16", PALETTE["base"], "s", "MLP-6D h=16 (518 p)"),
        ("MLP h=4", PALETTE["grey"], "^", "MLP h=4 (76 p)"),
    ]

    e39 = dump("e39")
    fig, ax = plt.subplots(figsize=(4.68, 3.22))
    for key, color, marker, label in series:
        means = [data[f"E4:N{n}:{key}"]["mean"] for n in sizes]
        stds = [data[f"E4:N{n}:{key}"]["std"] for n in sizes]
        ax.errorbar(sizes, means, yerr=stds, color=color, marker=marker, ms=7,
                    lw=1.4, capsize=3, label=label)

    for key, color, lab in (("MLP6D16_tuned", PALETTE["base"], "MLP-6D h=16 (tuned lr)"),
                            ("MLPQuat64_tuned", "#56B4E9", "MLP-Quat h=64 (tuned lr)")):
        means = [e39[f"E39:N{n}:{key}"]["mean"] for n in sizes]
        ax.plot(sizes, means, color=color, ls="--", lw=1.2, label=lab)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_yticks([2, 3, 5, 10, 20, 30])
    ax.set_yticklabels(["2", "3", "5", "10", "20", "30"])
    ax.set_xlim(38, 1700)
    ax.set_ylim(1.0, 60)
    ax.set_xlabel("Training samples N")
    ax.set_ylabel("Geodesic error (deg)")
    ax.annotate("LawKAN-SU(2)@50 $\\approx$ fixed 6D@1600\n32$\\times$ vs fixed, $\\approx$7$\\times$ vs tuned",
                xy=(50, data["E4:N50:LawKAN"]["mean"]), xycoords="data",
                xytext=(0.02, 0.02), textcoords="axes fraction",
                fontsize=7, va="bottom", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8,
                                shrinkA=3, shrinkB=6))
    ax.legend(loc="upper right", fontsize=7.5)

    fig.savefig(OUT / "fig05_sample_efficiency.pdf")


if __name__ == "__main__":
    main()
