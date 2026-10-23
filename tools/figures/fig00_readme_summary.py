# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig00_readme_summary.py -- four-panel README summary.

(a) synthetic composition: geodesic error vs parameters (E1/E38), with the
    matched Bayes floor (E40);
(b) real FR3 calibration: exact dense composer, Gauss--Newton reference and the
    isolated 306-p two-slot correction (x7bench/x35);
(c) FR3 paired differences against the two references, 95% CI;
(d) cross-vendor paired differences against the classical reference, 95% CI.

All data are resolved through repro/canonical.json. Run from the repo root:
    python tools/figures/fig00_readme_summary.py
"""
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from figstyle import apply, PALETTE

ROOT = Path(__file__).resolve().parents[2]
CANON = json.loads((ROOT / "repro" / "canonical.json").read_text(encoding="utf-8"))
OUT = ROOT / "assets"
FLOOR = 2.733
apply()


def canonical(block):
    return json.loads((ROOT / "repro" / "json" / CANON[block]["file"])
                      .read_text(encoding="utf-8"))


def paired(ref, corr):
    a = np.asarray(ref["per_seed"], dtype=float)
    b = np.asarray(corr["per_seed"], dtype=float)
    if a.ndim > 1:
        a = a[:, 0]
    if b.ndim > 1:
        b = b[:, 0]
    d = a - b
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return d.mean(), d.mean() - tcrit * se, d.mean() + tcrit * se, stats.ttest_rel(a, b).pvalue


def main():
    run = canonical("run_main")
    e38 = canonical("e38")
    bench = canonical("x7bench")
    x35 = canonical("x35")
    x36 = canonical("x36seq")
    mk = canonical("mk1")
    mkfr3 = canonical("mk1_fr3")
    x38b = canonical("x38b")
    rc1 = canonical("rc1")
    ns = [20, 40, 60, 80]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9))
    ax_a, ax_b, ax_c = axes

    # ---------------- (a) synthetic Pareto ----------------
    law = run["E1:QuatKAN K=1"]
    pts = [
        (law["params"], law["mean"], law["std"], "LawKAN-SU(2), edge: Fourier (K=1), node: exact", PALETTE["ours"], "o"),
        (run["E1:MLP-Quat h=32"]["params"], run["E1:MLP-Quat h=32"]["mean"],
         run["E1:MLP-Quat h=32"]["std"], "MLP + quaternion head h=32", PALETTE["base"], "s"),
        (run["E1:MLP-6D h=32"]["params"], run["E1:MLP-6D h=32"]["mean"],
         run["E1:MLP-6D h=32"]["std"], "MLP + 6D head h=32", PALETTE["base2"], "^"),
        (run["E1:MLP-9D+SVD h=16"]["params"], run["E1:MLP-9D+SVD h=16"]["mean"],
         run["E1:MLP-9D+SVD h=16"]["std"], "MLP + 9D-SVD head h=16", PALETTE["grey"], "D"),
        (e38["E38:MLPQuat64 lr3e-3"]["params"], e38["E38:MLPQuat64 lr3e-3"]["mean"],
         e38["E38:MLPQuat64 lr3e-3"]["std"], "MLP + quaternion head h=64, tuned", "#56B4E9", "P"),
        (e38["E38:MLP4 lr1e-2"]["params"], e38["E38:MLP4 lr1e-2"]["mean"],
         e38["E38:MLP4 lr1e-2"]["std"], "MLP h=4 (atan2), tuned", "0.55", "v"),
    ]
    for x, y, sd, label, color, marker in pts:
        ax_a.errorbar([x], [y], yerr=[sd], fmt=marker, color=color, capsize=2,
                      ms=5.5, label=label)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(45.7, 6000)
    ax_a.set_ylim(2.45, 36.8)
    ax_a.set_xticks([50, 100, 200, 500, 1000, 2000, 5000])
    ax_a.set_xticklabels(["50", "100", "200", "500", "1000", "2000", "5000"])
    ax_a.set_yticks([3, 5, 10, 20, 30])
    ax_a.set_yticklabels(["3", "5", "10", "20", "30"])
    ax_a.minorticks_off()
    ax_a.axhline(FLOOR, color=PALETTE["grey"], ls=":", lw=1.0)
    ax_a.text(700, 2.72, "Bayes floor 2.733°", color="0.35", fontsize=6.5)
    ax_a.set_xlabel("parameters (log)")
    ax_a.set_ylabel("geodesic error, deg (log)")
    ax_a.set_title("(a) synthetic composition", fontsize=8.5)
    ax_a.legend(fontsize=5.8, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.26), ncol=2)
    ax_a.annotate("54 p (18 sparse)", xy=(law["params"], law["mean"]),
                  xytext=(120, 2.92), fontsize=6.2,
                  arrowprops=dict(arrowstyle="-", lw=0.6, color="0.4"))

    # ---------------- (b) absolute FR3 curves ----------------
    series = [
        ("LawKAN-SU(2), node: exact (54 p)", "X7bench:n{n}:ours", bench, PALETTE["grey"], "-", "o"),
        ("per-joint screw, GN (54 p)", "X7bench:n{n}:gn", bench, PALETTE["base2"], "--", "s"),
        ("LawKAN-SU(2), two-slot $\\psi$ (306 p), isolated", "X35:n{n}:isolated", x35, PALETTE["ours"], "-", "D"),
    ]
    for label, pattern, src, color, ls, marker in series:
        mu = [src[pattern.format(n=n)]["pos_mm_mean"] for n in ns]
        sd = [src[pattern.format(n=n)]["pos_mm_std"] for n in ns]
        ax_b.errorbar(ns, mu, yerr=sd, color=color, ls=ls, marker=marker, capsize=2,
                      lw=1.2, label=label)
    ax_b.set_xlim(15, 85)
    ax_b.set_xticks(ns)
    ax_b.set_xticklabels([str(n) for n in ns])
    ax_b.set_xlabel("training poses $N$")
    ax_b.set_ylabel("position error (mm)")
    ax_b.set_title("(b) real FR3 calibration", fontsize=8.5)
    ax_b.legend(fontsize=6.0, frameon=False, loc="upper center",
                bbox_to_anchor=(0.5, -0.26), ncol=1)
    ax_b.text(0.03, 0.95, "error bars: ±1 seed SD", transform=ax_b.transAxes,
              fontsize=5.5, color="0.35", va="top")

    # ---------------- (c) vendor paired differences ----------------
    labels = ["ABB\nN=320", "KUKA", "Panda", "Kinova", "FR3 unit"]
    diffs = [
        paired(rc1["RC1:N320:gn"], rc1["RC1:N320:isolated"]),
        paired(mk["MK1:kuka14:dense:rms_mm"], x36["X36:kuka14:twoslot:rms_mm"]),
        paired(mk["MK1:panda_1:dense:rms_mm"], x36["X36:panda_1:twoslot:rms_mm"]),
        paired(mk["MK1:kinova:dense:rms_mm"], x38b["X38b:kinova:iso_pj_kernel:rms_mm"]),
        paired(mkfr3["MK1:fr3:dense:rms_mm"], x36["X36:fr3:twoslot:rms_mm"]),
    ]
    x = np.arange(5)
    means = np.array([d[0] for d in diffs])
    lo = np.array([d[0] - d[1] for d in diffs])
    hi = np.array([d[2] - d[0] for d in diffs])
    colors = [PALETTE["ours"] if m > 0 else PALETTE["grey"] for m in means]
    colors[3] = PALETTE["nodeonly"]
    ax_c.bar(x, means, 0.60, yerr=[lo, hi], capsize=3, color=colors,
             edgecolor="k", linewidth=0.35, zorder=3)
    ax_c.axhline(0, color="k", lw=0.7)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, fontsize=6.2)
    ax_c.set_ylabel("reference − LawKAN $\\psi$ (mm), 95% CI")
    ax_c.set_title("(c) correction vs classical reference", fontsize=8.5)
    ax_c.set_ylim(-0.40, 0.38)
    ptxt = ["p=4e-4", "p=0.023", "p=0.007", "p=0.10 (ns)",
            "p=0.028 (dense wins)"]
    for xi, m, up, text in zip(x, means, hi, ptxt):
        y = m + up + 0.018 if m >= 0 else m - up - 0.035
        ax_c.text(xi, y, text, ha="center", fontsize=5.8, color="0.2")
    ax_c.text(0.02, 0.96, "correction better", transform=ax_c.transAxes,
              fontsize=5.8, color=PALETTE["ours"], va="top")

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.30, top=0.91, wspace=0.36)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "readme_summary.png", dpi=220, bbox_inches="tight")
    print("Saved:", OUT / "readme_summary.png")


if __name__ == "__main__":
    main()
