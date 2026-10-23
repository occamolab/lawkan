#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""check_numbers.py -- verify paper numbers against the canonical dumps.

Resolves every block through the manifest `repro/canonical.json` (never by
"latest filename"). CHECKS is the curated list of the paper's headline numbers:
(label, block, key path, expected value, tolerance).

Run:  python repro/check_numbers.py [--repo /path/to/lawkan]
Exit: 0 -- all checks pass; 1 -- mismatches or a missing dump.

A mismatch is a TODO: either the paper number or the dump must be updated, and
the corresponding CHECKS entry must change with it.
"""
import argparse
import json
import os
import sys

CHECKS = [
    # --- core matrix / battery (run_main) ---
    ("Table matrix: LawKAN K=1", "run_main", "E1:QuatKAN K=1", 2.78, 0.05),
    ("Table matrix: MLP-Quat h=32", "run_main", "E1:MLP-Quat h=32", 3.36, 0.05),
    ("Table matrix: MLP-6D h=16", "run_main", "E1:MLP-6D h=16", 4.49, 0.05),
    ("Table matrix: MLP h=4 (atan2)", "run_main", "E1:MLP h=4 (atan2)", 15.14, 0.10),
    ("E3.1 n=3: LawKAN", "run_main", "E3.1:QuatKAN n=3", 3.19, 0.05),
    ("E3.1 n=3: MLP-Quat h=32", "run_main", "E3.1:MLP-Quat h=32", 6.60, 0.10),
    ("E3.2 sigma=0.01: LawKAN", "run_main", "E3.2:0.01:QuatKAN", 0.96, 0.05),
    ("E3.2 sigma=0.2: LawKAN", "run_main", "E3.2:0.2:QuatKAN", 18.07, 0.10),
    ("E3.3 N=3200: LawKAN", "run_main", "E3.3:N3200:QuatKAN", 2.77, 0.05),
    ("E3.4 both poles: LawKAN", "run_main", "E3.4:QuatKAN:both_poles", 3.46, 0.15),
    ("E7a null: LawKAN", "run_main", "E7a:QuatKAN K=1", 23.87, 0.50),
    ("E7b random: LawKAN", "run_main", "E7b:QuatKAN K=1", 38.21, 0.50),
    ("E8 n=4: LawKAN", "run_main", "E8:n4:QuatKAN", 3.79, 0.10),
    ("E11 Lie baseline", "run_main", "E11:MLP-Lie h=32", 3.28, 0.05),
    ("E13 K=2", "run_main", "E13:K2", 2.88, 0.05),
    # --- extensions (run_ext) ---
    ("E1 RPMG (B1-fixed)", "run_ext", "X1:MLP-Quat h=32 + RPMG", 3.42, 0.05),
    ("E15 QuatIMU", "run_ext", "X3:E15:QuatIMU", 0.082, 0.005),
    ("E15 MLP+RPMG", "run_ext", "X3:E15:MLP h=32 + RPMG", 9.21, 0.05),
    ("FK LawKANArm", "run_ext", "X3:FK:QuatKANArm", 0.159, 0.02),
    ("FK MLP+RPMG", "run_ext", "X3:FK:MLP h=32 + RPMG", 0.672, 0.02),
    ("E18 BCH1", "run_ext", "E18:QuatKAN-BCH1 (+1/2 cross)", 4.00, 0.05),
    # --- sparse edge (e19) ---
    ("Sparse 18p", "e19", "E19.1:sparse (18p)", 2.75, 0.05),
    ("Dense 54p", "e19", "E19.1:dense K=1 (54p)", 2.79, 0.05),
    # --- discovery / SE(3) (replications) ---
    ("Discovery m=6", "e21", "E21:pruned m=6", 0.0026, 0.0002),
    ("SE(3) structured", "e22", "E22:SE3-structured (analytic + exp)", 4.8e-4, 0.5e-4),
    ("SE(3) dense", "e22", "E22:SE3-dense (linear edge + exp)", 0.388, 0.001),
    ("SE(3) MLP h=64", "e22", "E22:MLP h=64", 0.329, 0.002),
    # --- order rule (E20, seeds 10-14 = Table 10/P35 source) ---
    ("Table 10 K1 eps=0", "e20", "E20:eps=0 (canonical):K1", 2.78, 0.05),
    ("Table 10 K2 harmonic", "e20", "E20:eps=0.3 (harmonic):K2", 3.31, 0.05),
    ("Table 10 noiseless K2", "e20", "E20:noiseless:K2", 0.17, 0.02),
    ("E20 s0 K1 eps=0 (consistency)", "e20s0", "E20:eps=0 (canonical):K1", 2.79, 0.05),
    ("E20 s0 K2 harmonic (consistency)", "e20s0", "E20:eps=0.3 (harmonic):K2", 3.32, 0.05),
    ("E23 S1", "e23", "E23:S1={a,p}:sp0.5", 0.387, 0.002),
    ("E23 S4", "e23", "E23:S4=full:sp0.5", 6.1e-4, 5e-5),
    # --- mechanism / components ---
    ("E25 state-quad", "e25", "E25:StateKAN-quad", 0.0818, 0.001),
    ("E25 state-linear", "e25", "E25:StateKAN-linear", 7.13, 0.05),
    ("E25 MLP h=32", "e25", "E25:MLP h=32", 0.2765, 0.005),
    ("E25b FF-quad", "e25b", "E25b:FF-quad", 1.58, 0.05),
    ("E29 tensor n=3", "e29", "E29:trained_tensor_n3", 0.14, 0.10),
    ("E29 fixed n=2", "e29", "E29:fixed_n2", 24.4, 0.5),
    ("E30 LawKAN", "e30", "E30:QuatKAN-SU(2)", 6.12, 0.10),
    ("E30 Lie head", "e30", "E30:Lie head h=32", 6.86, 0.10),
    # --- node isolation / tuning / metric floor (phase 2) ---
    ("E31 node+KAN", "e31", "E31:node+KAN (K=1)", 2.78, 0.05),
    ("E31 node+MLP h=8", "e31", "E31:node+MLP h=8", 2.97, 0.05),
    ("E31 control plain MLP h=32", "e31", "E31:plain MLP-Quat h=32", 3.36, 0.05),
    ("E31 control single-exp", "e31", "E31:single-exp MLP-Lie h=32", 3.29, 0.05),
    ("E31 node+linear (refuted)", "e31", "E31:node+linear", 13.31, 0.10),
    ("E31b MLP h=4 2L (below threshold)", "e31b", "E31b:node+MLP h=4 (2L, 110p)", 3.69, 0.10),
    ("E31b MLP h=4 1L", "e31b", "E31b:node+MLP h=4 (1L, 70p)", 5.56, 0.20),
    ("E31c sparse 18p", "e31c", "E31c:node+sparse (18p)", 2.747, 0.02),
    ("E31 node+MLP h=32", "e31", "E31:node+MLP h=32", 2.900, 0.05),
    ("E31 node+MLP h=2 (1L)", "e31", "E31:node+MLP h=2 (1L)", 22.9, 0.5),
    ("E32 best cell h=64/lr=3e-3", "e32", "E32:h=64,lr=0.003", 3.26, 0.05),
    ("E33 floor n=2 (metric, corrected)", "e33", "E33:n2:intrinsic", 2.697, 0.01),
    ("E33 floor n=3 (metric, corrected)", "e33", "E33:n3:intrinsic", 3.057, 0.01),
    ("E33 posterior spread n=2 (diagnostic)", "e33", "E33:n2:posterior_spread", 1.902, 0.01),
    ("E33b metric-loss LawKAN", "e33b", "E33b:QuatKAN K=1, metric loss", 2.79, 0.05),
    ("E33c stable-metric LawKAN", "e33c", "E33c:QuatKAN K=1, stable metric", 2.79, 0.05),
    ("E31d pruned m=12 (72p)", "e31d", "E31d:pruned m=12 (72p)", 2.81, 0.05),
    ("E31d pruned m=3 (18p, refuted)", "e31d", "E31d:pruned m=3 (18p)", 16.24, 0.5),
    ("E19b random mask 18p", "e19b", "E19b:random-mask (18p)", 32.89, 0.10),
    ("E36 MC SO(3) floor", "e36", "E36:floor_intrinsic", 6.050, 0.02),
    ("E36 target-noise-only bound", "e36", "E36:target_noise_only", 5.477, 0.05),
    # --- Table 3 (arm calibration) ---
    ("T3 exact n=80", "x7bench", "X7bench:n80:ours.pos_mm_mean", 1.400, 0.02),
    ("T3 GN n=80", "x7bench", "X7bench:n80:gn.pos_mm_mean", 1.331, 0.02),
    ("T3 PoE n=20", "x7bench", "X7bench:n20:poe.pos_mm_mean", 1.724, 0.03),
    ("T3 two-slot n=60", "x35", "X35:n60:isolated", 1.144, 0.01),
    ("T3 two-slot n=80", "x35", "X35:n80:isolated", 1.118, 0.01),
    ("T3 joint n=80", "x35", "X35:n80:joint", 1.094, 0.01),
    ("T3 selected MLP n=80", "x22", "X22:n80:selected_mlp", 1.329, 0.02),
    ("T3 exact node n=80", "x24", "X24:n80:exact", 1.400, 0.02),
    # --- Table 4 (socket) ---
    ("T4 KUKA dense", "x38", "X38:kuka14:dense:rms_mm", 0.260, 0.01),
    ("T4 KUKA reduced edge", "x38", "X38:kuka14:iso_edgeb:rms_mm", 0.166, 0.005),
    ("T4 KUKA per-joint two-slot", "x38", "X38:kuka14:iso_pj:rms_mm", 0.190, 0.005),
    ("T4 Panda dense", "x38", "X38:panda_1:dense:rms_mm", 0.490, 0.01),
    ("T4 Panda reduced edge", "x38", "X38:panda_1:iso_edgeb:rms_mm", 0.330, 0.01),
    ("T4 Panda per-joint two-slot", "x38", "X38:panda_1:iso_pj:rms_mm", 0.390, 0.01),
    ("T4 Kinova dense", "x38", "X38:kinova:dense:rms_mm", 1.978, 0.01),
    ("T4 Kinova reduced edge", "x38", "X38:kinova:iso_edgeb:rms_mm", 2.325, 0.01),
    ("T4 Kinova per-joint two-slot", "x38", "X38:kinova:iso_pj:rms_mm", 2.397, 0.01),
    ("T4 FR3 dense", "x38", "X38:fr3:dense:rms_mm", 0.287, 0.01),
    ("T4 FR3 reduced edge", "x38", "X38:fr3:iso_edgeb:rms_mm", 0.441, 0.01),
    ("T4 FR3 per-joint two-slot", "x38", "X38:fr3:iso_pj:rms_mm", 0.310, 0.01),
    ("T4 FR3 seq two-slot", "x36seq", "X36:fr3:twoslot:rms_mm", 0.441, 0.01),
    # --- real data ---
    ("X5 c-5s sparse", "x5", "X5:c-5s:QuatIMU-sparse", 1.189, 0.01),
    ("X5 c-5s integrator", "x5", "X5:c-5s:BiasIntegrator", 1.191, 0.01),
    ("X6 LM c-1s", "x6", "X6:c-1s:LM-(M,b)", 1.362, 0.01),
    ("X6 learned c-1s", "x6", "X6:c-1s:Calibrated", 1.371, 0.01),
    ("X7 LawKANArm n=80", "x7", "X7:n80:QuatKANArm.pos_mm_mean", 1.454, 0.02),
    ("X7 PoE-LS n=80", "x7", "X7:n80:PoE-LS.pos_mm_mean", 1.516, 0.02),
    ("X8 GRU b-1s", "x8", "X8:b-1s:GRU h=32", 14.905, 0.1),
    ("X10 FR3 N=80", "x10", "philnad.aggregate.N80.screw_dense.pos_mean.mean", 1.32, 0.05),
    ("X10 UR10 N=80", "x10", "ipa_A.aggregate.N80.screw_dense.pos_mean.mean", 0.43, 0.02),
    ("X10 UR5 N=80", "x10", "pauljac_UR5.aggregate.N80.screw_dense_random.pos_mean.mean", 0.135, 0.01),
    # --- October 2026 validity package ---
    ("G2bc fig N=60 isolated", "g2bc_fig", "G2f:fourier:n60:isolated.pos_mm_mean", 1.165, 0.02),
    ("G2bc fig N=80 isolated", "g2bc_fig", "G2f:fourier:n80:isolated.pos_mm_mean", 1.140, 0.02),
    ("MK1 KUKA isolated", "mk1", "MK1:kuka14:iso:rms_mm", 0.187, 0.02),
    ("MK1 Panda isolated", "mk1", "MK1:panda_1:iso:rms_mm", 0.365, 0.02),
    ("MK1 Kinova isolated", "mk1", "MK1:kinova:iso:rms_mm", 2.269, 0.03),
    ("MK1 FR3 unit dense", "mk1_fr3", "MK1:fr3:dense:rms_mm", 0.287, 0.01),
    ("MK1 FR3 unit isolated", "mk1_fr3", "MK1:fr3:iso:rms_mm", 0.450, 0.02),
    ("MK1 5-seed KUKA isolated", "mk1_fr3", "MK1:kuka14:iso:rms_mm", 0.209, 0.01),
    ("MK1 5-seed Panda isolated", "mk1_fr3", "MK1:panda_1:iso:rms_mm", 0.290, 0.01),
    ("MK1c shuffle KUKA", "mk1c", "MK1c:kuka14:iso:test_rms_mm", 60.47, 0.5),
    ("MK1c shuffle Panda", "mk1c", "MK1c:panda_1:iso:test_rms_mm", 59.28, 0.5),
    ("ABB N=80 isolated", "rc1", "RC1:N80:isolated.per_seed", 1.370, 0.02),
    ("ABB N=160 isolated", "rc1", "RC1:N160:isolated.per_seed", 1.105, 0.02),
    ("ABB N=320 isolated", "rc1", "RC1:N320:isolated.per_seed", 1.057, 0.02),
    ("ABB 5-seed N=40 isolated", "rc1_5seed", "RC1:N40:isolated.per_seed", 1.928, 0.02),
    ("ABB 5-seed N=40 exact", "rc1_5seed", "RC1:N40:exact.per_seed", 1.995, 0.02),
    ("ABB 5-seed N=80 GN", "rc1_5seed", "RC1:N80:gn.per_seed", 1.431, 0.02),
    ("POOL Fisher chi2", "pool", "POOL2:primary.chi2", 58.529, 0.1),
    ("POOL extended chi2", "pool", "POOL2:extended.chi2", 78.059, 0.1),
    ("X6b TinyGC a-short", "x6b", "X6b:a-short:TinyGC.mean", 0.5887, 0.002),
    ("X6b LM a-short", "x6b", "X6b:a-short:LM.mean", 0.5884, 0.002),
    ("X18 coning floor", "x18", "X18:coning_left:floor_deg", 0.1668, 0.005),
    ("X18 coning quad", "x18", "X18:coning_left:quad_deg", 0.1283, 0.005),
    ("X18b constrained gain", "x18b", "X18b:coning_left:gain_pct", 3.15, 0.2),
    ("X19 physics N=20", "x19", "X19:n20:physics.mean", 1.535, 0.02),
    ("X19 generic N=20", "x19", "X19:n20:generic.mean", 2.489, 0.03),
    ("X19 generic N=60", "x19", "X19:n60:generic.mean", 1.165, 0.02),
    ("X20 sign fraction", "x20", "X20:sign_frac", 1.0, 0.001),
    ("X20 removal", "x20", "X20:gain_pct", 97.90, 0.5),
    ("X21 permutation p", "x21", "X21:perm_p", 0.918, 0.02),
    # --- fairness audit (E38): tuned opponents and per-edge lr ---
    ("E38 LawKAN lr1e-2", "e38", "E38:LawKAN lr1e-2", 2.784, 0.02),
    ("E38 LawKAN lr3e-3", "e38", "E38:LawKAN lr3e-3", 2.824, 0.03),
    ("E38 tuned MLP-Quat h=64", "e38", "E38:MLPQuat64 lr3e-3", 3.252, 0.02),
    ("E38 tuned MLP-6D h=16", "e38", "E38:MLP6D16 lr1e-2", 3.997, 0.10),
    ("E38 tuned MLP h=4", "e38", "E38:MLP4 lr1e-2", 14.406, 0.50),
    ("E38 node+MLP h=32 lr1e-3", "e38", "E38:nodeMLP32 lr1e-3", 2.856, 0.02),
    ("E39 LawKAN N=50", "e39", "E39:N50:LawKAN", 4.237, 0.10),
    ("E39 tuned MLP-Quat N=400", "e39", "E39:N400:MLPQuat64_tuned", 3.816, 0.05),
    ("E39 tuned 6D N=800", "e39", "E39:N800:MLP6D16_tuned", 3.997, 0.10),
    ("E40 matched floor", "e40", "E40:floor", 2.733, 0.01),
    ("E40 gap to floor", "e40", "E40:gap", 0.0504, 0.01),
    # --- revision / pole-fix branch ---
    ("X37 dense wd0 spread", "x37", "X37:c-5s:dense:wd0.spread", 73.94, 0.1),
    ("X37 dense wd3e-3 spread", "x37", "X37:c-5s:dense:wd0.003.spread", 76.77, 0.1),
    ("X37 sparse spread", "x37", "X37:c-5s:sparse:wd0.spread", 0.0167, 0.002),
    ("X46 N80 MLP ep1000", "x46", "X46:A1:N80:mlp:ep1000", 1.242, 0.01),
    ("X46 N80 MLP ep4000", "x46", "X46:A1:N80:mlp:ep4000", 1.329, 0.01),
    ("X46 N60 MLP ep2000", "x46", "X46:A1:N60:mlp:ep2000", 1.250, 0.01),
    ("X46 N60 MLP ep4000", "x46", "X46:A1:N60:mlp:ep4000", 1.352, 0.01),
    ("X52 base N20 max", "x52", "X52:base:N20.max_mean", 24.12, 0.1),
    ("X52 clip N20 max", "x52", "X52:clip:N20.max_mean", 24.12, 0.1),
    ("X52 lr1e3 N20 max", "x52", "X52:lr1e3:N20.max_mean", 4.918, 0.05),
    ("X52 base N20 final", "x52", "X52:base:N20.final_mean", 1.454, 0.01),
    ("X54 base max", "x54", "results.base.max", 26.04, 0.05),
    ("X54 base final", "x54", "results.base.final", 1.09, 0.01),
    ("X54 lr1e3 max", "x54", "results.lr1e3.max", 4.73, 0.05),
    ("X55 q14 max", "x55", "X53-fix.max", 13.18, 0.05),
    ("X55 q14 final", "x55", "X53-fix.final", 1.285, 0.01),
    ("X57 k=n final", "x57", "cells.k=n.final", 1.604, 0.01),
    ("X57 canon max", "x57", "cells.canon.max", 9.57, 0.05),
    ("X58 k=20 final", "x58", "cells.k=20.final", 1.285, 0.01),
    ("X59 k=20 max", "x59", "cells.k=20.max", 1.30, 0.02),
    ("X59 k=20 final", "x59", "cells.k=20.final", 1.238, 0.01),
    ("X60 geo final", "x60", "cells.geo.final", 1.272, 0.01),
    ("X60 chord final", "x60", "cells.chord.final", 1.391, 0.01),
    ("X47 KUKA joint", "x47", "X47:kuka14.joint_mean", 0.185, 0.005),
    ("X47 KUKA seq", "x47", "X47:kuka14.seq_mean", 0.166, 0.005),
    ("X47 Kinova joint", "x47", "X47:kinova.joint_mean", 2.145, 0.01),
    ("X47 Kinova seq", "x47", "X47:kinova.seq_mean", 2.325, 0.01),
    ("X47 FR3 joint", "x47", "X47:fr3.joint_mean", 0.502, 0.01),
    ("X47 FR3 seq", "x47", "X47:fr3.seq_mean", 0.441, 0.01),
    ("G3b two-slot orthogonal share", "g3b", "G3:fr3t:isolated.orthogonal_share", 0.8798, 0.01),
    ("X36 seq KUKA", "x36seq", "X36:kuka14:twoslot:rms_mm", 0.166, 0.005),
    ("X36 seq Panda", "x36seq", "X36:panda_1:twoslot:rms_mm", 0.330, 0.01),
    ("X36 seq Kinova", "x36seq", "X36:kinova:twoslot:rms_mm", 2.325, 0.01),
    ("X38b Kinova kernel", "x38b", "X38b:kinova:iso_pj_kernel:rms_mm", 1.908, 0.01),
    ("X38c KUKA global", "x38c", "X38c:kuka14:iso_pj_global:rms_mm", 0.198, 0.005),
    ("X38c KUKA kernel", "x38c", "X38c:kuka14:iso_pj_kernel:rms_mm", 0.218, 0.005),
]


FIELDS = ("mean", "geo_mean", "rot_deg_mean", "pos_mm_mean", "text")


def walk(d, path):
    """Resolve a dump entry. Accepts either a plain key (may contain dots,
    e.g. 'E3.4:QuatKAN:both_poles') or '<key>.<field>' where field is one of
    FIELDS; otherwise falls back to the default priority order."""
    if path in d:
        cur = d[path]
    else:
        key, _, field = path.rpartition(".")
        if key in d and isinstance(d[key], dict) and field in d[key]:
            cur = d[key].get(field)
            return float(cur) if isinstance(cur, (int, float)) else cur
        if key in d and field in FIELDS:
            cur = d[key].get(field)
            return float(cur) if isinstance(cur, (int, float)) else cur
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
    if isinstance(cur, (int, float)):
        return float(cur)
    if isinstance(cur, dict):
        for f in FIELDS:
            if f in cur:
                v = cur[f]
                return float(v) if isinstance(v, (int, float)) else v
    if isinstance(cur, list):
        return cur
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_repo = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=default_repo)
    args = ap.parse_args()
    jdir = os.path.join(args.repo, "repro", "json")
    man_path = os.path.join(args.repo, "repro", "canonical.json")
    man = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}

    cache = {}
    fails, missing, pending = [], [], []
    print(f"{'check':38s} {'paper':>10s} {'dump':>10s} {'diff':>9s}  verdict")
    for label, block, key, expected, tol in CHECKS:
        entry = man.get(block, {})
        fname = entry.get("file")
        if fname is None:
            pending.append((label, f"no canonical dump for '{block}'"))
            print(f"{label:38s} {expected:10.4g} {'--':>10s} {'--':>9s}  PENDING ({block})")
            continue
        fpath = os.path.join(jdir, fname)
        if not os.path.exists(fpath):
            missing.append((label, f"{fname} not found"))
            print(f"{label:38s} {expected:10.4g} {'--':>10s} {'--':>9s}  MISSING ({fname})")
            continue
        if fpath not in cache:
            cache[fpath] = json.load(open(fpath, encoding="utf-8"))
        got = walk(cache[fpath], key)
        if got is None:
            missing.append((label, f"key '{key}' not in {fname}"))
            print(f"{label:38s} {expected:10.4g} {'--':>10s} {'--':>9s}  KEY MISSING")
            continue
        if isinstance(got, list) and got and all(isinstance(x, (int, float)) for x in got):
            got = sum(got) / len(got)
        if isinstance(got, str):
            print(f"{label:38s} {expected!s:>10s} {got[:10]:>10s} {'--':>9s}  FAIL (text)")
            fails.append((label, block, expected, got, tol))
            continue
        diff = abs(got - expected)
        ok = diff <= tol
        print(f"{label:38s} {expected:10.4g} {got:10.4g} {diff:9.2g}  {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append((label, block, expected, got, tol))

    print()
    if pending:
        print(f"pending dumps: {len(pending)}")
        for label, why in pending:
            print(f"  - {label}: {why}")
    if missing:
        print(f"missing dumps/keys: {len(missing)}")
        for label, why in missing:
            print(f"  - {label}: {why}")
    if fails:
        print(f"FAILED: {len(fails)}")
        for label, block, exp, got, tol in fails:
            print(f"  - {label} [{block}]: paper {exp:g}, dump {got:g} (tol {tol:g})")
        return 1
    print(f"all {len(CHECKS) - len(missing) - len(pending)} comparable checks OK"
          + (f", {len(pending)} pending (no canonical dump yet)" if pending else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
