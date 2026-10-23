#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X22_learned_baseline.py — one strong learned-calibration opponent for the FR3
law-based calibration (reviewer request: no strong learned baseline).

Baseline: X7.MLPResidual (nominal FK with a fitted base + an end-effector-frame
MLP residual), capacity/lr selected on a validation split carved from training,
then refit on the full training set. Lean grid (hidden {128,256} x lr {1e-2,1e-3})
with the test-oracle bound only at N>=60. E34.train_recipe protocol, position
metric (mm), same X7 splits rng(1000+seed), seeds 0-4, N=20/40/60/80.

Registered predictions (plan 2026-10-13, before the run):
  P-LB1: the validation-selected MLP does not beat the law model (relaxed/ours)
         at N>=60 (paired, 5 seeds);
  P-LB2: at N=20 the validation-selected MLP is not better than the exact model;
  P-LB3 (boundary): even the test-oracle best MLP does not beat the law model at
         N>=60 -- the gain is structural, not the absence of strong opponents.

References on the same splits: X7 bench (ours = exact dense, gn = X7_gn_ls),
X13 relaxed_dense.

Output: repro/json/X22_learned_baseline.<ts>.json via repro/dump.
Run:  python X22_learned_baseline.py [--quick]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}

HIDDENS = (128, 256)
LRS = (1e-2, 1e-3)


def load_latest(pattern):
    cands = [p for p in glob.glob(os.path.join(_REPO, 'repro', 'json', pattern))
             if 'smoke' not in p]
    return json.load(open(sorted(cands, key=os.path.getmtime)[-1]))


def fit_eval(C, E, q_tr, oq_tr, ot_tr, q_ev, oq_ev, ot_ev,
             hidden, lr, seed, epochs):
    """Train a residual MLP on (q_tr, oq_tr, ot_tr); return held-out pos (mm)."""
    m = X7.MLPResidual(C, E, hidden=hidden)
    m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, epochs, lr, seed,
                            clip=0.0, ang_weight=1.0, warm_epochs=0,
                            loss_form='cosine2')
    return GN.eval_pos(m, q_ev, oq_ev, ot_ev)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--val-frac', type=float, default=0.25)
    ap.add_argument('--oracle-min', type=int, default=60,
                    help='compute the test-oracle bound only for N >= this')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.epochs = '20', 1, 300
    t0 = time.time()

    q_np, obs_np = X7.load_philnad_dataset(args.data)
    C, E = X7.extract_nominal_chain()
    q_all = torch.tensor(q_np, dtype=DT)
    oq_all = X7.mat_to_quat(torch.tensor(obs_np[:, :3, :3], dtype=DT))
    ot_all = torch.tensor(obs_np[:, :3, 3], dtype=DT)
    n_total = len(q_np)
    sizes = [int(s) for s in args.sizes.split(',')]

    bench = load_latest('X7_bench_main.*.json')
    x13 = load_latest('X13_learned_node.*.json')
    ref = {}
    for n in (20, 40, 60, 80):
        ref[n] = {k: np.array(bench[f'X7bench:n{n}:{k}']['per_seed'])[:, 0]
                  for k in ('ours', 'gn')}
        ref[n]['relaxed'] = np.array(x13[f'X13:n{n}:relaxed_dense']['per_seed'])

    grid = [(h, lr) for h in HIDDENS for lr in LRS]
    res = {n: {'selected': [], 'oracle': [], 'cfg': []} for n in sizes}

    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            n_val = max(2, int(round(args.val_frac * n_tr)))
            tr_fit, tr_val = tr[:n_tr - n_val], tr[n_tr - n_val:]

            best, best_cfg = None, None
            for (h, lr) in grid:
                v = fit_eval(C, E, q_all[tr_fit], oq_all[tr_fit], ot_all[tr_fit],
                             q_all[tr_val], oq_all[tr_val], ot_all[tr_val],
                             h, lr, s, args.epochs)
                if best is None or v < best:
                    best, best_cfg = v, (h, lr)

            h, lr = best_cfg
            te_sel = fit_eval(C, E, q_all[tr], oq_all[tr], ot_all[tr],
                              q_all[te], oq_all[te], ot_all[te],
                              h, lr, s, args.epochs)
            te_or = None
            if n_tr >= args.oracle_min:
                te_or = min(fit_eval(C, E, q_all[tr], oq_all[tr], ot_all[tr],
                                     q_all[te], oq_all[te], ot_all[te],
                                     hh, ll, s, args.epochs)
                            for (hh, ll) in grid)

            res[n_tr]['selected'].append(te_sel)
            res[n_tr]['oracle'].append(te_or)
            res[n_tr]['cfg'].append(best_cfg)
            or_txt = f'oracle {te_or:.3f} ' if te_or is not None else ''
            print(f'[X22 seed {s} n={n_tr}] selected {te_sel:.3f} '
                  f'(cfg h={h},lr={lr}) {or_txt}| '
                  f'ours {ref[n_tr]["ours"][s]:.3f} '
                  f'relaxed {ref[n_tr]["relaxed"][s]:.3f}', flush=True)
        print(f'[X22] seed {s} done', flush=True)

    for n_tr in sizes:
        for key in ('selected', 'oracle'):
            if any(v is None for v in res[n_tr][key]):
                continue
            a = np.array(res[n_tr][key])
            RESULTS[f'X22:n{n_tr}:{key}_mlp'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                per_seed=[float(x) for x in a])
            for rname in ('ours', 'gn', 'relaxed'):
                rr = np.array(ref[n_tr][rname])
                if len(a) != len(rr) or len(a) < 2:
                    continue
                t, p = stats.ttest_rel(a, rr)
                RESULTS[f'X22:paired:n{n_tr}:{key}_vs_{rname}'] = dict(
                    diff=float(a.mean() - rr.mean()), t=float(t), p=float(p))
        RESULTS[f'X22:cfg:n{n_tr}'] = dict(
            selected=[[float(c[0]), float(c[1])] for c in res[n_tr]['cfg']])
    RESULTS['X22:grid'] = dict(hidden=list(HIDDENS), lr=list(LRS),
                               epochs=args.epochs)
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('dump ->', p)


if __name__ == '__main__':
    main()
