#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
S3_matched_mlp.py — parameter-matched residual-MLP control for the learned node.

Question: is the X13 gain (relaxed law, 396p, beating dense/GN at N>=60) about
the *placement* (a correction to the composition law) or just about capacity?
This runs X7.MLPResidual with a hidden width chosen so that its parameter count
matches the relaxed node (~384p vs 396p), on the same splits/seeds/protocol.

Registered predictions (plan 2026-09-29, before the run):
  P-S3a: relaxed node (X13, 396p) beats the matched MLP (384p) at N>=60;
  P-S3b: at N=20 the matched MLP is >= relaxed (less structural bias);
  P-S3c (boundary): if matched MLP ~ relaxed at N>=60, the gain is capacity,
        not the learned-law placement.

Protocol: X7 splits rng(1000+seed), seeds 0-4, N=20/40/60/80, E34.train_recipe
(4000 epochs, lr 5e-3, cosine2, clip=0, ang_weight=1), position metric.
References: X13 relaxed/nodeonly per-seed, dense/GN from the bench dump.

Output: repro/json/S3_matched_mlp.<ts>.json via repro/dump.
Run:  python S3_matched_mlp.py [--quick]
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


def load_latest(pattern):
    cands = [p for p in glob.glob(os.path.join(_REPO, 'repro', 'json', pattern))
             if 'smoke' not in p]
    return json.load(open(sorted(cands, key=os.path.getmtime)[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--hidden', type=int, default=13)
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

    m0 = X7.MLPResidual(C, E, hidden=args.hidden)
    n_params = sum(p.numel() for p in m0.parameters())
    print(f'[S3] MLPResidual h={args.hidden}: {n_params} params (relaxed node 396)',
          flush=True)

    bench = load_latest('X7_bench_main.*.json')
    x13 = load_latest('X13_learned_node.*.json')
    ref = {n: {k: np.array(bench[f'X7bench:n{n}:{k}']['per_seed'])[:, 0]
               for k in ('ours', 'gn')} for n in (20, 40, 60, 80)}
    for n in (20, 40, 60, 80):
        ref[n]['relaxed'] = np.array(x13[f'X13:n{n}:relaxed_dense']['per_seed'])

    res = {n: [] for n in sizes}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            m = X7.MLPResidual(C, E, hidden=args.hidden)
            m, _ = E34.train_recipe(m, q_tr, oq_tr, ot_tr, args.epochs,
                                    5e-3, s, clip=0.0, ang_weight=1.0,
                                    warm_epochs=0, loss_form='cosine2')
            res[n_tr].append(GN.eval_pos(m, q_te, oq_te, ot_te))
            print(f'[S3 seed {s} n={n_tr}] matched_mlp {res[n_tr][-1][0]:.3f} '
                  f'(relaxed {ref[n_tr]["relaxed"][s]:.3f})', flush=True)
        print(f'[S3] seed {s} done', flush=True)

    for n_tr in sizes:
        a = np.array([v[0] for v in res[n_tr]])
        RESULTS[f'S3:n{n_tr}:matched_mlp'] = dict(
            pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
            params=int(n_params), per_seed=[float(x) for x in a])
        for rname in ('relaxed', 'ours', 'gn'):
            rr = ref[n_tr][rname]
            if len(a) != len(rr) or len(a) < 2:
                continue
            t, p = stats.ttest_rel(a, rr)
            RESULTS[f'S3:paired:n{n_tr}:matched_mlp_vs_{rname}'] = dict(
                diff=float(a.mean() - rr.mean()), t=float(t), p=float(p))
    RESULTS['S3:params'] = dict(matched_mlp=int(n_params), relaxed=396)
    p = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                  seeds=args.seeds, epochs=args.epochs)
    print('dump ->', p)


if __name__ == '__main__':
    main()
