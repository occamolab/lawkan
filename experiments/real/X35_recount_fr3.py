#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X35_recount_fr3.py — recount the FR3 relaxed-psi rows (Table 3) under the final
two-slot psi head (registered R-1..R-4).

Two-slot head: psi(f) = 0.1*(W.f + sum_k w_k exp(-||f-c_k||^2/2 bw^2)), both slots
zero-init, global slot dosed per N (X26c), kernel undosed. Centers = k-means of the
pair features seen on the frozen exact edge.

Variants: isolated (frozen exact edge, only psi trained) and joint (edge + psi
trained together; the compensation control). FR3, X7 splits, seeds 0-4, N=20/40/60/80.

Registered (plan 2026-10-14): R-1 every recounted cell <= old (X33/X34 gave the
small-N gain already); R-4 the two-slot matches/beats the Fourier head.

Output: repro/json/X35_recount_fr3.<ts>.json.
Run:  python X35_recount_fr3.py [--quick]
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X7_robot_calibration as X7          # noqa: E402
import E34_x7_recipe as E34                 # noqa: E402
import X14_fourier_law as X14               # noqa: E402
import X7_gn_ls as GN                       # noqa: E402
import X24_fr3_gate as X24                  # noqa: E402
import X34_fr3_kernel as X34                # noqa: E402
import X33_fr3_twoslot as X33               # noqa: E402
from scipy import stats
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}
DOSE = {20: 3e-3, 40: 3e-4, 60: 1e-4, 80: 0.0}


def train_all(m, q_tr, oq_tr, ot_tr, epochs, lr, seed, dose=0.0):
    """Train every parameter (joint), with L2 only on the global slot."""
    for p in m.parameters():
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
    for _ in range(epochs):
        opt.zero_grad()
        q_pred, t_pred = m(q_tr)
        dp = (t_pred - ot_tr).norm(dim=-1)
        dot = (q_pred * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        if dose:
            loss = loss + dose * (m.psi.global_lin.weight ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--centers', type=int, default=20)
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
    from scipy.cluster.vq import kmeans2

    variants = ('isolated', 'joint')
    res = {n: {v: [] for v in variants} for n in sizes}

    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            dose = DOSE.get(int(n_tr), 0.0)
            exact = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)
            feats = X34.collect_feats(copy.deepcopy(exact), q_tr)
            cen, _ = kmeans2(feats.numpy(), args.centers, minit='++', seed=s)
            centers = torch.tensor(cen, dtype=DT)
            dat = torch.cdist(centers, centers)
            bw = float(torch.median(dat[dat > 0])) if (dat > 0).any() else 1.0

            # isolated: frozen edge, two-slot psi
            mi = X33.make_two_slot(exact, centers, bw)
            X33.train(mi, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s,
                      slots={'global', 'kernel'}, l2_global=dose)
            res[n_tr]['isolated'].append(GN.eval_pos(mi, q_te, oq_te, ot_te)[0])

            # joint: edge + two-slot psi together
            mj = X14.FourierLawArm(C, E, edge_dense=True)
            mj.psi = X33.TwoSlotPsi(centers, bw)
            train_all(mj, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s, dose=dose)
            res[n_tr]['joint'].append(GN.eval_pos(mj, q_te, oq_te, ot_te)[0])
            print(f'[X35 seed {s} n={n_tr}] isolated {res[n_tr]["isolated"][-1]:.3f} '
                  f'joint {res[n_tr]["joint"][-1]:.3f}', flush=True)
        print(f'[X35] seed {s} done', flush=True)

    for n in sizes:
        for v in variants:
            a = np.array(res[n][v])
            RESULTS[f'X35:n{n}:{v}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                per_seed=[float(x) for x in a])
    RESULTS['X35:protocol'] = dict(sizes=sizes, seeds=args.seeds, epochs=args.epochs,
                                   lr=args.lr, centers=args.centers, dose=DOSE,
                                   head='two-slot', tag='final-parameterization')
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
