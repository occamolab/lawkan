#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
X26c_fr3_lambda.py — L-4 on FR3: sweep the L2 weight-decay of the isolated psi
across training size N (registered before run).

Kinova's socket protocol has no N knob, so the parameters-per-pose question (L-4)
is tested on the FR3 arm, where sub-sampling poses is natural (X7 splits). The
isolated Fourier-psi (linear, no tanh) is trained with an L2 penalty lambda in
{0,1e-4,3e-4,1e-3,3e-3,1e-2} at N=20 and N=80; prediction L-4: the fixing lambda*
is smaller at N=80 than at N=20 (2.3 vs 9.3 parameters/pose).

Registered prediction (plan 2026-10-13, before run):
  L-4: lambda*(N=80) < lambda*(N=20); the window exists at both N but shifts.

Output: repro/json/X26c_fr3_lambda.<ts>.json.
Run:  python X26c_fr3_lambda.py [--quick]
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
from repro.dump import save_dump            # noqa: E402

DT = E34.DT
RESULTS = {}


def phase2_l2(exact, q_tr, oq_tr, ot_tr, epochs, lr, seed, l2=0.0):
    m = copy.deepcopy(exact)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)
    for _ in range(epochs):
        opt.zero_grad()
        q_pred, t_pred = m(q_tr)
        dp = (t_pred - ot_tr).norm(dim=-1)
        dot = (q_pred * oq_tr).sum(-1).abs()
        loss = (dp ** 2).mean() + 2.0 * (1.0 - dot.clamp(max=1 - 1e-7)).mean()
        if l2:
            loss = loss + l2 * sum((p ** 2).sum() for p in m.psi.parameters())
        loss.backward()
        opt.step()
        sched.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_REPO, 'data', 'calib_data.pickle'))
    ap.add_argument('--sizes', default='20,40,60,80')
    ap.add_argument('--l2-list', default='0,1e-4,3e-4,1e-3,3e-3,1e-2')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=5e-3)
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
    l2s = [float(x) for x in args.l2_list.split(',')]

    res = {(n, x): [] for n in sizes for x in l2s}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n_total)
        for n_tr in sizes:
            tr, te = perm[:n_tr], perm[n_tr:]
            q_tr, oq_tr, ot_tr = q_all[tr], oq_all[tr], ot_all[tr]
            q_te, oq_te, ot_te = q_all[te], oq_all[te], ot_all[te]
            ex = X24.phase1_exact(C, E, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s)
            for x in l2s:
                m = phase2_l2(ex, q_tr, oq_tr, ot_tr, args.epochs, args.lr, s, l2=x)
                res[(n_tr, x)].append(GN.eval_pos(m, q_te, oq_te, ot_te)[0])
            print(f'[X26c seed {s} n={n_tr}] ' +
                  ' '.join(f'l2={x:g}:{res[(n_tr, x)][-1]:.3f}' for x in l2s), flush=True)
        print(f'[X26c] seed {s} done', flush=True)

    for n in sizes:
        for x in l2s:
            a = np.array(res[(n, x)])
            RESULTS[f'X26c:n{n}:l2_{x:g}'] = dict(
                pos_mm_mean=float(a.mean()), pos_mm_std=float(a.std()),
                per_seed=[float(v) for v in a])
    RESULTS['X26c:protocol'] = dict(sizes=sizes, l2=l2s, seeds=args.seeds,
                                    epochs=args.epochs, lr=args.lr, psi='fourier',
                                    split='X7 rng(1000+seed)')
    RESULTS['_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.epochs)
    print('dump:', out)


if __name__ == '__main__':
    main()
