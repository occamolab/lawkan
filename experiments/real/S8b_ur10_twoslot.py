#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
S8b_ur10_twoslot.py — S8 (UR10 IPA) relaxed law under the FINAL two-slot head
(registered before run).

S8 is the identifiability failure mode (UR10, full 6D): the joint optimum is
compensated rather than flat. Here the relaxed head is the two-slot, so R-3
(compensation reproduces) is checked on the final head.

Registered prediction (plan R-3): the two-slot reproduces S8 --- UR10 does not beat
the exact/GN references (identifiability boundary unchanged).

Output: repro/json/S8b_ur10_twoslot.<ts>.json via repro/dump.
Run:  python S8b_ur10_twoslot.py [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from scipy.cluster.vq import kmeans2

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import X10_fk_calib_real as X10             # noqa: E402
import S8_ur10_relaxed as S8                # noqa: E402
import X36_recount_socket as X36            # noqa: E402
from repro.dump import save_dump           # noqa: E402

DT = torch.float64
RESULTS = {}


def collect14(m, q):
    fs = []

    def hook(mod, inp, out):
        fs.append(inp[0].detach().cpu())

    h = m.psi.register_forward_hook(hook)
    with torch.no_grad():
        m(torch.tensor(q, dtype=DT))
    h.remove()
    return torch.cat(fs, 0)


def make_2slot(xi, M0, q_tr, seed, k=20):
    m = S8.ScrewCalibRelax(xi, M0, mode='dense')
    feats = collect14(m, q_tr)
    assert feats.shape[1] == 14, ('ur10 feat dim', feats.shape)
    cen, _ = kmeans2(feats.numpy(), k, minit='++', seed=seed)
    centers = torch.tensor(cen, dtype=DT)
    dc = torch.cdist(centers, centers)
    bw = float(torch.median(dc[dc > 0]))
    m.psi = X36.TwoSlotPsiSocket(14, centers, bw)
    return m


def fit_model(m, q_tr, T_obs2_tr, tool_init, steps, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        m.tool.copy_(torch.tensor(tool_init, dtype=m.tool.dtype))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    qt = torch.tensor(q_tr, dtype=DT)
    pt = torch.tensor(T_obs2_tr[:, :3, 3], dtype=DT)
    for _ in range(steps):
        opt.zero_grad()
        Tp = m(qt)
        loss = 10.0 * ((Tp[:, :3, 3] - pt) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()
        sched.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.join(_REPO, 'data'))
    ap.add_argument('--series', default='A')
    ap.add_argument('--sizes', default='20,40,80')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed_start', type=int, default=50)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.seeds, args.steps = '80', 1, 200
    t0 = time.time()

    data = X10.load_ipa(args.data_dir, args.series)
    robot, xi, M0 = X10.get_robot_tensors(data['robot'])
    q_all, T_all = data['q'], data['T']
    n = len(q_all)
    Tf_all = X10.fk_np(robot, q_all)
    sizes = [int(s) for s in args.sizes.split(',')]
    canon, gn = S8.load_refs()

    for s in range(args.seed_start, args.seed_start + args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(n)
        for N in sizes:
            tr, te = perm[:N], perm[N:]
            B0, C0 = X10.gap_init(Tf_all[tr], T_all[tr], data['goal'][tr], full=True)
            B, C = X10.fit_gaps_torch(Tf_all[tr], T_all[tr], B0, C0,
                                      steps=args.steps, full=True, seed=s)
            T_obs2_tr = np.array([np.linalg.inv(B) @ t for t in T_all[tr]])
            T_obs2_te = np.array([np.linalg.inv(B) @ t for t in T_all[te]])
            m = make_2slot(xi, M0, q_all[tr], s)
            fit_model(m, q_all[tr], T_obs2_tr, X10.log_se3_np(C), args.steps, s)
            rec = X10.pose_metrics(X10.apply_sc(m, q_all[te]), T_obs2_te, full=True)
            exact = canon['per_seed']['seed%d' % s]['N%d' % N]['screw_dense']['pos_mean']
            gnm = gn['per_seed']['seed%d' % s]['N%d' % N]['poe_ls']['pos_mean']
            RESULTS[f'S8b:seed{s}:N{N}:twoslot'] = float(rec['pos_mean'])
            RESULTS[f'S8b:seed{s}:N{N}:rot'] = float(rec['rot_mean'])
            RESULTS[f'S8b:seed{s}:N{N}:exact'] = float(exact)
            RESULTS[f'S8b:seed{s}:N{N}:gn'] = float(gnm)
            print(f'[S8b seed {s} N={N}] twoslot {rec["pos_mean"]:.3f} '
                  f'exact {exact:.3f} gn {gnm:.3f}', flush=True)

    RESULTS['S8b:_meta_local'] = dict(wall_s=time.time() - t0)
    out = save_dump(RESULTS, __file__, quick=args.quick, args=args,
                    seeds=args.seeds, epochs=args.steps)
    print('dump:', out)


if __name__ == '__main__':
    main()
